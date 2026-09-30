"""Subnet device discovery: ICMP sweep -> ARP table -> reverse DNS -> TCP fingerprint -> classify.

Uses only the OS ``ping`` and ``arp``/``ip`` binaries plus plain TCP connects, so it
runs unprivileged on Windows and Linux. Only sweep networks you are authorised to scan.
"""

from __future__ import annotations

import ipaddress
import logging
import re
import socket
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Optional, Protocol

from . import netinfo
from .models import Device, DeviceClass, normalize_mac
from .oui import OuiDatabase

log = logging.getLogger(__name__)

# Small, targeted probe set; each port is a strong class signal.
FINGERPRINT_PORTS: tuple[int, ...] = (22, 80, 135, 139, 443, 445, 548, 631, 3389, 5900, 7000, 9100, 62078)

_MOBILE_HOST_KEYS = ("iphone", "ipad", "android", "galaxy", "pixel", "oneplus", "redmi", "huawei", "phone", "oppo", "vivo")
_PC_HOST_KEYS = (
    "desktop", "laptop", "-pc", "pc-", "workstation", "macbook", "thinkpad", "latitude", "precision",
    "elitebook", "probook", "surface", "-nb", "-ws", "-lt-", "-lt", "-dt-", "-wks",
)
_SERVER_HOST_KEYS = ("srv", "server", "nas", "db", "vm-", "-vm", "esxi", "proxmox", "docker")
_PRINTER_HOST_KEYS = ("printer", "print", "npi", "brn", "epson", "canon", "hp-", "mfp")


class DeviceScanner(Protocol):
    def scan(self) -> list[Device]: ...


@dataclass
class ScanConfig:
    subnet: ipaddress.IPv4Network
    local_ip: str
    local_mac: Optional[str]
    gateway: Optional[str]
    ping_timeout_ms: int = 400
    port_timeout_s: float = 0.35
    workers: int = 64
    probe_ports: bool = True
    resolve_hostnames: bool = True

    def __post_init__(self) -> None:
        if not 50 <= self.ping_timeout_ms <= 5000:
            raise ValueError("ping_timeout_ms must be within 50..5000")
        if not 1 <= self.workers <= 256:
            raise ValueError("workers must be within 1..256")


# --------------------------------------------------------------------------- primitives

_RTT_RE = re.compile(r"time[=<]\s*([\d.]+)\s*ms", re.I)


def ping(ip: str, timeout_ms: int) -> Optional[float]:
    """Return RTT in ms if the host answered ICMP echo, else None."""
    if netinfo.IS_WINDOWS:
        args = ["ping", "-n", "1", "-w", str(timeout_ms), ip]
    else:
        args = ["ping", "-c", "1", "-W", str(max(1, round(timeout_ms / 1000))), ip]
    try:
        out = netinfo.run_cmd(args, timeout=timeout_ms / 1000 + 2.0)
    except RuntimeError:
        return None
    replied = ("TTL=" in out.upper()) or ("bytes from" in out)
    if not replied:
        return None
    m = _RTT_RE.search(out)
    return float(m.group(1)) if m else 0.0


def read_arp_table() -> dict[str, str]:
    """Return ``{ip: mac}`` from the OS neighbour cache, excluding broadcast/multicast entries."""
    table: dict[str, str] = {}
    try:
        if netinfo.IS_WINDOWS:
            out = netinfo.run_cmd(["arp", "-a"])
            pattern = re.compile(r"(\d+\.\d+\.\d+\.\d+)\s+([0-9a-fA-F]{2}(?:[-:][0-9a-fA-F]{2}){5})\s+(\w+)")
            for ip, mac, _kind in pattern.findall(out):
                table[ip] = normalize_mac(mac)
        else:
            out = netinfo.run_cmd(["ip", "-4", "neigh", "show"])
            pattern = re.compile(r"(\d+\.\d+\.\d+\.\d+)\s+dev\s+\S+\s+lladdr\s+([0-9a-fA-F:]{17})\s+(\w+)")
            for ip, mac, state in pattern.findall(out):
                if state.upper() != "FAILED":
                    table[ip] = normalize_mac(mac)
    except (RuntimeError, ValueError) as exc:
        log.warning("ARP table read failed: %s", exc)
    return {
        ip: mac
        for ip, mac in table.items()
        if mac != "ff:ff:ff:ff:ff:ff" and not mac.startswith("01:00:5e") and not mac.startswith("33:33")
    }


def tcp_open(ip: str, port: int, timeout: float) -> bool:
    try:
        with socket.create_connection((ip, port), timeout=timeout):
            return True
    except (OSError, socket.timeout):
        return False


def reverse_dns(ip: str, timeout: float = 1.5) -> Optional[str]:
    """gethostbyaddr has no timeout; bound it with a worker thread."""
    with ThreadPoolExecutor(max_workers=1) as pool:
        fut = pool.submit(socket.gethostbyaddr, ip)
        try:
            return fut.result(timeout=timeout)[0]
        except Exception:  # noqa: BLE001 - herror/timeout/gaierror all mean "no name"
            return None


# --------------------------------------------------------------------------- classification

def classify(device: Device) -> DeviceClass:
    """Rule-ordered classifier. Strong evidence (gateway, service ports) beats vendor hints."""
    host = (device.hostname or "").lower()
    ports = set(device.open_ports)
    hint = device.vendor_hint or ""

    if device.is_gateway:
        return DeviceClass.ROUTER
    if device.is_self:
        return DeviceClass.WORKSTATION
    if 62078 in ports or any(k in host for k in _MOBILE_HOST_KEYS):
        return DeviceClass.MOBILE
    if ports & {9100, 631} or any(k in host for k in _PRINTER_HOST_KEYS):
        return DeviceClass.PRINTER
    if ports & {3389, 445, 135, 139, 5900}:
        return DeviceClass.SERVER if any(k in host for k in _SERVER_HOST_KEYS) else DeviceClass.WORKSTATION
    if any(k in host for k in _PC_HOST_KEYS):
        return DeviceClass.WORKSTATION
    if hint == "network" or (ports & {80, 443} and not ports & {22} and hint == "network"):
        return DeviceClass.ROUTER
    if 22 in ports or any(k in host for k in _SERVER_HOST_KEYS) or hint == "virtual":
        return DeviceClass.SERVER
    if hint == "mobile":
        # Apple prefixes are shared by Macs and iPhones; AFP/AirPlay ports indicate a Mac.
        if ports & {548, 7000, 5900}:
            return DeviceClass.WORKSTATION
        return DeviceClass.MOBILE
    if hint == "pc":
        return DeviceClass.WORKSTATION
    if hint == "printer":
        return DeviceClass.PRINTER
    if hint == "iot" or ports & {80, 443}:
        return DeviceClass.IOT
    return DeviceClass.UNKNOWN


# --------------------------------------------------------------------------- scanner

class SubnetScanner:
    def __init__(self, config: ScanConfig, oui: OuiDatabase) -> None:
        self.config = config
        self.oui = oui

    def scan(self) -> list[Device]:
        cfg = self.config
        started = time.monotonic()
        hosts = [str(h) for h in cfg.subnet.hosts()]
        log.info("sweeping %s (%d hosts, %d workers)", cfg.subnet, len(hosts), cfg.workers)

        rtts: dict[str, float] = {}
        with ThreadPoolExecutor(max_workers=cfg.workers) as pool:
            futures = {pool.submit(ping, ip, cfg.ping_timeout_ms): ip for ip in hosts}
            for fut in as_completed(futures):
                ip = futures[fut]
                try:
                    rtt = fut.result()
                except Exception as exc:  # noqa: BLE001
                    log.debug("ping %s raised: %s", ip, exc)
                    continue
                if rtt is not None:
                    rtts[ip] = rtt

        arp = read_arp_table()
        # Hosts that answered ARP but dropped ICMP (common for phones) are still alive.
        alive = set(rtts) | {ip for ip in arp if ipaddress.ip_address(ip) in cfg.subnet}
        alive.discard(cfg.local_ip)
        log.info("alive: %d (icmp=%d, arp-only=%d)", len(alive), len(rtts), len(alive) - len(set(rtts) - {cfg.local_ip}))

        devices: list[Device] = [self._self_device()]
        with ThreadPoolExecutor(max_workers=min(cfg.workers, 32)) as pool:
            futures = {pool.submit(self._enrich, ip, rtts.get(ip), arp.get(ip)): ip for ip in sorted(alive, key=ipaddress.ip_address)}
            for fut in as_completed(futures):
                try:
                    devices.append(fut.result())
                except Exception as exc:  # noqa: BLE001
                    log.warning("enrich %s failed: %s", futures[fut], exc)

        devices.sort(key=lambda d: ipaddress.ip_address(d.ip))
        log.info("scan complete: %d devices in %.1fs", len(devices), time.monotonic() - started)
        return devices

    def _self_device(self) -> Device:
        vendor, hint = self.oui.lookup(self.config.local_mac)
        dev = Device(
            ip=self.config.local_ip,
            mac=self.config.local_mac,
            hostname=socket.gethostname(),
            vendor=vendor,
            vendor_hint=hint,
            rtt_ms=0.0,
            is_self=True,
        )
        dev.device_class = classify(dev)
        return dev

    def _enrich(self, ip: str, rtt: Optional[float], mac: Optional[str]) -> Device:
        cfg = self.config
        vendor, hint = self.oui.lookup(mac)
        dev = Device(ip=ip, mac=mac, vendor=vendor, vendor_hint=hint, rtt_ms=rtt, is_gateway=(ip == cfg.gateway))
        if cfg.resolve_hostnames:
            dev.hostname = reverse_dns(ip)
        if cfg.probe_ports:
            dev.open_ports = [p for p in FINGERPRINT_PORTS if tcp_open(ip, p, cfg.port_timeout_s)]
        dev.device_class = classify(dev)
        return dev
