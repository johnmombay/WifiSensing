"""Host network discovery and WiFi signal sampling (Windows + Linux).

No elevated privileges are required. All data comes from OS utilities:
``ipconfig``/``netsh wlan`` on Windows, ``ip route``/``/proc/net/wireless`` on Linux.
"""

from __future__ import annotations

import ipaddress
import logging
import platform
import re
import socket
import subprocess
import time
import uuid
from typing import Optional, Protocol

from .models import SignalSample, normalize_mac

log = logging.getLogger(__name__)

IS_WINDOWS = platform.system() == "Windows"
IS_LINUX = platform.system() == "Linux"

_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)


def run_cmd(args: list[str], timeout: float = 5.0) -> str:
    """Run an OS command and return stdout (decoded leniently). Raises on failure."""
    try:
        proc = subprocess.run(
            args,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            creationflags=_NO_WINDOW,
            check=False,
        )
    except FileNotFoundError as exc:
        raise RuntimeError(f"command not found: {args[0]}") from exc
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"command timed out after {timeout}s: {' '.join(args)}") from exc
    if proc.returncode != 0 and not proc.stdout:
        raise RuntimeError(f"{' '.join(args)} exited {proc.returncode}: {proc.stderr.strip()}")
    return proc.stdout


# --------------------------------------------------------------------------- host / subnet

def local_ipv4() -> str:
    """Primary outbound IPv4 address. Uses a UDP socket; no packets are sent."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.connect(("192.0.2.1", 9))  # TEST-NET-1, never routed; only selects the interface
        ip = sock.getsockname()[0]
    except OSError as exc:
        raise RuntimeError("could not determine local IPv4 address (no default route?)") from exc
    finally:
        sock.close()
    if ip.startswith("127."):
        raise RuntimeError("local address resolved to loopback; is a network interface up?")
    return ip


def local_mac() -> Optional[str]:
    node = uuid.getnode()
    if (node >> 40) & 0x01:  # multicast bit set -> Python fell back to a random value
        return None
    return normalize_mac(f"{node:012x}")


def parse_subnet(spec: Optional[str], local_ip: str, max_prefix_hosts: int = 1024) -> ipaddress.IPv4Network:
    """Validate a user-supplied CIDR or derive a /24 from the local IP."""
    if spec:
        try:
            net = ipaddress.ip_network(spec, strict=False)
        except ValueError as exc:
            raise ValueError(f"invalid subnet {spec!r}: {exc}") from exc
        if not isinstance(net, ipaddress.IPv4Network):
            raise ValueError("only IPv4 subnets are supported")
    else:
        net = ipaddress.ip_network(f"{local_ip}/24", strict=False)
    if net.num_addresses > max_prefix_hosts:
        raise ValueError(
            f"subnet {net} has {net.num_addresses} addresses; refusing to sweep more than "
            f"{max_prefix_hosts}. Pass a narrower --subnet."
        )
    return net


def default_gateway(local_ip: str) -> Optional[str]:
    """Best-effort default gateway for the interface holding ``local_ip``."""
    try:
        if IS_WINDOWS:
            gateways = parse_ipconfig_gateways(run_cmd(["ipconfig"]))
            net = ipaddress.ip_network(f"{local_ip}/24", strict=False)
            for gw in gateways:
                if ipaddress.ip_address(gw) in net:
                    return gw
            return gateways[0] if gateways else None
        if IS_LINUX:
            out = run_cmd(["ip", "route", "show", "default"])
            m = re.search(r"default via (\d+\.\d+\.\d+\.\d+)", out)
            return m.group(1) if m else None
    except RuntimeError as exc:
        log.warning("gateway lookup failed: %s", exc)
    return None


_IPV4_RE = re.compile(r"(\d+\.\d+\.\d+\.\d+)")


def parse_ipconfig_gateways(text: str) -> list[str]:
    """IPv4 default gateways from ``ipconfig``; handles the IPv6-first layout where the
    IPv4 address sits alone on the following indented continuation line."""
    found: list[str] = []
    in_gateway = False
    for line in text.splitlines():
        if re.match(r"^\s*Default Gateway", line):
            in_gateway = True
            m = _IPV4_RE.search(line.split(":", 1)[1] if ":" in line else "")
            if m:
                found.append(m.group(1))
            continue
        if in_gateway and re.match(r"^\s+\S", line) and ":" not in line.split(".")[0]:
            m = _IPV4_RE.fullmatch(line.strip())
            if m:
                found.append(m.group(1))
                continue
        in_gateway = False
    return found


# --------------------------------------------------------------------------- signal sources

class SignalSource(Protocol):
    def sample(self) -> Optional[SignalSample]: ...


def percent_to_dbm(percent: float) -> float:
    """Windows reports link quality in %; the common inverse of its mapping is dBm = pct/2 - 100."""
    return max(-100.0, min(-50.0, percent / 2.0 - 100.0))


class WindowsNetshSignal:
    """Samples ``netsh wlan show interfaces``."""

    _signal_re = re.compile(r"^\s*Signal\s*:\s*(\d+)\s*%", re.M)
    _ssid_re = re.compile(r"^\s*SSID\s*:\s*(.+?)\s*$", re.M)
    _bssid_re = re.compile(r"^\s*(?:AP )?BSSID\s*:\s*([0-9a-fA-F:-]{17})", re.M)
    _rate_re = re.compile(r"^\s*Receive rate \(Mbps\)\s*:\s*([\d.]+)", re.M)
    _state_re = re.compile(r"^\s*State\s*:\s*(\w+)", re.M)

    def sample(self) -> Optional[SignalSample]:
        try:
            out = run_cmd(["netsh", "wlan", "show", "interfaces"], timeout=4.0)
        except RuntimeError as exc:
            log.debug("netsh failed: %s", exc)
            return None
        state = self._state_re.search(out)
        if state and state.group(1).lower() != "connected":
            log.debug("wlan interface state: %s", state.group(1))
            return None
        sig = self._signal_re.search(out)
        if not sig:
            return None
        ssid = self._ssid_re.search(out)
        bssid = self._bssid_re.search(out)
        rate = self._rate_re.search(out)
        return SignalSample(
            timestamp=time.time(),
            rssi_dbm=percent_to_dbm(float(sig.group(1))),
            ssid=ssid.group(1) if ssid else None,
            bssid=normalize_mac(bssid.group(1)) if bssid else None,
            link_mbps=float(rate.group(1)) if rate else None,
        )


class LinuxProcSignal:
    """Samples ``/proc/net/wireless`` (level column, dBm) and ``iw`` for SSID/BSSID."""

    _ssid_cache: tuple[Optional[str], Optional[str]] = (None, None)
    _ssid_cache_at: float = 0.0

    def sample(self) -> Optional[SignalSample]:
        try:
            with open("/proc/net/wireless", encoding="utf-8") as fh:
                lines = fh.readlines()[2:]
        except OSError as exc:
            log.debug("/proc/net/wireless unavailable: %s", exc)
            return None
        for line in lines:
            parts = line.split()
            if len(parts) < 4:
                continue
            iface = parts[0].rstrip(":")
            try:
                level = float(parts[3].rstrip("."))
            except ValueError:
                continue
            if level > 0:  # some drivers report quality units rather than dBm
                level = percent_to_dbm(level)
            ssid, bssid = self._ssid_info(iface)
            return SignalSample(timestamp=time.time(), rssi_dbm=level, ssid=ssid, bssid=bssid)
        return None

    def _ssid_info(self, iface: str) -> tuple[Optional[str], Optional[str]]:
        if time.time() - self._ssid_cache_at < 30:
            return self._ssid_cache
        ssid = bssid = None
        try:
            out = run_cmd(["iw", "dev", iface, "link"], timeout=3.0)
            m = re.search(r"Connected to ([0-9a-f:]{17})", out)
            bssid = normalize_mac(m.group(1)) if m else None
            m = re.search(r"SSID:\s*(.+)", out)
            ssid = m.group(1).strip() if m else None
        except RuntimeError:
            pass
        self._ssid_cache, self._ssid_cache_at = (ssid, bssid), time.time()
        return self._ssid_cache


class NullSignal:
    def sample(self) -> Optional[SignalSample]:
        return None


def platform_signal_source() -> SignalSource:
    if IS_WINDOWS:
        return WindowsNetshSignal()
    if IS_LINUX:
        return LinuxProcSignal()
    log.warning("no WiFi signal source for platform %s; presence sensing disabled", platform.system())
    return NullSignal()
