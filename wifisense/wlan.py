"""Multi-BSSID scanning: every access-point radio in range with its RSSI in dBm.

Windows : native WLAN API via ctypes (``WlanScan`` + ``WlanGetNetworkBssList``), which
          forces a fresh scan and reports true dBm. Falls back to ``netsh wlan show networks``
          (percent, refreshed only when the OS scans on its own).
Linux   : ``nmcli dev wifi rescan`` + ``nmcli -t dev wifi list`` (NetworkManager, no root).
"""

from __future__ import annotations

import ctypes
import logging
import platform
import re
import threading
import time
from typing import Optional, Protocol

from .models import BssReading, normalize_mac
from .netinfo import percent_to_dbm, run_cmd

log = logging.getLogger(__name__)

IS_WINDOWS = platform.system() == "Windows"
IS_LINUX = platform.system() == "Linux"


class BssScanner(Protocol):
    def scan(self) -> list[BssReading]: ...


class NullBssScanner:
    def scan(self) -> list[BssReading]:
        return []


# --------------------------------------------------------------------------- Windows native API

if IS_WINDOWS:
    import ctypes.wintypes as wt

    class _GUID(ctypes.Structure):
        _fields_ = [("Data1", wt.DWORD), ("Data2", wt.WORD), ("Data3", wt.WORD), ("Data4", ctypes.c_ubyte * 8)]

    class _IFACE(ctypes.Structure):
        _fields_ = [("guid", _GUID), ("desc", wt.WCHAR * 256), ("state", wt.DWORD)]

    class _IFACE_LIST(ctypes.Structure):
        _fields_ = [("n", wt.DWORD), ("idx", wt.DWORD), ("ifaces", _IFACE * 1)]

    class _DOT11_SSID(ctypes.Structure):
        _fields_ = [("len", wt.ULONG), ("ssid", ctypes.c_ubyte * 32)]

    class _BSS_ENTRY(ctypes.Structure):
        # WLAN_BSS_ENTRY, wlanapi.h
        _fields_ = [
            ("ssid", _DOT11_SSID), ("phy_id", wt.ULONG), ("bssid", ctypes.c_ubyte * 6),
            ("bss_type", wt.DWORD), ("phy_type", wt.DWORD), ("rssi", ctypes.c_long), ("link_quality", wt.ULONG),
            ("in_reg_domain", ctypes.c_ubyte), ("beacon_period", wt.USHORT),
            ("timestamp", ctypes.c_ulonglong), ("host_timestamp", ctypes.c_ulonglong),
            ("capability", wt.USHORT), ("center_freq_khz", wt.ULONG),
            ("rate_set_len", wt.ULONG), ("rate_set", wt.USHORT * 126),
            ("ie_offset", wt.ULONG), ("ie_size", wt.ULONG),
        ]

    class _BSS_LIST(ctypes.Structure):
        _fields_ = [("total_size", wt.DWORD), ("n", wt.DWORD), ("entries", _BSS_ENTRY * 1)]


class WindowsWlanApi:
    """Forces a scan and reads the BSS list through wlanapi.dll. Thread-safe."""

    def __init__(self, settle_s: float = 2.5) -> None:
        if not IS_WINDOWS:
            raise RuntimeError("WindowsWlanApi is Windows-only")
        if not 0.5 <= settle_s <= 10:
            raise ValueError("settle_s must be within 0.5..10")
        self.settle_s = settle_s
        self._lock = threading.Lock()
        self._api = ctypes.windll.wlanapi
        self._handle = wt.HANDLE()
        self._guid: Optional[_GUID] = None
        self._fallback = WindowsNetshBss()
        self._open()

    def _open(self) -> None:
        version = wt.DWORD()
        rc = self._api.WlanOpenHandle(2, None, ctypes.byref(version), ctypes.byref(self._handle))
        if rc != 0:
            raise RuntimeError(f"WlanOpenHandle failed: {rc}")
        plist = ctypes.POINTER(_IFACE_LIST)()
        rc = self._api.WlanEnumInterfaces(self._handle, None, ctypes.byref(plist))
        if rc != 0:
            raise RuntimeError(f"WlanEnumInterfaces failed: {rc}")
        try:
            if plist.contents.n < 1:
                raise RuntimeError("no WLAN interfaces present")
            iface = plist.contents.ifaces[0]
            self._guid = _GUID.from_buffer_copy(iface.guid)
            log.info("WLAN interface: %s", iface.desc)
        finally:
            self._api.WlanFreeMemory(plist)

    def close(self) -> None:
        if self._handle:
            self._api.WlanCloseHandle(self._handle, None)
            self._handle = wt.HANDLE()

    def __del__(self) -> None:  # pragma: no cover
        try:
            self.close()
        except Exception:  # noqa: BLE001
            pass

    def scan(self) -> list[BssReading]:
        with self._lock:
            try:
                rc = self._api.WlanScan(self._handle, ctypes.byref(self._guid), None, None, None)
                if rc != 0:
                    log.debug("WlanScan rc=%d (continuing with cached list)", rc)
                else:
                    time.sleep(self.settle_s)
                return self._read_bss_list()
            except Exception as exc:  # noqa: BLE001
                log.warning("native WLAN scan failed (%s); using netsh fallback", exc)
                return self._fallback.scan()

    def _read_bss_list(self) -> list[BssReading]:
        plist = ctypes.POINTER(_BSS_LIST)()
        rc = self._api.WlanGetNetworkBssList(self._handle, ctypes.byref(self._guid), None, 3, False, None, ctypes.byref(plist))
        if rc != 0:
            raise RuntimeError(f"WlanGetNetworkBssList failed: {rc}")
        try:
            n = plist.contents.n
            entries = ctypes.cast(ctypes.addressof(plist.contents.entries), ctypes.POINTER(_BSS_ENTRY * n)).contents
            now = time.time()
            out: list[BssReading] = []
            for e in entries:
                bssid = ":".join(f"{b:02x}" for b in e.bssid)
                if bssid == "00:00:00:00:00:00":
                    continue
                ssid = bytes(e.ssid.ssid[: min(e.ssid.len, 32)]).decode("utf-8", errors="replace")
                out.append(BssReading(bssid, ssid, float(e.rssi), int(e.center_freq_khz // 1000) or None, now))
            return out
        finally:
            self._api.WlanFreeMemory(plist)


class WindowsNetshBss:
    """Parses ``netsh wlan show networks mode=bssid`` (cached OS scan, percent -> dBm)."""

    _ssid_re = re.compile(r"^SSID \d+\s*:\s?(.*)$")
    _bssid_re = re.compile(r"^\s+BSSID \d+\s*:\s*([0-9a-fA-F:-]{17})")
    _signal_re = re.compile(r"^\s+Signal\s*:\s*(\d+)%")
    _channel_re = re.compile(r"^\s+Channel\s*:\s*(\d+)")

    def scan(self) -> list[BssReading]:
        try:
            out = run_cmd(["netsh", "wlan", "show", "networks", "mode=bssid"], timeout=8.0)
        except RuntimeError as exc:
            log.debug("netsh networks failed: %s", exc)
            return []
        return parse_netsh_networks(out)


def parse_netsh_networks(text: str, now: Optional[float] = None) -> list[BssReading]:
    now = now or time.time()
    readings: list[BssReading] = []
    ssid, bssid = "", None
    for line in text.splitlines():
        m = WindowsNetshBss._ssid_re.match(line)
        if m:
            ssid = m.group(1).strip()
            continue
        m = WindowsNetshBss._bssid_re.match(line)
        if m:
            bssid = normalize_mac(m.group(1))
            continue
        m = WindowsNetshBss._signal_re.match(line)
        if m and bssid:
            readings.append(BssReading(bssid, ssid, percent_to_dbm(float(m.group(1))), None, now))
            continue
        m = WindowsNetshBss._channel_re.match(line)
        if m and bssid and readings and readings[-1].bssid == bssid:
            r = readings[-1]
            readings[-1] = BssReading(r.bssid, r.ssid, r.rssi_dbm, channel_to_mhz(int(m.group(1))), r.timestamp)
    return readings


def channel_to_mhz(channel: Optional[int]) -> Optional[int]:
    if channel is None:
        return None
    if 1 <= channel <= 13:
        return 2407 + 5 * channel
    if channel == 14:
        return 2484
    if 32 <= channel <= 177:
        return 5000 + 5 * channel
    return None


# --------------------------------------------------------------------------- Linux

class LinuxNmcliBss:
    def scan(self) -> list[BssReading]:
        try:
            run_cmd(["nmcli", "dev", "wifi", "rescan"], timeout=6.0)
        except RuntimeError as exc:
            log.debug("nmcli rescan: %s", exc)
        time.sleep(1.5)
        try:
            out = run_cmd(["nmcli", "-t", "-f", "BSSID,SSID,SIGNAL,FREQ", "dev", "wifi", "list"], timeout=6.0)
        except RuntimeError as exc:
            log.debug("nmcli list failed: %s", exc)
            return []
        return parse_nmcli(out)


def parse_nmcli(text: str, now: Optional[float] = None) -> list[BssReading]:
    now = now or time.time()
    readings: list[BssReading] = []
    for line in text.splitlines():
        # nmcli -t escapes the colons inside the BSSID with a backslash
        parts = re.split(r"(?<!\\):", line.strip())
        if len(parts) < 4:
            continue
        try:
            bssid = normalize_mac(parts[0].replace("\\:", ":"))
            signal = float(parts[2])
            freq = int(re.sub(r"\D", "", parts[3]) or 0) or None
        except ValueError:
            continue
        readings.append(BssReading(bssid, parts[1], percent_to_dbm(signal), freq, now))
    return readings


def platform_bss_scanner() -> BssScanner:
    if IS_WINDOWS:
        try:
            return WindowsWlanApi()
        except (RuntimeError, OSError) as exc:
            log.warning("native WLAN API unavailable (%s); using netsh", exc)
            return WindowsNetshBss()
    if IS_LINUX:
        return LinuxNmcliBss()
    log.warning("no BSS scanner for platform %s; multi-link sensing disabled", platform.system())
    return NullBssScanner()
