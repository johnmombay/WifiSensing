"""MAC OUI (vendor prefix) lookup.

A curated built-in table covers common office hardware. For full coverage load
the IEEE registry with :func:`load_ieee_file` (``oui.txt`` or ``oui.csv`` from
https://standards-oui.ieee.org/).
"""

from __future__ import annotations

import csv
import logging
import re
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)

# Category hints: pc | mobile | network | iot | printer | virtual
_BUILTIN: dict[str, tuple[str, str]] = {
    # Apple (laptops and handsets share prefixes; port probing refines the class)
    "f0:18:98": ("Apple", "mobile"), "3c:22:fb": ("Apple", "mobile"), "a4:83:e7": ("Apple", "mobile"),
    "f8:ff:c2": ("Apple", "mobile"), "dc:a9:04": ("Apple", "mobile"), "ac:bc:32": ("Apple", "mobile"),
    "b8:e8:56": ("Apple", "mobile"), "28:cf:e9": ("Apple", "mobile"), "d0:03:4b": ("Apple", "mobile"),
    "00:17:f2": ("Apple", "mobile"),
    # Samsung / Xiaomi / Huawei / Google / OnePlus / Motorola handsets
    "8c:f5:a3": ("Samsung", "mobile"), "00:12:47": ("Samsung", "mobile"), "f4:7b:5e": ("Samsung", "mobile"),
    "5c:49:7d": ("Samsung", "mobile"), "64:09:80": ("Xiaomi", "mobile"), "28:6c:07": ("Xiaomi", "mobile"),
    "f8:a4:5f": ("Xiaomi", "mobile"), "7c:1d:d9": ("Xiaomi", "mobile"), "00:e0:fc": ("Huawei", "mobile"),
    "e8:cd:2d": ("Huawei", "mobile"), "f4:f5:d8": ("Google", "mobile"), "54:60:09": ("Google", "mobile"),
    "3c:5a:b4": ("Google", "mobile"), "94:65:2d": ("OnePlus", "mobile"), "c0:ee:fb": ("OnePlus", "mobile"),
    "f8:e0:79": ("Motorola", "mobile"), "40:78:6a": ("Motorola", "mobile"),
    # PC vendors / NIC chipsets
    "00:15:17": ("Intel", "pc"), "00:1b:21": ("Intel", "pc"), "3c:97:0e": ("Intel", "pc"),
    "f8:63:3f": ("Intel", "pc"), "a4:34:d9": ("Intel", "pc"), "8c:8d:28": ("Intel", "pc"),
    "34:e6:ad": ("Intel", "pc"), "00:14:22": ("Dell", "pc"), "f8:b1:56": ("Dell", "pc"),
    "18:03:73": ("Dell", "pc"), "d4:be:d9": ("Dell", "pc"), "b0:83:fe": ("Dell", "pc"),
    "f4:8e:38": ("Dell", "pc"), "34:17:eb": ("Dell", "pc"), "e4:54:e8": ("Dell", "pc"),
    "3c:d9:2b": ("HP", "pc"), "9c:8e:99": ("HP", "pc"), "00:1a:4b": ("HP", "pc"),
    "f4:ce:46": ("HP", "pc"), "10:1f:74": ("HP", "pc"), "28:d2:44": ("Lenovo", "pc"),
    "54:e1:ad": ("Lenovo", "pc"), "50:7b:9d": ("Lenovo", "pc"), "28:18:78": ("Microsoft", "pc"),
    "7c:1e:52": ("Microsoft", "pc"), "98:5f:d3": ("Microsoft", "pc"),
    # Network infrastructure
    "00:00:0c": ("Cisco", "network"), "88:15:44": ("Cisco Meraki", "network"), "e0:55:3d": ("Cisco Meraki", "network"),
    "ac:17:c8": ("Cisco Meraki", "network"), "24:a4:3c": ("Ubiquiti", "network"), "f0:9f:c2": ("Ubiquiti", "network"),
    "80:2a:a8": ("Ubiquiti", "network"), "74:83:c2": ("Ubiquiti", "network"), "fc:ec:da": ("Ubiquiti", "network"),
    "68:d7:9a": ("Ubiquiti", "network"), "50:c7:bf": ("TP-Link", "network"), "f4:f2:6d": ("TP-Link", "network"),
    "b0:4e:26": ("TP-Link", "network"), "20:e5:2a": ("Netgear", "network"), "a4:2b:8c": ("Netgear", "network"),
    "c0:3f:0e": ("Netgear", "network"), "9c:d3:6d": ("Netgear", "network"), "24:de:c6": ("Aruba", "network"),
    "00:0b:86": ("Aruba", "network"), "94:b4:0f": ("Aruba", "network"), "d8:c7:c8": ("Aruba", "network"),
    "c4:10:8a": ("Ruckus", "network"), "58:93:96": ("Ruckus", "network"),
    # IoT / SBC / smart devices
    "b8:27:eb": ("Raspberry Pi", "iot"), "dc:a6:32": ("Raspberry Pi", "iot"), "e4:5f:01": ("Raspberry Pi", "iot"),
    "d8:3a:dd": ("Raspberry Pi", "iot"), "24:6f:28": ("Espressif", "iot"), "30:ae:a4": ("Espressif", "iot"),
    "a4:cf:12": ("Espressif", "iot"), "3c:71:bf": ("Espressif", "iot"), "84:cc:a8": ("Espressif", "iot"),
    "24:0a:c4": ("Espressif", "iot"), "fc:65:de": ("Amazon", "iot"), "74:c2:46": ("Amazon", "iot"),
    "a0:02:dc": ("Amazon", "iot"), "40:b4:cd": ("Amazon", "iot"), "44:65:0d": ("Amazon", "iot"),
    "5c:aa:fd": ("Sonos", "iot"), "94:9f:3e": ("Sonos", "iot"), "b8:e9:37": ("Sonos", "iot"),
    # Printers
    "00:80:77": ("Brother", "printer"), "30:05:5c": ("Brother", "printer"), "00:1e:8f": ("Canon", "printer"),
    "00:bb:c1": ("Canon", "printer"), "00:26:ab": ("Epson", "printer"),
    # Virtualisation
    "00:50:56": ("VMware", "virtual"), "00:0c:29": ("VMware", "virtual"), "00:05:69": ("VMware", "virtual"),
    "00:15:5d": ("Hyper-V", "virtual"), "08:00:27": ("VirtualBox", "virtual"),
}

_VENDOR_HINT_RULES: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"apple", re.I), "mobile"),
    (re.compile(r"samsung|xiaomi|huawei|oneplus|oppo|vivo|realme|motorola|google|lg electronics|sony mobile|nokia", re.I), "mobile"),
    (re.compile(r"intel|dell|hewlett|hp inc|lenovo|asus|acer|micro-star|gigabyte|microsoft|fujitsu|toshiba|realtek|qualcomm", re.I), "pc"),
    (re.compile(r"cisco|meraki|ubiquiti|tp-link|netgear|aruba|ruckus|juniper|mikrotik|fortinet|d-link|zyxel|extreme", re.I), "network"),
    (re.compile(r"raspberry|espressif|amazon|sonos|ring|nest|tuya|philips|hikvision|dahua|axis", re.I), "iot"),
    (re.compile(r"brother|canon|epson|xerox|ricoh|kyocera|lexmark|konica", re.I), "printer"),
    (re.compile(r"vmware|hyper-v|virtualbox|xen|parallels", re.I), "virtual"),
]


class OuiDatabase:
    """Vendor lookup by the first three MAC octets."""

    def __init__(self, table: Optional[dict[str, tuple[str, str]]] = None) -> None:
        self._table: dict[str, tuple[str, str]] = dict(_BUILTIN)
        if table:
            self._table.update(table)

    def __len__(self) -> int:
        return len(self._table)

    def lookup(self, mac: Optional[str]) -> tuple[Optional[str], Optional[str]]:
        """Return ``(vendor, hint)`` or ``(None, None)`` when unknown.

        Locally-administered (randomised) MACs are reported as ``("Randomised", None)``.
        """
        if not mac or len(mac) < 8:
            return None, None
        prefix = mac[:8].lower()
        entry = self._table.get(prefix)
        if entry:
            return entry
        try:
            first_octet = int(mac[:2], 16)
        except ValueError:
            return None, None
        if first_octet & 0x02:
            return "Randomised", None
        return None, None

    @classmethod
    def load_ieee_file(cls, path: Path | str) -> "OuiDatabase":
        """Load an IEEE ``oui.txt`` or ``oui.csv`` registry file on top of the built-in table."""
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"OUI file not found: {path}")
        table: dict[str, tuple[str, str]] = {}
        if path.suffix.lower() == ".csv":
            with path.open(newline="", encoding="utf-8", errors="replace") as fh:
                reader = csv.reader(fh)
                for row in reader:
                    if len(row) < 3 or row[0] == "Registry":
                        continue
                    assignment, org = row[1].strip(), row[2].strip()
                    if len(assignment) != 6:
                        continue
                    prefix = ":".join(assignment[i : i + 2] for i in range(0, 6, 2)).lower()
                    table[prefix] = (org, _hint_for(org))
        else:
            pattern = re.compile(r"^([0-9A-Fa-f]{2})-([0-9A-Fa-f]{2})-([0-9A-Fa-f]{2})\s+\(hex\)\s+(.+?)\s*$")
            with path.open(encoding="utf-8", errors="replace") as fh:
                for line in fh:
                    m = pattern.match(line)
                    if m:
                        prefix = ":".join(g.lower() for g in m.groups()[:3])
                        org = m.group(4)
                        table[prefix] = (org, _hint_for(org))
        if not table:
            raise ValueError(f"no OUI entries parsed from {path}")
        log.info("loaded %d OUI entries from %s", len(table), path)
        return cls(table)


def _hint_for(org: str) -> str:
    for pattern, hint in _VENDOR_HINT_RULES:
        if pattern.search(org):
            return hint
    return ""
