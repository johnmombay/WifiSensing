import pytest

from wifisense import netinfo, scanner
from wifisense.models import Device, DeviceClass, normalize_mac
from wifisense.oui import OuiDatabase

WIN_ARP = """
Interface: 192.168.1.10 --- 0x5
  Internet Address      Physical Address      Type
  192.168.1.1           aa-bb-cc-dd-ee-01     dynamic
  192.168.1.20          F8-B1-56-AA-01-01     dynamic
  192.168.1.255         ff-ff-ff-ff-ff-ff     static
  224.0.0.251           01-00-5e-00-00-fb     static
"""

LINUX_NEIGH = """192.168.1.1 dev wlan0 lladdr aa:bb:cc:dd:ee:01 REACHABLE
192.168.1.20 dev wlan0 lladdr f8:b1:56:aa:01:01 STALE
192.168.1.30 dev wlan0  FAILED
"""


@pytest.mark.parametrize("raw", ["AA-BB-CC-DD-EE-FF", "aa:bb:cc:dd:ee:ff", "aabb.ccdd.eeff", "AABBCCDDEEFF"])
def test_normalize_mac(raw):
    assert normalize_mac(raw) == "aa:bb:cc:dd:ee:ff"


def test_normalize_mac_rejects_garbage():
    with pytest.raises(ValueError):
        normalize_mac("not-a-mac")


def test_arp_windows(monkeypatch):
    monkeypatch.setattr(netinfo, "IS_WINDOWS", True)
    monkeypatch.setattr(netinfo, "run_cmd", lambda args, timeout=5.0: WIN_ARP)
    table = scanner.read_arp_table()
    assert table == {"192.168.1.1": "aa:bb:cc:dd:ee:01", "192.168.1.20": "f8:b1:56:aa:01:01"}


def test_arp_linux(monkeypatch):
    monkeypatch.setattr(netinfo, "IS_WINDOWS", False)
    monkeypatch.setattr(netinfo, "run_cmd", lambda args, timeout=5.0: LINUX_NEIGH)
    table = scanner.read_arp_table()
    assert table == {"192.168.1.1": "aa:bb:cc:dd:ee:01", "192.168.1.20": "f8:b1:56:aa:01:01"}


def test_ping_parses_rtt(monkeypatch):
    monkeypatch.setattr(netinfo, "IS_WINDOWS", True)
    monkeypatch.setattr(netinfo, "run_cmd", lambda args, timeout=5.0: "Reply from 192.168.1.1: bytes=32 time=3ms TTL=64")
    assert scanner.ping("192.168.1.1", 400) == 3.0


def test_ping_no_reply(monkeypatch):
    monkeypatch.setattr(netinfo, "IS_WINDOWS", True)
    monkeypatch.setattr(netinfo, "run_cmd", lambda args, timeout=5.0: "Request timed out.")
    assert scanner.ping("192.168.1.99", 400) is None


@pytest.mark.parametrize(
    "kwargs, expected",
    [
        ({"is_gateway": True}, DeviceClass.ROUTER),
        ({"open_ports": [62078], "vendor_hint": "mobile"}, DeviceClass.MOBILE),
        ({"hostname": "Galaxy-S24"}, DeviceClass.MOBILE),
        ({"open_ports": [135, 445, 3389]}, DeviceClass.WORKSTATION),
        ({"hostname": "ACME-LT-00210"}, DeviceClass.WORKSTATION),  # corporate asset-tag naming
        ({"hostname": "jdoe-pc"}, DeviceClass.WORKSTATION),
        ({"open_ports": [22, 445], "hostname": "srv-files01"}, DeviceClass.SERVER),
        ({"open_ports": [9100, 80]}, DeviceClass.PRINTER),
        ({"vendor_hint": "mobile", "open_ports": [548, 7000]}, DeviceClass.WORKSTATION),  # Mac, not iPhone
        ({"vendor_hint": "pc"}, DeviceClass.WORKSTATION),
        ({"vendor_hint": "network"}, DeviceClass.ROUTER),
        ({"vendor_hint": "iot", "open_ports": [80]}, DeviceClass.IOT),
        ({}, DeviceClass.UNKNOWN),
    ],
)
def test_classify(kwargs, expected):
    dev = Device(ip="192.168.1.50", **kwargs)
    assert scanner.classify(dev) is expected


IPCONFIG = """
Ethernet adapter Ethernet:

   Media State . . . . . . . . . . . : Media disconnected

Wireless LAN adapter Wi-Fi:

   IPv6 Address. . . . . . . . . . . : 2001:db8::1
   IPv4 Address. . . . . . . . . . . : 192.168.1.61
   Subnet Mask . . . . . . . . . . . : 255.255.255.0
   Default Gateway . . . . . . . . . : fe80::1%12
                                       192.168.1.1

Ethernet adapter vEthernet (Default Switch):

   IPv4 Address. . . . . . . . . . . : 172.30.0.1
   Default Gateway . . . . . . . . . : 172.30.0.254
"""


def test_parse_ipconfig_gateways_handles_continuation_lines():
    assert netinfo.parse_ipconfig_gateways(IPCONFIG) == ["192.168.1.1", "172.30.0.254"]


def test_oui_lookup_builtin_and_randomised():
    db = OuiDatabase()
    assert db.lookup("f8:b1:56:aa:01:01") == ("Dell", "pc")
    assert db.lookup("d6:12:9f:ee:0c:0c") == ("Randomised", None)
    assert db.lookup(None) == (None, None)


def test_oui_load_ieee_txt(tmp_path):
    f = tmp_path / "oui.txt"
    f.write_text("00-1B-21   (hex)\t\tIntel Corporate\nAB-CD-EF   (hex)\t\tAcme Mobile Handsets\n", encoding="utf-8")
    db = OuiDatabase.load_ieee_file(f)
    assert db.lookup("ab:cd:ef:00:00:01")[0] == "Acme Mobile Handsets"
