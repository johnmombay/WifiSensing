from wifisense.wlan import channel_to_mhz, parse_netsh_networks, parse_nmcli

NETSH = """
Interface name : Wi-Fi
There are 2 networks currently visible.

SSID 1 : OfficeNet
    Network type            : Infrastructure
    Authentication          : WPA2-Personal
    Encryption              : CCMP
    BSSID 1                 : aa:bb:cc:dd:ee:01
         Signal             : 90%
         Radio type         : 802.11ax
         Band               : 5 GHz
         Channel            : 36
    BSSID 2                 : aa:bb:cc:dd:ee:02
         Signal             : 40%
         Radio type         : 802.11n
         Band               : 2.4 GHz
         Channel            : 6

SSID 2 :
    Network type            : Infrastructure
    BSSID 1                 : aa:bb:cc:dd:ee:03
         Signal             : 20%
         Channel            : 11
"""

NMCLI = r"""AA\:BB\:CC\:DD\:EE\:01:OfficeNet:90:5180 MHz
AA\:BB\:CC\:DD\:EE\:02:Guest:40:2437 MHz
garbage line
"""


def test_parse_netsh_networks():
    r = parse_netsh_networks(NETSH, now=1.0)
    assert [x.bssid for x in r] == ["aa:bb:cc:dd:ee:01", "aa:bb:cc:dd:ee:02", "aa:bb:cc:dd:ee:03"]
    assert r[0].ssid == "OfficeNet" and r[0].rssi_dbm == -55.0 and r[0].freq_mhz == 5180
    assert r[1].freq_mhz == 2437 and r[2].ssid == ""


def test_parse_nmcli():
    r = parse_nmcli(NMCLI, now=1.0)
    assert len(r) == 2
    assert r[0].bssid == "aa:bb:cc:dd:ee:01" and r[0].ssid == "OfficeNet" and r[0].freq_mhz == 5180
    assert r[1].rssi_dbm == -80.0


def test_channel_to_mhz():
    assert channel_to_mhz(1) == 2412 and channel_to_mhz(14) == 2484 and channel_to_mhz(36) == 5180
    assert channel_to_mhz(None) is None and channel_to_mhz(200) is None
