"""wifisense - WiFi-based device discovery and presence sensing with a live map.

Dependencies
------------
Runtime : Python >= 3.10, matplotlib >= 3.7 (rendering only)
OS tools: Windows - ping, arp, ipconfig, netsh (built in)
          Linux   - ping, ip, /proc/net/wireless (iproute2, iputils)
Privileges: none (no raw sockets; ICMP via the OS ping binary)
"""

__version__ = "0.1.0"
