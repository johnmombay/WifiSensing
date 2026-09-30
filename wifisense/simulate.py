"""Deterministic simulation backends for demos and tests (no network access).

``SimWorld`` holds a synthetic office: three access points, interior walls with doors,
a stationary host and a person who walks a loop and pauses. Radio propagation uses the
log-distance model plus a fixed loss per wall crossed, so both the multi-link motion
sensing and the terrain inference have real structure to find.
"""

from __future__ import annotations

import math
import random
import time
from typing import Optional

from .models import BssReading, Device, SignalSample
from .oui import OuiDatabase
from .scanner import classify
from .survey import Survey, SurveyPoint

# (x1, y1, x2, y2) in metres. Gaps between collinear segments are doors.
SIM_WALLS: list[tuple[float, float, float, float]] = [
    (5.2, 0.0, 5.2, 3.3), (5.2, 4.5, 5.2, 8.0),          # partition between desk areas, door at y 3.3..4.5
    (7.6, 4.4, 7.6, 6.0), (7.6, 7.0, 7.6, 8.0),          # meeting-room west wall, door at y 6.0..7.0
    (7.6, 4.4, 10.4, 4.4), (11.4, 4.4, 12.0, 4.4),       # meeting-room south wall, door at x 10.4..11.4
]
SIM_DOORS: list[tuple[float, float]] = [(5.2, 3.9), (7.6, 6.5), (10.9, 4.4)]
SIM_APS: dict[str, tuple[float, float, str]] = {
    "24:a4:3c:11:22:33": (6.0, 4.2, "SIM-OFFICE"),
    "24:a4:3c:11:22:44": (1.5, 7.2, "SIM-OFFICE-W"),
    "24:a4:3c:11:22:55": (10.6, 1.0, "SIM-OFFICE-E"),
}
SIM_MAIN_BSSID = "24:a4:3c:11:22:33"
SIM_HOST_XY = (5.5, 5.0)

_P0_DBM = -35.0
_PATH_LOSS_N = 2.2
_WALL_LOSS_DB = 6.0


def _ccw(ax: float, ay: float, bx: float, by: float, cx: float, cy: float) -> bool:
    return (cy - ay) * (bx - ax) > (by - ay) * (cx - ax)


def segments_intersect(a: tuple[float, float], b: tuple[float, float], c: tuple[float, float], d: tuple[float, float]) -> bool:
    return (_ccw(*a, *c, *d) != _ccw(*b, *c, *d)) and (_ccw(*a, *b, *c) != _ccw(*a, *b, *d))


def walls_crossed(p: tuple[float, float], q: tuple[float, float], walls=SIM_WALLS) -> int:
    return sum(1 for (x1, y1, x2, y2) in walls if segments_intersect(p, q, (x1, y1), (x2, y2)))


def point_segment_distance(p: tuple[float, float], a: tuple[float, float], b: tuple[float, float]) -> float:
    ax, ay, bx, by = a[0], a[1], b[0], b[1]
    dx, dy = bx - ax, by - ay
    if dx == dy == 0:
        return math.hypot(p[0] - ax, p[1] - ay)
    t = max(0.0, min(1.0, ((p[0] - ax) * dx + (p[1] - ay) * dy) / (dx * dx + dy * dy)))
    return math.hypot(p[0] - (ax + t * dx), p[1] - (ay + t * dy))


class SimWorld:
    """Shared clock, walker and propagation for all simulated sensors."""

    _PATH = [(2.0, 2.0), (2.0, 6.0), (4.2, 6.0), (4.2, 3.9), (6.5, 3.9), (6.5, 2.0), (9.5, 2.0), (9.5, 3.0)]
    _SPEED = 0.6      # m/s
    _WALK_S = 22.0
    _PAUSE_S = 14.0

    def __init__(self, seed: int = 7, host_xy: tuple[float, float] = SIM_HOST_XY) -> None:
        self.rng = random.Random(seed)
        self.host_xy = host_xy
        self.walls = SIM_WALLS
        self.aps = SIM_APS
        self._t0 = time.time()
        self._path_len = [math.dist(a, b) for a, b in zip(self._PATH, self._PATH[1:] + self._PATH[:1])]

    def walker_xy(self, now: Optional[float] = None) -> tuple[float, float]:
        t = (now or time.time()) - self._t0
        cycle = self._WALK_S + self._PAUSE_S
        cycles, phase = divmod(t, cycle)
        walked = (cycles * self._WALK_S + min(phase, self._WALK_S)) * self._SPEED
        total = sum(self._path_len)
        s = walked % total
        for (a, b), seg in zip(zip(self._PATH, self._PATH[1:] + self._PATH[:1]), self._path_len):
            if s <= seg:
                f = s / seg if seg else 0.0
                return (a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f)
            s -= seg
        return self._PATH[0]

    def walker_moving(self, now: Optional[float] = None) -> bool:
        t = (now or time.time()) - self._t0
        return (t % (self._WALK_S + self._PAUSE_S)) < self._WALK_S

    def rssi_at(self, ap_xy: tuple[float, float], pos: tuple[float, float], noise_db: float = 0.8) -> float:
        d = max(0.5, math.dist(ap_xy, pos))
        loss = 10.0 * _PATH_LOSS_N * math.log10(d) + _WALL_LOSS_DB * walls_crossed(ap_xy, pos, self.walls)
        return _P0_DBM - loss + self.rng.gauss(0.0, noise_db)

    def link_sigma(self, ap_xy: tuple[float, float], quiet: float, disturbed: float, now: Optional[float] = None) -> float:
        """Extra RSSI noise on the host<->AP link when the walker is near its path and moving."""
        if not self.walker_moving(now):
            return quiet
        if point_segment_distance(self.walker_xy(now), self.host_xy, ap_xy) < 1.0:
            return disturbed
        return quiet


# --------------------------------------------------------------------------- devices

_SIM_HOSTS: list[dict] = [
    {"ip": "192.168.50.1", "mac": SIM_MAIN_BSSID, "hostname": "unifi-ap", "ports": [80, 443], "rtt": 1.2, "gateway": True},
    {"ip": "192.168.50.10", "mac": "f8:b1:56:aa:01:01", "hostname": "DESKTOP-FIN01", "ports": [135, 139, 445, 3389], "rtt": 2.1},
    {"ip": "192.168.50.11", "mac": "3c:d9:2b:aa:02:02", "hostname": "LAPTOP-OPS02", "ports": [135, 445], "rtt": 4.8},
    {"ip": "192.168.50.12", "mac": "28:d2:44:aa:03:03", "hostname": "THINKPAD-HR03", "ports": [135, 445], "rtt": 6.0},
    {"ip": "192.168.50.13", "mac": "f0:18:98:aa:04:04", "hostname": "MacBook-Pro", "ports": [548, 7000], "rtt": 3.3},
    {"ip": "192.168.50.14", "mac": "f8:b1:56:aa:05:05", "hostname": "MREYES-LAPTOP", "ports": [135, 445], "rtt": 5.1},
    {"ip": "192.168.50.20", "mac": "00:50:56:bb:05:05", "hostname": "srv-files01", "ports": [22, 443, 445], "rtt": 0.9},
    {"ip": "192.168.50.101", "mac": "a4:83:e7:cc:06:06", "hostname": "iPhone-Ana", "ports": [62078], "rtt": 18.0},
    {"ip": "192.168.50.102", "mac": "8c:f5:a3:cc:07:07", "hostname": "Galaxy-S24", "ports": [], "rtt": 31.0},
    {"ip": "192.168.50.103", "mac": "64:09:80:cc:08:08", "hostname": None, "ports": [], "rtt": None},
    {"ip": "192.168.50.104", "mac": "f4:f5:d8:cc:09:09", "hostname": "Pixel-8", "ports": [], "rtt": 12.0},
    {"ip": "192.168.50.150", "mac": "9c:8e:99:dd:0a:0a", "hostname": "NPI3F2A1C", "ports": [80, 9100, 631], "rtt": 2.4},
    {"ip": "192.168.50.160", "mac": "24:6f:28:dd:0b:0b", "hostname": None, "ports": [80], "rtt": 9.5},
    {"ip": "192.168.50.200", "mac": "d6:12:9f:ee:0c:0c", "hostname": None, "ports": [], "rtt": None},
]


class SimulatedScanner:
    """Returns a fixed office population with light RTT jitter and one phone that comes and goes."""

    def __init__(self, oui: OuiDatabase, local_ip: str = "192.168.50.42", seed: int = 7) -> None:
        self.oui = oui
        self.local_ip = local_ip
        self._rng = random.Random(seed)
        self._scans = 0

    def scan(self) -> list[Device]:
        self._scans += 1
        devices = [self._make({"ip": self.local_ip, "mac": "a4:34:d9:ff:42:42", "hostname": "THIS-HOST", "ports": [], "rtt": 0.0, "self": True})]
        for spec in _SIM_HOSTS:
            if spec["ip"] == "192.168.50.104" and self._scans % 3 == 0:
                continue  # Pixel leaves the room every third scan
            devices.append(self._make(spec))
        return devices

    def _make(self, spec: dict) -> Device:
        vendor, hint = self.oui.lookup(spec["mac"])
        rtt = spec["rtt"]
        if rtt is not None:
            rtt = max(0.0, rtt + self._rng.gauss(0.0, 0.15 * max(rtt, 1.0)))
        dev = Device(
            ip=spec["ip"],
            mac=spec["mac"],
            hostname=spec.get("hostname"),
            vendor=vendor,
            vendor_hint=hint,
            open_ports=list(spec["ports"]),
            rtt_ms=rtt,
            is_self=spec.get("self", False),
            is_gateway=spec.get("gateway", False),
        )
        dev.device_class = classify(dev)
        return dev


# --------------------------------------------------------------------------- radio

class SimulatedSignal:
    """Connected link (host -> main AP): quiet 0.5 dB noise, 2.4 dB when the walker crosses its path."""

    def __init__(self, world: SimWorld) -> None:
        self.world = world
        self._base = world.rssi_at(world.aps[SIM_MAIN_BSSID][:2], world.host_xy, noise_db=0.0)

    def sample(self) -> Optional[SignalSample]:
        now = time.time()
        sigma = self.world.link_sigma(self.world.aps[SIM_MAIN_BSSID][:2], 0.5, 2.4, now)
        return SignalSample(now, round(self._base + self.world.rng.gauss(0.0, sigma), 1), "SIM-OFFICE", SIM_MAIN_BSSID, 433.0)


class SimulatedBssScanner:
    """All three office radios plus two weak neighbours, measured at the host position."""

    _NEIGHBOURS = {"f4:6f:ed:00:00:01": "NEIGHBOUR-1", "28:77:77:00:00:02": "NEIGHBOUR-2"}

    def __init__(self, world: SimWorld) -> None:
        self.world = world

    def scan(self) -> list[BssReading]:
        now = time.time()
        out: list[BssReading] = []
        for bssid, (x, y, ssid) in self.world.aps.items():
            sigma = self.world.link_sigma((x, y), 0.6, 2.6, now)
            out.append(BssReading(bssid, ssid, round(self.world.rssi_at((x, y), self.world.host_xy, sigma), 1), 2437, now))
        for bssid, ssid in self._NEIGHBOURS.items():
            out.append(BssReading(bssid, ssid, round(-88.0 + self.world.rng.gauss(0.0, 1.0), 1), 2462, now))
        return out


def generate_survey(world: SimWorld, width: float, height: float, step: float = 1.0, survey: Optional[Survey] = None) -> Survey:
    """Walk a grid over the plan and record what each radio measures there."""
    if not 0.25 <= step <= 4.0:
        raise ValueError("step must be within 0.25..4.0 m")
    if survey is None:
        survey = Survey()
    y = step / 2
    while y < height:
        x = step / 2
        while x < width:
            readings = [
                BssReading(bssid, ssid, round(world.rssi_at((ax, ay), (x, y)), 1), 2437, time.time())
                for bssid, (ax, ay, ssid) in world.aps.items()
            ]
            survey.points.append(SurveyPoint(x, y, time.time(), {r.bssid: r.rssi_dbm for r in readings}, {r.bssid: r.ssid for r in readings}))
            x += step
        y += step
    survey.save()
    return survey
