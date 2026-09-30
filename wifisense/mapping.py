"""Floor-plan model and device placement.

Placement strategy
------------------
1. ``floorplan``: device key (MAC, IP, or hostname) listed in the floor plan -> fixed coordinates.
   The host itself goes to the plan's ``host`` position when one is given.
2. ``anchor``: the default gateway is drawn at the access-point position.
3. ``estimate``: everything else is placed on a ring around the AP. Radius is derived from
   ICMP round-trip time (a weak proximity/link-quality proxy, clearly labelled as such);
   angle is a stable hash of the device identity so devices do not jump between frames.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from .models import Device, PlacedDevice, normalize_mac

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Zone:
    name: str
    x: float
    y: float
    w: float
    h: float


@dataclass(frozen=True)
class KnownDevice:
    label: str
    x: float
    y: float


@dataclass
class FloorPlan:
    name: str = "Coverage area"
    width: float = 12.0
    height: float = 8.0
    ap_x: float = 6.0
    ap_y: float = 4.0
    zones: list[Zone] = field(default_factory=list)
    known: dict[str, KnownDevice] = field(default_factory=dict)
    access_points: dict[str, tuple[float, float, str]] = field(default_factory=dict)   # bssid -> (x, y, label)
    host: Optional[tuple[float, float]] = None

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError("floor plan width/height must be positive")
        if not (0 <= self.ap_x <= self.width and 0 <= self.ap_y <= self.height):
            raise ValueError("access point must lie inside the floor plan")
        for z in self.zones:
            if z.w <= 0 or z.h <= 0:
                raise ValueError(f"zone {z.name!r} has non-positive size")
        for bssid, (x, y, _) in self.access_points.items():
            if not (0 <= x <= self.width and 0 <= y <= self.height):
                raise ValueError(f"access point {bssid} lies outside the floor plan")
        if self.host and not (0 <= self.host[0] <= self.width and 0 <= self.host[1] <= self.height):
            raise ValueError("host position lies outside the floor plan")
        self.known = {k.lower(): v for k, v in self.known.items()}
        self.access_points = {normalize_mac(k): v for k, v in self.access_points.items()}

    @classmethod
    def load(cls, path: Path | str) -> "FloorPlan":
        path = Path(path)
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            raise
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"cannot parse floor plan {path}: {exc}") from exc
        try:
            width = float(raw.get("width_m", 12.0))
            height = float(raw.get("height_m", 8.0))
            ap = raw.get("access_point", {})
            zones = [Zone(z["name"], float(z["x"]), float(z["y"]), float(z["w"]), float(z["h"])) for z in raw.get("zones", [])]
            known = {
                str(key): KnownDevice(str(v.get("label", key)), float(v["x"]), float(v["y"]))
                for key, v in raw.get("known_devices", {}).items()
            }
            aps = {
                str(key): (float(v["x"]), float(v["y"]), str(v.get("label", key)))
                for key, v in raw.get("access_points", {}).items()
            }
            host_raw = raw.get("host")
            host = (float(host_raw["x"]), float(host_raw["y"])) if host_raw else None
            plan = cls(
                name=str(raw.get("name", "Coverage area")),
                width=width,
                height=height,
                ap_x=float(ap.get("x", width / 2)),
                ap_y=float(ap.get("y", height / 2)),
                zones=zones,
                known=known,
                access_points=aps,
                host=host,
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"invalid floor plan {path}: {exc}") from exc
        log.info("floor plan %r loaded: %.1fx%.1f m, %d zones, %d known devices, %d known APs",
                 plan.name, plan.width, plan.height, len(zones), len(known), len(aps))
        return plan


class Mapper:
    def __init__(self, plan: FloorPlan, ring_min_frac: float = 0.12, ring_max_frac: float = 0.42, min_separation: float = 0.6) -> None:
        if not 0 < ring_min_frac < ring_max_frac <= 0.5:
            raise ValueError("ring fractions must satisfy 0 < min < max <= 0.5")
        self.plan = plan
        self._scale = min(plan.width, plan.height)
        self.r_min = ring_min_frac * self._scale
        self.r_max = ring_max_frac * self._scale
        self.min_sep = min_separation

    def place(self, devices: list[Device]) -> list[PlacedDevice]:
        placed: list[PlacedDevice] = []
        estimates: list[PlacedDevice] = []
        for dev in devices:
            known = self._lookup_known(dev)
            if known:
                placed.append(PlacedDevice(dev, known.x, known.y, "floorplan"))
            elif dev.is_self and self.plan.host:
                placed.append(PlacedDevice(dev, self.plan.host[0], self.plan.host[1], "floorplan"))
            elif dev.is_gateway:
                placed.append(PlacedDevice(dev, self.plan.ap_x, self.plan.ap_y, "anchor"))
            else:
                estimates.append(self._estimate(dev, placed + estimates))
        return placed + estimates

    def _lookup_known(self, dev: Device) -> Optional[KnownDevice]:
        for key in (dev.mac, dev.ip, dev.hostname, (dev.hostname or "").split(".")[0]):
            if key and key.lower() in self.plan.known:
                return self.plan.known[key.lower()]
        return None

    def _estimate(self, dev: Device, occupied: list[PlacedDevice]) -> PlacedDevice:
        key = dev.mac or dev.ip
        radius = self.r_min + (self.r_max - self.r_min) * self._rtt_norm(dev.rtt_ms, key)
        angle = self._stable_angle(key)
        step = math.radians(12.0)
        # Walk around the ring; after each full turn step the radius out (then in) so a
        # saturated ring spills into neighbouring rings instead of stacking markers.
        # If no slot satisfies min_sep, keep the candidate with the largest clearance.
        best_xy, best_clearance = (self.plan.ap_x, self.plan.ap_y), -1.0
        base_radius = radius
        for attempt in range(360):
            x, y = self._clamp(self.plan.ap_x + radius * math.cos(angle), self.plan.ap_y + radius * math.sin(angle))
            clearance = min((math.hypot(x - o.x, y - o.y) for o in occupied), default=math.inf)
            if clearance >= self.min_sep:
                return PlacedDevice(dev, x, y, "estimate")
            if clearance > best_clearance:
                best_xy, best_clearance = (x, y), clearance
            angle += step
            if (attempt + 1) % 30 == 0:
                turn = (attempt + 1) // 30
                delta = self.min_sep * 0.8 * ((turn + 1) // 2) * (1 if turn % 2 else -1)
                radius = max(self.r_min * 0.5, min(self._scale * 0.5, base_radius + delta))
        log.debug("no clear slot for %s; using best clearance %.2f m", key, best_clearance)
        return PlacedDevice(dev, best_xy[0], best_xy[1], "estimate")

    def _rtt_norm(self, rtt_ms: Optional[float], key: str = "") -> float:
        if rtt_ms is None:
            # Answered ARP only (typically a sleepy handset or a firewalled laptop):
            # spread deterministically across the outer band instead of one ring.
            return 0.6 + 0.3 * self._stable_fraction(key)
        return max(0.0, min(1.0, math.log10(1.0 + max(0.0, rtt_ms)) / math.log10(51.0)))

    @staticmethod
    def _stable_fraction(key: str) -> float:
        digest = hashlib.blake2b(key.encode("utf-8"), digest_size=4, person=b"radius").digest()
        return int.from_bytes(digest, "big") / 0xFFFFFFFF

    @staticmethod
    def _stable_angle(key: str) -> float:
        digest = hashlib.blake2b(key.encode("utf-8"), digest_size=4).digest()
        return math.radians(int.from_bytes(digest, "big") % 360)

    def _clamp(self, x: float, y: float) -> tuple[float, float]:
        pad = 0.3
        return (max(pad, min(self.plan.width - pad, x)), max(pad, min(self.plan.height - pad, y)))
