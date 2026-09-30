"""Core data types shared across the package."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Optional

_MAC_RE = re.compile(r"^[0-9a-f]{2}(:[0-9a-f]{2}){5}$")


class DeviceClass(str, Enum):
    ROUTER = "router"
    WORKSTATION = "workstation"
    SERVER = "server"
    MOBILE = "mobile"
    PRINTER = "printer"
    IOT = "iot"
    UNKNOWN = "unknown"

    @property
    def is_person_proxy(self) -> bool:
        """Mobile handsets travel with people; treat them as presence proxies."""
        return self is DeviceClass.MOBILE

    @property
    def is_computer(self) -> bool:
        return self in (DeviceClass.WORKSTATION, DeviceClass.SERVER)


class MotionState(str, Enum):
    CALIBRATING = "calibrating"
    STILL = "still"
    LOW = "low"
    ACTIVE = "active"
    UNAVAILABLE = "unavailable"


class Confidence(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


def normalize_mac(raw: str) -> str:
    """Normalise ``AA-BB-CC-DD-EE-FF`` / ``aabb.ccdd.eeff`` / ``aa:bb:..`` to lowercase colon form."""
    hexdigits = re.sub(r"[^0-9a-fA-F]", "", raw).lower()
    if len(hexdigits) != 12:
        raise ValueError(f"invalid MAC address: {raw!r}")
    mac = ":".join(hexdigits[i : i + 2] for i in range(0, 12, 2))
    if not _MAC_RE.match(mac):  # pragma: no cover - defensive
        raise ValueError(f"invalid MAC address: {raw!r}")
    return mac


@dataclass
class Device:
    ip: str
    mac: Optional[str] = None
    hostname: Optional[str] = None
    vendor: Optional[str] = None
    vendor_hint: Optional[str] = None
    open_ports: list[int] = field(default_factory=list)
    rtt_ms: Optional[float] = None
    device_class: DeviceClass = DeviceClass.UNKNOWN
    is_self: bool = False
    is_gateway: bool = False
    last_seen: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def label(self) -> str:
        if self.hostname:
            return self.hostname.split(".")[0]
        if self.vendor:
            return f"{self.vendor} ({self.ip.rsplit('.', 1)[-1]})"
        return self.ip

    def to_dict(self) -> dict:
        return {
            "ip": self.ip,
            "mac": self.mac,
            "hostname": self.hostname,
            "vendor": self.vendor,
            "open_ports": list(self.open_ports),
            "rtt_ms": self.rtt_ms,
            "class": self.device_class.value,
            "is_self": self.is_self,
            "is_gateway": self.is_gateway,
            "last_seen": self.last_seen.isoformat(),
        }


@dataclass(frozen=True)
class SignalSample:
    timestamp: float
    rssi_dbm: float
    ssid: Optional[str] = None
    bssid: Optional[str] = None
    link_mbps: Optional[float] = None


@dataclass(frozen=True)
class BssReading:
    """One access-point radio as seen by a scan."""
    bssid: str
    ssid: str
    rssi_dbm: float
    freq_mhz: Optional[int]
    timestamp: float


@dataclass(frozen=True)
class PresenceReading:
    state: MotionState
    motion_score: float          # 0.0 (still) .. 1.0 (strong motion)
    rssi_dbm: Optional[float]
    rssi_std_db: Optional[float]
    noise_floor_db: Optional[float]
    samples: int


@dataclass(frozen=True)
class LinkReading:
    """Motion state of one host<->AP radio link."""
    bssid: str
    ssid: str
    rssi_dbm: Optional[float]
    motion_score: float
    state: MotionState
    ap_xy: Optional[tuple[float, float]]
    connected: bool = False

    def to_dict(self) -> dict:
        return {
            "bssid": self.bssid, "ssid": self.ssid, "rssi_dbm": self.rssi_dbm,
            "motion_score": round(self.motion_score, 3), "state": self.state.value,
            "ap_xy": list(self.ap_xy) if self.ap_xy else None, "connected": self.connected,
        }


@dataclass(frozen=True)
class PersonCandidate:
    id: str
    x: float
    y: float
    confidence: Confidence
    source: str          # phone | handset-probable | laptop-personal | laptop | motion
    evidence: str
    device_ip: Optional[str] = None

    def to_dict(self) -> dict:
        return {"id": self.id, "x": round(self.x, 2), "y": round(self.y, 2), "confidence": self.confidence.value,
                "source": self.source, "evidence": self.evidence, "device_ip": self.device_ip}


@dataclass(frozen=True)
class PlacedDevice:
    device: Device
    x: float
    y: float
    placed_by: str   # "floorplan" | "estimate" | "anchor"


@dataclass(frozen=True)
class WallSegment:
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float

    def to_dict(self) -> dict:
        return {"x1": round(self.x1, 2), "y1": round(self.y1, 2), "x2": round(self.x2, 2), "y2": round(self.y2, 2),
                "confidence": round(self.confidence, 2)}


@dataclass(frozen=True)
class Opening:
    x: float
    y: float
    confidence: float

    def to_dict(self) -> dict:
        return {"x": round(self.x, 2), "y": round(self.y, 2), "confidence": round(self.confidence, 2)}


@dataclass(frozen=True)
class ApEstimate:
    bssid: str
    label: str
    x: float
    y: float
    method: str          # "floorplan" | "fitted"
    p0_dbm: float
    residual_db: float
    points: int

    def to_dict(self) -> dict:
        return {"bssid": self.bssid, "label": self.label, "x": round(self.x, 2), "y": round(self.y, 2),
                "method": self.method, "p0_dbm": round(self.p0_dbm, 1), "residual_db": round(self.residual_db, 2),
                "points": self.points}


@dataclass
class TerrainModel:
    cell: float
    width: float
    height: float
    obstruction: list[list[Optional[float]]]   # [row][col], 0..1, None where no survey coverage
    excess_db: list[list[Optional[float]]]     # mean excess attenuation vs free space
    walls: list[WallSegment]
    openings: list[Opening]
    aps: dict[str, ApEstimate]
    survey_points: int

    def to_dict(self) -> dict:
        return {
            "cell_m": self.cell, "survey_points": self.survey_points,
            "access_points": {k: v.to_dict() for k, v in self.aps.items()},
            "walls": [w.to_dict() for w in self.walls],
            "openings": [o.to_dict() for o in self.openings],
        }


@dataclass
class SensingState:
    timestamp: datetime
    devices: list[Device]
    placed: list[PlacedDevice]
    presence: PresenceReading
    rssi_history: list[tuple[float, float]]
    ssid: Optional[str]
    bssid: Optional[str]
    subnet: str
    last_scan: Optional[datetime]
    scan_in_progress: bool
    links: list[LinkReading] = field(default_factory=list)
    people: list[PersonCandidate] = field(default_factory=list)
    terrain: Optional[TerrainModel] = None
    host_xy: Optional[tuple[float, float]] = None
    survey_points: int = 0
    survey_enabled: bool = False

    @property
    def mobiles(self) -> int:
        return sum(1 for d in self.devices if d.device_class.is_person_proxy)

    @property
    def computers(self) -> int:
        return sum(1 for d in self.devices if d.device_class.is_computer)

    @property
    def likely_people(self) -> int:
        return sum(1 for p in self.people if p.confidence in (Confidence.HIGH, Confidence.MEDIUM))

    @property
    def potential_people(self) -> int:
        return len(self.people)

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp.isoformat(),
            "ssid": self.ssid,
            "bssid": self.bssid,
            "subnet": self.subnet,
            "last_scan": self.last_scan.isoformat() if self.last_scan else None,
            "host_xy": list(self.host_xy) if self.host_xy else None,
            "summary": {
                "devices": len(self.devices),
                "computers": self.computers,
                "mobiles": self.mobiles,
                "likely_people": self.likely_people,
                "potential_people": self.potential_people,
                "motion_state": self.presence.state.value,
                "motion_score": round(self.presence.motion_score, 3),
                "links_monitored": len(self.links),
                "survey_points": self.survey_points,
            },
            "presence": {
                "rssi_dbm": self.presence.rssi_dbm,
                "rssi_std_db": self.presence.rssi_std_db,
                "noise_floor_db": self.presence.noise_floor_db,
                "samples": self.presence.samples,
            },
            "links": [l.to_dict() for l in self.links],
            "people": [p.to_dict() for p in self.people],
            "terrain": self.terrain.to_dict() if self.terrain else None,
            "devices": [
                {**p.device.to_dict(), "x": round(p.x, 2), "y": round(p.y, 2), "placed_by": p.placed_by}
                for p in self.placed
            ],
        }
