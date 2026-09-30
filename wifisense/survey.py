"""Survey points: where the host stood and what every AP radio measured there."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from .models import BssReading

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class SurveyPoint:
    x: float
    y: float
    timestamp: float
    rssi: dict[str, float]      # bssid -> dBm
    ssids: dict[str, str] = field(default_factory=dict)


class Survey:
    def __init__(self, path: Optional[Path | str] = None) -> None:
        self.path = Path(path) if path else None
        self.points: list[SurveyPoint] = []
        if self.path and self.path.exists():
            self.load()

    def __len__(self) -> int:
        return len(self.points)

    def add(self, x: float, y: float, readings: list[BssReading]) -> SurveyPoint:
        if not readings:
            raise ValueError("survey point needs at least one BSS reading")
        pt = SurveyPoint(float(x), float(y), time.time(), {r.bssid: r.rssi_dbm for r in readings},
                         {r.bssid: r.ssid for r in readings if r.ssid})
        self.points.append(pt)
        self.save()
        log.info("survey point %d at (%.2f, %.2f): %d radios", len(self.points), x, y, len(pt.rssi))
        return pt

    def undo(self) -> Optional[SurveyPoint]:
        if not self.points:
            return None
        pt = self.points.pop()
        self.save()
        log.info("survey point removed; %d remain", len(self.points))
        return pt

    def bssids(self) -> set[str]:
        return {b for p in self.points for b in p.rssi}

    def ssid_for(self, bssid: str) -> str:
        for p in reversed(self.points):
            if p.ssids.get(bssid):
                return p.ssids[bssid]
        return ""

    def load(self) -> None:
        assert self.path is not None
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
            self.points = [
                SurveyPoint(float(p["x"]), float(p["y"]), float(p.get("t", 0)),
                            {str(k): float(v) for k, v in p["rssi"].items()},
                            {str(k): str(v) for k, v in p.get("ssids", {}).items()})
                for p in raw.get("points", [])
            ]
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise ValueError(f"cannot read survey {self.path}: {exc}") from exc
        log.info("survey loaded: %d points from %s", len(self.points), self.path)

    def save(self) -> None:
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"version": 1, "points": [
            {"x": p.x, "y": p.y, "t": p.timestamp, "rssi": p.rssi, "ssids": p.ssids} for p in self.points
        ]}
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(json.dumps(payload, indent=1), encoding="utf-8")
        tmp.replace(self.path)
