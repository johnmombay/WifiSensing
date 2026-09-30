"""Per-access-point link motion monitoring.

Each radio in range is a separate propagation path through the room. Running the RSSI
variance detector on every strong link turns one laptop into a crude sensor array: motion
shows up first on the links whose path the person crosses, which gives a direction.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

from .models import BssReading, LinkReading, MotionState, SignalSample
from .presence import PresenceConfig, PresenceDetector
from .wlan import BssScanner

log = logging.getLogger(__name__)


@dataclass
class LinkConfig:
    max_links: int = 12
    min_rssi_dbm: float = -85.0
    window: int = 8
    calibration_samples: int = 6
    sensitivity_db: float = 3.0   # scan-to-scan RSSI is noisier than the connected link

    def __post_init__(self) -> None:
        if not 1 <= self.max_links <= 64:
            raise ValueError("max_links must be within 1..64")
        if not -100 <= self.min_rssi_dbm <= -30:
            raise ValueError("min_rssi_dbm must be within -100..-30")


class LinkMonitor:
    def __init__(self, scanner: BssScanner, config: Optional[LinkConfig] = None) -> None:
        self.scanner = scanner
        self.cfg = config or LinkConfig()
        self._detectors: dict[str, PresenceDetector] = {}
        self._ssids: dict[str, str] = {}
        self._latest: list[BssReading] = []

    @property
    def latest_scan(self) -> list[BssReading]:
        return list(self._latest)

    def sample(self) -> list[LinkReading]:
        readings = self.scanner.scan()
        self._latest = readings
        strong = sorted((r for r in readings if r.rssi_dbm >= self.cfg.min_rssi_dbm), key=lambda r: -r.rssi_dbm)
        tracked = strong[: self.cfg.max_links]
        seen = {r.bssid for r in tracked}

        out: list[LinkReading] = []
        for r in tracked:
            det = self._detectors.get(r.bssid)
            if det is None:
                det = self._detectors[r.bssid] = PresenceDetector(
                    PresenceConfig(window=self.cfg.window, calibration_samples=self.cfg.calibration_samples,
                                   sensitivity_db=self.cfg.sensitivity_db, history=60)
                )
            self._ssids[r.bssid] = r.ssid or self._ssids.get(r.bssid, "")
            pr = det.update(SignalSample(r.timestamp, r.rssi_dbm, r.ssid, r.bssid))
            out.append(LinkReading(r.bssid, self._ssids[r.bssid], r.rssi_dbm, pr.motion_score, pr.state, None))

        for bssid, det in list(self._detectors.items()):
            if bssid not in seen:
                pr = det.update(None)
                if pr.state is MotionState.UNAVAILABLE and pr.samples == 0:
                    del self._detectors[bssid]
        return out
