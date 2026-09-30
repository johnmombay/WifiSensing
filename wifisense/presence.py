"""RSSI-variance presence/motion detector.

Physics: a person moving between (or near) the client and the access point perturbs
multipath propagation, which shows up as increased short-term variance of the received
signal strength. A static room has a tight noise floor; motion widens it.

This is single-link, single-antenna sensing. It reports *whether* the environment is
changing, not *where*. Per-person localisation needs CSI-capable hardware.
"""

from __future__ import annotations

import logging
import statistics
from collections import deque
from dataclasses import dataclass
from typing import Optional

from .models import MotionState, PresenceReading, SignalSample

log = logging.getLogger(__name__)


@dataclass
class PresenceConfig:
    window: int = 15               # samples in the variance window
    calibration_samples: int = 10  # samples before reporting a state
    sensitivity_db: float = 2.0    # std-dev rise above the noise floor that maps to score 1.0
    low_threshold: float = 0.25
    active_threshold: float = 0.55
    history: int = 300             # samples kept for the sparkline

    def __post_init__(self) -> None:
        if self.window < 3:
            raise ValueError("window must be >= 3")
        if self.calibration_samples < 3:
            raise ValueError("calibration_samples must be >= 3")
        if self.sensitivity_db <= 0:
            raise ValueError("sensitivity_db must be > 0")
        if not 0 < self.low_threshold < self.active_threshold <= 1.0:
            raise ValueError("thresholds must satisfy 0 < low < active <= 1")


class PresenceDetector:
    def __init__(self, config: Optional[PresenceConfig] = None) -> None:
        self.cfg = config or PresenceConfig()
        self._window: deque[float] = deque(maxlen=self.cfg.window)
        self._history: deque[tuple[float, float]] = deque(maxlen=self.cfg.history)
        self._noise_floor: Optional[float] = None
        self._total = 0
        self._misses = 0

    @property
    def history(self) -> list[tuple[float, float]]:
        return list(self._history)

    def update(self, sample: Optional[SignalSample]) -> PresenceReading:
        if sample is None:
            self._misses += 1
            state = MotionState.UNAVAILABLE if self._misses >= 3 or self._total == 0 else self._current_state()
            return PresenceReading(state, 0.0, None, self._window_std(), self._noise_floor, self._total)

        self._misses = 0
        self._total += 1
        self._window.append(sample.rssi_dbm)
        self._history.append((sample.timestamp, sample.rssi_dbm))

        std = self._window_std()
        if self._total < self.cfg.calibration_samples or std is None:
            return PresenceReading(MotionState.CALIBRATING, 0.0, sample.rssi_dbm, std, self._noise_floor, self._total)

        # Adaptive floor: track the quietest window seen, drifting slowly upward so a
        # one-off dead-still period does not pin the floor at zero forever.
        if self._noise_floor is None:
            self._noise_floor = std
        elif std < self._noise_floor:
            self._noise_floor = std
        else:
            self._noise_floor += 0.01 * (std - self._noise_floor)

        score = (std - self._noise_floor) / self.cfg.sensitivity_db
        score = max(0.0, min(1.0, score))
        return PresenceReading(self._state_for(score), score, sample.rssi_dbm, std, self._noise_floor, self._total)

    def _window_std(self) -> Optional[float]:
        if len(self._window) < 3:
            return None
        return statistics.pstdev(self._window)

    def _state_for(self, score: float) -> MotionState:
        if score >= self.cfg.active_threshold:
            return MotionState.ACTIVE
        if score >= self.cfg.low_threshold:
            return MotionState.LOW
        return MotionState.STILL

    def _current_state(self) -> MotionState:
        std = self._window_std()
        if std is None or self._noise_floor is None:
            return MotionState.CALIBRATING
        return self._state_for(max(0.0, min(1.0, (std - self._noise_floor) / self.cfg.sensitivity_db)))
