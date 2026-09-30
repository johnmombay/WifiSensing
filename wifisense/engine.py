"""Orchestrates device scans, link sampling, survey capture and terrain inference into a thread-safe state."""

from __future__ import annotations

import logging
import threading
import time
from datetime import datetime, timezone
from typing import Optional

from .links import LinkMonitor
from .mapping import FloorPlan, Mapper
from .models import Device, LinkReading, MotionState, PlacedDevice, PresenceReading, SensingState, TerrainModel
from .netinfo import SignalSource
from .people import estimate_people
from .presence import PresenceDetector
from .scanner import DeviceScanner
from .survey import Survey
from .terrain import build_terrain

log = logging.getLogger(__name__)


class SensingEngine:
    def __init__(
        self,
        scanner: DeviceScanner,
        signal_source: SignalSource,
        detector: PresenceDetector,
        mapper: Mapper,
        subnet: str,
        scan_interval: float = 60.0,
        sample_interval: float = 1.0,
        link_monitor: Optional[LinkMonitor] = None,
        link_interval: float = 5.0,
        survey: Optional[Survey] = None,
        terrain_cell: float = 0.25,
    ) -> None:
        if scan_interval < 5:
            raise ValueError("scan_interval must be >= 5 s")
        if not 0.2 <= sample_interval <= 30:
            raise ValueError("sample_interval must be within 0.2..30 s")
        if not 1.0 <= link_interval <= 120:
            raise ValueError("link_interval must be within 1..120 s")
        self.scanner = scanner
        self.signal_source = signal_source
        self.detector = detector
        self.mapper = mapper
        self.plan: FloorPlan = mapper.plan
        self.subnet = subnet
        self.scan_interval = scan_interval
        self.sample_interval = sample_interval
        self.link_monitor = link_monitor
        self.link_interval = link_interval
        self.survey = survey
        self.terrain_cell = terrain_cell

        self._lock = threading.Lock()
        self._terrain_lock = threading.Lock()
        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []
        self._devices: list[Device] = []
        self._placed: list[PlacedDevice] = []
        self._last_scan: Optional[datetime] = None
        self._scanning = False
        self._presence = PresenceReading(MotionState.CALIBRATING, 0.0, None, None, None, 0)
        self._ssid: Optional[str] = None
        self._bssid: Optional[str] = None
        self._links: list[LinkReading] = []
        self._terrain: Optional[TerrainModel] = None
        self._host_xy: Optional[tuple[float, float]] = self.plan.host
        if self.survey is not None and len(self.survey):
            last = self.survey.points[-1]
            self._host_xy = self._host_xy or (last.x, last.y)
            self._rebuild_terrain_async()

    # ------------------------------------------------------------------ lifecycle
    def start(self) -> None:
        if self._threads:
            raise RuntimeError("engine already started")
        self._stop.clear()
        self._threads = [
            threading.Thread(target=self._scan_loop, name="wifisense-scan", daemon=True),
            threading.Thread(target=self._sample_loop, name="wifisense-sample", daemon=True),
        ]
        if self.link_monitor is not None:
            self._threads.append(threading.Thread(target=self._link_loop, name="wifisense-links", daemon=True))
        for t in self._threads:
            t.start()
        log.info("engine started (scan %.0fs, sample %.1fs, links %s)", self.scan_interval, self.sample_interval,
                 f"{self.link_interval:.0f}s" if self.link_monitor else "off")

    def stop(self, timeout: float = 5.0) -> None:
        self._stop.set()
        for t in self._threads:
            t.join(timeout)
        self._threads = []
        log.info("engine stopped")

    def run_once(self, samples: int = 12, link_samples: int = 6) -> SensingState:
        """Synchronous single pass: one scan, ``samples`` signal readings, ``link_samples`` link scans."""
        if samples < 1:
            raise ValueError("samples must be >= 1")
        self._do_scan()
        for _ in range(samples):
            self._do_sample()
            time.sleep(self.sample_interval)
        if self.link_monitor is not None:
            for _ in range(max(0, link_samples)):
                self._do_links()
        self._wait_terrain()
        return self.snapshot()

    # ------------------------------------------------------------------ survey / terrain
    def record_survey_point(self, x: float, y: float, scans: int = 2) -> int:
        """Record a survey point at the host's current location (a click on the map).

        ``scans`` fresh radio scans are averaged per BSSID to knock down per-scan RSSI noise.
        """
        if self.survey is None:
            raise RuntimeError("survey mode is not enabled")
        if not (0 <= x <= self.plan.width and 0 <= y <= self.plan.height):
            raise ValueError("survey point outside the floor plan")
        readings = self._averaged_scan(scans) if self.link_monitor is not None else []
        if not readings:
            sample = self.signal_source.sample()
            if sample is None or sample.bssid is None:
                raise RuntimeError("no radio readings available for the survey point")
            from .models import BssReading
            readings = [BssReading(sample.bssid, sample.ssid or "", sample.rssi_dbm, None, sample.timestamp)]
        self.survey.add(x, y, readings)
        with self._lock:
            self._host_xy = (x, y)
        self._rebuild_terrain_async()
        return len(self.survey)

    def _averaged_scan(self, scans: int) -> list:
        from .models import BssReading

        assert self.link_monitor is not None
        acc: dict[str, list[float]] = {}
        ssids: dict[str, str] = {}
        freqs: dict[str, Optional[int]] = {}
        for _ in range(max(1, scans)):
            for r in self.link_monitor.scanner.scan():
                acc.setdefault(r.bssid, []).append(r.rssi_dbm)
                ssids[r.bssid] = r.ssid or ssids.get(r.bssid, "")
                freqs[r.bssid] = r.freq_mhz
        now = time.time()
        return [BssReading(b, ssids[b], sum(v) / len(v), freqs[b], now) for b, v in acc.items()]

    def undo_survey_point(self) -> int:
        if self.survey is None:
            raise RuntimeError("survey mode is not enabled")
        self.survey.undo()
        with self._lock:
            if self.survey.points:
                last = self.survey.points[-1]
                self._host_xy = (last.x, last.y)
            else:
                self._host_xy = self.plan.host
        self._rebuild_terrain_async()
        return len(self.survey)

    def _rebuild_terrain_async(self) -> None:
        threading.Thread(target=self._rebuild_terrain, name="wifisense-terrain", daemon=True).start()

    def _rebuild_terrain(self) -> None:
        if self.survey is None:
            return
        with self._terrain_lock:
            try:
                terrain = build_terrain(self.survey, self.plan.width, self.plan.height, self.plan.access_points, cell=self.terrain_cell)
            except Exception:  # noqa: BLE001
                log.exception("terrain build failed")
                return
        with self._lock:
            self._terrain = terrain

    def _wait_terrain(self, timeout: float = 30.0) -> None:
        deadline = time.monotonic() + timeout
        for t in threading.enumerate():
            if t.name == "wifisense-terrain":
                t.join(max(0.0, deadline - time.monotonic()))

    # ------------------------------------------------------------------ workers
    def _scan_loop(self) -> None:
        while not self._stop.is_set():
            self._do_scan()
            self._stop.wait(self.scan_interval)

    def _sample_loop(self) -> None:
        while not self._stop.is_set():
            started = time.monotonic()
            self._do_sample()
            self._stop.wait(max(0.0, self.sample_interval - (time.monotonic() - started)))

    def _link_loop(self) -> None:
        while not self._stop.is_set():
            started = time.monotonic()
            self._do_links()
            self._stop.wait(max(0.0, self.link_interval - (time.monotonic() - started)))

    def _do_scan(self) -> None:
        with self._lock:
            self._scanning = True
        try:
            devices = self.scanner.scan()
        except Exception:  # noqa: BLE001 - keep the loop alive; surface via log
            log.exception("device scan failed")
            devices = None
        # Placement is O(n^2) on dense subnets; compute it once per scan, not per frame.
        placed = self.mapper.place(devices) if devices is not None else None
        with self._lock:
            self._scanning = False
            if devices is not None and placed is not None:
                self._devices = devices
                self._placed = placed
                self._last_scan = datetime.now(timezone.utc)

    def _do_sample(self) -> None:
        try:
            sample = self.signal_source.sample()
        except Exception:  # noqa: BLE001
            log.exception("signal sample failed")
            sample = None
        with self._lock:
            self._presence = self.detector.update(sample)
            if sample is not None:
                self._ssid, self._bssid = sample.ssid, sample.bssid

    def _do_links(self) -> None:
        assert self.link_monitor is not None
        try:
            links = self.link_monitor.sample()
        except Exception:  # noqa: BLE001
            log.exception("link sample failed")
            return
        with self._lock:
            self._links = links

    # ------------------------------------------------------------------ state
    def _ap_position(self, bssid: Optional[str], terrain: Optional[TerrainModel]) -> Optional[tuple[float, float]]:
        if bssid is None:
            return None
        if bssid in self.plan.access_points:
            x, y, _ = self.plan.access_points[bssid]
            return (x, y)
        if terrain and bssid in terrain.aps:
            return (terrain.aps[bssid].x, terrain.aps[bssid].y)
        if bssid == self._bssid:
            return (self.plan.ap_x, self.plan.ap_y)
        return None

    def snapshot(self) -> SensingState:
        with self._lock:
            devices = list(self._devices)
            placed = list(self._placed)
            presence = self._presence
            history = self.detector.history
            last_scan, scanning = self._last_scan, self._scanning
            ssid, bssid = self._ssid, self._bssid
            raw_links = list(self._links)
            terrain = self._terrain
            host_xy = self._host_xy

        if host_xy is None:
            host_xy = next(((p.x, p.y) for p in placed if p.device.is_self), None)

        links: list[LinkReading] = []
        if bssid is not None and presence.state is not MotionState.UNAVAILABLE:
            links.append(LinkReading(bssid, ssid or "", presence.rssi_dbm, presence.motion_score, presence.state,
                                     self._ap_position(bssid, terrain), connected=True))
        for l in raw_links:
            if l.bssid == bssid:
                continue
            links.append(LinkReading(l.bssid, l.ssid, l.rssi_dbm, l.motion_score, l.state, self._ap_position(l.bssid, terrain)))

        people = estimate_people(placed, links, host_xy)
        return SensingState(
            timestamp=datetime.now(timezone.utc),
            devices=devices,
            placed=placed,
            presence=presence,
            rssi_history=history,
            ssid=ssid,
            bssid=bssid,
            subnet=self.subnet,
            last_scan=last_scan,
            scan_in_progress=scanning,
            links=links,
            people=people,
            terrain=terrain,
            host_xy=host_xy,
            survey_points=len(self.survey) if self.survey is not None else 0,
            survey_enabled=self.survey is not None,
        )
