import random

from wifisense.links import LinkConfig, LinkMonitor
from wifisense.models import BssReading, MotionState


class FakeScanner:
    def __init__(self):
        self.rng = random.Random(1)
        self.sigma = {"aa": 0.3, "bb": 0.3}
        self.t = 0.0

    def scan(self):
        self.t += 5
        return [
            BssReading("aa", "A", -55 + self.rng.gauss(0, self.sigma["aa"]), 2437, self.t),
            BssReading("bb", "B", -70 + self.rng.gauss(0, self.sigma["bb"]), 2437, self.t),
            BssReading("cc", "weak", -92 + self.rng.gauss(0, 0.3), 2437, self.t),
        ]


def test_link_monitor_tracks_strong_links_and_detects_motion():
    scanner = FakeScanner()
    mon = LinkMonitor(scanner, LinkConfig(window=8, calibration_samples=6, sensitivity_db=3.0))
    for _ in range(25):
        readings = mon.sample()
    assert {r.bssid for r in readings} == {"aa", "bb"}          # -92 dBm neighbour filtered
    assert all(r.state is MotionState.STILL for r in readings)
    scanner.sigma["aa"] = 4.0
    for _ in range(10):
        readings = mon.sample()
    by = {r.bssid: r for r in readings}
    assert by["aa"].state is MotionState.ACTIVE and by["aa"].motion_score > 0.55
    assert by["bb"].state is MotionState.STILL
    assert mon.latest_scan and len(mon.latest_scan) == 3
