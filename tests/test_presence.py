import random

from wifisense.models import MotionState, SignalSample
from wifisense.presence import PresenceConfig, PresenceDetector


def _feed(detector, values, t0=0.0):
    reading = None
    for i, v in enumerate(values):
        reading = detector.update(SignalSample(timestamp=t0 + i, rssi_dbm=v))
    return reading


def test_calibrating_then_still():
    det = PresenceDetector(PresenceConfig(window=10, calibration_samples=10))
    rng = random.Random(1)
    quiet = [-58 + rng.gauss(0, 0.4) for _ in range(40)]
    early = _feed(det, quiet[:5])
    assert early.state is MotionState.CALIBRATING
    late = _feed(det, quiet[5:], t0=5)
    assert late.state is MotionState.STILL
    assert late.motion_score < 0.25


def test_motion_raises_score():
    det = PresenceDetector(PresenceConfig(window=10, calibration_samples=10, sensitivity_db=2.0))
    rng = random.Random(2)
    _feed(det, [-58 + rng.gauss(0, 0.4) for _ in range(40)])
    reading = _feed(det, [-58 + rng.gauss(0, 3.0) for _ in range(20)], t0=40)
    assert reading.state is MotionState.ACTIVE
    assert reading.motion_score > 0.55


def test_unavailable_after_misses():
    det = PresenceDetector()
    assert det.update(None).state is MotionState.UNAVAILABLE
    _feed(det, [-60.0] * 15)
    det.update(None)
    det.update(None)
    assert det.update(None).state is MotionState.UNAVAILABLE


def test_history_bounded():
    det = PresenceDetector(PresenceConfig(history=20))
    _feed(det, [-60.0] * 50)
    assert len(det.history) == 20
