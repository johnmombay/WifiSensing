from wifisense.models import Confidence, Device, DeviceClass, LinkReading, MotionState, PlacedDevice
from wifisense.people import estimate_people


def _placed(ip, cls, x=1.0, y=1.0, **kw):
    d = Device(ip=ip, device_class=cls, **kw)
    return PlacedDevice(d, x, y, "estimate")


def test_device_evidence_ladder():
    placed = [
        _placed("10.0.0.1", DeviceClass.MOBILE, hostname="iPhone"),
        _placed("10.0.0.2", DeviceClass.UNKNOWN, vendor="Randomised"),
        _placed("10.0.0.3", DeviceClass.WORKSTATION, hostname="MREYES-LAPTOP"),
        _placed("10.0.0.4", DeviceClass.WORKSTATION, hostname="UPPI-LT-00210"),
        _placed("10.0.0.5", DeviceClass.PRINTER),
        _placed("10.0.0.6", DeviceClass.UNKNOWN, vendor="Randomised", open_ports=[80]),
    ]
    people = {p.device_ip: p for p in estimate_people(placed, [], None)}
    assert people["10.0.0.1"].confidence is Confidence.HIGH
    assert people["10.0.0.2"].confidence is Confidence.MEDIUM and people["10.0.0.2"].source == "handset-probable"
    assert people["10.0.0.3"].confidence is Confidence.MEDIUM and people["10.0.0.3"].source == "laptop-personal"
    assert people["10.0.0.4"].confidence is Confidence.LOW
    assert "10.0.0.5" not in people
    assert "10.0.0.6" not in people


def test_self_and_gateway_are_not_people():
    placed = [PlacedDevice(Device(ip="1.1.1.1", device_class=DeviceClass.WORKSTATION, is_self=True), 0, 0, "estimate"),
              PlacedDevice(Device(ip="1.1.1.2", device_class=DeviceClass.ROUTER, is_gateway=True), 0, 0, "anchor")]
    assert estimate_people(placed, [], (0, 0)) == []


def test_motion_candidate_between_host_and_perturbed_aps():
    links = [
        LinkReading("aa", "AP-A", -60, 0.8, MotionState.ACTIVE, (10.0, 0.0)),
        LinkReading("bb", "AP-B", -65, 0.7, MotionState.ACTIVE, (0.0, 10.0)),
        LinkReading("cc", "AP-C", -70, 0.05, MotionState.STILL, (10.0, 10.0)),
    ]
    people = estimate_people([], links, (0.0, 0.0))
    assert len(people) == 1
    m = people[0]
    assert m.source == "motion" and m.confidence is Confidence.HIGH
    # weighted midpoint of (5,0) and (0,5)
    assert abs(m.x - 5 * 0.8 / 1.5) < 1e-6 and abs(m.y - 5 * 0.7 / 1.5) < 1e-6
    assert "AP-C" not in m.evidence


def test_motion_without_ap_positions_lands_near_host():
    links = [LinkReading("aa", "AP-A", -60, 0.6, MotionState.ACTIVE, None)]
    people = estimate_people([], links, (3.0, 3.0))
    assert len(people) == 1 and people[0].confidence is Confidence.MEDIUM
    assert abs(people[0].x - 3.9) < 1e-6


def test_motion_merged_into_nearby_phone():
    phone = _placed("10.0.0.1", DeviceClass.MOBILE, x=4.6, y=4.6)
    links = [LinkReading("aa", "AP-A", -60, 0.9, MotionState.ACTIVE, (10.0, 10.0))]
    people = estimate_people([phone], links, (0.0, 0.0))   # midpoint (5,5) is within 1.2 m of the phone glyph
    assert [p.source for p in people] == ["phone"]


def test_no_motion_when_host_unknown():
    links = [LinkReading("aa", "AP-A", -60, 0.9, MotionState.ACTIVE, (1, 1))]
    assert estimate_people([], links, None) == []
