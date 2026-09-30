import json

import pytest

from wifisense.mapping import FloorPlan, KnownDevice, Mapper, Zone
from wifisense.models import Device


def _plan():
    return FloorPlan(
        width=12, height=8, ap_x=6, ap_y=4,
        zones=[Zone("A", 0.5, 0.5, 3, 3)],
        known={"F8:B1:56:AA:01:01": KnownDevice("desk", 1.0, 1.0), "srv-files01": KnownDevice("srv", 10.0, 1.0)},
    )


def test_known_device_placed_by_mac_and_hostname():
    placed = Mapper(_plan()).place([
        Device(ip="10.0.0.5", mac="f8:b1:56:aa:01:01"),
        Device(ip="10.0.0.6", hostname="srv-files01.corp.local"),
    ])
    assert {(p.x, p.y, p.placed_by) for p in placed} == {(1.0, 1.0, "floorplan"), (10.0, 1.0, "floorplan")}


def test_gateway_anchored_at_ap():
    p = Mapper(_plan()).place([Device(ip="10.0.0.1", is_gateway=True)])[0]
    assert (p.x, p.y, p.placed_by) == (6.0, 4.0, "anchor")


def test_self_placed_at_host_position():
    plan = FloorPlan(width=12, height=8, host=(2.5, 3.5))
    p = Mapper(plan).place([Device(ip="10.0.0.9", is_self=True)])[0]
    assert (p.x, p.y, p.placed_by) == (2.5, 3.5, "floorplan")


def test_estimates_deterministic_inside_bounds_and_separated():
    mapper = Mapper(_plan())
    devices = [Device(ip=f"10.0.0.{i}", mac=f"aa:bb:cc:00:00:{i:02x}", rtt_ms=float(i)) for i in range(2, 30)]
    first = mapper.place(devices)
    second = mapper.place(devices)
    assert [(p.x, p.y) for p in first] == [(p.x, p.y) for p in second]
    for p in first:
        assert p.placed_by == "estimate"
        assert 0 <= p.x <= 12 and 0 <= p.y <= 8
    coords = [(p.x, p.y) for p in first]
    close_pairs = sum(1 for i in range(len(coords)) for j in range(i + 1, len(coords)) if ((coords[i][0] - coords[j][0]) ** 2 + (coords[i][1] - coords[j][1]) ** 2) ** 0.5 < 0.3)
    assert close_pairs == 0


def test_rtt_orders_radius():
    mapper = Mapper(_plan())
    assert mapper._rtt_norm(0.0) == 0.0
    assert mapper._rtt_norm(50.0) == pytest.approx(1.0)
    assert 0.0 < mapper._rtt_norm(5.0) < mapper._rtt_norm(30.0)
    assert 0.6 <= mapper._rtt_norm(None, "aa:bb:cc:00:00:01") <= 0.9
    assert mapper._rtt_norm(None, "k1") == mapper._rtt_norm(None, "k1")


def test_saturated_ring_spills_without_stacking():
    mapper = Mapper(_plan())
    devices = [Device(ip=f"10.0.1.{i}", mac=f"aa:bb:cc:01:00:{i:02x}", rtt_ms=None) for i in range(1, 120)]
    placed = mapper.place(devices)
    coords = [(p.x, p.y) for p in placed]
    assert len(set((round(x, 2), round(y, 2)) for x, y in coords)) == len(coords)
    for x, y in coords:
        assert 0 <= x <= 12 and 0 <= y <= 8


def test_load_floorplan(tmp_path):
    f = tmp_path / "plan.json"
    f.write_text(json.dumps({"name": "T", "width_m": 5, "height_m": 4, "access_point": {"x": 2, "y": 2},
                             "host": {"x": 1, "y": 1},
                             "access_points": {"AA-BB-CC-DD-EE-01": {"x": 2, "y": 2, "label": "main"}},
                             "zones": [{"name": "z", "x": 0, "y": 0, "w": 1, "h": 1}],
                             "known_devices": {"AA:BB:CC:DD:EE:FF": {"label": "d", "x": 1, "y": 1}}}), encoding="utf-8")
    plan = FloorPlan.load(f)
    assert plan.name == "T" and plan.width == 5 and len(plan.zones) == 1
    assert "aa:bb:cc:dd:ee:ff" in plan.known
    assert plan.host == (1.0, 1.0)
    assert plan.access_points["aa:bb:cc:dd:ee:01"] == (2.0, 2.0, "main")


def test_invalid_floorplan_rejected(tmp_path):
    f = tmp_path / "bad.json"
    f.write_text('{"width_m": -1}', encoding="utf-8")
    with pytest.raises(ValueError):
        FloorPlan.load(f)
    with pytest.raises(ValueError):
        FloorPlan(width=4, height=4, ap_x=9, ap_y=1)
    with pytest.raises(ValueError):
        FloorPlan(width=4, height=4, host=(5, 1))
