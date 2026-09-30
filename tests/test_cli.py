import pytest

from wifisense.cli import _build_engine, build_parser, parse_room


@pytest.mark.parametrize("spec, expected", [("8x6", (8.0, 6.0)), (" 7.5 X 4 ", (7.5, 4.0)), ("10*12", (10.0, 12.0))])
def test_parse_room(spec, expected):
    assert parse_room(spec) == expected


@pytest.mark.parametrize("spec", ["8", "8x", "axb", "0.5x3", "300x3", ""])
def test_parse_room_rejects(spec):
    with pytest.raises(ValueError):
        parse_room(spec)


def test_room_without_floorplan_sets_plan_size():
    args = build_parser().parse_args(["--simulate", "--room", "8x6"])
    engine, plan, mapper = _build_engine(args)
    assert (plan.width, plan.height, plan.ap_x, plan.ap_y) == (8.0, 6.0, 4.0, 3.0)
    assert plan.access_points == {} and plan.host is None


def test_default_plan_without_floorplan():
    engine, plan, _ = _build_engine(build_parser().parse_args(["--simulate"]))
    assert (plan.width, plan.height) == (12.0, 8.0)
