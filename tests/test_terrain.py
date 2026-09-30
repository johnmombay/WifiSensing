import math

import numpy as np

from wifisense.simulate import SIM_APS, SIM_DOORS, SIM_WALLS, SimWorld, generate_survey, point_segment_distance, segments_intersect, walls_crossed
from wifisense.survey import Survey
from wifisense.terrain import build_terrain, localize_ap


def _survey(step=1.0):
    return generate_survey(SimWorld(seed=3), 12.0, 8.0, step=step)


def test_geometry_helpers():
    assert segments_intersect((0, 0), (2, 2), (0, 2), (2, 0))
    assert not segments_intersect((0, 0), (1, 1), (2, 2), (3, 3))
    assert walls_crossed((4.0, 2.0), (6.5, 2.0)) == 1          # crosses the partition
    assert walls_crossed((4.0, 3.9), (6.5, 3.9)) == 0          # through the door
    assert point_segment_distance((0, 1), (-1, 0), (1, 0)) == 1.0


def test_localize_ap_free_space():
    rng = np.random.default_rng(0)
    xs = rng.uniform(0, 12, 40); ys = rng.uniform(0, 8, 40)
    rssi = -40 - 22 * np.log10(np.maximum(np.hypot(xs - 8.0, ys - 3.0), 0.5)) + rng.normal(0, 0.5, 40)
    x, y, p0, resid = localize_ap(xs, ys, rssi, 12.0, 8.0)
    assert math.hypot(x - 8.0, y - 3.0) < 0.6
    assert abs(p0 + 40) < 1.5 and resid < 1.5


def test_build_terrain_finds_walls_and_aps():
    survey = _survey(step=1.0)
    known = {"24:a4:3c:11:22:33": SIM_APS["24:a4:3c:11:22:33"]}
    t = build_terrain(survey, 12.0, 8.0, known, cell=0.25)
    assert t is not None and t.survey_points == len(survey)
    assert set(t.aps) == set(SIM_APS)
    assert t.aps["24:a4:3c:11:22:33"].method == "floorplan"
    for bssid, ap in t.aps.items():
        if ap.method == "fitted":
            tx, ty, _ = SIM_APS[bssid]
            assert math.hypot(ap.x - tx, ap.y - ty) < 2.0, f"{bssid} fitted {ap.x:.1f},{ap.y:.1f} vs {tx},{ty}"

    grid = np.array([[np.nan if v is None else v for v in row] for row in t.obstruction])
    ny, nx = grid.shape
    near, far = [], []
    for j in range(ny):
        for i in range(nx):
            if np.isnan(grid[j, i]):
                continue
            x, y = (i + 0.5) * t.cell, (j + 0.5) * t.cell
            d = min(point_segment_distance((x, y), (w[0], w[1]), (w[2], w[3])) for w in SIM_WALLS)
            (near if d < 0.4 else far if d > 1.2 else []).append(grid[j, i])
    assert np.mean(near) > 1.6 * np.mean(far), (np.mean(near), np.mean(far))
    assert t.walls, "expected at least one inferred wall segment"
    # every inferred wall should lie close to a true wall
    close = sum(1 for w in t.walls if min(point_segment_distance(((w.x1 + w.x2) / 2, (w.y1 + w.y2) / 2), (s[0], s[1]), (s[2], s[3])) for s in SIM_WALLS) < 0.8)
    assert close / len(t.walls) >= 0.6


def test_openings_near_true_doors():
    # 1 m spacing is the documented survey recommendation
    t = build_terrain(_survey(step=1.0), 12.0, 8.0, {}, cell=0.25)
    assert t is not None and t.openings
    hits = sum(1 for o in t.openings if min(math.hypot(o.x - dx, o.y - dy) for dx, dy in SIM_DOORS) < 1.0)
    assert hits >= 2, [(round(o.x, 2), round(o.y, 2)) for o in t.openings]
    assert hits >= len(t.openings) // 2   # false openings must not outnumber real ones


def test_terrain_needs_points():
    assert build_terrain(Survey(), 12.0, 8.0) is None
