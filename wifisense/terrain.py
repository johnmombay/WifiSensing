"""Terrain inference: access-point positions, obstruction map, walls and openings.

Method
------
1. For each radio with enough survey points, fit the log-distance path-loss model
   ``RSSI = P0 - 10 n log10(d)``. Unknown AP positions are grid-searched. The fit is
   iteratively reweighted so points behind walls (large positive residual) stop biasing it.
2. Excess attenuation at each survey point = model prediction - measurement. Interpolate
   (inverse-distance weighting) onto a grid and smooth. A wall between the AP and the
   point adds a step of several dB to this field.
3. Walls are where the excess field changes fastest: gradient magnitude normalised by the
   gradient a single wall would produce at the survey's point spacing. Layers are combined
   weighted by fit quality, with the zone right around each AP suppressed (position error
   dominates there). Ridge cells forming straight runs become wall segments.
4. Openings (doors) are short gaps between collinear wall runs.

The result is a hypothesis, not a floor plan. Resolution is bounded by survey-point spacing.
"""

from __future__ import annotations

import logging
import math
import warnings
from typing import Optional

import numpy as np

from .models import ApEstimate, Opening, TerrainModel, WallSegment
from .survey import Survey

log = logging.getLogger(__name__)

PATH_LOSS_EXPONENT = 2.2
MIN_DISTANCE_M = 0.5
WALL_LOSS_DB = 5.0          # a typical interior partition
ROBUST_SCALE_DB = 4.0       # residuals beyond this are down-weighted (likely behind a wall)
AP_SUPPRESS_M = (1.0, 2.2)  # score fades in between these distances from the AP


def _model(dist: np.ndarray) -> np.ndarray:
    return -10.0 * PATH_LOSS_EXPONENT * np.log10(np.maximum(dist, MIN_DISTANCE_M))


def _weights(rssi: np.ndarray) -> np.ndarray:
    return np.clip((rssi + 95.0) / 30.0, 0.2, 1.0)


def _robust(w: np.ndarray, residual: np.ndarray) -> np.ndarray:
    return w / (1.0 + (residual / ROBUST_SCALE_DB) ** 2)


def fit_p0(ap_xy: tuple[float, float], xs: np.ndarray, ys: np.ndarray, rssi: np.ndarray, iterations: int = 3) -> tuple[float, float]:
    """Weighted, iteratively reweighted P0 fit for a known AP position. Returns (p0, residual_rms)."""
    m = _model(np.hypot(xs - ap_xy[0], ys - ap_xy[1]))
    w = _weights(rssi)
    p0 = float(np.sum(w * (rssi - m)) / np.sum(w))
    for _ in range(iterations):
        w = _robust(_weights(rssi), rssi - p0 - m)
        p0 = float(np.sum(w * (rssi - m)) / np.sum(w))
    resid = float(math.sqrt(np.sum(w * (rssi - p0 - m) ** 2) / np.sum(w)))
    return p0, resid


def localize_ap(xs: np.ndarray, ys: np.ndarray, rssi: np.ndarray, width: float, height: float, iterations: int = 3) -> tuple[float, float, float, float]:
    """Grid-search the AP position minimising weighted residual; returns (x, y, p0, residual_rms)."""
    if len(xs) < 3:
        raise ValueError("need at least 3 points to localise an AP")

    def search(w: np.ndarray, x0: float, x1: float, y0: float, y1: float, step: float) -> tuple[float, float, float, float]:
        gx, gy = np.meshgrid(np.arange(x0, x1 + 1e-9, step), np.arange(y0, y1 + 1e-9, step))
        gx, gy = gx.ravel(), gy.ravel()
        m = _model(np.hypot(gx[:, None] - xs[None, :], gy[:, None] - ys[None, :]))
        p0 = (w[None, :] * (rssi[None, :] - m)).sum(axis=1) / w.sum()
        r = (w[None, :] * (rssi[None, :] - p0[:, None] - m) ** 2).sum(axis=1) / w.sum()
        i = int(np.argmin(r))
        return float(gx[i]), float(gy[i]), float(p0[i]), float(math.sqrt(r[i]))

    coarse = max(width, height) / 60.0
    w = _weights(rssi)
    result = (width / 2, height / 2, -50.0, 99.0)
    for _ in range(iterations):
        x, y, _, _ = search(w, 0.0, width, 0.0, height, coarse)
        result = search(w, max(0.0, x - coarse), min(width, x + coarse), max(0.0, y - coarse), min(height, y + coarse), coarse / 5.0)
        x, y, p0, _ = result
        w = _robust(_weights(rssi), rssi - p0 - _model(np.hypot(xs - x, ys - y)))
    return result


def _box_smooth(a: np.ndarray, k: int) -> np.ndarray:
    """NaN-aware k x k mean filter."""
    if k <= 1:
        return a
    pad = k // 2
    valid = ~np.isnan(a)
    vp = np.pad(np.where(valid, a, 0.0), pad)
    cp = np.pad(valid.astype(float), pad)
    h, w = a.shape
    s = np.zeros_like(a, dtype=float)
    c = np.zeros_like(a, dtype=float)
    for dy in range(k):
        for dx in range(k):
            s += vp[dy:dy + h, dx:dx + w]
            c += cp[dy:dy + h, dx:dx + w]
    return np.where(c > 0, s / np.maximum(c, 1.0), np.nan)


def _median_spacing(xs: np.ndarray, ys: np.ndarray) -> float:
    if len(xs) < 2:
        return 1.0
    d = np.hypot(xs[:, None] - xs[None, :], ys[:, None] - ys[None, :])
    np.fill_diagonal(d, np.inf)
    return float(np.median(d.min(axis=1)))


def build_terrain(
    survey: Survey,
    width: float,
    height: float,
    known_aps: Optional[dict[str, tuple[float, float, str]]] = None,
    cell: float = 0.25,
    min_points: int = 4,
    interp_radius: float = 2.5,
    wall_threshold: float = 0.5,
    min_run_cells: int = 4,
    door_max_cells: int = 6,
    smooth_cells: int = 3,
    wall_gradient_db_per_m: Optional[float] = None,
) -> Optional[TerrainModel]:
    """Infer terrain from a survey. ``wall_gradient_db_per_m`` overrides the automatic
    normaliser (the radial gradient that maps to obstruction score 1.0)."""
    if not 0.05 <= cell <= 2.0:
        raise ValueError("cell must be within 0.05..2.0 m")
    if len(survey) < min_points:
        return None
    known_aps = known_aps or {}

    nx, ny = int(math.ceil(width / cell)), int(math.ceil(height / cell))
    gx, gy = np.meshgrid((np.arange(nx) + 0.5) * cell, (np.arange(ny) + 0.5) * cell)   # shape (ny, nx)
    gxf, gyf = gx.ravel(), gy.ravel()

    all_x = np.array([p.x for p in survey.points])
    all_y = np.array([p.y for p in survey.points])
    spacing = _median_spacing(all_x, all_y)
    # Gradient one wall produces once its step is spread over the spacing plus the smoothing width.
    wall_gradient = wall_gradient_db_per_m or 0.7 * WALL_LOSS_DB / (spacing + smooth_cells * cell)

    aps: dict[str, ApEstimate] = {}
    excess_layers: list[np.ndarray] = []
    score_layers: list[np.ndarray] = []
    layer_weights: list[np.ndarray] = []

    for bssid in sorted(survey.bssids()):
        pts = [p for p in survey.points if bssid in p.rssi]
        if len(pts) < min_points:
            continue
        xs = np.array([p.x for p in pts])
        ys = np.array([p.y for p in pts])
        rssi = np.array([p.rssi[bssid] for p in pts])
        label = survey.ssid_for(bssid) or bssid[-8:]
        if bssid in known_aps:
            ax, ay, label = known_aps[bssid]
            p0, resid = fit_p0((ax, ay), xs, ys, rssi)
            method = "floorplan"
        else:
            try:
                ax, ay, p0, resid = localize_ap(xs, ys, rssi, width, height)
            except ValueError:
                continue
            method = "fitted"
        aps[bssid] = ApEstimate(bssid, label, ax, ay, method, p0, resid, len(pts))

        residual = (p0 + _model(np.hypot(xs - ax, ys - ay))) - rssi     # positive = more loss than free space
        d = np.hypot(gxf[:, None] - xs[None, :], gyf[:, None] - ys[None, :])
        wgt = np.where(d <= interp_radius, 1.0 / (d ** 2 + 0.05), 0.0)
        wsum = wgt.sum(axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            field = np.where(wsum > 0, (wgt * residual[None, :]).sum(axis=1) / wsum, np.nan).reshape(ny, nx)
        field = _box_smooth(field, smooth_cells)
        excess_layers.append(field)

        # Only the outward radial component counts: a wall adds loss as you move away from
        # the AP across it, while shadow edges fanning out through doorways change the field
        # sideways (tangentially) and would otherwise be mistaken for walls.
        gyv, gxv = np.gradient(field, cell)
        d_ap = np.hypot(gx - ax, gy - ay)
        rx, ry = (gx - ax) / np.maximum(d_ap, 1e-6), (gy - ay) / np.maximum(d_ap, 1e-6)
        radial = gxv * rx + gyv * ry
        score = np.clip(np.maximum(radial, 0.0) / wall_gradient, 0.0, 1.0)
        suppress = np.clip((d_ap - AP_SUPPRESS_M[0]) / (AP_SUPPRESS_M[1] - AP_SUPPRESS_M[0]), 0.0, 1.0)
        quality = 1.0 / (1.0 + (resid / ROBUST_SCALE_DB) ** 2)
        score_layers.append(score)
        layer_weights.append(np.where(np.isnan(score), 0.0, suppress * quality))

    if not score_layers:
        return None
    scores = np.nan_to_num(np.stack(score_layers), nan=0.0)
    weights = np.stack(layer_weights)
    wsum = weights.sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        obstruction = np.where(wsum > 0, (weights * scores).sum(axis=0) / wsum, np.nan)
    obstruction = _box_smooth(obstruction, smooth_cells)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        excess = np.nanmean(np.stack(excess_layers), axis=0)

    walls, openings = _extract_walls(obstruction, cell, wall_threshold, min_run_cells, door_max_cells)
    log.info("terrain: %d radios, %d walls, %d openings from %d survey points (spacing %.2f m)",
             len(aps), len(walls), len(openings), len(survey), spacing)
    return TerrainModel(cell, width, height, _to_lists(obstruction), _to_lists(excess), walls, openings, aps, len(survey))


def _to_lists(a: np.ndarray) -> list[list[Optional[float]]]:
    return [[None if np.isnan(v) else round(float(v), 3) for v in row] for row in a]


def _runs(mask_row: np.ndarray) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    start: Optional[int] = None
    for i, v in enumerate(mask_row):
        if v and start is None:
            start = i
        elif not v and start is not None:
            runs.append((start, i - 1))
            start = None
    if start is not None:
        runs.append((start, len(mask_row) - 1))
    return runs


def _dilate(mask: np.ndarray, axis: int) -> np.ndarray:
    """OR each cell with its two neighbours along ``axis`` (no wrap-around)."""
    p = np.pad(mask, 1)
    if axis == 0:
        return mask | p[:-2, 1:-1] | p[2:, 1:-1]
    return mask | p[1:-1, :-2] | p[1:-1, 2:]


def _extract_walls(score: np.ndarray, cell: float, thr: float, min_run: int, door_max: int) -> tuple[list[WallSegment], list[Opening]]:
    s = np.nan_to_num(score, nan=0.0)
    hot = s >= thr
    pad = np.pad(s, 2, mode="edge")
    up, down = pad[:-4, 2:-2], pad[4:, 2:-2]        # two cells away across rows
    left, right = pad[2:-2, :-4], pad[2:-2, 4:]      # two cells away across columns
    # A ridge must stand proud of its surroundings in the crossing direction; a plateau
    # (the flat top of a thick ridge in the other orientation) is not a wall.
    prominence = 0.08
    ridge_h = hot & (s - np.maximum(up, down) >= prominence)     # peak across rows -> horizontal wall
    ridge_v = hot & (s - np.maximum(left, right) >= prominence)  # peak across cols -> vertical wall
    # A ridge that drifts by one cell would otherwise break into pieces with false gaps.
    ridge_h = _dilate(ridge_h, axis=0)
    ridge_v = _dilate(ridge_v, axis=1)
    walls: list[WallSegment] = []
    openings: list[Opening] = []

    def emit(runs: list[tuple[int, int]], fixed: int, horizontal: bool) -> None:
        kept = [(a, b) for a, b in runs if b - a + 1 >= min_run]
        f = (fixed + 0.5) * cell
        for a, b in kept:
            seg = s[fixed, a:b + 1] if horizontal else s[a:b + 1, fixed]
            conf = float(seg.mean())
            if horizontal:
                walls.append(WallSegment(a * cell, f, (b + 1) * cell, f, conf))
            else:
                walls.append(WallSegment(f, a * cell, f, (b + 1) * cell, conf))
        for (a1, b1), (a2, b2) in zip(kept, kept[1:]):
            gap = a2 - b1 - 1
            if 1 <= gap <= door_max:
                g = ((b1 + a2 + 1) / 2.0) * cell
                seg = s[fixed, a1:b2 + 1] if horizontal else s[a1:b2 + 1, fixed]
                conf = min(1.0, float(seg.mean()))
                openings.append(Opening(g, f, conf) if horizontal else Opening(f, g, conf))

    for j in range(s.shape[0]):
        emit(_runs(ridge_h[j, :]), j, True)
    for i in range(s.shape[1]):
        emit(_runs(ridge_v[:, i]), i, False)
    return _merge_segments(walls, lateral=1.6 * cell), _dedupe_openings(openings, radius=3 * cell)


def _merge_segments(walls: list[WallSegment], lateral: float) -> list[WallSegment]:
    """Merge parallel segments that lie within ``lateral`` of each other and overlap in extent."""
    def horizontal(w: WallSegment) -> bool:
        return abs(w.y1 - w.y2) < 1e-9

    merged: list[WallSegment] = []
    for w in sorted(walls, key=lambda w: -w.confidence):
        for k, m in enumerate(merged):
            if horizontal(w) != horizontal(m):
                continue
            if horizontal(w):
                if abs(w.y1 - m.y1) <= lateral and w.x1 <= m.x2 + lateral and m.x1 <= w.x2 + lateral:
                    merged[k] = WallSegment(min(w.x1, m.x1), m.y1, max(w.x2, m.x2), m.y1, max(w.confidence, m.confidence))
                    break
            else:
                if abs(w.x1 - m.x1) <= lateral and w.y1 <= m.y2 + lateral and m.y1 <= w.y2 + lateral:
                    merged[k] = WallSegment(m.x1, min(w.y1, m.y1), m.x1, max(w.y2, m.y2), max(w.confidence, m.confidence))
                    break
        else:
            merged.append(w)
    return merged


def _dedupe_openings(openings: list[Opening], radius: float) -> list[Opening]:
    kept: list[Opening] = []
    for o in sorted(openings, key=lambda o: -o.confidence):
        if all(math.hypot(o.x - k.x, o.y - k.y) > radius for k in kept):
            kept.append(o)
    return kept
