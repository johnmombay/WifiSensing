"""matplotlib rendering: live window or static snapshot of a :class:`SensingState`.

Live-window keys: left-click on the map = "I am standing here" (records a survey point when
survey mode is on), ``u`` = undo last survey point, ``t`` = toggle the terrain overlay.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Callable, Optional

import matplotlib
import numpy as np

from .mapping import FloorPlan
from .models import Confidence, DeviceClass, MotionState, PersonCandidate, SensingState

log = logging.getLogger(__name__)

# marker, colour, legend label
CLASS_STYLE: dict[DeviceClass, tuple[str, str, str]] = {
    DeviceClass.ROUTER: ("*", "#c0392b", "Router / AP"),
    DeviceClass.WORKSTATION: ("s", "#2471a3", "Workstation"),
    DeviceClass.SERVER: ("D", "#7d3c98", "Server"),
    DeviceClass.MOBILE: ("o", "#1e8449", "Mobile"),
    DeviceClass.PRINTER: ("^", "#d68910", "Printer"),
    DeviceClass.IOT: ("v", "#7b4f2e", "IoT"),
    DeviceClass.UNKNOWN: ("X", "#7f8c8d", "Unknown"),
}

MOTION_COLOUR: dict[MotionState, str] = {
    MotionState.STILL: "#27ae60",
    MotionState.LOW: "#f39c12",
    MotionState.ACTIVE: "#e74c3c",
    MotionState.CALIBRATING: "#95a5a6",
    MotionState.UNAVAILABLE: "#bdc3c7",
}

PERSON_COLOUR = "#8e44ad"
MOTION_PERSON_COLOUR = "#e67e22"
PERSON_ALPHA: dict[Confidence, float] = {Confidence.HIGH: 0.95, Confidence.MEDIUM: 0.6, Confidence.LOW: 0.32}


def configure_backend(headless: bool) -> None:
    if headless:
        matplotlib.use("Agg")


class MapRenderer:
    def __init__(
        self,
        plan: FloorPlan,
        r_min: float,
        r_max: float,
        survey_enabled: bool = False,
        on_survey_click: Optional[Callable[[float, float], None]] = None,
        on_undo: Optional[Callable[[], None]] = None,
    ) -> None:
        import matplotlib.pyplot as plt
        from matplotlib.gridspec import GridSpec

        self.plan = plan
        self.r_min, self.r_max = r_min, r_max
        self.survey_enabled = survey_enabled
        self.show_terrain = True
        self._on_survey_click = on_survey_click
        self._on_undo = on_undo
        self._status = ""
        self._status_at = 0.0

        self.fig = plt.figure(figsize=(14.5, 8), facecolor="white")
        gs = GridSpec(3, 2, width_ratios=[2.3, 1.0], height_ratios=[1.1, 1.0, 1.5], figure=self.fig, wspace=0.18, hspace=0.5)
        self.ax_map = self.fig.add_subplot(gs[:, 0])
        self.ax_links = self.fig.add_subplot(gs[0, 1])
        self.ax_rssi = self.fig.add_subplot(gs[1, 1])
        self.ax_list = self.fig.add_subplot(gs[2, 1])
        self.fig.canvas.mpl_connect("button_press_event", self._on_click)
        self.fig.canvas.mpl_connect("key_press_event", self._on_key)

    # ------------------------------------------------------------------ interaction
    def _on_click(self, event) -> None:
        if not self.survey_enabled or self._on_survey_click is None:
            return
        if event.inaxes is not self.ax_map or event.button != 1 or event.xdata is None:
            return
        toolbar = getattr(self.fig.canvas, "toolbar", None)
        if toolbar is not None and getattr(toolbar, "mode", ""):
            self._set_status("pan/zoom tool is active; deselect it in the toolbar to record survey points")
            return
        try:
            self._on_survey_click(float(event.xdata), float(event.ydata))
            self._set_status(f"survey point recorded at ({event.xdata:.1f}, {event.ydata:.1f})")
        except Exception as exc:  # noqa: BLE001
            log.warning("survey click failed: %s", exc)
            self._set_status(f"survey failed: {exc}")

    def _on_key(self, event) -> None:
        if event.key == "u" and self._on_undo is not None:
            try:
                self._on_undo()
                self._set_status("last survey point removed")
            except Exception as exc:  # noqa: BLE001
                self._set_status(f"undo failed: {exc}")
        elif event.key == "t":
            self.show_terrain = not self.show_terrain
            self._set_status(f"terrain overlay {'on' if self.show_terrain else 'off'}")

    def _set_status(self, text: str) -> None:
        self._status, self._status_at = text, time.monotonic()

    # ------------------------------------------------------------------ drawing
    def draw(self, state: SensingState) -> None:
        self._draw_map(state)
        self._draw_links(state)
        self._draw_rssi(state)
        self._draw_list(state)
        self.fig.suptitle(self._title(state), fontsize=11.5, fontweight="bold", y=0.985)

    def _title(self, state: SensingState) -> str:
        scan = "scanning..." if state.scan_in_progress else (state.last_scan.astimezone().strftime("%H:%M:%S") if state.last_scan else "pending")
        ssid = state.ssid or "no WiFi link"
        return (f"wifisense  |  {ssid}  |  {state.subnet}  |  devices {len(state.devices)}  computers {state.computers}  "
                f"phones {state.mobiles}  |  people likely {state.likely_people} / potential {state.potential_people}  |  last scan {scan}")

    def _draw_map(self, state: SensingState) -> None:
        from matplotlib.lines import Line2D
        from matplotlib.patches import Circle, Ellipse, Rectangle

        plan = self.plan
        ax = self.ax_map
        ax.clear()
        ax.set_xlim(0, plan.width)
        ax.set_ylim(0, plan.height)
        ax.set_aspect("equal")
        ax.set_facecolor("#fbfbfb")
        ax.set_xlabel("metres")
        ax.set_ylabel("metres")
        ax.set_title(plan.name, fontsize=10, loc="left", color="#555555")
        ax.grid(True, color="#e6e6e6", linewidth=0.6)
        ax.add_patch(Rectangle((0, 0), plan.width, plan.height, fill=False, linewidth=1.5, edgecolor="#333333"))

        for z in plan.zones:
            ax.add_patch(Rectangle((z.x, z.y), z.w, z.h, facecolor="#eef3f8", edgecolor="#b8c6d6", linewidth=1.0, zorder=1))
            ax.text(z.x + 0.1, z.y + z.h - 0.15, z.name, fontsize=8, color="#5d6d7e", va="top", zorder=2)

        if self.show_terrain and state.terrain is not None:
            self._draw_terrain(ax, state)

        ap_xy = (plan.ap_x, plan.ap_y)
        colour = MOTION_COLOUR[state.presence.state]
        alpha = 0.05 + 0.25 * state.presence.motion_score
        ax.add_patch(Circle(ap_xy, self.r_max * 1.15, facecolor=colour, edgecolor="none", alpha=alpha, zorder=1.5))
        for r in (self.r_min, self.r_max):
            ax.add_patch(Circle(ap_xy, r, fill=False, linestyle="--", linewidth=0.7, edgecolor="#9aa5b1", zorder=2))
        ax.text(ap_xy[0] + self.r_max + 0.1, ap_xy[1], "est. ring\n(RTT)", fontsize=7, color="#9aa5b1", va="center")

        # Known AP positions from the floor plan
        for bssid, (x, y, label) in plan.access_points.items():
            ax.plot(x, y, marker="*", markersize=16, color=CLASS_STYLE[DeviceClass.ROUTER][1], zorder=5, linestyle="none")
            ax.text(x, y + 0.3, label, fontsize=7, ha="center", color="#7b241c", zorder=6)
        ax.plot(*ap_xy, marker="*", markersize=20, color=CLASS_STYLE[DeviceClass.ROUTER][1], zorder=5)
        ax.text(ap_xy[0], ap_xy[1] + 0.35, "AP", fontsize=8, ha="center", fontweight="bold", zorder=6)

        # Perturbed links: draw the host->AP path so the direction of motion is visible
        if state.host_xy:
            for l in state.links:
                if l.ap_xy and l.motion_score >= 0.35 and l.state in (MotionState.LOW, MotionState.ACTIVE):
                    ax.plot([state.host_xy[0], l.ap_xy[0]], [state.host_xy[1], l.ap_xy[1]], color=MOTION_PERSON_COLOUR,
                            linewidth=1.0 + 3.0 * l.motion_score, alpha=0.35, zorder=3)

        total = len(state.placed)
        for p in state.placed:
            if p.device.is_gateway:
                continue
            marker, colour, _ = CLASS_STYLE[p.device.device_class]
            size = 11 if p.device.device_class is DeviceClass.MOBILE else 10
            edge = "#000000" if p.placed_by == "floorplan" else colour
            ax.plot(p.x, p.y, marker=marker, markersize=size, color=colour, markeredgecolor=edge, markeredgewidth=1.2,
                    alpha=0.95 if p.placed_by == "floorplan" else 0.75, linestyle="none", zorder=4)
            show_label = (
                total <= 40 or p.placed_by == "floorplan" or p.device.is_self
                or (total <= 80 and p.device.device_class is not DeviceClass.UNKNOWN)
                or p.device.device_class in (DeviceClass.MOBILE, DeviceClass.ROUTER)
            )
            if show_label:
                ax.annotate(p.device.label, (p.x, p.y), xytext=(0, -11), textcoords="offset points",
                            fontsize=7, ha="center", va="top", color="#2c3e50", zorder=6)

        for person in state.people:
            self._draw_person(ax, person, Circle, Ellipse)

        if state.host_xy:
            ax.add_patch(Circle(state.host_xy, 0.34, fill=False, edgecolor="#2471a3", linewidth=1.8, zorder=7))
            ax.text(state.host_xy[0], state.host_xy[1] + 0.45, "you", fontsize=7, ha="center", color="#2471a3", zorder=7)

        if total > 40:
            ax.text(0.99, 0.01, f"{total} devices: labels thinned (see list / --json)", transform=ax.transAxes,
                    fontsize=7, ha="right", va="bottom", color="#7f8c8d")
        hint = []
        if state.survey_enabled:
            hint.append(f"survey: {state.survey_points} pts  |  click map = I am here  |  u = undo  |  t = terrain overlay")
            if state.terrain is None:
                hint.append(f"terrain needs >= 4 survey points spread around the room (have {state.survey_points}); "
                            "stand somewhere, click that spot, move ~1 m, repeat")
            elif not state.terrain.walls:
                hint.append("no wall stands out yet: add points on both sides of each wall, ~1 m apart")
        if self._status and time.monotonic() - self._status_at < 6:
            hint.append(self._status)
        if hint:
            ax.text(0.01, 0.01, "\n".join(hint), transform=ax.transAxes, fontsize=7, ha="left", va="bottom", color="#1f618d",
                    bbox=dict(facecolor="white", alpha=0.75, edgecolor="none", pad=2))

        handles = [Line2D([0], [0], marker=m, color=c, linestyle="none", markersize=8, label=lbl) for m, c, lbl in CLASS_STYLE.values()]
        handles += [
            Line2D([0], [0], marker="o", color=PERSON_COLOUR, linestyle="none", markersize=8, label="Person (likely)"),
            Line2D([0], [0], marker="o", color=PERSON_COLOUR, alpha=0.35, linestyle="none", markersize=8, label="Person (potential)"),
            Line2D([0], [0], marker="o", color=MOTION_PERSON_COLOUR, linestyle="none", markersize=8, label="Motion (unidentified)"),
            Line2D([0], [0], color="#2c3e50", linewidth=4, label="Inferred wall"),
            Line2D([0], [0], marker="s", color="#27ae60", linestyle="none", markersize=8, label="Inferred opening"),
        ]
        ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.0, -0.075), ncol=4, fontsize=7, frameon=False)

    def _draw_person(self, ax, person: PersonCandidate, Circle, Ellipse) -> None:
        motion = person.source == "motion"
        colour = MOTION_PERSON_COLOUR if motion else PERSON_COLOUR
        alpha = PERSON_ALPHA[person.confidence]
        ax.add_patch(Circle((person.x, person.y + 0.2), 0.13, facecolor=colour, edgecolor="white", linewidth=0.8, alpha=alpha, zorder=8))
        ax.add_patch(Ellipse((person.x, person.y - 0.1), 0.36, 0.44, facecolor=colour, edgecolor="white", linewidth=0.8, alpha=alpha, zorder=8))
        if motion:
            ax.add_patch(Circle((person.x, person.y), 0.75, fill=False, edgecolor=colour, linestyle="--", linewidth=1.2, alpha=0.8, zorder=8))
        tag = {"phone": "phone", "handset-probable": "handset?", "laptop-personal": "laptop owner", "laptop": "laptop?", "motion": "motion"}[person.source]
        if person.confidence is Confidence.LOW:
            tag += " (low)"
        ax.text(person.x, person.y + 0.42, tag, fontsize=6.5, ha="center", color=colour, alpha=min(1.0, alpha + 0.25), zorder=9)

    def _draw_terrain(self, ax, state: SensingState) -> None:
        t = state.terrain
        assert t is not None
        grid = np.array([[np.nan if v is None else v for v in row] for row in t.obstruction], dtype=float)
        if grid.size:
            masked = np.ma.masked_invalid(grid)
            ax.imshow(masked, extent=(0, t.width, 0, t.height), origin="lower", cmap="Greys", vmin=0, vmax=1,
                      alpha=0.45, interpolation="nearest", zorder=1.2)
        for w in t.walls:
            ax.plot([w.x1, w.x2], [w.y1, w.y2], color="#2c3e50", linewidth=2.5 + 3.0 * w.confidence, alpha=0.5 + 0.5 * w.confidence,
                    solid_capstyle="round", zorder=2.5)
        for o in t.openings:
            ax.plot(o.x, o.y, marker="s", markersize=9, color="#27ae60", markeredgecolor="white", linestyle="none", zorder=2.6)
        for ap in t.aps.values():
            if ap.method == "fitted":
                ax.plot(ap.x, ap.y, marker="*", markersize=14, markerfacecolor="none", markeredgecolor="#c0392b",
                        markeredgewidth=1.5, linestyle="none", zorder=5)
                ax.text(ap.x, ap.y + 0.3, f"{ap.label} (fit ±{ap.residual_db:.0f} dB)", fontsize=6.5, ha="center", color="#7b241c", zorder=6)

    def _draw_links(self, state: SensingState) -> None:
        ax = self.ax_links
        ax.clear()
        links = sorted(state.links, key=lambda l: (not l.connected, -(l.rssi_dbm or -200)))[:8]
        ax.set_title(f"Motion per radio link ({len(state.links)} monitored)", fontsize=9, loc="left")
        if not links:
            ax.text(0.5, 0.5, "no link readings yet", ha="center", va="center", transform=ax.transAxes, fontsize=8, color="#7f8c8d")
            ax.set_xticks([]); ax.set_yticks([])
            return
        names = [("* " if l.connected else "") + ((l.ssid or l.bssid[-8:])[:14]) for l in links]
        ys = np.arange(len(links))[::-1]
        ax.barh(ys, [1.0] * len(links), color="#ecf0f1", height=0.6)
        ax.barh(ys, [l.motion_score for l in links], color=[MOTION_COLOUR[l.state] for l in links], height=0.6)
        for y, l in zip(ys, links):
            ax.text(1.01, y, f"{l.rssi_dbm:.0f}" if l.rssi_dbm is not None else "-", fontsize=6.5, va="center", color="#555555")
        ax.set_yticks(ys)
        ax.set_yticklabels(names, fontsize=7)
        ax.set_xlim(0, 1.08)
        ax.set_xticks([0, 0.25, 0.55, 1.0])
        ax.set_xticklabels(["still", "low", "active", "dBm"], fontsize=7)
        for x in (0.25, 0.55):
            ax.axvline(x, color="#7f8c8d", linewidth=0.6, linestyle=":")
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)

    def _draw_rssi(self, state: SensingState) -> None:
        ax = self.ax_rssi
        ax.clear()
        pr = state.presence
        ax.set_title(f"Connected link RSSI  |  {pr.state.value} {pr.motion_score:.2f}", fontsize=9, loc="left", color=MOTION_COLOUR[pr.state])
        ax.set_ylabel("dBm", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(True, color="#e6e6e6", linewidth=0.6)
        if len(state.rssi_history) >= 2:
            now = state.rssi_history[-1][0]
            xs = [t - now for t, _ in state.rssi_history]
            ys = [v for _, v in state.rssi_history]
            ax.plot(xs, ys, color="#2471a3", linewidth=1.0)
            ax.set_ylim(min(ys) - 3, max(ys) + 3)
            ax.set_xlim(min(xs), 0)
            ax.set_xlabel("seconds ago", fontsize=8)
        else:
            ax.text(0.5, 0.5, "no signal samples", ha="center", va="center", transform=ax.transAxes, fontsize=8, color="#7f8c8d")

    def _draw_list(self, state: SensingState, max_rows: int = 15) -> None:
        ax = self.ax_list
        ax.clear()
        ax.axis("off")
        ax.set_title("People and devices", fontsize=9, loc="left")
        rows: list[tuple[str, str, str]] = []
        order = {Confidence.HIGH: 0, Confidence.MEDIUM: 1, Confidence.LOW: 2}
        for p in sorted(state.people, key=lambda p: order[p.confidence]):
            colour = MOTION_PERSON_COLOUR if p.source == "motion" else PERSON_COLOUR
            rows.append(("o", colour, f"{p.confidence.value:<7}{p.evidence[:38]}"))
        cls_order = [DeviceClass.ROUTER, DeviceClass.WORKSTATION, DeviceClass.SERVER, DeviceClass.MOBILE, DeviceClass.PRINTER, DeviceClass.IOT, DeviceClass.UNKNOWN]
        for p in sorted(state.placed, key=lambda p: (cls_order.index(p.device.device_class), p.device.ip)):
            marker, colour, _ = CLASS_STYLE[p.device.device_class]
            rows.append((marker, colour, f"{p.device.label[:20]:<20} {p.device.ip:<15} {p.device.vendor or ''}"))
        y, step = 0.97, 0.95 / max_rows
        for marker, colour, text in rows[:max_rows]:
            ax.plot(0.02, y, marker=marker, color=colour, markersize=6, transform=ax.transAxes, linestyle="none", clip_on=False)
            ax.text(0.06, y, text, fontsize=7, family="monospace", va="center", transform=ax.transAxes)
            y -= step
        if len(rows) > max_rows:
            ax.text(0.06, y, f"... {len(rows) - max_rows} more (see --json)", fontsize=7, va="center", transform=ax.transAxes, color="#7f8c8d")


def render_snapshot(renderer: MapRenderer, state: SensingState, path: Path | str, dpi: int = 130) -> Path:
    path = Path(path)
    if path.suffix.lower() not in (".png", ".svg", ".pdf"):
        raise ValueError("snapshot must be .png, .svg or .pdf")
    path.parent.mkdir(parents=True, exist_ok=True)
    renderer.draw(state)
    renderer.fig.savefig(path, dpi=dpi, bbox_inches="tight")
    log.info("snapshot written: %s", path)
    return path


def run_live(renderer: MapRenderer, snapshot_fn: Callable[[], SensingState], refresh_s: float = 1.0, on_close: Optional[Callable[[], None]] = None) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    def _frame(_i: int) -> None:
        try:
            renderer.draw(snapshot_fn())
        except Exception:  # noqa: BLE001 - never let a draw error kill the window
            log.exception("frame render failed")

    if on_close:
        renderer.fig.canvas.mpl_connect("close_event", lambda _evt: on_close())
    anim = FuncAnimation(renderer.fig, _frame, interval=int(refresh_s * 1000), cache_frame_data=False)
    renderer.fig._wifisense_anim = anim  # keep a reference so the animation is not garbage-collected
    started = time.monotonic()
    plt.show()
    log.info("window closed after %.0fs", time.monotonic() - started)
