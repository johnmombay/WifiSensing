"""Command-line entry point.

Examples
--------
    python -m wifisense                                  # live map of the local /24
    python -m wifisense --simulate --simulate-survey     # full demo incl. terrain, no network
    python -m wifisense --survey office.survey.json --floorplan office.json
    python -m wifisense --once --snapshot map.png --json state.json
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from pathlib import Path
from typing import Optional

from . import __version__
from .engine import SensingEngine
from .links import LinkConfig, LinkMonitor
from .mapping import FloorPlan, Mapper
from .models import SensingState
from .oui import OuiDatabase
from .presence import PresenceConfig, PresenceDetector
from .survey import Survey

log = logging.getLogger("wifisense")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="wifisense", description="WiFi device discovery, presence sensing and terrain inference with a live map.")
    p.add_argument("--version", action="version", version=f"wifisense {__version__}")
    p.add_argument("--subnet", help="CIDR to sweep (default: local /24). Max 1024 addresses.")
    p.add_argument("--floorplan", type=Path, help="Floor-plan JSON (see floorplan.example.json). Optional.")
    p.add_argument("--room", metavar="WxH", help="Area size in metres without a floor-plan file, e.g. 8x6 (default 12x8).")
    p.add_argument("--simulate", action="store_true", help="Use simulated devices, radios and a walking person; no network access.")
    p.add_argument("--simulate-survey", action="store_true", help="With --simulate: auto-generate a 1 m survey grid so terrain inference has data.")
    p.add_argument("--survey", type=Path, help="Enable survey mode; points are persisted to this JSON file.")
    p.add_argument("--once", action="store_true", help="Single scan + short sample window, then exit (no window).")
    p.add_argument("--samples", type=int, default=12, help="Signal samples to take in --once mode (default 12).")
    p.add_argument("--snapshot", type=Path, help="Write the map to this .png/.svg/.pdf (implies headless in --once mode).")
    p.add_argument("--json", type=Path, dest="json_out", help="Write the sensing state as JSON to this path.")
    p.add_argument("--scan-interval", type=float, default=60.0, help="Seconds between device sweeps (default 60).")
    p.add_argument("--sample-interval", type=float, default=1.0, help="Seconds between connected-link samples (default 1.0).")
    p.add_argument("--link-interval", type=float, default=5.0, help="Seconds between multi-radio scans (default 5).")
    p.add_argument("--no-links", action="store_true", help="Disable multi-radio link monitoring.")
    p.add_argument("--sensitivity", type=float, default=2.0, help="RSSI std-dev rise (dB) that maps to motion score 1.0.")
    p.add_argument("--terrain-cell", type=float, default=0.25, help="Terrain grid cell size in metres (default 0.25).")
    p.add_argument("--workers", type=int, default=64, help="Parallel ping workers (default 64).")
    p.add_argument("--no-ports", action="store_true", help="Skip TCP fingerprinting (quieter on the network).")
    p.add_argument("--no-dns", action="store_true", help="Skip reverse DNS lookups.")
    p.add_argument("--oui-file", type=Path, help="IEEE oui.txt/oui.csv for full vendor coverage.")
    p.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p


def parse_room(spec: str) -> tuple[float, float]:
    """Parse ``WxH`` metres, e.g. ``8x6`` or ``7.5x4``."""
    m = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*[xX*]\s*(\d+(?:\.\d+)?)\s*", spec or "")
    if not m:
        raise ValueError(f"--room must look like WxH in metres, e.g. 8x6 (got {spec!r})")
    width, height = float(m.group(1)), float(m.group(2))
    if not (1.0 <= width <= 200.0 and 1.0 <= height <= 200.0):
        raise ValueError("--room sides must be within 1..200 m")
    return width, height


def _configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level),
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stderr,
    )


def _build_engine(args: argparse.Namespace) -> tuple[SensingEngine, FloorPlan, Mapper]:
    oui = OuiDatabase.load_ieee_file(args.oui_file) if args.oui_file else OuiDatabase()
    if args.floorplan:
        plan = FloorPlan.load(args.floorplan)
    else:
        width, height = parse_room(args.room) if args.room else (12.0, 8.0)
        plan = FloorPlan(name=f"Coverage area {width:g} x {height:g} m", width=width, height=height, ap_x=width / 2, ap_y=height / 2)
    mapper = Mapper(plan)
    detector = PresenceDetector(PresenceConfig(sensitivity_db=args.sensitivity))
    survey: Optional[Survey] = Survey(args.survey) if args.survey else None

    if args.simulate:
        from .simulate import SIM_HOST_XY, SimulatedBssScanner, SimulatedScanner, SimulatedSignal, SimWorld, generate_survey

        world = SimWorld(host_xy=plan.host or SIM_HOST_XY)
        scanner = SimulatedScanner(oui)
        signal = SimulatedSignal(world)
        link_monitor = None if args.no_links else LinkMonitor(SimulatedBssScanner(world))
        if args.simulate_survey:
            if survey is None:
                survey = Survey()
            if len(survey) == 0:
                generate_survey(world, plan.width, plan.height, step=1.0, survey=survey)
                log.info("simulated survey: %d points", len(survey))
        subnet = "192.168.50.0/24 (simulated)"
    else:
        if args.simulate_survey:
            raise ValueError("--simulate-survey requires --simulate")
        from . import netinfo
        from .scanner import ScanConfig, SubnetScanner
        from .wlan import platform_bss_scanner

        local_ip = netinfo.local_ipv4()
        net = netinfo.parse_subnet(args.subnet, local_ip)
        gateway = netinfo.default_gateway(local_ip)
        log.info("host %s on %s, gateway %s", local_ip, net, gateway or "unknown")
        cfg = ScanConfig(
            subnet=net,
            local_ip=local_ip,
            local_mac=netinfo.local_mac(),
            gateway=gateway,
            workers=args.workers,
            probe_ports=not args.no_ports,
            resolve_hostnames=not args.no_dns,
        )
        scanner = SubnetScanner(cfg, oui)
        signal = netinfo.platform_signal_source()
        link_monitor = None if args.no_links else LinkMonitor(platform_bss_scanner(), LinkConfig())
        subnet = str(net)

    engine = SensingEngine(
        scanner, signal, detector, mapper, subnet, args.scan_interval, args.sample_interval,
        link_monitor=link_monitor, link_interval=args.link_interval, survey=survey, terrain_cell=args.terrain_cell,
    )
    return engine, plan, mapper


def _write_json(state: SensingState, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state.to_dict(), indent=2), encoding="utf-8")
    log.info("state written: %s", path)


def _print_summary(state: SensingState) -> None:
    s = state.to_dict()["summary"]
    print(f"\nSSID: {state.ssid or '-'}   subnet: {state.subnet}   motion: {s['motion_state']} ({s['motion_score']:.2f})   links: {s['links_monitored']}")
    print(f"devices: {s['devices']}   computers: {s['computers']}   phones: {s['mobiles']}   "
          f"people likely: {s['likely_people']}   potential: {s['potential_people']}   survey points: {s['survey_points']}\n")
    if state.people:
        print(f"{'CONF':<8}{'SOURCE':<18}{'POS':<14}EVIDENCE")
        for p in state.people:
            print(f"{p.confidence.value:<8}{p.source:<18}{f'({p.x:.1f},{p.y:.1f})':<14}{p.evidence}")
        print()
    if state.terrain:
        t = state.terrain
        print(f"terrain: {len(t.aps)} radios fitted, {len(t.walls)} wall segments, {len(t.openings)} openings")
        for ap in t.aps.values():
            print(f"  AP {ap.label:<16} ({ap.x:.1f},{ap.y:.1f}) {ap.method:<10} P0 {ap.p0_dbm:.0f} dBm  resid {ap.residual_db:.1f} dB  pts {ap.points}")
        print()
    print(f"{'CLASS':<12}{'LABEL':<24}{'IP':<16}{'MAC':<19}{'VENDOR':<14}{'RTT':>7}  PORTS")
    for p in state.placed:
        d = p.device
        rtt = f"{d.rtt_ms:.1f}" if d.rtt_ms is not None else "-"
        ports = ",".join(map(str, d.open_ports)) or "-"
        print(f"{d.device_class.value:<12}{d.label[:23]:<24}{d.ip:<16}{d.mac or '-':<19}{(d.vendor or '-')[:13]:<14}{rtt:>7}  {ports}")
    print()


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    _configure_logging(args.log_level)

    if args.samples < 1:
        log.error("--samples must be >= 1")
        return 2
    if args.snapshot and args.snapshot.suffix.lower() not in (".png", ".svg", ".pdf"):
        log.error("--snapshot must end in .png, .svg or .pdf")
        return 2

    try:
        engine, plan, mapper = _build_engine(args)
    except (ValueError, RuntimeError, FileNotFoundError) as exc:
        log.error("%s", exc)
        return 2

    from . import render

    render.configure_backend(headless=args.once)
    renderer = render.MapRenderer(
        plan, mapper.r_min, mapper.r_max,
        survey_enabled=engine.survey is not None,
        on_survey_click=engine.record_survey_point,
        on_undo=engine.undo_survey_point,
    )

    if args.once:
        try:
            state = engine.run_once(samples=args.samples)
        except KeyboardInterrupt:
            log.warning("interrupted")
            return 130
        _print_summary(state)
        if args.json_out:
            _write_json(state, args.json_out)
        if args.snapshot:
            render.render_snapshot(renderer, state, args.snapshot)
        return 0

    engine.start()
    try:
        render.run_live(renderer, engine.snapshot, refresh_s=max(0.5, args.sample_interval), on_close=engine.stop)
    except KeyboardInterrupt:
        log.warning("interrupted")
    finally:
        engine.stop()
        state = engine.snapshot()
        if args.json_out:
            _write_json(state, args.json_out)
        if args.snapshot:
            render.render_snapshot(renderer, state, args.snapshot)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
