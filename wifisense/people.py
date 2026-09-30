"""Potential-people estimation from device evidence and multi-link motion.

Evidence ladder (strongest first):
  high    - a phone (iOS sync port, handset hostname or handset vendor OUI)
  medium  - a randomised-MAC host with no services and no name (typical sleeping handset)
  medium  - a laptop named after a person (JDOE-PC) that is online on WiFi
  low     - any other workstation online on WiFi (asset-tag laptops)
  motion  - links between this host and known AP positions whose RSSI variance rose;
            the disturbance is placed at the score-weighted midpoint of those links
"""

from __future__ import annotations

import math
import re
from typing import Optional

from .models import Confidence, DeviceClass, LinkReading, MotionState, PersonCandidate, PlacedDevice

_PERSONAL_RE = re.compile(r"^([a-z]{3,}\d{0,4})[-_](pc|laptop|nb|lt|mbp|macbook|desktop|ws)$", re.I)
_ASSET_TAG_RE = re.compile(r"^[a-z]{2,6}[-_](lt|nb|dt|ws|pc|wks)[-_]?\d{2,}$", re.I)

_OFFSET = 0.38  # metres; person glyph sits beside its device


def estimate_people(
    placed: list[PlacedDevice],
    links: list[LinkReading],
    host_xy: Optional[tuple[float, float]],
    min_link_score: float = 0.35,
    merge_radius: float = 1.2,
) -> list[PersonCandidate]:
    people: list[PersonCandidate] = []

    for p in placed:
        d = p.device
        if d.is_self or d.is_gateway:
            continue
        host = d.hostname or ""
        if d.device_class is DeviceClass.MOBILE:
            conf, source, why = Confidence.HIGH, "phone", "mobile handset on the network"
        elif d.device_class is DeviceClass.UNKNOWN and d.vendor == "Randomised" and not d.open_ports and not host:
            conf, source, why = Confidence.MEDIUM, "handset-probable", "randomised MAC, no name, no services"
        elif d.device_class is DeviceClass.WORKSTATION and _PERSONAL_RE.match(host) and not _ASSET_TAG_RE.match(host):
            conf, source, why = Confidence.MEDIUM, "laptop-personal", f"personal laptop {host} online"
        elif d.device_class is DeviceClass.WORKSTATION:
            conf, source, why = Confidence.LOW, "laptop", "workstation online on WiFi"
        else:
            continue
        people.append(PersonCandidate(f"dev:{d.ip}", p.x + _OFFSET, p.y + _OFFSET, conf, source, why, d.ip))

    motion = _motion_candidate(links, host_xy, min_link_score)
    if motion is not None:
        near_strong = any(
            c.confidence is Confidence.HIGH and math.hypot(c.x - motion.x, c.y - motion.y) < merge_radius for c in people
        )
        if not near_strong:
            people.append(motion)
    return people


def _motion_candidate(links: list[LinkReading], host_xy: Optional[tuple[float, float]], min_score: float) -> Optional[PersonCandidate]:
    if host_xy is None:
        return None
    active = [l for l in links if l.motion_score >= min_score and l.state in (MotionState.LOW, MotionState.ACTIVE)]
    if not active:
        return None
    located = [l for l in active if l.ap_xy is not None]
    if located:
        w = sum(l.motion_score for l in located)
        x = sum(l.motion_score * (host_xy[0] + l.ap_xy[0]) / 2 for l in located) / w
        y = sum(l.motion_score * (host_xy[1] + l.ap_xy[1]) / 2 for l in located) / w
        peak = max(l.motion_score for l in located)
        conf = Confidence.HIGH if len(located) >= 2 and peak >= 0.6 else Confidence.MEDIUM
        names = ", ".join((l.ssid or l.bssid[-5:]) + f" {l.motion_score:.2f}" for l in sorted(located, key=lambda l: -l.motion_score)[:3])
        return PersonCandidate("motion", x, y, conf, "motion", f"{len(located)} link(s) perturbed: {names}")
    # Motion seen but no AP positions known: report it next to the host.
    peak = max(l.motion_score for l in active)
    return PersonCandidate("motion", host_xy[0] + 0.9, host_xy[1] + 0.9, Confidence.MEDIUM, "motion",
                           f"{len(active)} link(s) perturbed (peak {peak:.2f}); AP positions unknown")
