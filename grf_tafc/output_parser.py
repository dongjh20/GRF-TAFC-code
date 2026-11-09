"""Output parser P (Eq. 12) and rule-based formation validation (Sec. 3.2.3)."""
import math
import re
from dataclasses import dataclass, field
from typing import List, Optional

from .formation import Point, is_finite_point, normalize, parse_points

TAGS = ("formation", "knowledge", "experience", "tool", "results")


def extract_tag(text: str, tag: str) -> Optional[str]:
    m = re.search(rf"<{tag}>\s*(.*?)\s*</{tag}>", text or "", re.DOTALL | re.IGNORECASE)
    return m.group(1).strip() if m else None


@dataclass
class ParsedOutput:
    raw: str
    tags: dict
    formation: Optional[List[Point]]   # normalized; None if the tag is missing


def parse_generation_output(text: str) -> ParsedOutput:
    tags = {t: extract_tag(text, t) for t in TAGS}
    formation = normalize(parse_points(tags["formation"])) if tags["formation"] is not None else None
    return ParsedOutput(text, tags, formation)


@dataclass
class ValidationResult:
    ok: bool
    errors: List[str] = field(default_factory=list)
    physical: List[tuple] = field(default_factory=list)   # (X, Y) metres


class FormationValidator:
    """Formation-level admissibility checks. Dynamic feasibility is left to the trajectory optimizer."""

    def __init__(self, cfg):
        self.cfg = cfg

    def to_physical(self, points: List[Point]) -> List[tuple]:
        half = self.cfg.desired_following_distance_m / 2.0
        return [(x * half, (y - 1) * self.cfg.lane_width_m) for x, y in points]

    def validate(self, points: Optional[List[Point]], num_cavs: int, num_lanes: int,
                 constraints: dict) -> ValidationResult:
        if points is None:
            return ValidationResult(False, ["missing <formation></formation> tags"])
        errors = []
        if len(points) != num_cavs:
            errors.append(f"expected exactly {num_cavs} target positions, got {len(points)}")
        if not all(is_finite_point(p) for p in points):
            errors.append("non-finite coordinates")
            return ValidationResult(False, errors)

        blocked = set(constraints.get("lanes_to_clear", [])) | set(constraints.get("unavailable_lanes", []))
        for x, y in points:
            if abs(y - round(y)) > 1e-6 or not 1 <= round(y) <= num_lanes:
                errors.append(f"({x:g},{y:g}): y must be an integer lane index in 1..{num_lanes}")
            elif int(round(y)) in blocked:
                errors.append(f"({x:g},{y:g}) lies in Lane {int(round(y))}, which must be kept clear "
                              f"(high-priority vehicle / incident / unavailable)")

        phys = self.to_physical(points)
        dmin = self.cfg.min_inter_cav_distance_m
        for i in range(len(phys)):
            for j in range(i + 1, len(phys)):
                d = math.dist(phys[i], phys[j])
                if d < dmin:
                    errors.append(f"targets {points[i]} and {points[j]} are {d:.2f} m apart (< {dmin} m)")
        for ob in constraints.get("obstacles_m", []):
            for p, q in zip(points, phys):
                if math.dist(q, ob) < self.cfg.min_obstacle_clearance_m:
                    errors.append(f"target {p} violates obstacle clearance to {ob}")
        return ValidationResult(not errors, errors, phys)
