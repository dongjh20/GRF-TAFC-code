"""Formation representation helpers (dimensionless relative coordinates, Sec. 3.2.1)."""
import math
import re
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

Point = Tuple[float, float]   # (x: longitudinal, y: lane index)

_NUMBER_WORDS = ["Zero", "One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight",
                 "Nine", "Ten", "Eleven", "Twelve"]
_POINT_RE = re.compile(r"\(\s*([-+]?\d+(?:\.\d+)?)\s*,\s*([-+]?\d+(?:\.\d+)?)\s*\)")


def number_word(n: int, lowercase: bool = False) -> str:
    word = _NUMBER_WORDS[n] if 0 <= n < len(_NUMBER_WORDS) else str(n)
    return word.lower() if lowercase else word


def _fmt(v: float) -> str:
    return str(int(round(v))) if abs(v - round(v)) < 1e-9 else f"{v:g}"


def format_formation(points: Sequence[Point]) -> str:
    """[(1,1), (3,1), ...] -- the exact textual form used in the prompts."""
    return "[" + ", ".join(f"({_fmt(x)},{_fmt(y)})" for x, y in points) + "]"


def parse_points(text: str) -> List[Point]:
    return [(float(a), float(b)) for a, b in _POINT_RE.findall(text or "")]


def normalize(points: Sequence[Point]) -> List[Point]:
    """Shift so that min x = 0, then sort by (y, x) -- deterministic post-processing."""
    if not points:
        return []
    x0 = min(p[0] for p in points)
    return sorted(((x - x0, y) for x, y in points), key=lambda p: (p[1], p[0]))


def lane_counts(points: Sequence[Point], num_lanes: int) -> List[int]:
    return [sum(1 for _, y in points if int(round(y)) == lane) for lane in range(1, num_lanes + 1)]


def min_lane_changes(prev: Sequence[Point], new: Sequence[Point], num_lanes: int) -> int:
    """Minimum total number of single-lane moves needed to go from `prev` to `new`.

    This is the 1-D earth mover's distance between lane-wise distributions; e.g.
    (2,3,2) -> (4,3,0) needs 4 lane changes (Pattern A) and (2,3,2) -> (3,4,0) needs 3 (Pattern B).
    """
    a, b = lane_counts(prev, num_lanes), lane_counts(new, num_lanes)
    ca = cb = total = 0
    for i in range(num_lanes):
        ca += a[i]
        cb += b[i]
        total += abs(ca - cb)
    return total


@dataclass
class FormationState:
    """System-derived context extracted from the collective CAV state (Sec. 3.2.1, item 1)."""
    num_cavs: int
    num_lanes: int
    previous_formation: List[Point]
    same_lane_spacing: float = 2.0

    def occupied_distribution(self) -> Tuple[Tuple[int, ...], int, int]:
        counts = lane_counts(self.previous_formation, self.num_lanes)
        occupied = [i + 1 for i, c in enumerate(counts) if c > 0] or [1]
        first, last = occupied[0], occupied[-1]
        return tuple(counts[first - 1:last]), first, last


def rule_based_interlaced(num_cavs: int, usable_lanes: Sequence[int],
                          previous: Optional[Sequence[Point]] = None,
                          spacing: float = 2.0) -> List[Point]:
    """Fallback generator (Sec. 3.2.3): a basic interlaced formation over the usable lanes.

    CAVs are spread as evenly as possible; the extra CAVs go to the lanes that already
    held more CAVs in the previous formation (fewer lane changes). Adjacent usable lanes are
    staggered by half the spacing, with the most populated lane anchored at x = 0.
    """
    lanes = sorted(usable_lanes)
    if not lanes:
        return []
    base, rem = divmod(num_cavs, len(lanes))
    prev_count = {l: sum(1 for _, y in (previous or []) if int(round(y)) == l) for l in lanes}
    extra = sorted(lanes, key=lambda l: (-prev_count[l], l))[:rem]
    counts = {l: base + (1 if l in extra else 0) for l in lanes}
    anchor = max(range(len(lanes)), key=lambda k: counts[lanes[k]])
    points = []
    for k, lane in enumerate(lanes):
        offset = ((k - anchor) % 2) * spacing / 2
        points += [(offset + j * spacing, float(lane)) for j in range(counts[lane])]
    return normalize(points)


def is_finite_point(p: Point) -> bool:
    return all(math.isfinite(v) for v in p)
