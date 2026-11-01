"""Mock execution evaluator, used when no execution module is running.

The lane-change count is computed from the formations; transition time/distance are
placeholders taken from the paper's reflection example. Replace with real measurements.
"""
from typing import List

from .formation import Point, lane_counts, min_lane_changes
from .reflection import ExecutionOutcome


def mock_outcome(previous: List[Point], new: List[Point], num_lanes: int,
                 lanes_to_clear: List[int]) -> ExecutionOutcome:
    counts = lane_counts(new, num_lanes)
    cleared = [l for l in lanes_to_clear if 1 <= l <= num_lanes and counts[l - 1] == 0]
    result = "Successful convergence"
    if cleared:
        result += "; the user task input to yield to the ambulance in Lane " + \
                  ", ".join(map(str, cleared)) + " was satisfied."
    else:
        result += "."
    return ExecutionOutcome(task_results=result, safety_checks="No collision, no near-collision.",
                            lane_changes=min_lane_changes(previous, new, num_lanes),
                            transition_time_s=9.14, transition_distance_m=208.49)
