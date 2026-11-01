"""Execution-driven reflection module (Sec. 3.3): reflection context -> insight (Eq. 13) -> memory update (Eq. 14)."""
import json
import re
from dataclasses import dataclass
from typing import Optional, Tuple

from google.genai import types

from .config import GRFConfig
from .experience_memory import ExperienceMemory
from .generator import GenerationRecord
from .output_parser import extract_tag
from .prompt_composer import GroundedPromptComposer


@dataclass
class ExecutionOutcome:
    """Execution outcome O_k summarized by the execution evaluator."""
    task_results: str            # task satisfaction and termination status
    safety_checks: str           # safety and traffic-rule compliance
    lane_changes: Optional[int] = None
    transition_time_s: Optional[float] = None
    transition_distance_m: Optional[float] = None

    @classmethod
    def from_dict(cls, d: dict) -> "ExecutionOutcome":
        return cls(d.get("task_results", "Unknown"), d.get("safety_checks", "Unknown"),
                   d.get("lane_changes"), d.get("transition_time_s"), d.get("transition_distance_m"))

    def performance_text(self) -> str:
        f = lambda v, fmt: "N/A" if v is None else fmt.format(v)
        return (f"Number of lane changes: {f(self.lane_changes, '{}')}, "
                f"transition time: {f(self.transition_time_s, '{:.2f} s')}, "
                f"transition distance: {f(self.transition_distance_m, '{:.2f} m')}.")


# Outcome used in the paper's reflection example (Pattern A, ambulance in Lane 3).
PAPER_EXAMPLE_OUTCOME = ExecutionOutcome(
    task_results="Successful convergence; the user task input to yield to the ambulance in Lane 3 was satisfied.",
    safety_checks="No collision, no near-collision.",
    lane_changes=4, transition_time_s=9.14, transition_distance_m=208.49)


@dataclass
class ReflectionInsight:
    key: Optional[str]
    value: Optional[str]
    raw_output: str
    inserted: bool = False
    update_note: str = ""


def parse_key_value(text: str) -> Tuple[Optional[str], Optional[str]]:
    """Parse the <experience> payload. The prompt only asks for 'a key-value pair', so accept
    JSON ({"key":..,"value":..} or {"<key>": "<value>"}), 'Key: .. Value: ..', or 'key -> value'."""
    if not text:
        return None, None
    s = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M).replace("**", "").strip()
    try:
        obj = json.loads(s)
        if isinstance(obj, dict):
            low = {k.lower(): v for k, v in obj.items()}
            if "key" in low and "value" in low:
                return str(low["key"]).strip(), str(low["value"]).strip()
            if len(obj) == 1:
                k, v = next(iter(obj.items()))
                return str(k).strip(), str(v).strip()
    except (json.JSONDecodeError, TypeError):
        pass
    m = re.search(r"\bkey\b\s*[:=]\s*(.+?)\s*[,;]?\s*\bvalue\b\s*[:=]\s*(.+)", s, re.I | re.S)
    if m:
        return m.group(1).strip(" \"'[]"), m.group(2).strip(" \"'[]")
    for sep in ("->", "=>", "\n"):
        if sep in s:
            k, v = s.split(sep, 1)
            return k.strip(" \"'"), v.strip(" \"'")
    return None, None


class ReflectionModule:
    def __init__(self, client, cfg: GRFConfig, memory: ExperienceMemory, verbose: bool = True):
        self.client, self.cfg, self.memory, self.verbose = client, cfg, memory, verbose

    def reflect(self, record: GenerationRecord, outcome: ExecutionOutcome) -> ReflectionInsight:
        prompt = GroundedPromptComposer.reflection_text(record, outcome)
        resp = self.client.models.generate_content(
            model=self.cfg.reflection_model, contents=prompt,
            config=types.GenerateContentConfig(
                temperature=self.cfg.temperature, top_p=self.cfg.top_p,
                thinking_config=types.ThinkingConfig(thinking_level=self.cfg.thinking_level)))
        raw = resp.text or ""
        key, value = parse_key_value(extract_tag(raw, "experience") or "")
        insight = ReflectionInsight(key, value, raw)
        if key and value:
            insight.inserted, insight.update_note = self.memory.update(key, value, source="reflection")
        else:
            insight.update_note = "could not parse a key-value pair from <experience>; memory unchanged"
        if self.verbose:
            print(f"[reflection] key  : {key}\n[reflection] value: {value}\n[reflection] memory: {insight.update_note}")
        return insight
