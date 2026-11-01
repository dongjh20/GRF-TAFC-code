"""Task-adaptive formation generator (Sec. 3.2): composer -> MLLM + tools/retrieval -> parser/validator."""
import time
from dataclasses import dataclass, field
from typing import List, Optional

from google.genai import types

from .config import GRFConfig
from .formation import FormationState, Point, rule_based_interlaced
from .output_parser import FormationValidator, parse_generation_output
from .prompt_composer import GroundedPromptComposer, TaskInput, describe_task_context, describe_user_input
from .tools import ToolRegistry, to_text

RETRIEVAL = ("explore_knowledge_graph", "explore_experience")


@dataclass
class GenerationRecord:
    """Everything the reflection module needs about 'what was requested' and 'how it was generated'."""
    task_context: str
    user_task_input: str
    formation: Optional[List[Point]]
    source: str                         # mllm / rule_based / safe_stop
    retrieved_knowledge: str = ""
    retrieved_experience: str = ""
    invoked_tools: str = ""
    tool_outputs: str = ""
    raw_output: str = ""
    tool_trace: list = field(default_factory=list)   # what was actually executed
    attempts: int = 0
    validation_errors: list = field(default_factory=list)
    elapsed_s: float = 0.0


class FormationGenerator:
    def __init__(self, client, cfg: GRFConfig, tools: ToolRegistry, env,
                 composer: Optional[GroundedPromptComposer] = None,
                 validator: Optional[FormationValidator] = None, verbose: bool = True):
        self.client, self.cfg, self.tools, self.env = client, cfg, tools, env
        self.composer = composer or GroundedPromptComposer(cfg.default_same_lane_spacing)
        self.validator = validator or FormationValidator(cfg)
        self.verbose = verbose

    def log(self, *a):
        if self.verbose:
            print(*a, flush=True)

    # ------------------------------------------------------------------
    def generate(self, state: FormationState, task: TaskInput) -> GenerationRecord:
        t0 = time.perf_counter()
        constraints = self.env.constraints()
        base = dict(task_context=describe_task_context(state), user_task_input=describe_user_input(task))
        errors: List[str] = []

        for attempt in range(1, self.cfg.max_invalid_outputs + 1):
            self.log(f"[generator] attempt {attempt}/{self.cfg.max_invalid_outputs}")
            parts = self.composer.generation_parts(state, task, errors or None)
            raw, trace = self._reason_with_tools(parts)
            parsed = parse_generation_output(raw)
            vr = self.validator.validate(parsed.formation, state.num_cavs, state.num_lanes, constraints)
            if vr.ok:
                t = parsed.tags
                # Generation context is taken from the predefined output tags (Sec. 3.3);
                # if the MLLM omitted a tag, fall back to the actual execution trace.
                return GenerationRecord(
                    **base, formation=parsed.formation, source="mllm",
                    retrieved_knowledge=t["knowledge"] or self._trace_text(trace, "explore_knowledge_graph"),
                    retrieved_experience=t["experience"] or self._trace_text(trace, "explore_experience"),
                    invoked_tools=t["tool"] or ", ".join(c["name"] for c in trace if c["name"] not in RETRIEVAL),
                    tool_outputs=t["results"] or self._trace_text(trace, None),
                    raw_output=raw, tool_trace=trace, attempts=attempt,
                    validation_errors=errors, elapsed_s=time.perf_counter() - t0)
            errors = vr.errors
            self.log(f"[generator] output rejected: {errors}")

        # Three consecutive invalid outputs -> rule-based interlaced formation over usable lanes.
        blocked = set(constraints["lanes_to_clear"]) | set(constraints["unavailable_lanes"])
        usable = [l for l in range(1, state.num_lanes + 1) if l not in blocked]
        fb = rule_based_interlaced(state.num_cavs, usable, state.previous_formation, state.same_lane_spacing)
        vr = self.validator.validate(fb or None, state.num_cavs, state.num_lanes, constraints)
        source = "rule_based" if (usable and vr.ok) else "safe_stop"
        self.log(f"[generator] falling back -> {source}")
        return GenerationRecord(**base, formation=fb if source == "rule_based" else None, source=source,
                                attempts=self.cfg.max_invalid_outputs, validation_errors=errors,
                                elapsed_s=time.perf_counter() - t0)

    # ------------------------------------------------------------------
    def _reason_with_tools(self, user_parts):
        """MLLM reasoning loop (Eq. 11). Retrieved knowledge and tool outputs are fed back
        iteratively into the context until the MLLM produces its final answer."""
        config = types.GenerateContentConfig(
            temperature=self.cfg.temperature, top_p=self.cfg.top_p,
            thinking_config=types.ThinkingConfig(thinking_level=self.cfg.thinking_level),
            tools=self.tools.gemini_tools(),
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        )
        history = [types.Content(role="user", parts=user_parts)]
        trace = []
        for _ in range(self.cfg.max_tool_rounds):
            resp = self.client.models.generate_content(
                model=self.cfg.generation_model, contents=history, config=config)
            calls = resp.function_calls or []
            if not calls:
                return self._text(resp), trace
            # keep the model turn as-is (preserves Gemini thought signatures)
            history.append(resp.candidates[0].content)
            replies = []
            for fc in calls:
                args = dict(fc.args or {})
                result = self.tools.execute(fc.name, args)
                trace.append({"name": fc.name, "args": args, "result": result})
                self.log(f"[tool] {fc.name}({args}) -> {to_text(result)}")
                part = types.Part.from_function_response(name=fc.name, response={"result": result})
                if getattr(fc, "id", None):
                    part.function_response.id = fc.id
                replies.append(part)
            history.append(types.Content(role="user", parts=replies))
        return "", trace   # too many tool rounds -> treated as an invalid output

    @staticmethod
    def _text(resp) -> str:
        parts = (resp.candidates[0].content.parts or []) if resp.candidates else []
        return "".join(p.text for p in parts if getattr(p, "text", None) and not getattr(p, "thought", False))

    @staticmethod
    def _trace_text(trace, name) -> str:
        # name=None selects the real-time tools (everything except knowledge retrieval)
        sel = [c for c in trace if ((c["name"] == name) if name else (c["name"] not in RETRIEVAL))]
        return "; ".join(to_text(c["result"]) for c in sel)
