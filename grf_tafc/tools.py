"""External resources exposed to the MLLM as callable functions (Sec. 3.2.2).

* Real-time tool library  : time-sensitive facts (here a mocked ambulance-lane query).
* Knowledge database      : explore_knowledge_graph (domain knowledge) and
                            explore_experience (experience memory).

As in the paper, function descriptions and parameter schemas are supplied through
Gemini function declarations, not repeated in the text prompt. The MLLM chooses the
function and its arguments; this application executes it and returns the result.
"""
import json
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List

from google.genai import types

from .experience_memory import ExperienceMemory
from .knowledge_graph import DomainKnowledgeGraph


@dataclass
class MockTrafficEnvironment:
    """Stand-in for roadside units / V2X / connected platforms.

    TODO(real system): replace with live sources, e.g. a ROS 2 subscription to the
    connected-ambulance position, incident reports, or perception output.
    """
    ambulance_lane: int = 3                                 # 0 = no ambulance nearby
    unavailable_lanes: List[int] = field(default_factory=list)
    obstacles_m: List[tuple] = field(default_factory=list)  # (X, Y) in the formation frame, metres

    def get_nearest_ambulance_lane(self) -> int:
        return self.ambulance_lane

    def constraints(self) -> dict:
        """Ground truth used by the output validator, independent of what the MLLM queried."""
        cleared = [self.ambulance_lane] if self.ambulance_lane else []
        return {"lanes_to_clear": cleared,
                "unavailable_lanes": list(self.unavailable_lanes),
                "obstacles_m": list(self.obstacles_m)}


@dataclass
class Tool:
    declaration: types.FunctionDeclaration
    fn: Callable[..., Any]


class ToolRegistry:
    """Holds the tool_list registered with the MLLM and executes the calls it makes."""

    def __init__(self, env: MockTrafficEnvironment, kg: DomainKnowledgeGraph,
                 memory: ExperienceMemory, experience_top_k: int = 2,
                 experience_min_similarity: float = 0.0):
        self.tools: Dict[str, Tool] = {}

        # ---- real-time tool library ------------------------------------
        self.register(
            "get_nearest_ambulance_lane",
            "Query the real-time traffic information system for the nearest approaching ambulance. "
            "Returns the index of the lane it currently occupies, using the same lane indexing as the "
            "formation coordinates (y = lane index). Returns 0 if no ambulance is nearby.",
            {"type": "object", "properties": {}},
            lambda: env.get_nearest_ambulance_lane(),
        )

        # ---- knowledge database: domain knowledge graph -------------------
        self.register(
            "explore_knowledge_graph",
            "Retrieve domain knowledge (human-intent interpretation, traffic regulations, formation-level "
            "safety) from a knowledge graph. Starts from a seed entity and follows relations for up to "
            "max_hops hops. Returns the traversed (head, relation, tail) triples and the knowledge texts.",
            {"type": "object",
             "properties": {
                 "entity": {"type": "string", "description": "Task-relevant seed entity, e.g. 'ambulance'."},
                 "relation": {"type": "string", "description": "Optional comma-separated relation types to "
                              "follow (e.g. 'IS_A,REQUIRES'). Empty string = all relations."},
                 "max_hops": {"type": "integer", "description": "Maximum number of hops (default 2)."}},
             "required": ["entity"]},
            lambda entity, relation="", max_hops=2: kg.explore(entity, relation, int(max_hops)),
        )

        # ---- knowledge database: experience memory -------------------------
        def explore_experience(query: str, top_k: int = experience_top_k):
            hits = memory.retrieve(query, int(top_k), experience_min_similarity)
            return {"experiences": [{"key": e.key, "value": e.value, "similarity": round(s, 3)}
                                    for e, s in hits]}

        self.register(
            "explore_experience",
            "Retrieve the most relevant experience-memory entries. Each entry is a key-value pair: the key "
            "is a previous task context and the value is a reflection-derived recommendation. Ranked by "
            "embedding cosine similarity between the query and the stored keys.",
            {"type": "object",
             "properties": {
                 "query": {"type": "string", "description": "Current task and operating context, e.g. "
                           "'Reduction of available lanes from three to two for a seven-CAV formation'."},
                 "top_k": {"type": "integer", "description": "Number of entries to return."}},
             "required": ["query"]},
            explore_experience,
        )

    def register(self, name: str, description: str, schema: dict, fn: Callable[..., Any]):
        decl = types.FunctionDeclaration(name=name, description=description,
                                         parameters_json_schema=schema)
        self.tools[name] = Tool(decl, fn)

    def gemini_tools(self) -> List[types.Tool]:
        return [types.Tool(function_declarations=[t.declaration for t in self.tools.values()])]

    def execute(self, name: str, args: dict) -> Any:
        if name not in self.tools:
            return {"error": f"unknown function '{name}'"}
        try:
            return self.tools[name].fn(**(args or {}))
        except Exception as e:  # report errors back to the MLLM instead of crashing
            return {"error": f"{type(e).__name__}: {e}"}


def to_text(result: Any) -> str:
    return result if isinstance(result, str) else json.dumps(result, ensure_ascii=False)
