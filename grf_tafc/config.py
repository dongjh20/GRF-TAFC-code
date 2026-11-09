"""Central configuration for the GRF-TAFC formation generator and reflection module.

Fill in API_KEY below, or (recommended) export GEMINI_API_KEY / GOOGLE_API_KEY.
"""
import os
from dataclasses import dataclass, field
from pathlib import Path

# ---------------------------------------------------------------------------
# Put your own Gemini API key here, or leave it empty and set the environment
# variable GEMINI_API_KEY (or GOOGLE_API_KEY). The environment variable wins.
# ---------------------------------------------------------------------------
API_KEY = ""

PACKAGE_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PACKAGE_ROOT / "data"          # offline-built resources shipped with the repo
RUNTIME_DIR = PACKAGE_ROOT / "runtime"    # files written during operation (git-ignored)


def get_api_key() -> str:
    key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or API_KEY
    if not key:
        raise RuntimeError(
            "No Gemini API key found. Set GEMINI_API_KEY (or GOOGLE_API_KEY), "
            "or fill API_KEY in grf_tafc/config.py."
        )
    return key


@dataclass
class GRFConfig:
    # --- MLLMs (Sec. 3.2 uses Gemini 3.1 Pro for generation; Sec. 4.1 uses it for reflection) ---
    generation_model: str = "gemini-3.1-pro-preview"
    reflection_model: str = "gemini-3.1-pro-preview"
    embedding_model: str = "gemini-embedding-001"
    thinking_level: str = "low"          # low / medium / high
    temperature: float = 1.0
    top_p: float = 0.95
    max_tool_rounds: int = 8             # max function-calling rounds per generation attempt

    # --- Output validation & fallback (Sec. 3.2.3) ---
    max_invalid_outputs: int = 3         # after 3 invalid outputs -> rule-based generator
    default_same_lane_spacing: float = 2.0   # relative x-spacing between same-lane CAVs

    # --- Relative -> physical conversion used by the validator ---
    # A relative x-step of 1 equals half the desired within-lane following distance.
    desired_following_distance_m: float = 10.0
    lane_width_m: float = 3.5
    min_inter_cav_distance_m: float = 5.0
    min_obstacle_clearance_m: float = 5.0

    # --- Knowledge database (Sec. 3.2.2) ---
    knowledge_graph_path: Path = DATA_DIR / "domain_knowledge_graph.json"
    experience_seed_path: Path = DATA_DIR / "experience_memory_seed.json"
    experience_memory_path: Path = RUNTIME_DIR / "experience_memory.json"
    embedding_cache_path: Path = RUNTIME_DIR / "embedding_cache.json"
    experience_top_k: int = 2
    experience_min_similarity: float = 0.55
    # Semantic de-duplication in the memory update (Eq. 14)
    dedup_key_similarity: float = 0.85
    dedup_value_similarity: float = 0.92

    # --- ROS 2 interface (execution module <-> cloud) ---
    formation_topic: str = "/grf_tafc/desired_formation"     # we publish
    outcome_topic: str = "/grf_tafc/execution_outcome"       # we subscribe
    outcome_timeout_s: float = 120.0

    log_dir: Path = field(default_factory=lambda: RUNTIME_DIR / "logs")
