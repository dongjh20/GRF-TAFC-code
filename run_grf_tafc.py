#!/usr/bin/env python3
"""Closed loop of GRF-TAFC (Eq. 4): generate -> execute -> reflect -> update experience memory.

Examples
  # one task, execution outcome from the ROS 2 topic (start the execution side / mock publisher first)
  python3 run_grf_tafc.py --task "Watch out for the ambulance!" --outcome ros

  # one task, no ROS: use the built-in mock execution evaluator
  python3 run_grf_tafc.py --task "Watch out for the ambulance!" --outcome mock

  # multimodal input
  python3 run_grf_tafc.py --task "Form this shape." --image desired_shape.jpg --outcome mock
  python3 run_grf_tafc.py --audio user_task_input_1.wav --outcome mock

  # interactive: enter successive tasks; previous formation is carried over
  python3 run_grf_tafc.py --outcome mock
"""
import argparse
import json
import time

from google import genai

from grf_tafc.config import GRFConfig, get_api_key
from grf_tafc.experience_memory import ExperienceMemory, GeminiEmbedder, HashingEmbedder
from grf_tafc.formation import FormationState, format_formation, parse_points
from grf_tafc.generator import FormationGenerator
from grf_tafc.knowledge_graph import DomainKnowledgeGraph
from grf_tafc.mock_execution import mock_outcome
from grf_tafc.prompt_composer import TaskInput
from grf_tafc.reflection import ReflectionModule
from grf_tafc.tools import MockTrafficEnvironment, ToolRegistry

DEFAULT_PREVIOUS = "[(1,1), (3,1), (0,2), (2,2), (4,2), (1,3), (3,3)]"


def build_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", help="text task input (omit for interactive mode)")
    ap.add_argument("--image", help="image task input, e.g. a desired formation shape")
    ap.add_argument("--audio", help="speech task input (.wav/.mp3/.m4a/.ogg)")
    ap.add_argument("--num-lanes", type=int, default=3)
    ap.add_argument("--previous-formation", default=DEFAULT_PREVIOUS)
    ap.add_argument("--ambulance-lane", type=int, default=3, help="mocked real-time tool output (0 = none)")
    ap.add_argument("--outcome", choices=["ros", "mock", "none"], default="ros",
                    help="where the execution outcome comes from; 'none' skips reflection")
    ap.add_argument("--offline-embedder", action="store_true",
                    help="use a local bag-of-words embedder instead of the Gemini embedding API")
    return ap.parse_args()


def main():
    args = build_args()
    cfg = GRFConfig()
    client = genai.Client(api_key=get_api_key())

    embedder = HashingEmbedder() if args.offline_embedder else \
        GeminiEmbedder(client, cfg.embedding_model, cfg.embedding_cache_path)
    memory = ExperienceMemory(cfg.experience_memory_path, embedder, cfg.experience_seed_path,
                              cfg.dedup_key_similarity, cfg.dedup_value_similarity)
    kg = DomainKnowledgeGraph(cfg.knowledge_graph_path)
    env = MockTrafficEnvironment(ambulance_lane=args.ambulance_lane)
    tools = ToolRegistry(env, kg, memory, cfg.experience_top_k, cfg.experience_min_similarity)
    generator = FormationGenerator(client, cfg, tools, env)
    reflector = ReflectionModule(client, cfg, memory)

    bridge = None
    if args.outcome == "ros":
        from grf_tafc.ros_interface import RosBridge
        bridge = RosBridge(cfg.formation_topic, cfg.outcome_topic)

    prev = parse_points(args.previous_formation)
    state = FormationState(num_cavs=len(prev), num_lanes=args.num_lanes, previous_formation=prev,
                           same_lane_spacing=cfg.default_same_lane_spacing)
    cfg.log_dir.mkdir(parents=True, exist_ok=True)

    def tasks():
        if args.task or args.image or args.audio:
            yield TaskInput(args.task or "", args.image, args.audio)
            return
        while True:
            try:
                text = input("\nTask input (q to quit) >>> ").strip()
            except (EOFError, KeyboardInterrupt):
                return
            if text.lower() in ("q", "quit", "exit"):
                return
            if text:
                yield TaskInput(text)

    try:
        for task in tasks():
            # ---------------- formation generation (Sec. 3.2) ----------------
            record = generator.generate(state, task)
            print(f"\n[generator] source={record.source}  attempts={record.attempts}  "
                  f"time={record.elapsed_s:.2f}s")
            print(f"[generator] formation: {format_formation(record.formation) if record.formation else None}")
            if record.formation is None:
                print("[generator] SAFE STOP: no admissible formation.")
                continue

            # ---------------- execution (on-vehicle, via ROS 2) ----------------
            if args.outcome == "ros":
                bridge.publish_formation(record.formation)
                print(f"[execution] waiting for outcome on {cfg.outcome_topic} ...")
                outcome = bridge.wait_for_outcome(cfg.outcome_timeout_s)
            elif args.outcome == "mock":
                outcome = mock_outcome(state.previous_formation, record.formation, state.num_lanes,
                                       env.constraints()["lanes_to_clear"])
            else:
                outcome = None
            if outcome:
                print(f"[execution] {outcome.task_results} | {outcome.safety_checks} | "
                      f"{outcome.performance_text()}")

            # ---------------- execution-driven reflection (Sec. 3.3) ----------------
            insight = reflector.reflect(record, outcome) if outcome else None

            log = {"record": record.__dict__, "outcome": outcome.__dict__ if outcome else None,
                   "insight": insight.__dict__ if insight else None}
            path = cfg.log_dir / time.strftime("task_%Y%m%d_%H%M%S.json")
            path.write_text(json.dumps(log, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
            print(f"[log] {path}")

            state.previous_formation = record.formation   # becomes F_{k} for the next task
    finally:
        if bridge:
            bridge.shutdown()


if __name__ == "__main__":
    main()
