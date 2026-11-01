#!/usr/bin/env python3
"""Mock on-vehicle side: subscribes to the desired formation and replies with an execution outcome.

Stands in for the execution module + execution evaluator until the real planner is connected.
    python3 scripts/mock_execution_outcome_publisher.py --delay 2
"""
import argparse
import json
import sys
from pathlib import Path

import rclpy
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import Float64MultiArray, String

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from grf_tafc.config import GRFConfig  # noqa: E402
from grf_tafc.formation import parse_points  # noqa: E402
from grf_tafc.mock_execution import mock_outcome  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--delay", type=float, default=2.0, help="simulated transition duration before replying [s]")
    ap.add_argument("--num-lanes", type=int, default=3)
    ap.add_argument("--ambulance-lane", type=int, default=3)
    ap.add_argument("--previous-formation", default="[(1,1), (3,1), (0,2), (2,2), (4,2), (1,3), (3,3)]")
    args = ap.parse_args()
    cfg = GRFConfig()

    rclpy.init()
    node = rclpy.create_node("grf_tafc_mock_execution")
    qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL)
    pub = node.create_publisher(String, cfg.outcome_topic, 10)
    state = {"prev": parse_points(args.previous_formation), "timer": None}

    def on_formation(msg):
        d = list(msg.data)
        new = [(d[i], d[i + 1]) for i in range(0, len(d), 2)]
        node.get_logger().info(f"received formation {new}; replying in {args.delay}s")

        def reply():
            node.destroy_timer(state["timer"])
            out = mock_outcome(state["prev"], new, args.num_lanes,
                               [args.ambulance_lane] if args.ambulance_lane else [])
            pub.publish(String(data=json.dumps(out.__dict__)))
            node.get_logger().info(f"published outcome {out.__dict__}")
            state["prev"] = new

        state["timer"] = node.create_timer(args.delay, reply)

    node.create_subscription(Float64MultiArray, cfg.formation_topic, on_formation, qos)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
