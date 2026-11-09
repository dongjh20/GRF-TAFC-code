"""ROS 2 bridge between the cloud-side generator/reflection and the on-vehicle execution module.

Publishes : <formation_topic>  std_msgs/Float64MultiArray  data = [x1, y1, x2, y2, ...] (relative coords)
Subscribes: <outcome_topic>    std_msgs/String             JSON execution outcome, e.g.
    {"task_results": "Successful convergence; ...", "safety_checks": "No collision, no near-collision.",
     "lane_changes": 4, "transition_time_s": 9.14, "transition_distance_m": 208.49}

TODO(real system): the execution evaluator should monitor the formation state until convergence,
failure, or timeout and publish the summary above. scripts/mock_execution_outcome_publisher.py
plays that role for testing.
"""
import json
import time
from typing import List, Optional

from .formation import Point
from .reflection import ExecutionOutcome


class RosBridge:
    def __init__(self, formation_topic: str, outcome_topic: str):
        import rclpy
        from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
        from std_msgs.msg import Float64MultiArray, String

        self._rclpy, self._Float64MultiArray = rclpy, Float64MultiArray
        if not rclpy.ok():
            rclpy.init()
        self.node = rclpy.create_node("grf_tafc_formation_generator")
        qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                         durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.formation_pub = self.node.create_publisher(Float64MultiArray, formation_topic, qos)
        self._outcome: Optional[dict] = None
        self.node.create_subscription(String, outcome_topic, self._on_outcome, 10)

    def _on_outcome(self, msg):
        try:
            self._outcome = json.loads(msg.data)
        except json.JSONDecodeError:
            self.node.get_logger().warn(f"execution outcome is not valid JSON: {msg.data!r}")

    def publish_formation(self, points: List[Point]):
        self._outcome = None   # only accept outcomes that arrive after this formation
        msg = self._Float64MultiArray()
        msg.data = [float(v) for p in points for v in p]
        self.formation_pub.publish(msg)
        self.node.get_logger().info(f"published desired formation: {points}")

    def wait_for_outcome(self, timeout_s: float) -> ExecutionOutcome:
        deadline = time.monotonic() + timeout_s
        while self._outcome is None and time.monotonic() < deadline:
            self._rclpy.spin_once(self.node, timeout_sec=0.1)
        if self._outcome is None:
            return ExecutionOutcome(task_results=f"Timeout: no execution outcome within {timeout_s:.0f} s.",
                                    safety_checks="Unknown.")
        return ExecutionOutcome.from_dict(self._outcome)

    def shutdown(self):
        self.node.destroy_node()
        if self._rclpy.ok():
            self._rclpy.shutdown()
