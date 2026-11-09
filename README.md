# GRF-TAFC: Task-Adaptive Formation Generation with Reflection

This repository provides the task-adaptive formation generator and the execution-driven reflection module of GRF-TAFC (*Choreographing CAV Formations on the Fly with Reflection*). It is built on the Gemini API.

## Quick start

```bash
pip install -r requirements.txt
export GEMINI_API_KEY=<your key>          # or fill API_KEY in grf_tafc/config.py
```

Run without ROS. A built-in mock evaluator provides the execution outcome:

```bash
python3 run_grf_tafc.py --task "Watch out for the ambulance!" --outcome mock
```

Run with ROS 2 (tested on Humble). In terminal 1, start a mock execution side:

```bash
source /opt/ros/humble/setup.bash
python3 scripts/mock_execution_outcome_publisher.py --delay 2
```

In terminal 2:

```bash
source /opt/ros/humble/setup.bash
python3 run_grf_tafc.py --task "Watch out for the ambulance!" --outcome ros
```

Other task inputs are `--image shape.jpg` and `--audio request.wav`. Omit `--task` to enter tasks interactively.

## ROS 2 interface

| Topic | Type | Direction | Content |
|---|---|---|---|
| `/grf_tafc/desired_formation` | `std_msgs/Float64MultiArray` | published | `[x1, y1, x2, y2, ...]`: relative coordinates, where y is the lane index |
| `/grf_tafc/execution_outcome` | `std_msgs/String` (JSON) | subscribed | `{"task_results", "safety_checks", "lane_changes", "transition_time_s", "transition_distance_m"}` |

To connect a real execution module, subscribe to the formation topic and publish the outcome JSON once the formation has converged, failed, or timed out.
