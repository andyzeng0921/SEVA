# Integration guide

The portable policy and offline tests run without ROS. The native supervisor
targets Linux with ROS 2 Jazzy and a separately commissioned robot stack. No
installation or test command in the README starts physical navigation.

## Upstream frontier bridge

Install ROS packages providing `rclcpp`, `tf2_ros`, `nav_msgs`, `std_srvs`, Nav2,
SLAM Toolbox and the upstream frontier package's declared dependencies. The
legacy supervisor also imports `explore_lite_msgs`; that message package is an
external dependency even when the reward strategy is selected.

```bash
python scripts/fetch_upstream.py --group exploration
source /opt/ros/jazzy/setup.bash
mkdir -p ros2_ws
cd ros2_ws
colcon build --base-paths ../upstream/frontier-exploration-ros2-20260914 \
  ../integrations/exploration306/native_bridge \
  --packages-up-to zeng306_frontier_bridge
source install/setup.bash
cd ..
export SEVA_FRONTIER_BRIDGE="$PWD/ros2_ws/install/zeng306_frontier_bridge/lib/zeng306_frontier_bridge/candidate_bridge"
```

The bridge exposes `/zeng306/frontier_candidates` as a read-only `Trigger`
service. It has no velocity publisher or navigation action client. WFD and MRTSP
remain in the pinned upstream library. Use the upstream ROS test instructions to
test that library separately.

## Native interfaces and session contract

The adapter expects native `/compute_path_to_pose`, `/is_path_valid` and
`/navigate_to_pose`, a `map` to `base_link` transform, `/map`, native SLAM graph
markers, dual-lidar and odometry feedback, battery status and the deployed
supervisor's posture/control-ownership signals. The exact topic subscriptions and
freshness checks are in `deployment/coverage_session.py` and `session_health.py`.
Map, footprint, topic and transform configuration must match the actual platform.

`config/robot/` preserves the experimental parameter values as templates. The
`/opt/seva/session` behavior-tree path and `/opt/robot` SDK locations are deployment
placeholders. They are not automatically installed. Calibrated footprint, sensor
mounts, posture feedback, Collision Monitor, SDK transport and service launch
ownership must be supplied by the robot deployment.

The session supervisor expects this writable local structure:

```text
session/
  config/        reward.306.json, native-navigation-reward.xml, platform parameters
  evidence/
  logs/
  maps/
  bags/
  observations/
  semantics/
  reward/
  capture-semantic-start.sh
```

Copy `deployment/reward.306.json` and `deployment/native-navigation-reward.xml`
into `session/config/`. For another map, set `graph_first_live_node` to the first
node belonging to that exploration interval; `188` is the original experiment's
seed-map boundary, not a universal constant. The corresponding capture association
filter in `capture_semantic.py` must use the same boundary.

Set `SEVA_NAV_UNIT` to the existing owned navigation systemd unit. The supervisor
checks that unit and stops it on failure. `SEVA_LEGACY_EXPLORER` is only needed for
the optional legacy strategy. With the native stack and message types installed,
the supervisor's default mode performs a read-only health check:

```bash
export PYTHONPATH="$PWD/src:$PWD/integrations${PYTHONPATH:+:$PYTHONPATH}"
python integrations/exploration306/deployment/coverage_session.py /path/to/session
```

Physical execution is an explicit `--run --strategy reward` mode. It requires the
already commissioned stack, fresh telemetry, unique control ownership, a prepared
robot and the native collision checks. Changing a score or VLM response never
authorizes bypassing these requirements. The gateway's motion commissioning flag
remains false independently of the exploration supervisor.

## Camera SDK binding

`upstream306.observations` reads fresh synchronized RGB-D through the robot's
existing `camera_config.py` and `shm_camera.py`. These SDK files are not redistributed.
Supply licensed copies in the ignored `integrations/vendor306/` directory and a
`sources.json` mapping each filename to its SHA-256. The reader verifies those
files before import. Both color and depth producer timestamps are checked; an
injected `reader_pair` is used by offline tests.

## Vision-language service

The example configuration uses `qwen3.5-35b-a3b` and an OpenAI-compatible chat
endpoint. Configure credentials only through environment variables or the ignored
local `config.yaml`:

```bash
export ZENG_CONFIG="$PWD/config.yaml"
export SEVA_VLM_URL="http://127.0.0.1:8000/v1/chat/completions"
export SEVA_VLM_API_KEY="your-local-service-key"
```

`SEVA_LLM_BASE_URL` and `SEVA_LLM_API_KEY` configure the optional language planner.
`capture-semantic-start.sh` in a commissioned session should load its ROS/SDK
environment and invoke `python -m exploration306.capture_semantic /path/to/session`.
Capture runs after stationary feedback. The schema stores image labels,
normalized boxes, view quality, uncertainty and a viewpoint/graph association;
it does not estimate object world coordinates or authorize navigation.

## Isolated ROS contract checks

After sourcing ROS and installing Python dependencies, run these on a development
machine with no robot connection. They use fake planning/navigation responses:

```bash
export PYTHONPATH="$PWD/src:$PWD/integrations${PYTHONPATH:+:$PYTHONPATH}"
export ROS_DOMAIN_ID=224
export ROS_LOCALHOST_ONLY=1
python -m unittest exploration306.test_reward_runner exploration306.test_native_observation -v
PYTHONPATH="$PWD/integrations/exploration306/deployment:$PYTHONPATH" \
  python -m unittest discover -s integrations/exploration306/deployment -p 'test_*.py' -v
```

`capture_inputs.py`, `validate_native.py` and `build_semantic_report.py` are original
capture/replay/report source. Their recordings and map inputs are excluded from
this release. Native C++/ROS and hardware validation requires that separate
environment; portable pytest results do not establish physical performance.
