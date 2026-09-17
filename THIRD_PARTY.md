# Upstream components

SEVA integrates existing algorithms. It does not reimplement frontier search,
SLAM, path planning, or robot control. Third-party source, binaries, weights and
datasets are not redistributed in this repository.

| Component | Use | License information in the inspected source |
| --- | --- | --- |
| [frontier_exploration_ros2](https://github.com/mertgulerx/frontier_exploration_ros2) | WFD, visible frontier gain, MRTSP ordering | Apache-2.0 |
| [Stretch AI](https://github.com/hello-robot/stretch_ai) | Task/Operation protocol | Apache-2.0; preserve upstream notices |
| [LIMP](https://github.com/benedictquartey/robotlimp) | Optional temporal-logic progression | MIT |
| [REFLECT](https://github.com/real-stanford/reflect) | Optional failure summaries | MIT |
| [PhysMem](https://github.com/haoyangli16/PhysMem) | Separate task-gateway experience memory | MIT declared in `pyproject.toml`; the pinned tree has no standalone LICENSE |
| [BUMBLE](https://github.com/UT-Austin-RobIn/BUMBLE) | Optional skill-selection prompts | No explicit license found in the pinned tree; consult the authors for reuse terms |
| [SLAM Toolbox](https://github.com/SteveMacenski/slam_toolbox) | External lidar mapping and pose graph | Install separately under its upstream terms |
| [Navigation2](https://github.com/ros-navigation/navigation2) | External Smac2D, RPP, Collision Monitor | Install separately under its upstream terms |

Exact revisions for the source-loaded components are recorded in
[`dependencies/sources.lock.json`](dependencies/sources.lock.json). The adapter
checks entry-point hashes from `dependencies/entrypoints.sha256.json` before
loading those functions. No upstream licensing terms are replaced by SEVA's
Apache-2.0 license.

The robot's SDK, shared-memory camera reader, native ROS interfaces, calibrated
geometry and runtime launch configuration must be supplied by the robot owner.
The Qwen VLM is accessed through a configurable compatible API; no model weights
or hosted-service credentials are distributed. The included diagram and map
preview are project illustrations, not an experimental dataset.
