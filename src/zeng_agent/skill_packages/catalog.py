"""Data-driven skill package catalog exposed to the Agent."""
from __future__ import annotations
from dataclasses import asdict, dataclass, field
from typing import Any

@dataclass(frozen=True)
class SkillPackage:
    skill_id: str
    domain: str
    description: str
    intent: str
    permission: str = "read_only"
    requires_confirmation: bool = False
    safety_override: bool = False
    parameters: dict[str, dict[str, Any]] = field(default_factory=dict)
    preconditions: list[str] = field(default_factory=list)
    side_effects: str = ""
    version: str = "1.0.0"

    def to_agent_tool(self) -> dict[str, Any]:
        return asdict(self)

class SkillPackageCatalog:
    def __init__(self, packages: list[SkillPackage] | None = None) -> None:
        self._packages: dict[str, SkillPackage] = {}
        for package in packages or []:
            self.register(package)

    def register(self, package: SkillPackage) -> None:
        if package.skill_id in self._packages:
            raise ValueError(f"duplicate skill package: {package.skill_id}")
        self._packages[package.skill_id] = package

    def get(self, skill_id: str) -> SkillPackage | None:
        return self._packages.get(skill_id)

    def list(self) -> list[SkillPackage]:
        return sorted(self._packages.values(), key=lambda item: item.skill_id)

    def as_agent_tools(self) -> list[dict[str, Any]]:
        return [item.to_agent_tool() for item in self.list()]

    def by_intent(self, intent: str) -> SkillPackage | None:
        return next((item for item in self._packages.values() if item.intent == intent), None)

def _p(kind: str, description: str, **kwargs: Any) -> dict[str, Any]:
    value = {"type": kind, "description": description}
    value.update(kwargs)
    return value

def default_catalog(require_motion_confirmation: bool = True) -> SkillPackageCatalog:
    motion = require_motion_confirmation
    return SkillPackageCatalog([
        SkillPackage("list_waypoints", "navigation", "List available waypoint names.", "list_waypoints"),
        SkillPackage("robot_status", "system", "Read ROS2 nodes, topics, and robot status.", "robot_status"),
        SkillPackage("stop_navigation", "safety", "Stop active navigation as a safety override.", "stop_navigation", "safe_action", safety_override=True, side_effects="Stops robot navigation."),
        SkillPackage("navigate_to_waypoint", "navigation", "Navigate to a named waypoint.", "navigate_to_waypoint", "motion_action", motion, parameters={"waypoint": _p("string", "Waypoint name.", required=True)}, preconditions=["ROS2 navigation service is available."], side_effects="Moves the robot."),
        SkillPackage("save_waypoint", "navigation", "Save the current pose as a waypoint.", "save_waypoint", "motion_action", motion, parameters={"waypoint": _p("string", "New waypoint name.", required=True)}, side_effects="Writes waypoint data."),
        SkillPackage("arm_preset", "manipulation", "Run a configured arm pose preset.", "arm_preset", "motion_action", motion, parameters={"preset": _p("string", "Preset name.", required=True)}, preconditions=["Preset is listed in allowed_arm_presets."], side_effects="Moves the robot arm."),
        SkillPackage("arm_stop", "safety", "Stop robot arm motion.", "arm_stop", "safe_action", safety_override=True, side_effects="Stops the robot arm."),
        SkillPackage("arm_continuous_move", "manipulation", "Move the arm continuously in one direction.", "arm_continuous_move", "motion_action", motion, parameters={"direction": _p("enum", "Motion direction.", allowed=["up","down","left","right","forward","backward"]), "distance_m": _p("number", "Travel distance in meters.", minimum=0, maximum=0.5)}, side_effects="Moves the robot arm."),
        SkillPackage("arm_cross_wave", "manipulation", "Run the cross-wave arm gesture.", "arm_cross_wave", "motion_action", motion, side_effects="Moves both robot arms."),
        SkillPackage("vision_navigate", "navigation", "Navigate incrementally using the front camera.", "vision_navigate", "motion_action", motion, parameters={"goal_distance_m": _p("number", "Target distance in meters.", minimum=0, maximum=3.0)}, preconditions=["Front camera and vision service are available."], side_effects="Moves the robot using vision guidance."),
        SkillPackage("vision_turn", "navigation", "Check surroundings with vision before turning.", "vision_turn", "motion_action", motion, parameters={"direction": _p("enum", "Turn direction.", allowed=["left","right","auto"]), "angle_deg": _p("number", "Turn angle in degrees.", minimum=1, maximum=120)}, side_effects="Turns the robot."),
        SkillPackage("fire_search", "perception_navigation", "Search for and approach a fire hydrant using fresh camera frames.", "fire_search", "motion_action", motion, preconditions=["Front camera, vision model, and ROS2 chassis are available."], side_effects="Turns and advances the robot under vision guidance."),
    ])
