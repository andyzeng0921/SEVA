from __future__ import annotations

import time
from typing import Protocol

from .config import AgentConfig
from .models import ExecutionRequest, ExecutionResult


class ControlBackend(Protocol):
    def navigate_to_waypoint(self, name: str) -> ExecutionResult: ...
    def save_waypoint(self, name: str) -> ExecutionResult: ...
    def list_waypoints(self) -> ExecutionResult: ...
    def stop_navigation(self) -> ExecutionResult: ...
    def arm_preset(self, action_name: str) -> ExecutionResult: ...
    def arm_stop(self) -> ExecutionResult: ...
    def arm_continuous_move(self, direction: str, amount: float, limb: str) -> ExecutionResult: ...
    def arm_cross_wave(self, phase: str) -> ExecutionResult: ...
    def vision_navigate(self, goal_distance: float) -> ExecutionResult: ...
    def vision_turn(self, direction: str, angle_deg: float) -> ExecutionResult: ...
    def fire_search(self, instruction: str) -> ExecutionResult: ...
    def robot_status(self) -> ExecutionResult: ...


class DryRunBackend:
    def __init__(self, waypoint_names: list[str], allowed_arm_presets: list[str]) -> None:
        self._waypoints = sorted(set(waypoint_names))
        self._allowed_arm_presets = set(allowed_arm_presets)

    def navigate_to_waypoint(self, name: str) -> ExecutionResult:
        if self._waypoints and name not in self._waypoints:
            return ExecutionResult(success=False, message=f"Unknown waypoint: {name}")
        return ExecutionResult(
            success=True,
            message=f"Dry-run accepted navigation to {name}",
            data={"mode": "dry_run", "action": "navigate_to_waypoint", "name": name},
        )

    def save_waypoint(self, name: str) -> ExecutionResult:
        return ExecutionResult(
            success=True,
            message=f"Dry-run accepted save waypoint {name}",
            data={"mode": "dry_run", "action": "save_waypoint", "name": name},
        )

    def list_waypoints(self) -> ExecutionResult:
        return ExecutionResult(
            success=True,
            message="Dry-run waypoint list",
            data={"mode": "dry_run", "action": "list_waypoints", "names": self._waypoints},
        )

    def stop_navigation(self) -> ExecutionResult:
        return ExecutionResult(
            success=True,
            message="Dry-run stop navigation accepted",
            data={"mode": "dry_run", "action": "stop_navigation"},
        )

    def arm_preset(self, action_name: str) -> ExecutionResult:
        if action_name not in self._allowed_arm_presets:
            return ExecutionResult(success=False, message=f"Unsupported arm preset: {action_name}")
        return ExecutionResult(
            success=True,
            message=f"Dry-run arm preset accepted: {action_name}",
            data={"mode": "dry_run", "action": "arm_preset", "preset": action_name},
        )

    def arm_stop(self) -> ExecutionResult:
        return ExecutionResult(
            success=True,
            message="Dry-run arm stop accepted",
            data={"mode": "dry_run", "action": "arm_stop"},
        )

    def arm_continuous_move(self, direction: str, amount: float, limb: str) -> ExecutionResult:
        return ExecutionResult(
            success=True,
            message=f"Dry-run continuous move: {limb} arm {direction} by {amount}m",
            data={
                "mode": "dry_run",
                "action": "arm_continuous_move",
                "direction": direction,
                "amount_meters": amount,
                "limb": limb,
            },
        )

    def arm_cross_wave(self, phase: str) -> ExecutionResult:
        label = "左臂上↑ 右臂下↓" if phase == "left_up" else "左臂下↓ 右臂上↑"
        return ExecutionResult(
            success=True,
            message=f"Dry-run cross wave: {label}",
            data={"mode": "dry_run", "action": "arm_cross_wave", "phase": phase},
        )

    def vision_navigate(self, goal_distance: float) -> ExecutionResult:
        return ExecutionResult(
            success=True,
            message=f"Dry-run vision navigate: {goal_distance}m",
            data={"mode": "dry_run", "action": "vision_navigate", "goal_distance_m": goal_distance},
        )

    def vision_turn(self, direction: str, angle_deg: float) -> ExecutionResult:
        return ExecutionResult(
            success=True,
            message=f"Dry-run vision turn: {direction} {angle_deg}°",
            data={"mode": "dry_run", "action": "vision_turn", "direction": direction, "angle_deg": angle_deg},
        )

    def fire_search(self, instruction: str) -> ExecutionResult:
        return ExecutionResult(
            success=True,
            message="Dry-run fire hydrant search accepted",
            data={"mode": "dry_run", "action": "fire_search", "instruction": instruction},
        )

    def robot_status(self) -> ExecutionResult:
        return ExecutionResult(
            success=True,
            message="Dry-run robot status",
            data={"mode": "dry_run", "action": "robot_status"},
        )


class AgentExecutor:
    def __init__(
        self,
        config: AgentConfig,
        waypoint_names: list[str] | None = None,
        backend: ControlBackend | None = None,
    ) -> None:
        self.config = config
        self.backend = backend or DryRunBackend(
            waypoint_names=waypoint_names or config.waypoint_names,
            allowed_arm_presets=config.allowed_arm_presets,
        )

    def execute(self, request: ExecutionRequest) -> ExecutionResult:
        started = time.perf_counter()
        result = self._execute_inner(request)
        result.latency_ms = round((time.perf_counter() - started) * 1000.0, 2)
        return result

    def _execute_inner(self, request: ExecutionRequest) -> ExecutionResult:
        if request.intent == "navigate_to_waypoint":
            return self.backend.navigate_to_waypoint(str(request.validated_slots["name"]))
        if request.intent == "save_waypoint":
            return self.backend.save_waypoint(str(request.validated_slots["name"]))
        if request.intent == "list_waypoints":
            return self.backend.list_waypoints()
        if request.intent == "stop_navigation":
            return self.backend.stop_navigation()
        if request.intent == "arm_preset":
            return self.backend.arm_preset(str(request.validated_slots["action_name"]))
        if request.intent == "arm_stop":
            return self.backend.arm_stop()
        if request.intent == "arm_continuous_move":
            return self.backend.arm_continuous_move(
                direction=str(request.validated_slots["direction"]),
                amount=float(request.validated_slots["amount_meters"]),
                limb=str(request.validated_slots.get("limb", "left")),
            )
        if request.intent == "arm_cross_wave":
            return self.backend.arm_cross_wave(
                phase=str(request.validated_slots.get("phase", "left_up")),
            )
        if request.intent == "vision_navigate":
            return self.backend.vision_navigate(
                goal_distance=float(request.validated_slots.get("goal_distance_m", 2.0)),
            )
        if request.intent == "vision_turn":
            return self.backend.vision_turn(
                direction=str(request.validated_slots.get("direction", "left")),
                angle_deg=float(request.validated_slots.get("angle_deg", 30)),
            )
        if request.intent == "fire_search":
            return self.backend.fire_search(
                instruction=str(request.validated_slots.get("instruction", "搜索并靠近消防栓")),
            )
        if request.intent == "robot_status":
            return self.backend.robot_status()
        return ExecutionResult(success=False, message=f"Unsupported intent: {request.intent}")
