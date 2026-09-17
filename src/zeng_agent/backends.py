from __future__ import annotations

import json
import os
import re
import shlex
import socket
import subprocess
from typing import Callable

from .config import AgentConfig
from .models import ExecutionResult

Runner = Callable[[str], tuple[int, str, str]]


def default_runner(command: str) -> tuple[int, str, str]:
    result = subprocess.run(command, shell=True, capture_output=True, text=True, check=False)
    return result.returncode, result.stdout, result.stderr


class Ros2ShellBackend:
    def __init__(self, config: AgentConfig, runner: Runner | None = None) -> None:
        self.config = config
        self.runner = runner or default_runner

    def _ros_shell(self, inner: str) -> str:
        script = (
            f"source {shlex.quote(self.config.ros2.ros_setup)} >/dev/null 2>&1 "
            f"&& source {shlex.quote(self.config.ros2.workspace_setup)} >/dev/null 2>&1 && {inner}"
        )
        return f"timeout 30s bash -c {shlex.quote(script)}"

    def _run(self, inner: str, action: str) -> ExecutionResult:
        code, stdout, stderr = self.runner(self._ros_shell(inner))
        if code != 0:
            return ExecutionResult(success=False, message=stderr.strip() or f"{action} failed")
        return ExecutionResult(success=True, message="ok", data={"action": action, "stdout": stdout.strip()})

    def navigate_to_waypoint(self, name: str) -> ExecutionResult:
        return self._run(
            f"ros2 service call {self.config.ros2.navigate_service} multi_lidar_nav/srv/NavigateToWaypoint "
            f"'{{name: {name}}}'",
            "navigate_to_waypoint",
        )

    def save_waypoint(self, name: str) -> ExecutionResult:
        return self._run(
            f"ros2 service call {self.config.ros2.save_waypoint_service} multi_lidar_nav/srv/SaveWaypoint "
            f"'{{name: {name}}}'",
            "save_waypoint",
        )

    def list_waypoints(self) -> ExecutionResult:
        result = self._run(
            f"ros2 service call {self.config.ros2.list_waypoints_service} multi_lidar_nav/srv/ListWaypoints '{{}}'",
            "list_waypoints",
        )
        if not result.success:
            return result
        stdout = result.data.get("stdout", "")
        match = re.search(r"names=(\[[^\]]*\])", stdout)
        if match:
            import ast
            names = ast.literal_eval(match.group(1))
        else:
            names = re.findall(r"^-\s*([\w\-]+)", stdout, re.MULTILINE)
        result.data["names"] = names
        return result

    def stop_navigation(self) -> ExecutionResult:
        return self._run(
            "ros2 topic pub --once "
            f"{self.config.ros2.stop_topic} geometry_msgs/msg/Twist "
            "'{linear: {x: 0.0, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 0.0}}'",
            "stop_navigation",
        )

    def arm_preset(self, action_name: str) -> ExecutionResult:
        return ExecutionResult(success=False, message=f"Arm preset {action_name} should be handled by ArmCommandBackend")

    def arm_stop(self) -> ExecutionResult:
        return ExecutionResult(success=False, message="Arm stop should be handled by ArmCommandBackend")

    def arm_continuous_move(self, direction: str, amount: float, limb: str) -> ExecutionResult:
        return ExecutionResult(success=False, message="Arm continuous move should be handled by ArmContinuousBackend")

    def arm_cross_wave(self, phase: str) -> ExecutionResult:
        return ExecutionResult(success=False, message="Arm cross wave should be handled by ArmContinuousBackend")

    def robot_status(self) -> ExecutionResult:
        return self._run("ros2 node list --no-daemon && ros2 topic list --no-daemon", "robot_status")


class ArmCommandBackend:
    def __init__(self, config: AgentConfig, runner: Runner | None = None) -> None:
        self.config = config
        self.runner = runner or default_runner

    def arm_preset(self, action_name: str) -> ExecutionResult:
        command = self.config.arm_preset_commands.get(action_name)
        if not command:
            return ExecutionResult(success=False, message=f"No command configured for arm preset: {action_name}")
        code, stdout, stderr = self.runner(command)
        if code != 0:
            return ExecutionResult(success=False, message=stderr.strip() or f"Arm preset failed: {action_name}")
        return ExecutionResult(success=True, message="ok", data={"action": "arm_preset", "preset": action_name, "stdout": stdout.strip()})

    def arm_stop(self) -> ExecutionResult:
        command = self.config.arm_preset_commands.get("stop")
        if not command:
            return ExecutionResult(success=False, message="No command configured for arm stop")
        code, stdout, stderr = self.runner(command)
        if code != 0:
            return ExecutionResult(success=False, message=stderr.strip() or "Arm stop failed")
        return ExecutionResult(success=True, message="ok", data={"action": "arm_stop", "stdout": stdout.strip()})


class ArmContinuousBackend:
    """连续运动控制后端 — 发布连续位姿指令到 ROS2

    调用 publish_arm_continuous.py 脚本实现:
      1. 读取当前 EEF 位姿 (ROS2 topic)
      2. 计算目标位姿 (当前 + 方向 × 距离)
      3. 发布到 move_eef_pose_in_robot_frame 或 move_up_down_z
    """

    def __init__(self, config: AgentConfig, runner: Runner | None = None) -> None:
        self.config = config
        self.runner = runner or default_runner

    def _build_command(self, direction: str, amount: float, limb: str) -> str:
        script_path = os.path.join(os.path.dirname(__file__), "..", "..", "scripts", "publish_arm_continuous.py")
        script_path = os.path.abspath(script_path)
        inner = (
            f"cd /opt/seva && "
            f"export LD_LIBRARY_PATH=/opt/robot/env/lib:$LD_LIBRARY_PATH && "
            f". {shlex.quote(self.config.ros2.ros_setup)} && "
            f"/opt/robot/env/bin/python {script_path} "
            f"--direction {direction} --amount {amount} --limb {limb} "
            f"--topic-node-id {shlex.quote(self.config.topic_node_id)}"
        )
        return f"bash -c {shlex.quote(inner)}"

    def _build_cross_wave_command(self, phase: str) -> str:
        script_path = os.path.join(os.path.dirname(__file__), "..", "..", "scripts", "publish_arm_continuous.py")
        script_path = os.path.abspath(script_path)
        inner = (
            f"cd /opt/seva && "
            f"export LD_LIBRARY_PATH=/opt/robot/env/lib:$LD_LIBRARY_PATH && "
            f". {shlex.quote(self.config.ros2.ros_setup)} && "
            f"/opt/robot/env/bin/python {script_path} "
            f"--mode cross_wave --phase {phase} "
            f"--topic-node-id {shlex.quote(self.config.topic_node_id)}"
        )
        return f"bash -c {shlex.quote(inner)}"

    def arm_continuous_move(self, direction: str, amount: float, limb: str) -> ExecutionResult:
        command = self._build_command(direction, amount, limb)
        code, stdout, stderr = self.runner(command)
        if code != 0:
            return ExecutionResult(success=False, message=stderr.strip() or f"Continuous move failed: {direction} {amount}m")
        try:
            result = json.loads(stdout.strip())
            if result.get("success"):
                return ExecutionResult(success=True, message=result["message"], data=result.get("data", {}))
            return ExecutionResult(success=False, message=result.get("message", "Unknown error"))
        except json.JSONDecodeError:
            return ExecutionResult(success=True, message=f"Continuous move sent: {limb} {direction} {amount}m", data={"raw": stdout.strip()})

    def arm_cross_wave(self, phase: str) -> ExecutionResult:
        # "full_cycle" — 执行两个相位
        phases_to_run = ["left_up", "right_up"] if phase == "full_cycle" else [phase]

        results = []
        for p in phases_to_run:
            command = self._build_cross_wave_command(p)
            code, stdout, stderr = self.runner(command)
            if code != 0:
                return ExecutionResult(success=False, message=stderr.strip() or f"Cross wave {p} failed")
            try:
                r = json.loads(stdout.strip())
                results.append(r)
            except json.JSONDecodeError:
                results.append({"raw": stdout.strip()})
            # 等待运动完成
            if len(phases_to_run) > 1 and p != phases_to_run[-1]:
                import time
                time.sleep(5.0)

        last = results[-1]
        if isinstance(last, dict) and last.get("success"):
            label = "完整交叉挥手" if phase == "full_cycle" else last["message"]
            return ExecutionResult(success=True, message=label, data={"phases": results})
        return ExecutionResult(success=False, message=str(last))

    def arm_preset(self, action_name: str) -> ExecutionResult:
        return ExecutionResult(success=False, message=f"Arm preset {action_name} should be handled by ArmCommandBackend")

    def arm_stop(self) -> ExecutionResult:
        return ExecutionResult(success=False, message="Arm stop should be handled by ArmCommandBackend")


class VisionNavigateBackend:
    """视觉导航后端 — 融合视觉+底盘的多步导航"""
    
    def __init__(self, config: AgentConfig) -> None:
        self.config = config
        self._navigator = None

    def _ensure_navigator(self):
        if self._navigator is not None:
            return self._navigator
        from .vision_navigator import VisionNavigator, VisionNavConfig
        self._navigator = VisionNavigator(VisionNavConfig(
            api_key=self.config.fire_search.vision_api_key,
            api_url=self.config.fire_search.vision_api_url,
            vision_model=self.config.fire_search.vision_model,
            dry_run=self.config.dry_run,
            step_forward_distance=0.20,
            goal_distance=2.0,
            require_confirmation=True,
        ))
        return self._navigator

    def vision_navigate(self, goal_distance: float = 2.0) -> ExecutionResult:
        try:
            nav = self._ensure_navigator()
            nav.config.goal_distance = goal_distance
            # 重置状态
            from .vision_navigator import NavState
            nav.state = NavState()

            if self.config.dry_run:
                result = nav.preview_step()
                return ExecutionResult(
                    success=True,
                    message=f"Dry-run vision nav: {result['decision']}",
                    data=result,
                )

            final_state = nav.run_steps()
            nav.shutdown()

            return ExecutionResult(
                success=final_state.success,
                message=final_state.message or f"导航完成: {final_state.total_distance_m:.2f}m, {final_state.step}步",
                data={
                    "total_distance_m": round(final_state.total_distance_m, 3),
                    "steps": final_state.step,
                    "last_decision": final_state.last_decision.value,
                    "history": [h for h in final_state.decision_history[-5:]],
                },
            )
        except Exception as e:
            return ExecutionResult(success=False, message=f"视觉导航失败: {e}")


class TurnBackend:
    """视觉转身后端 — 先检测空间，再执行转身"""

    def __init__(self, config: AgentConfig) -> None:
        self.config = config
        self._skill = None

    def _ensure_skill(self):
        if self._skill is not None:
            return self._skill
        from .turn_skill import TurnConfig, TurnSkill
        self._skill = TurnSkill(TurnConfig(
            api_key=self.config.llm.api_key,
            dry_run=self.config.dry_run,
        ))
        return self._skill

    def vision_turn(self, direction: str, angle_deg: float = 30.0) -> ExecutionResult:
        try:
            skill = self._ensure_skill()
            result = skill.safe_turn(direction, angle_deg)
            return ExecutionResult(
                success=result.success,
                message=f"转身 {'成功' if result.success else '失败'}: {direction} {result.angle_actual_deg:.0f}°",
                data=result.to_dict(),
            )
        except Exception as e:
            return ExecutionResult(success=False, message=f"转身失败: {e}")


class FireSearchBackend:
    """Build and run the fail-closed fire-hydrant LangGraph workflow."""

    def __init__(self, config: AgentConfig) -> None:
        self.config = config

    def fire_search(self, instruction: str) -> ExecutionResult:
        cfg = self.config.fire_search
        if not cfg.enabled:
            return ExecutionResult(success=False, message="消防栓搜索功能未启用")
        if not cfg.vision_api_key:
            return ExecutionResult(success=False, message="消防栓搜索缺少 vision_api_key")
        try:
            from .fire_search import FireHydrantSearchAgent, FireSearchConfig
            from .skills.base import SkillRegistry
            from .skills.chassis import ChassisConfig, ChassisSkill
            from .skills.detect import DetectConfig, VisualDetectSkill
            from .skills.reset import RobotResetConfig, RobotResetSkill

            registry = SkillRegistry()
            registry.register_many([
                RobotResetSkill(RobotResetConfig(
                    command=self.config.arm_preset_commands.get("reset", ""),
                    dry_run=self.config.dry_run,
                    settle_seconds=cfg.reset_settle_seconds,
                )),
                VisualDetectSkill(DetectConfig(
                    api_key=cfg.vision_api_key,
                    api_url=cfg.vision_api_url,
                    model=cfg.vision_model,
                    camera_source=cfg.camera_source,
                    timeout=cfg.vision_timeout_seconds,
                    fresh_frame_timeout=cfg.fresh_frame_timeout_seconds,
                )),
                ChassisSkill(ChassisConfig(dry_run=self.config.dry_run)),
            ])
            agent = FireHydrantSearchAgent(
                registry,
                dry_run=self.config.dry_run,
                config=FireSearchConfig(
                    camera_source=cfg.camera_source,
                    max_scan_steps=cfg.max_scan_steps,
                    max_align_attempts=cfg.max_align_attempts,
                    max_approach_steps=cfg.max_approach_steps,
                    max_forward_step_m=cfg.max_forward_step_m,
                    max_total_forward_m=cfg.max_total_forward_m,
                ),
            )
            result = agent.search(instruction)
            return ExecutionResult(
                success=result.get("status") == "done",
                message=result.get("message", "消防栓搜索结束"),
                data=result,
            )
        except Exception as exc:
            return ExecutionResult(success=False, message=f"消防栓搜索失败: {exc}")


class CompositeControlBackend:
    def __init__(self, ros2_backend: Ros2ShellBackend, arm_backend: ArmCommandBackend,
                 continuous_backend: ArmContinuousBackend | None = None,
                 vision_nav_backend: VisionNavigateBackend | None = None,
                 turn_backend: TurnBackend | None = None,
                 fire_search_backend: FireSearchBackend | None = None) -> None:
        self.ros2_backend = ros2_backend
        self.arm_backend = arm_backend
        self.continuous_backend = continuous_backend or ArmContinuousBackend(config=arm_backend.config)
        self.vision_nav_backend = vision_nav_backend or VisionNavigateBackend(config=arm_backend.config)
        self.turn_backend = turn_backend or TurnBackend(config=arm_backend.config)
        self.fire_search_backend = fire_search_backend or FireSearchBackend(config=arm_backend.config)

    def navigate_to_waypoint(self, name: str) -> ExecutionResult:
        return self.ros2_backend.navigate_to_waypoint(name)

    def save_waypoint(self, name: str) -> ExecutionResult:
        return self.ros2_backend.save_waypoint(name)

    def list_waypoints(self) -> ExecutionResult:
        return self.ros2_backend.list_waypoints()

    def stop_navigation(self) -> ExecutionResult:
        return self.ros2_backend.stop_navigation()

    def arm_preset(self, action_name: str) -> ExecutionResult:
        return self.arm_backend.arm_preset(action_name)

    def arm_stop(self) -> ExecutionResult:
        return self.arm_backend.arm_stop()

    def arm_continuous_move(self, direction: str, amount: float, limb: str) -> ExecutionResult:
        return self.continuous_backend.arm_continuous_move(direction, amount, limb)

    def arm_cross_wave(self, phase: str) -> ExecutionResult:
        return self.continuous_backend.arm_cross_wave(phase)

    def vision_navigate(self, goal_distance: float) -> ExecutionResult:
        return self.vision_nav_backend.vision_navigate(goal_distance)

    def vision_turn(self, direction: str, angle_deg: float) -> ExecutionResult:
        return self.turn_backend.vision_turn(direction, angle_deg)

    def fire_search(self, instruction: str) -> ExecutionResult:
        return self.fire_search_backend.fire_search(instruction)

    def robot_status(self) -> ExecutionResult:
        return self.ros2_backend.robot_status()


def tcp_probe(host: str, port: int, timeout: float = 1.0) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False
