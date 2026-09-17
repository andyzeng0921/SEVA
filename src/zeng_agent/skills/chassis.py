"""
chassis.py — 底盘控制 Skill

封装底盘移动能力: 前进、后退、左转、右转、停止、旋转扫描。
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

from .base import Skill, SkillInput, SkillOutput


@dataclass
class ChassisConfig:
    """底盘参数配置"""
    step_distance: float = 0.20     # 默认步进距离 (m)
    turn_angle: float = 90.0         # 默认转角 (°)
    move_speed: float = 0.15         # 线速度 (m/s)
    turn_speed: float = 0.5          # 角速度 (rad/s)
    move_timeout: float = 5.0        # 移动超时 (s)
    dry_run: bool = True             # True=不实际控制底盘


class ChassisSkill(Skill):
    """底盘控制技能

    输入参数:
      - action: "forward" | "backward" | "turn_left" | "turn_right" | "stop" | "rotate_to"
      - distance: 前进/后退距离 (m), 默认 0.2
      - angle: 转角 (°), 默认 90
      - target_angle: rotate_to 目标角度 (°), 绝对方向 0-360

    输出:
      - success: bool
      - data: {odom_info, action, ...}
    """

    name = "chassis"
    description = "底盘运动控制: 前进/后退/转向/停止/旋转到指定角度"
    version = "1.0.0"

    def __init__(self, config: Optional[ChassisConfig] = None):
        self.config = config or ChassisConfig()
        self._controller = None
        self._current_heading: float = 0.0  # 当前朝向 (°)

    def _ensure_controller(self):
        if self._controller is not None:
            return
        from ..chassis_controller import ChassisController, ChassisConfig as CC
        self._controller = ChassisController(CC(
            default_linear_speed=self.config.move_speed,
            default_angular_speed=self.config.turn_speed,
            step_forward_distance=self.config.step_distance,
            turn_angle=math.radians(self.config.turn_angle),
        ))

    def execute(self, input: SkillInput) -> SkillOutput:
        action = input.get("action", "forward")
        distance = float(input.get("distance", self.config.step_distance))
        angle = float(input.get("angle", self.config.turn_angle))

        if self.config.dry_run:
            return SkillOutput(
                success=True,
                data={"action": action, "distance": distance, "angle": angle, "mode": "dry_run"},
                message=f"DRY_RUN: {action} {distance}m / {angle}°",
            )

        self._ensure_controller()

        try:
            if action == "forward":
                result = self._controller.move_forward(distance)
                self._current_heading = self._current_heading  # 方向不变
            elif action == "backward":
                result = self._controller.move_backward(distance)
            elif action == "turn_left":
                result = self._controller.turn_left(angle)
                self._current_heading = (self._current_heading + angle) % 360
            elif action == "turn_right":
                result = self._controller.turn_right(angle)
                self._current_heading = (self._current_heading - angle) % 360
            elif action == "rotate_to":
                target = float(input.get("target_angle", 0))
                delta = (target - self._current_heading + 360) % 360
                if delta <= 180:
                    result = self._controller.turn_left(delta)
                else:
                    result = self._controller.turn_right(360 - delta)
                self._current_heading = target
            elif action == "stop":
                result = self._controller.stop()
            else:
                return SkillOutput(success=False, error=f"未知动作: {action}")

            return SkillOutput(
                success=result.get("success", False),
                data={"action": action, "heading": self._current_heading, "odom": result},
                message=f"{action} {'完成' if result.get('success') else '失败'}",
            )
        except Exception as e:
            return SkillOutput(success=False, error=str(e), message=f"底盘 {action} 异常")

    def shutdown(self):
        if self._controller:
            self._controller.shutdown()
            self._controller = None
