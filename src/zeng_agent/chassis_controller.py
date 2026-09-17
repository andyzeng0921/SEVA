"""
chassis_controller.py — 底盘直接控制模块 (bridge 版本)

通过子进程 cmd_vel_bridge.py 与 ROS2 节点通信，
使用 GVController.publish_velocity_command() 控制底盘。
避免 CycloneDDS 多 participant 冲突。
"""
from __future__ import annotations

import json
import math
import subprocess
import time
from dataclasses import dataclass


@dataclass
class ChassisConfig:
    """底盘控制配置"""
    bridge_script: str = "/opt/seva/scripts/cmd_vel_bridge.py"
    python_bin: str = "/opt/robot/env/bin/python"

    # 运动参数
    default_linear_speed: float = 0.10   # m/s
    default_angular_speed: float = 0.30   # rad/s
    step_forward_distance: float = 0.20   # m
    turn_angle: float = math.radians(30)  # rad

    # 安全参数
    max_linear_speed: float = 0.30
    max_angular_speed: float = 0.80


class ChassisController:
    """底盘桥接控制器 — 通过 cmd_vel_bridge 子进程

    用法:
        ctrl = ChassisController(ChassisConfig())
        ctrl.move_forward(0.2)
        ctrl.turn_left(30)
        ctrl.stop()
        ctrl.shutdown()
    """

    def __init__(self, config: ChassisConfig) -> None:
        self.config = config
        self._proc: subprocess.Popen | None = None

    def _ensure_bridge(self) -> subprocess.Popen:
        if self._proc is not None and self._proc.poll() is None:
            return self._proc

        self._proc = subprocess.Popen(
            [self.config.python_bin, self.config.bridge_script],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL, text=True, bufsize=1,
        )
        # 循环读取直到 "cmd_vel_bridge ready"，跳过 ROS2 日志行
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            line = self._proc.stdout.readline()
            if not line:
                time.sleep(0.1)
                continue
            if "cmd_vel_bridge ready" in line:
                return self._proc
        raise RuntimeError("Bridge init timeout")

    def _send_cmd(self, cmd: dict) -> dict:
        """发送指令并等待响应"""
        proc = self._ensure_bridge()
        proc.stdin.write(json.dumps(cmd) + "\n")
        proc.stdin.flush()
        response = proc.stdout.readline()
        return json.loads(response)

    def move_forward(self, distance_m: float | None = None) -> dict:
        """小幅前进"""
        dist = distance_m if distance_m is not None else self.config.step_forward_distance
        return self._move_linear(dist)

    def move_backward(self, distance_m: float | None = None) -> dict:
        """小幅后退"""
        dist = distance_m if distance_m is not None else self.config.step_forward_distance
        return self._move_linear(-abs(dist))

    def _move_linear(self, distance_m: float) -> dict:
        speed = min(abs(self.config.default_linear_speed), self.config.max_linear_speed)
        if distance_m < 0:
            speed = -speed
        duration = abs(distance_m) / abs(speed) if speed != 0 else 0

        result = self._send_cmd({
            "action": "move",
            "linear_x": speed,
            "angular_z": 0.0,
            "duration_s": duration,
        })

        return {
            "success": result.get("success", False),
            "distance_target": distance_m,
            "distance_actual": round(speed * duration, 4),
            "duration_s": round(duration, 2),
            "linear_speed": round(speed, 3),
            "bridge_response": result,
        }

    def turn_left(self, angle_deg: float | None = None) -> dict:
        return self._turn(angle_deg, direction=1)

    def turn_right(self, angle_deg: float | None = None) -> dict:
        return self._turn(angle_deg, direction=-1)

    def _turn(self, angle_deg: float | None, direction: int) -> dict:
        if angle_deg is None:
            angle_deg = math.degrees(self.config.turn_angle)
        target_rad = math.radians(angle_deg)
        angular_speed = min(abs(self.config.default_angular_speed), self.config.max_angular_speed) * direction
        duration = abs(target_rad / abs(angular_speed)) if angular_speed != 0 else 0

        result = self._send_cmd({
            "action": "move",
            "linear_x": 0.0,
            "angular_z": angular_speed,
            "duration_s": duration,
        })

        return {
            "success": result.get("success", False),
            "angle_target_deg": angle_deg,
            "duration_s": round(duration, 2),
            "direction": "left" if direction > 0 else "right",
            "bridge_response": result,
        }

    def stop(self) -> dict:
        try:
            result = self._send_cmd({"action": "stop", "linear_x": 0.0, "angular_z": 0.0, "duration_s": 0.0})
            return {"success": result.get("success", False), "message": "stopped"}
        except Exception:
            return {"success": False, "message": "bridge unavailable"}

    def shutdown(self) -> None:
        self.stop()
        if self._proc:
            try:
                self._proc.stdin.close()
                self._proc.wait(timeout=3)
            except Exception:
                self._proc.kill()
            self._proc = None
