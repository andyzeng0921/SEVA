"""Whole-body power-on/reset skill."""
from __future__ import annotations

import subprocess
import time
import os
from dataclasses import dataclass

from .base import Skill, SkillInput, SkillOutput


@dataclass
class RobotResetConfig:
    command: str
    dry_run: bool = True
    command_timeout_seconds: float = 10.0
    settle_seconds: float = 9.0
    restart_vision_service: bool = True
    vision_service_name: str = "vision-service.service"
    camera_ready_timeout_seconds: float = 20.0


class RobotResetSkill(Skill):
    name = "robot_reset"
    description = "Enable all body joints, restore standing posture, and zero the neck"
    version = "1.0.0"

    def __init__(self, config: RobotResetConfig):
        self.config = config

    @staticmethod
    def _user_systemd_env() -> dict[str, str]:
        uid = os.getuid()
        env = dict(os.environ)
        env.setdefault("XDG_RUNTIME_DIR", f"/run/user/{uid}")
        env.setdefault("DBUS_SESSION_BUS_ADDRESS", f"unix:path=/run/user/{uid}/bus")
        return env

    def _vision_service(self, action: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["systemctl", "--user", action, self.config.vision_service_name],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
            env=self._user_systemd_env(),
        )

    def _wait_for_front_camera(self) -> bool:
        from ..vision_provider import VisionConfig, VisionProvider

        deadline = time.monotonic() + self.config.camera_ready_timeout_seconds
        while time.monotonic() < deadline:
            provider = VisionProvider(VisionConfig(
                camera_source="rgbd_color",
                fresh_frame_timeout=1.0,
            ))
            _, metadata, _ = provider.capture_live_frame()
            if metadata is not None:
                return True
            time.sleep(0.5)
        return False

    def execute(self, input: SkillInput) -> SkillOutput:
        if self.config.dry_run:
            return SkillOutput(success=True, data={"mode": "dry_run"}, message="DRY_RUN: robot reset")
        if not self.config.command:
            return SkillOutput(success=False, error="未配置机器人复位命令")
        vision_stopped = False
        try:
            if self.config.restart_vision_service:
                stopped = self._vision_service("stop")
                if stopped.returncode != 0:
                    return SkillOutput(success=False, error=stopped.stderr.strip() or "无法暂停视觉服务")
                vision_stopped = True
                time.sleep(0.8)
            completed = subprocess.run(
                self.config.command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=self.config.command_timeout_seconds,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            return SkillOutput(success=False, error=f"机器人复位命令失败: {exc}")
        finally:
            if vision_stopped:
                self._vision_service("start")
        if completed.returncode != 0:
            return SkillOutput(
                success=False,
                error=completed.stderr.strip() or completed.stdout.strip() or "机器人复位发布失败",
            )
        time.sleep(max(0.0, self.config.settle_seconds))
        if self.config.restart_vision_service and not self._wait_for_front_camera():
            return SkillOutput(success=False, error="机器人复位完成，但前向RGBD相机未在超时内恢复")
        return SkillOutput(
            success=True,
            data={"settle_seconds": self.config.settle_seconds, "publisher_output": completed.stdout.strip()},
            message="机器人已上电复位，颈部已等待归零",
        )
