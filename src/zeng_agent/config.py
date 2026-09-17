from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import os
import yaml


@dataclass(slots=True)
class LocalAsrConfig:
    device: str = "default"
    sample_rate: int = 16000
    channels: int = 1
    duration_seconds: int = 4
    python_bin: str = "python3"
    transcriber_script: str = "./scripts/run_whisper_cpp.py"
    whisper_binary: str = "./vendor/whisper.cpp/build/bin/whisper-cli"
    whisper_model: str = "./models/ggml-base.bin"
    transcriber_command: str | None = None
    temp_dir: str = "/tmp/zeng-agent"

    def resolve_paths(self, project_root: Path) -> None:
        """将相对路径解析为基于项目根目录的绝对路径"""
        if self.transcriber_command and "{TRANSCRIBER_SCRIPT}" in self.transcriber_command:
            self.transcriber_command = self.transcriber_command.replace(
                "{PYTHON_BIN}", self.python_bin
            ).replace(
                "{TRANSCRIBER_SCRIPT}", str((project_root / self.transcriber_script).resolve())
            ).replace(
                "{WHISPER_BINARY}", str((project_root / self.whisper_binary).resolve())
            ).replace(
                "{WHISPER_MODEL}", str((project_root / self.whisper_model).resolve())
            )


@dataclass(slots=True)
class RemoteAsrConfig:
    enabled: bool = False
    base_url: str = ""
    timeout_seconds: int = 20


@dataclass(slots=True)
class RelayConfig:
    ws_url: str = "ws://127.0.0.1:3000/ws"
    tcp_host: str = "127.0.0.1"
    tcp_port: int = 3000


@dataclass(slots=True)
class Ros2Config:
    ros_setup: str = "/opt/ros/jazzy/setup.bash"
    workspace_setup: str = "./ros2_ws/install/setup.bash"
    navigate_service: str = "/navigate_to_waypoint"
    save_waypoint_service: str = "/save_waypoint"
    list_waypoints_service: str = "/list_waypoints"
    delete_waypoint_service: str = "/delete_waypoint"
    stop_topic: str = "/topic_gv_target_cmd_vel_0_283"

    def resolve_paths(self, project_root: Path) -> None:
        """将相对路径解析为基于项目根目录的绝对路径"""
        if self.workspace_setup.startswith("./"):
            resolved = (project_root / self.workspace_setup).resolve()
            self.workspace_setup = str(resolved)


@dataclass(slots=True)
class LlmConfig:
    enabled: bool = False
    api_key: str = ""
    base_url: str = "https://api.deepseek.com"
    model: str = "deepseek-v4-flash"
    timeout_seconds: int = 30
    enable_thinking: bool | None = None


@dataclass(slots=True)
class BrainConfig:
    enabled: bool = True
    framework: str = "langgraph_ready"
    state_backend: str = "memory"
    robot_profile: str = "zhidongweilai_s2"
    require_confirmation_for_motion: bool = True
    max_steps: int = 8


@dataclass(slots=True)
class FireSearchRuntimeConfig:
    enabled: bool = True
    camera_source: str = "auto_front"
    vision_api_url: str = "http://127.0.0.1:8000/v1/chat/completions"
    vision_api_key: str = ""
    vision_model: str = "qwen3.5-35b-a3b"
    vision_timeout_seconds: int = 30
    fresh_frame_timeout_seconds: float = 2.0
    reset_settle_seconds: float = 9.0
    max_scan_steps: int = 8
    max_align_attempts: int = 4
    max_approach_steps: int = 6
    max_forward_step_m: float = 0.5
    max_total_forward_m: float = 3.0


@dataclass(slots=True)
class OpenClawRobotConfig:
    enabled: bool = True
    max_actions: int = 36
    max_motion_actions: int = 24
    max_elapsed_seconds: float = 600.0
    max_turn_angle: float = 120.0
    max_total_rotation_deg: float = 1080.0
    max_forward_step_m: float = 0.5
    max_total_forward_m: float = 3.0
    alignment_tolerance_deg: float = 3.0


@dataclass(slots=True)
class AgentConfig:
    dry_run: bool = True
    host: str = "127.0.0.1"
    port: int = 8765
    log_level: str = "info"
    topic_node_id: str = "0_283"
    waypoint_names: list[str] = field(default_factory=list)
    allowed_arm_presets: list[str] = field(
        default_factory=lambda: [
            "reset",
            "stop",
            "zero_position",
            "wave",
            "package_pos",
            "standup_pos",
            "left_open_gripper",
            "left_close_gripper",
            "right_open_gripper",
            "right_close_gripper",
        ]
    )
    arm_preset_commands: dict[str, str] = field(default_factory=dict)
    local_asr: LocalAsrConfig = field(default_factory=LocalAsrConfig)
    remote_asr: RemoteAsrConfig = field(default_factory=RemoteAsrConfig)
    relay: RelayConfig = field(default_factory=RelayConfig)
    ros2: Ros2Config = field(default_factory=Ros2Config)
    llm: LlmConfig = field(default_factory=LlmConfig)
    brain: BrainConfig = field(default_factory=BrainConfig)
    fire_search: FireSearchRuntimeConfig = field(default_factory=FireSearchRuntimeConfig)
    openclaw_robot: OpenClawRobotConfig = field(default_factory=OpenClawRobotConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> AgentConfig:
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        return cls.from_dict(raw, project_root=Path(path).resolve().parent)

    @classmethod
    def from_dict(cls, data: dict[str, Any], project_root: Path | None = None) -> AgentConfig:
        local_asr = LocalAsrConfig(**data.get("local_asr", {}))
        remote_asr = RemoteAsrConfig(**data.get("remote_asr", {}))
        relay = RelayConfig(**data.get("relay", {}))
        ros2 = Ros2Config(**data.get("ros2", {}))
        llm = LlmConfig(**data.get("llm", {}))
        brain = BrainConfig(**data.get("brain", {}))
        fire_search = FireSearchRuntimeConfig(**data.get("fire_search", {}))
        openclaw_robot = OpenClawRobotConfig(**data.get("openclaw_robot", {}))
        payload = {
            k: v
            for k, v in data.items()
            if k not in {"local_asr", "remote_asr", "relay", "ros2", "llm", "brain", "fire_search", "openclaw_robot"}
        }
        cfg = cls(
            **payload,
            local_asr=local_asr,
            remote_asr=remote_asr,
            relay=relay,
            ros2=ros2,
            llm=llm,
            brain=brain,
            fire_search=fire_search,
            openclaw_robot=openclaw_robot,
        )
        # Deployment credentials stay outside tracked configuration.
        cfg.llm.api_key = os.environ.get("SEVA_LLM_API_KEY", cfg.llm.api_key)
        cfg.llm.base_url = os.environ.get("SEVA_LLM_BASE_URL", cfg.llm.base_url)
        cfg.fire_search.vision_api_key = os.environ.get("SEVA_VLM_API_KEY", cfg.fire_search.vision_api_key)
        cfg.fire_search.vision_api_url = os.environ.get("SEVA_VLM_URL", cfg.fire_search.vision_api_url)
        # 解析相对路径（基于项目根目录）
        if project_root is not None:
            cfg.local_asr.resolve_paths(project_root)
            cfg.ros2.resolve_paths(project_root)
        return cfg
