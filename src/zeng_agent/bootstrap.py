from __future__ import annotations

from pathlib import Path

from .asr import LocalMicASRProvider
from .backends import ArmCommandBackend, ArmContinuousBackend, CompositeControlBackend, Ros2ShellBackend, Runner
from .brain import BrainOrchestrator
from .config import AgentConfig
from .executor import AgentExecutor
from .llm_planner import LLMTaskPlanner
from .runtime_setup import build_arm_preset_commands
from .nlu import IntentParser
from .service import RobotAgentService
from .openclaw_controller import OpenClawRobotController


def build_service(config: AgentConfig, waypoint_names: list[str] | None = None, runner: Runner | None = None) -> RobotAgentService:
    local_asr_provider = None
    if config.local_asr.transcriber_command:
        local_asr_provider = LocalMicASRProvider(
            device=config.local_asr.device,
            sample_rate=config.local_asr.sample_rate,
            transcriber_command=config.local_asr.transcriber_command,
            channels=config.local_asr.channels,
            duration_seconds=config.local_asr.duration_seconds,
        )

    backend = None
    if not config.dry_run:
        if not config.arm_preset_commands:
            python_bin = "/opt/seva/scripts/run_ros_python.sh"
            helper_script = str(Path("/opt/seva/scripts/publish_arm_command.py"))
            live_presets = [name for name in config.allowed_arm_presets if name not in {"reset", "stop"}]
            config.arm_preset_commands = build_arm_preset_commands(
                python_bin=python_bin,
                helper_script=helper_script,
                topic_node_id=config.topic_node_id,
                presets=live_presets,
            )
        backend = CompositeControlBackend(
            ros2_backend=Ros2ShellBackend(config=config, runner=runner),
            arm_backend=ArmCommandBackend(config=config, runner=runner),
            continuous_backend=ArmContinuousBackend(config=config, runner=runner),
        )

    executor = AgentExecutor(config=config, waypoint_names=waypoint_names or config.waypoint_names, backend=backend)

    # LLM 规划器（DeepSeek）
    llm_planner = None
    if config.llm.enabled and config.llm.api_key:
        llm_planner = LLMTaskPlanner(
            config=config.llm,
            allowed_arm_presets=config.allowed_arm_presets,
            waypoint_names=waypoint_names or config.waypoint_names,
        )

    brain_orchestrator = BrainOrchestrator(
        config=config,
        parser=IntentParser(),
        executor=executor,
        llm_planner=llm_planner,
    )
    openclaw_controller = OpenClawRobotController(config) if config.openclaw_robot.enabled else None
    return RobotAgentService(
        config=config,
        waypoint_names=waypoint_names or config.waypoint_names,
        executor=executor,
        local_asr_provider=local_asr_provider,
        brain_orchestrator=brain_orchestrator,
        openclaw_controller=openclaw_controller,
    )
