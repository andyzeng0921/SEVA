from __future__ import annotations

from dataclasses import asdict
from typing import Any

from .asr import LocalMicASRProvider
from .config import AgentConfig
from .executor import AgentExecutor
from .models import CommandText, ExecutionRequest, ParsedIntent
from .nlu import IntentParser
from .brain import BrainOrchestrator
from .openclaw_controller import OpenClawRobotController


class RobotAgentService:
    def __init__(
        self,
        config: AgentConfig,
        waypoint_names: list[str] | None = None,
        executor: AgentExecutor | None = None,
        parser: IntentParser | None = None,
        local_asr_provider: LocalMicASRProvider | None = None,
        brain_orchestrator: BrainOrchestrator | None = None,
        openclaw_controller: OpenClawRobotController | None = None,
    ) -> None:
        self.config = config
        self.parser = parser or IntentParser()
        self.executor = executor or AgentExecutor(config=config, waypoint_names=waypoint_names or config.waypoint_names)
        self._waypoint_names = waypoint_names or config.waypoint_names
        self.local_asr_provider = local_asr_provider
        self.brain_orchestrator = brain_orchestrator
        self.openclaw_controller = openclaw_controller

    def status(self) -> dict[str, Any]:
        return {
            "dry_run": self.config.dry_run,
            "components": {
                "nlu": "ok",
                "executor": "ok",
                "relay": "configured" if self.config.relay.ws_url else "disabled",
                "asr_local": "configured" if self.config.local_asr.device else "disabled",
                "asr_remote": "configured" if self.config.remote_asr.base_url else "disabled",
                "brain": "configured" if self.brain_orchestrator is not None else "disabled",
                "openclaw_robot": "configured" if self.openclaw_controller is not None else "disabled",
            },
            "waypoint_names": list(self._waypoint_names),
        }

    def openclaw_start(self, goal: str, reset: bool = True) -> dict[str, Any]:
        return self.openclaw_controller.start(goal, reset) if self.openclaw_controller else {"success": False, "message": "OpenClaw robot controller is disabled"}

    def openclaw_observe(self, session_id: str) -> dict[str, Any]:
        return self.openclaw_controller.observe(session_id) if self.openclaw_controller else {"success": False, "message": "OpenClaw robot controller is disabled"}

    def openclaw_turn(self, session_id: str, angle_deg: float) -> dict[str, Any]:
        return self.openclaw_controller.turn(session_id, angle_deg) if self.openclaw_controller else {"success": False, "message": "OpenClaw robot controller is disabled"}

    def openclaw_approach(self, session_id: str, distance_m: float) -> dict[str, Any]:
        return self.openclaw_controller.approach(session_id, distance_m) if self.openclaw_controller else {"success": False, "message": "OpenClaw robot controller is disabled"}

    def openclaw_stop(self, session_id: str | None, reason: str) -> dict[str, Any]:
        return self.openclaw_controller.stop(session_id, reason) if self.openclaw_controller else {"success": False, "message": "OpenClaw robot controller is disabled"}

    def openclaw_status(self) -> dict[str, Any]:
        return self.openclaw_controller.status() if self.openclaw_controller else {"enabled": False}

    def brain_status(self) -> dict[str, Any]:
        if self.brain_orchestrator is None:
            return {"enabled": False, "message": "Brain orchestrator is not configured"}
        return self.brain_orchestrator.status()

    def handle_brain_text_task(self, text: str, confirmed: bool = False) -> dict[str, Any]:
        if self.brain_orchestrator is None:
            return {
                "status": "blocked",
                "message": "Brain orchestrator is not configured",
                "steps": [],
            }
        return self.brain_orchestrator.run_text_task(text=text, confirmed=confirmed)

    def get_brain_task(self, task_id: str) -> dict[str, Any]:
        if self.brain_orchestrator is None:
            return {"found": False, "message": "Brain orchestrator is not configured"}
        task = self.brain_orchestrator.get_task(task_id)
        if task is None:
            return {"found": False, "message": f"Unknown task: {task_id}"}
        task["found"] = True
        return task

    def handle_text_command(self, text: str, source: str = "text", confidence: float = 1.0) -> dict[str, Any]:
        command = CommandText(text=text, source=source, confidence=confidence)
        parsed = self.parser.parse(command.text)
        validated = self._validate(parsed)
        if validated.get("blocked"):
            result = {
                "success": False,
                "message": validated["message"],
                "data": {"mode": "guardrail"},
                "latency_ms": 0.0,
            }
        else:
            execution = self.executor.execute(
                ExecutionRequest(intent=parsed.intent, validated_slots=validated["slots"], mode="dry_run" if self.config.dry_run else "live")
            )
            result = asdict(execution)
        return {
            "command": asdict(command),
            "parsed": asdict(parsed),
            "result": result,
        }

    def handle_local_asr_command(self) -> dict[str, Any]:
        if self.local_asr_provider is None:
            return {
                "result": {
                    "success": False,
                    "message": "Local ASR provider is not configured",
                    "data": {"mode": "guardrail"},
                    "latency_ms": 0.0,
                }
            }

        asr_result = self.local_asr_provider.capture_and_transcribe()
        if not asr_result.success or asr_result.command is None:
            return {
                "asr": {
                    "success": asr_result.success,
                    "message": asr_result.message,
                },
                "result": {
                    "success": False,
                    "message": asr_result.message,
                    "data": {"mode": "guardrail"},
                    "latency_ms": 0.0,
                },
            }

        handled = self.handle_text_command(
            asr_result.command.text,
            source=asr_result.command.source,
            confidence=asr_result.command.confidence,
        )
        handled["asr"] = {
            "success": asr_result.success,
            "message": asr_result.message,
        }
        return handled

    def _validate(self, parsed: ParsedIntent) -> dict[str, Any]:
        if parsed.intent == "unsupported":
            return {"blocked": True, "message": "Unsupported command in v1", "slots": parsed.slots}

        if parsed.intent == "arm_preset":
            action_name = str(parsed.slots["action_name"])
            if action_name not in self.config.allowed_arm_presets:
                return {"blocked": True, "message": f"Arm preset not allowed: {action_name}", "slots": parsed.slots}

        if parsed.intent == "navigate_to_waypoint":
            name = str(parsed.slots["name"])
            if self._waypoint_names and name not in self._waypoint_names:
                return {"blocked": True, "message": f"Unknown waypoint: {name}", "slots": parsed.slots}

        return {"blocked": False, "slots": parsed.slots}
