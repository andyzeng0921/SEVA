"""Capability-based robot controller for the OpenClaw agent."""
from __future__ import annotations

import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any

from .config import AgentConfig
from .skills.base import SkillInput, SkillOutput, SkillRegistry
from .skills.chassis import ChassisConfig, ChassisSkill
from .skills.detect import DetectConfig, VisualDetectSkill
from .skills.reset import RobotResetConfig, RobotResetSkill


@dataclass
class ExplorationSession:
    session_id: str
    goal: str
    started_monotonic: float
    status: str = "starting"
    actions: int = 0
    motion_actions: int = 0
    total_forward_m: float = 0.0
    total_rotation_deg: float = 0.0
    last_frame_id: str | None = None
    latest_observation: dict[str, Any] | None = None
    observation_valid_for_motion: bool = False
    history: list[dict[str, Any]] = field(default_factory=list)


class OpenClawRobotController:
    """Stateful safety kernel. The reasoning agent never receives raw ROS access."""

    def __init__(self, config: AgentConfig, registry: SkillRegistry | None = None) -> None:
        self.config = config
        self.limits = config.openclaw_robot
        self.registry = registry or self._build_registry(config)
        self._session: ExplorationSession | None = None
        self._lock = threading.RLock()

    @staticmethod
    def _build_registry(config: AgentConfig) -> SkillRegistry:
        cfg = config.fire_search
        registry = SkillRegistry()
        registry.register_many([
            RobotResetSkill(RobotResetConfig(
                command=config.arm_preset_commands.get("reset", ""),
                dry_run=config.dry_run,
                settle_seconds=cfg.reset_settle_seconds,
            )),
            VisualDetectSkill(DetectConfig(
                api_key=cfg.vision_api_key,
                api_url=cfg.vision_api_url,
                model=cfg.vision_model,
                camera_source=cfg.camera_source,
                timeout=cfg.vision_timeout_seconds,
                fresh_frame_timeout=cfg.fresh_frame_timeout_seconds,
                alignment_tolerance_deg=config.openclaw_robot.alignment_tolerance_deg,
            )),
            ChassisSkill(ChassisConfig(dry_run=config.dry_run)),
        ])
        return registry

    def _snapshot(self, session: ExplorationSession) -> dict[str, Any]:
        data = asdict(session)
        data.pop("started_monotonic", None)
        data["elapsed_seconds"] = round(time.monotonic() - session.started_monotonic, 1)
        data["limits"] = asdict(self.limits)
        return data

    def _fail(self, message: str, session: ExplorationSession | None = None) -> dict[str, Any]:
        return {"success": False, "message": message, "session": self._snapshot(session) if session else None}

    def _active(self, session_id: str) -> tuple[ExplorationSession | None, dict[str, Any] | None]:
        session = self._session
        if session is None or session.session_id != session_id:
            return None, self._fail("No matching exploration session")
        if session.status != "running":
            return None, self._fail(f"Session is not running: {session.status}", session)
        if time.monotonic() - session.started_monotonic > self.limits.max_elapsed_seconds:
            self._emergency_stop(session, "time budget exhausted")
            return None, self._fail("Exploration time budget exhausted", session)
        if session.actions >= self.limits.max_actions:
            self._emergency_stop(session, "action budget exhausted")
            return None, self._fail("Exploration action budget exhausted", session)
        return session, None

    def _record(self, session: ExplorationSession, action: str, data: dict[str, Any]) -> None:
        session.actions += 1
        session.history.append({"action": action, **data})
        session.history = session.history[-60:]

    def _emergency_stop(self, session: ExplorationSession, reason: str) -> SkillOutput:
        result = self.registry.execute("chassis", SkillInput({"action": "stop"}))
        session.status = "stopped"
        session.observation_valid_for_motion = False
        session.history.append({"action": "stop", "reason": reason, "success": result.success})
        return result

    def start(self, goal: str, reset: bool = True) -> dict[str, Any]:
        with self._lock:
            if not self.limits.enabled:
                return self._fail("OpenClaw robot control is disabled")
            if self._session and self._session.status == "running":
                self._emergency_stop(self._session, "superseded by a new session")
            session = ExplorationSession(
                session_id=uuid.uuid4().hex,
                goal=goal.strip() or "Explore safely",
                started_monotonic=time.monotonic(),
            )
            self._session = session
            if reset:
                result = self.registry.execute("robot_reset", SkillInput({"reason": "openclaw_session_start"}))
                session.history.append({"action": "reset", "success": result.success, "message": result.message})
                if not result.success:
                    session.status = "failed"
                    self._emergency_stop(session, "reset failed")
                    return self._fail(result.error or result.message or "Robot reset failed", session)
            session.status = "running"
            return {"success": True, "message": "Exploration session started", "session": self._snapshot(session)}

    def observe(self, session_id: str) -> dict[str, Any]:
        with self._lock:
            session, error = self._active(session_id)
            if error:
                return error
            assert session is not None
            result = self.registry.execute("visual_detect", SkillInput({
                "camera_source": self.config.fire_search.camera_source,
                "require_newer_than": session.last_frame_id,
                "save_image": True,
            }))
            if not result.success:
                session.observation_valid_for_motion = False
                self._record(session, "observe", {"success": False, "error": result.error})
                return self._fail(result.error or "Fresh visual observation failed", session)
            observation = dict(result.data)
            frame_id = str(observation.get("frame_id", ""))
            if not frame_id or frame_id == session.last_frame_id:
                session.observation_valid_for_motion = False
                self._record(session, "observe", {"success": False, "error": "frame was not new"})
                return self._fail("Vision provider did not return a new frame", session)
            session.last_frame_id = frame_id
            session.latest_observation = observation
            session.observation_valid_for_motion = bool(observation.get("camera_forward_ready"))
            self._record(session, "observe", {"success": True, "frame_id": frame_id, "observation": observation})
            return {"success": True, "observation": observation, "session": self._snapshot(session)}

    def _motion_allowed(self, session: ExplorationSession) -> dict[str, Any] | None:
        if session.motion_actions >= self.limits.max_motion_actions:
            self._emergency_stop(session, "motion budget exhausted")
            return self._fail("Exploration motion budget exhausted", session)
        if not session.observation_valid_for_motion or not session.latest_observation:
            return self._fail("A fresh forward-facing observation is required before motion", session)
        return None

    def turn(self, session_id: str, angle_deg: float) -> dict[str, Any]:
        with self._lock:
            session, error = self._active(session_id)
            if error:
                return error
            assert session is not None
            if (error := self._motion_allowed(session)) is not None:
                return error
            angle = max(-self.limits.max_turn_angle, min(self.limits.max_turn_angle, float(angle_deg)))
            if abs(angle) < 1.0:
                return self._fail("Turn angle must be at least 1 degree", session)
            if session.total_rotation_deg + abs(angle) > self.limits.max_total_rotation_deg:
                self._emergency_stop(session, "rotation budget exhausted")
                return self._fail("Cumulative rotation budget would be exceeded", session)
            action = "turn_right" if angle > 0 else "turn_left"
            result = self.registry.execute("chassis", SkillInput({"action": action, "angle": abs(angle)}))
            session.observation_valid_for_motion = False
            session.motion_actions += 1
            if result.success:
                session.total_rotation_deg += abs(angle)
            self._record(session, "turn", {"success": result.success, "angle_deg": angle})
            if not result.success:
                self._emergency_stop(session, "turn failed")
                return self._fail(result.error or result.message or "Turn failed", session)
            return {"success": True, "message": result.message, "session": self._snapshot(session)}

    def approach(self, session_id: str, distance_m: float) -> dict[str, Any]:
        with self._lock:
            session, error = self._active(session_id)
            if error:
                return error
            assert session is not None
            if (error := self._motion_allowed(session)) is not None:
                return error
            obs = session.latest_observation or {}
            aligned = abs(float(obs.get("turn_angle", 999.0))) <= self.limits.alignment_tolerance_deg
            ready = (
                obs.get("fire_hydrant") == "YES"
                and bool(obs.get("ready_to_approach"))
                and bool(obs.get("camera_forward_ready"))
                and obs.get("has_person") != "YES"
                and aligned
            )
            if not ready:
                return self._fail("Approach blocked: target is not safely centered and reachable", session)
            model_distance = max(0.0, float(obs.get("approach_distance_m", 0.0)))
            remaining = self.limits.max_total_forward_m - session.total_forward_m
            distance = min(float(distance_m), model_distance, self.limits.max_forward_step_m, remaining)
            if distance < 0.05:
                return self._fail("Approach distance is below the safe movement threshold", session)
            result = self.registry.execute("chassis", SkillInput({"action": "forward", "distance": distance}))
            session.observation_valid_for_motion = False
            session.motion_actions += 1
            if result.success:
                session.total_forward_m += distance
            self._record(session, "approach", {"success": result.success, "distance_m": round(distance, 3)})
            if not result.success:
                self._emergency_stop(session, "approach failed")
                return self._fail(result.error or result.message or "Approach failed", session)
            return {"success": True, "distance_m": round(distance, 3), "message": result.message, "session": self._snapshot(session)}

    def stop(self, session_id: str | None = None, reason: str = "requested") -> dict[str, Any]:
        with self._lock:
            session = self._session
            if session_id and (session is None or session.session_id != session_id):
                return self._fail("No matching exploration session")
            if session is None:
                result = self.registry.execute("chassis", SkillInput({"action": "stop"}))
                return {"success": result.success, "message": result.message or reason, "session": None}
            result = self._emergency_stop(session, reason)
            return {"success": result.success, "message": result.message or reason, "session": self._snapshot(session)}

    def status(self) -> dict[str, Any]:
        with self._lock:
            return {
                "enabled": self.limits.enabled,
                "orchestrator": "openclaw",
                "session": self._snapshot(self._session) if self._session else None,
            }
