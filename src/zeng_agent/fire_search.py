"""Fail-closed LangGraph workflow for finding and approaching a fire hydrant."""
from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Literal, TypedDict

from .skills.base import SkillInput, SkillOutput

logger = logging.getLogger(__name__)

# LangGraph 是可选依赖 — 仅在 FireHydrantSearchAgent 实例化时才需要
_LANGGRAPH_AVAILABLE = False
_LANGGRAPH_ERROR = ""
try:
    from langgraph.checkpoint.memory import MemorySaver as _MemorySaver  # noqa: F401
    from langgraph.graph import END as _END, START as _START, StateGraph as _StateGraph  # noqa: F401
    _LANGGRAPH_AVAILABLE = True
except ImportError as e:
    _LANGGRAPH_ERROR = str(e)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class FireSearchConfig:
    camera_source: str = "auto_front"
    min_confidence: float = 0.5
    alignment_tolerance_deg: float = 3.0
    min_turn_deg: float = 2.0
    max_turn_deg: float = 180.0
    max_scan_steps: int = 8
    max_align_attempts: int = 4
    max_approach_steps: int = 6
    max_forward_step_m: float = 0.5
    max_total_forward_m: float = 3.0


SearchStatus = Literal["running", "done", "failed"]


class FireSearchState(TypedDict, total=False):
    task_id: str
    text: str
    status: SearchStatus
    phase: str
    message: str
    created_at: str
    updated_at: str
    current_detection: dict[str, Any]
    observations: list[dict[str, Any]]
    history: list[dict[str, Any]]
    planned_actions: list[dict[str, Any]]
    camera_source: str
    last_frame_id: str
    require_new_frame: bool
    pending_turn: float
    turn_reason: str
    scan_steps: int
    align_attempts: int
    approach_steps: int
    total_forward_m: float
    target_found: bool
    terminated: bool


class FireHydrantSearchAgent:
    """One authoritative visual-search workflow.

    A forward command can only be reached from a fresh, front-camera decision
    that explicitly marks the hydrant centered and safe to approach.
    """

    def __init__(self, registry, dry_run: bool = False, config: FireSearchConfig | None = None):
        if not _LANGGRAPH_AVAILABLE:
            raise ImportError(
                f"FireHydrantSearchAgent 需要 langgraph 库。{_LANGGRAPH_ERROR or '请安装: pip install langgraph'}"
            )
        self.registry = registry
        self.dry_run = dry_run
        self.config = config or FireSearchConfig()
        self.checkpointer = _MemorySaver()
        self.graph = self._build_graph()
        self.app = self.graph.compile(checkpointer=self.checkpointer)

    def search(self, instruction: str = "搜索消防栓") -> dict[str, Any]:
        task_id = str(uuid.uuid4())
        now = _utc_now()
        initial: FireSearchState = {
            "task_id": task_id,
            "text": instruction,
            "status": "running",
            "phase": "observe",
            "message": "准备从头部前向共享内存获取图像",
            "created_at": now,
            "updated_at": now,
            "current_detection": {},
            "observations": [],
            "history": [],
            "planned_actions": [],
            "camera_source": self.config.camera_source,
            "last_frame_id": "",
            "require_new_frame": False,
            "pending_turn": 0.0,
            "turn_reason": "",
            "scan_steps": 0,
            "align_attempts": 0,
            "approach_steps": 0,
            "total_forward_m": 0.0,
            "target_found": False,
            "terminated": False,
        }
        try:
            final = self.app.invoke(
                initial,
                config={"configurable": {"thread_id": task_id}, "recursion_limit": 80},
            )
            return self._serialize(final)
        except Exception as exc:
            logger.exception("消防栓搜索异常")
            initial["status"] = "failed"
            initial["message"] = str(exc)
            return self._serialize(initial)

    def get_state(self, task_id: str) -> dict[str, Any] | None:
        try:
            snapshot = self.app.get_state({"configurable": {"thread_id": task_id}})
            if snapshot is None or not snapshot.values.get("task_id"):
                return None
            return self._serialize(snapshot.values)
        except Exception:
            return None

    @staticmethod
    def _serialize(state: dict[str, Any]) -> dict[str, Any]:
        return {
            "task_id": state.get("task_id", ""),
            "status": state.get("status", "failed"),
            "phase": state.get("phase", ""),
            "message": state.get("message", ""),
            "target_found": state.get("target_found", False),
            "camera_source": state.get("camera_source", ""),
            "last_frame_id": state.get("last_frame_id", ""),
            "scan_steps": state.get("scan_steps", 0),
            "align_attempts": state.get("align_attempts", 0),
            "approach_steps": state.get("approach_steps", 0),
            "total_forward_m": round(float(state.get("total_forward_m", 0.0)), 3),
            "current_detection": state.get("current_detection", {}),
            "observations": state.get("observations", []),
            "planned_actions": state.get("planned_actions", []),
            "history": state.get("history", []),
        }

    def _build_graph(self) -> _StateGraph:
        graph = _StateGraph(FireSearchState)
        graph.add_node("power_on_reset", self._node_power_on_reset)
        graph.add_node("observe", self._node_observe)
        graph.add_node("decide", self._node_decide)
        graph.add_node("turn", self._node_turn)
        graph.add_node("approach", self._node_approach)
        graph.add_node("finish", self._node_finish)
        graph.add_node("fail", self._node_fail)

        graph.add_edge(_START, "power_on_reset")
        graph.add_conditional_edges(
            "power_on_reset",
            self._route_after_reset,
            {"observe": "observe", "fail": "fail"},
        )
        graph.add_conditional_edges("observe", self._route_after_observe, {"decide": "decide", "fail": "fail"})
        graph.add_conditional_edges(
            "decide",
            self._route_after_decide,
            {"turn": "turn", "approach": "approach", "finish": "finish", "fail": "fail"},
        )
        graph.add_edge("turn", "observe")
        graph.add_conditional_edges(
            "approach",
            self._route_after_approach,
            {"observe": "observe", "finish": "finish", "fail": "fail"},
        )
        graph.add_edge("finish", _END)
        graph.add_edge("fail", _END)
        return graph

    def _node_power_on_reset(self, state: FireSearchState) -> dict[str, Any]:
        now = _utc_now()
        history = list(state.get("history", []))
        result = self.registry.execute("robot_reset", SkillInput({"reason": "fire_search_precondition"}))
        if not result.success:
            message = result.error or result.message or "机器人上电复位失败"
            history.append({"time": now, "phase": "power_on_reset", "success": False, "message": message})
            return {
                "status": "failed",
                "phase": "fail",
                "message": message,
                "history": history,
                "updated_at": now,
            }
        history.append({"time": now, "phase": "power_on_reset", "success": True, "message": result.message})
        return {
            "status": "running",
            "phase": "observe",
            "message": result.message or "机器人上电复位完成",
            "history": history,
            "require_new_frame": False,
            "updated_at": now,
        }

    def _node_observe(self, state: FireSearchState) -> dict[str, Any]:
        now = _utc_now()
        require_token = state.get("last_frame_id") if state.get("require_new_frame") and not self.dry_run else None
        camera_source = state.get("camera_source") or self.config.camera_source
        detected = self.registry.execute("visual_detect", SkillInput({
            "mode": "navigate" if state.get("target_found") else "search",
            "camera_source": camera_source,
            "require_newer_than": require_token,
            "save_image": True,
        }))
        history = list(state.get("history", []))
        if not detected.success:
            message = detected.error or detected.message or "视觉检测失败"
            history.append({"time": now, "phase": "observe", "success": False, "message": message})
            return {
                "status": "failed",
                "phase": "fail",
                "message": message,
                "history": history,
                "updated_at": now,
            }

        data = dict(detected.data)
        frame_id = str(data.get("frame_id", ""))
        source = str(data.get("camera_source", camera_source))
        if not frame_id or not data.get("frame_fresh", False):
            message = "视觉结果缺少新帧凭证，禁止运动"
            history.append({"time": now, "phase": "observe", "success": False, "message": message})
            return {"status": "failed", "phase": "fail", "message": message, "history": history, "updated_at": now}

        observations = list(state.get("observations", []))
        observations.append(data)
        history.append({
            "time": now,
            "phase": "observe",
            "success": True,
            "camera_source": source,
            "frame_id": frame_id,
            "image_path": data.get("image_path", ""),
            "description": data.get("description", ""),
        })
        return {
            "status": "running",
            "phase": "decide",
            "message": detected.message,
            "current_detection": data,
            "observations": observations,
            "history": history,
            "last_frame_id": frame_id,
            "camera_source": source,
            "require_new_frame": False,
            "target_found": state.get("target_found", False) or data.get("fire_hydrant") == "YES",
            "updated_at": now,
        }

    def _node_decide(self, state: FireSearchState) -> dict[str, Any]:
        now = _utc_now()
        data = state.get("current_detection", {})
        history = list(state.get("history", []))
        if not data.get("camera_forward_ready", False):
            message = "复位后相机仍未抬头或画面主要朝向地面，禁止底盘运动"
            history.append({"time": now, "phase": "decide", "decision": "fail", "message": message})
            return {"status": "failed", "phase": "fail", "message": message, "history": history, "updated_at": now}
        found = data.get("fire_hydrant") == "YES" and float(data.get("confidence", 0.0)) >= self.config.min_confidence
        turn = max(-self.config.max_turn_deg, min(self.config.max_turn_deg, float(data.get("turn_angle", 0.0))))

        if not found:
            scans = int(state.get("scan_steps", 0))
            if scans >= self.config.max_scan_steps or abs(turn) < self.config.min_turn_deg:
                message = f"扫描结束，未发现消防栓（{scans + 1}帧）"
                history.append({"time": now, "phase": "decide", "decision": "finish", "message": message})
                return {"phase": "finish", "message": message, "history": history, "updated_at": now}
            history.append({"time": now, "phase": "decide", "decision": "scan_turn", "angle": turn})
            return {
                "phase": "turn",
                "pending_turn": turn,
                "turn_reason": "scan",
                "history": history,
                "updated_at": now,
            }

        ready = bool(data.get("ready_to_approach", False))
        aligned = abs(turn) <= self.config.alignment_tolerance_deg
        if not aligned or not ready:
            attempts = int(state.get("align_attempts", 0))
            if attempts >= self.config.max_align_attempts:
                message = "达到对准尝试上限，消防栓仍未居中，禁止前进"
                history.append({"time": now, "phase": "decide", "decision": "fail", "message": message})
                return {"status": "failed", "phase": "fail", "message": message, "history": history, "updated_at": now}
            if abs(turn) < self.config.min_turn_deg:
                message = "视觉模型未确认可以前进，且没有有效校准转角"
                history.append({"time": now, "phase": "decide", "decision": "fail", "message": message})
                return {"status": "failed", "phase": "fail", "message": message, "history": history, "updated_at": now}
            history.append({"time": now, "phase": "decide", "decision": "align", "angle": turn})
            return {
                "phase": "turn",
                "pending_turn": turn,
                "turn_reason": "align",
                "history": history,
                "updated_at": now,
            }

        distance = max(0.0, float(data.get("approach_distance_m", 0.0)))
        if distance <= 0.01:
            message = "消防栓已正对，视觉模型判断无需继续前进"
            history.append({"time": now, "phase": "decide", "decision": "finish", "message": message})
            return {"phase": "finish", "message": message, "history": history, "updated_at": now}
        if int(state.get("approach_steps", 0)) >= self.config.max_approach_steps:
            message = "达到分段接近上限，停止前进并等待人工确认"
            return {"status": "failed", "phase": "fail", "message": message, "history": history, "updated_at": now}
        history.append({"time": now, "phase": "decide", "decision": "approach", "distance": distance})
        return {"phase": "approach", "history": history, "updated_at": now}

    def _motion(self, params: dict[str, Any]) -> SkillOutput:
        if self.dry_run:
            return SkillOutput(success=True, data={**params, "mode": "dry_run"}, message="DRY_RUN")
        return self.registry.execute("chassis", SkillInput(params))

    def _node_turn(self, state: FireSearchState) -> dict[str, Any]:
        now = _utc_now()
        angle = float(state.get("pending_turn", 0.0))
        action = "turn_right" if angle > 0 else "turn_left"
        params = {"action": action, "angle": round(abs(angle), 1)}
        result = self._motion(params)
        history = list(state.get("history", []))
        actions = list(state.get("planned_actions", []))
        actions.append(params)
        if not result.success:
            message = result.error or result.message or "底盘转向失败"
            history.append({"time": now, "phase": "turn", "success": False, "message": message})
            return {"status": "failed", "phase": "fail", "message": message, "history": history, "planned_actions": actions, "updated_at": now}
        reason = state.get("turn_reason", "scan")
        history.append({"time": now, "phase": "turn", "success": True, "reason": reason, **params})
        updates: dict[str, Any] = {
            "phase": "observe",
            "message": f"{action} {abs(angle):.1f}°，等待运动后的新帧",
            "history": history,
            "planned_actions": actions,
            "require_new_frame": not self.dry_run,
            "updated_at": now,
        }
        if reason == "align":
            updates["align_attempts"] = int(state.get("align_attempts", 0)) + 1
        else:
            updates["scan_steps"] = int(state.get("scan_steps", 0)) + 1
        return updates

    def _node_approach(self, state: FireSearchState) -> dict[str, Any]:
        now = _utc_now()
        data = state.get("current_detection", {})
        history = list(state.get("history", []))
        if not data.get("frame_fresh") or not data.get("ready_to_approach"):
            message = "前进门控失败：图像不是新帧或消防栓未对准"
            return {"status": "failed", "phase": "fail", "message": message, "history": history, "updated_at": now}
        if abs(float(data.get("turn_angle", 999))) > self.config.alignment_tolerance_deg:
            message = "前进门控失败：消防栓不在前向相机中央"
            return {"status": "failed", "phase": "fail", "message": message, "history": history, "updated_at": now}

        requested = max(0.0, float(data.get("approach_distance_m", 0.0)))
        remaining_budget = self.config.max_total_forward_m - float(state.get("total_forward_m", 0.0))
        distance = min(requested, self.config.max_forward_step_m, remaining_budget)
        if distance <= 0.01:
            message = "已达到前进安全预算，停止任务"
            return {"phase": "finish", "message": message, "history": history, "updated_at": now}
        params = {"action": "forward", "distance": round(distance, 2)}
        result = self._motion(params)
        actions = list(state.get("planned_actions", []))
        actions.append(params)
        if not result.success:
            message = result.error or result.message or "底盘前进失败"
            history.append({"time": now, "phase": "approach", "success": False, "message": message})
            return {"status": "failed", "phase": "fail", "message": message, "history": history, "planned_actions": actions, "updated_at": now}
        total = float(state.get("total_forward_m", 0.0)) + distance
        history.append({"time": now, "phase": "approach", "success": True, **params, "total_forward_m": total})
        return {
            "phase": "observe",
            "message": f"已前进{distance:.2f}m，重新取新帧校验",
            "history": history,
            "planned_actions": actions,
            "approach_steps": int(state.get("approach_steps", 0)) + 1,
            "total_forward_m": total,
            "require_new_frame": not self.dry_run,
            "updated_at": now,
        }

    def _node_finish(self, state: FireSearchState) -> dict[str, Any]:
        return {"status": "done", "phase": "done", "terminated": True, "updated_at": _utc_now()}

    def _node_fail(self, state: FireSearchState) -> dict[str, Any]:
        return {"status": "failed", "phase": "failed", "terminated": True, "updated_at": _utc_now()}

    @staticmethod
    def _route_after_reset(state: FireSearchState) -> str:
        return "fail" if state.get("status") == "failed" else "observe"

    @staticmethod
    def _route_after_observe(state: FireSearchState) -> str:
        return "fail" if state.get("status") == "failed" else "decide"

    @staticmethod
    def _route_after_decide(state: FireSearchState) -> str:
        phase = state.get("phase", "fail")
        return phase if phase in {"turn", "approach", "finish", "fail"} else "fail"

    @staticmethod
    def _route_after_approach(state: FireSearchState) -> str:
        phase = state.get("phase", "fail")
        return phase if phase in {"observe", "finish", "fail"} else "fail"
