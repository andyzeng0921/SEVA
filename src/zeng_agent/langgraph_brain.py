from __future__ import annotations

import uuid
from dataclasses import asdict
from datetime import datetime, timezone
from typing import Any, Literal, TypedDict

try:
    from langgraph.checkpoint.memory import MemorySaver
    from langgraph.graph import END, START, StateGraph
    _LANGGRAPH_AVAILABLE_LG = True
except ImportError:
    _LANGGRAPH_AVAILABLE_LG = False

from .config import AgentConfig
from .executor import AgentExecutor
from .llm_planner import LLMTaskPlanner
from .models import ExecutionRequest
from .nlu import IntentParser

TaskStatus = Literal["pending", "running", "blocked", "completed", "failed"]


class LangGraphState(TypedDict):
    """LangGraph 工作流的状态定义"""
    text: str
    confirmed: bool
    steps: list[dict[str, Any]]
    current_index: int
    task_status: TaskStatus
    message: str
    task_id: str
    created_at: str
    updated_at: str


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _step_to_dict(step: Any) -> dict[str, Any]:
    return asdict(step)


def _dict_to_step(data: dict[str, Any]) -> Any:
    from .brain import BrainStep

    return BrainStep(
        index=data["index"],
        text=data["text"],
        intent=data["intent"],
        slots=data["slots"],
        permission=data["permission"],
        status=data.get("status", "pending"),
        message=data.get("message", ""),
        result=data.get("result"),
    )


def _serialize_state(state: dict[str, Any]) -> dict[str, Any]:
    """将 LangGraph 状态序列化为对外一致的 JSON 格式"""
    return {
        "task_id": state["task_id"],
        "text": state["text"],
        "status": state["task_status"],
        "created_at": state["created_at"],
        "updated_at": state["updated_at"],
        "steps": state["steps"],
        "message": state["message"],
    }


class LangGraphBrain:
    """基于 LangGraph StateGraph 的机器人大脑

    工作流：
      START → plan_task → check_confirmation ──blocked──→ END
                                  │
                              execute_step ──continue──→ execute_step (loop)
                                  │
                            completed/failed → END
    """

    def __init__(
        self,
        config: AgentConfig,
        parser: IntentParser,
        executor: AgentExecutor,
        registry: SkillRegistry | None = None,
        llm_planner: LLMTaskPlanner | None = None,
    ) -> None:
        from .brain import SkillRegistry

        if not _LANGGRAPH_AVAILABLE_LG:
            raise ImportError("LangGraphBrain 需要 langgraph 库。请安装: pip install langgraph")

        self.config = config
        self.parser = parser
        self.executor = executor
        self.registry = registry or SkillRegistry.default(config.brain.require_confirmation_for_motion)
        self.llm_planner = llm_planner
        self._split_pattern = __import__("re").compile(
            r"(?:\s*(?:\u7136\u540e|\u518d|\u63a5\u7740|\u968f\u540e)\s*)|[\uff1b\n\u3001\uff0c]+"
        )

        self.checkpointer = MemorySaver()
        self.graph = self._build_graph()
        self.app = self.graph.compile(checkpointer=self.checkpointer)

    # ── 公开 API ──────────────────────────────────────────

    def status(self) -> dict[str, Any]:
        return {
            "enabled": self.config.brain.enabled,
            "framework": "langgraph",
            "state_backend": "langgraph_memory",
            "robot_profile": self.config.brain.robot_profile,
            "require_confirmation_for_motion": self.config.brain.require_confirmation_for_motion,
            "skills": self.registry.as_dicts(),
            "llm_enabled": self.llm_planner is not None and self.config.llm.enabled,
        }

    def run_text_task(self, text: str, confirmed: bool = False) -> dict[str, Any]:
        task_id = str(uuid.uuid4())
        thread_config = {"configurable": {"thread_id": task_id}}
        now = _utc_now()

        initial: LangGraphState = {
            "text": text,
            "confirmed": confirmed,
            "steps": [],
            "current_index": 0,
            "task_status": "pending",
            "message": "",
            "task_id": task_id,
            "created_at": now,
            "updated_at": now,
        }

        final = self.app.invoke(initial, config=thread_config)
        return _serialize_state(final)

    def get_task(self, task_id: str) -> dict[str, Any] | None:
        try:
            state = self.app.get_state({"configurable": {"thread_id": task_id}})
            if state is None:
                return None
            values = state.values
            if not values.get("task_id"):
                return None
            return _serialize_state(values)
        except Exception:
            return None

    # ── 图构建 ────────────────────────────────────────────

    def _build_graph(self) -> StateGraph:
        workflow = StateGraph(LangGraphState)

        workflow.add_node("plan_task", self._plan_task)
        workflow.add_node("check_confirmation", self._check_confirmation)
        workflow.add_node("execute_step", self._execute_step)

        workflow.add_edge(START, "plan_task")
        workflow.add_edge("plan_task", "check_confirmation")

        workflow.add_conditional_edges(
            "check_confirmation",
            self._route_after_confirmation,
            {"blocked": END, "execute": "execute_step"},
        )

        workflow.add_conditional_edges(
            "execute_step",
            self._route_after_execution,
            {"continue": "execute_step", "completed": END, "failed": END, "blocked": END},
        )

        return workflow

    # ── 节点函数 ──────────────────────────────────────────

    def _plan_task(self, state: LangGraphState) -> dict[str, Any]:
        """规划阶段：将文本拆解为结构化步骤"""
        text = state["text"]
        now = _utc_now()

        if not self.config.brain.enabled:
            return {
                "task_status": "blocked",
                "message": "Brain orchestrator is disabled",
                "steps": [],
                "updated_at": now,
            }

        # LLM 规划路径
        if self.llm_planner is not None and self.config.llm.enabled:
            plan = self.llm_planner.plan(text)
            if not plan["steps"]:
                return {
                    "task_status": "blocked",
                    "message": plan.get("message", "LLM 无法理解该指令"),
                    "steps": [],
                    "updated_at": now,
                }
            if len(plan["steps"]) > self.config.brain.max_steps:
                return {
                    "task_status": "blocked",
                    "message": f"Task has {len(plan['steps'])} steps; max_steps is {self.config.brain.max_steps}",
                    "steps": [],
                    "updated_at": now,
                }
            steps = self._build_steps_from_llm(plan["steps"])
        else:
            # 正则规划路径
            segments = [
                part.strip()
                for part in self._split_pattern.split(text.strip())
                if part.strip()
            ]
            if len(segments) > self.config.brain.max_steps:
                return {
                    "task_status": "blocked",
                    "message": f"Task has {len(segments)} steps; max_steps is {self.config.brain.max_steps}",
                    "steps": [],
                    "updated_at": now,
                }
            steps = self._plan_segments(segments)

        return {
            "steps": [_step_to_dict(s) for s in steps],
            "task_status": "running",
            "updated_at": now,
        }

    def _check_confirmation(self, state: LangGraphState) -> dict[str, Any]:
        """确认门控：运动动作需要显式确认"""
        now = _utc_now()

        if state["confirmed"]:
            return {"updated_at": now}

        for step_dict in state["steps"]:
            spec = self.registry.get(step_dict["intent"])
            if spec and spec.requires_confirmation and not spec.safety_override:
                step_dict["status"] = "blocked"
                step_dict["message"] = "Motion action requires explicit confirmation"
                return {
                    "task_status": "blocked",
                    "message": step_dict["message"],
                    "steps": state["steps"],
                    "updated_at": now,
                }

        return {"updated_at": now}

    def _execute_step(self, state: LangGraphState) -> dict[str, Any]:
        """执行当前步骤"""
        now = _utc_now()
        idx = state["current_index"]
        steps = list(state["steps"])

        if idx >= len(steps):
            return {
                "task_status": "completed",
                "message": "Task completed",
                "updated_at": now,
            }

        step_dict = steps[idx]

        # 检查 unsupported
        if step_dict["intent"] == "unsupported":
            step_dict["status"] = "blocked"
            step_dict["message"] = "Unsupported command in brain task"
            steps[idx] = step_dict
            return {
                "task_status": "blocked",
                "message": step_dict["message"],
                "steps": steps,
                "updated_at": now,
            }

        # 执行
        result = self.executor.execute(
            ExecutionRequest(
                intent=step_dict["intent"],
                validated_slots=step_dict["slots"],
                mode="dry_run" if self.config.dry_run else "live",
            )
        )
        step_dict["result"] = asdict(result)
        step_dict["status"] = "completed" if result.success else "failed"
        step_dict["message"] = result.message
        steps[idx] = step_dict

        if not result.success:
            return {
                "task_status": "failed",
                "message": result.message,
                "steps": steps,
                "updated_at": now,
            }

        next_idx = idx + 1
        is_last = next_idx >= len(steps)
        return {
            "current_index": next_idx,
            "task_status": "completed" if is_last else "running",
            "steps": steps,
            "updated_at": now,
        }

    # ── 路由函数 ──────────────────────────────────────────

    @staticmethod
    def _route_after_confirmation(state: LangGraphState) -> str:
        if state["task_status"] == "blocked":
            return "blocked"
        return "execute"

    @staticmethod
    def _route_after_execution(state: LangGraphState) -> str:
        status = state["task_status"]
        if status in ("completed", "failed", "blocked"):
            return status
        # 还有更多步骤
        if state["current_index"] < len(state["steps"]):
            return "continue"
        return "completed"

    # ── 内部工具方法 ──────────────────────────────────────

    def _plan_segments(self, segments: list[str]) -> list[Any]:
        from .brain import BrainStep

        steps: list[BrainStep] = []
        for index, segment in enumerate(segments, start=1):
            parsed = self.parser.parse(segment)
            spec = self.registry.get(parsed.intent)
            permission = spec.permission if spec else "safe_action"
            steps.append(
                BrainStep(
                    index=index,
                    text=segment,
                    intent=parsed.intent,
                    slots=parsed.slots,
                    permission=permission,
                )
            )
        return steps

    def _build_steps_from_llm(self, llm_steps: list[dict[str, Any]]) -> list[Any]:
        from .brain import BrainStep

        steps: list[BrainStep] = []
        for index, step_data in enumerate(llm_steps, start=1):
            intent = step_data.get("intent", "unsupported")
            spec = self.registry.get(intent)
            permission = spec.permission if spec else "safe_action"
            steps.append(
                BrainStep(
                    index=index,
                    text=step_data.get("text", ""),
                    intent=intent,
                    slots=step_data.get("slots", {}),
                    permission=permission,
                )
            )
        return steps
