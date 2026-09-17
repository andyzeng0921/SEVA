from __future__ import annotations

import re
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

from .config import AgentConfig
from .executor import AgentExecutor
from .llm_planner import LLMTaskPlanner
from .models import ExecutionRequest, utc_now_iso
from .nlu import IntentParser
from .skill_packages import default_catalog

# LangGraph 是可选依赖 — 仅在 framework=langgraph 且 langgraph 已安装时可用
_LANGGRAPH_AVAILABLE = False
try:
    from .langgraph_brain import LangGraphBrain
    _LANGGRAPH_AVAILABLE = True
except ImportError:
    LangGraphBrain = None  # type: ignore[assignment,misc]

PermissionLevel = Literal["read_only", "safe_action", "motion_action"]
TaskStatus = Literal["completed", "blocked", "failed"]


@dataclass(slots=True)
class SkillSpec:
    name: str
    intent: str
    permission: PermissionLevel
    description: str
    requires_confirmation: bool = False
    safety_override: bool = False


@dataclass(slots=True)
class BrainStep:
    index: int
    text: str
    intent: str
    slots: dict[str, Any]
    permission: PermissionLevel
    status: str = "pending"
    message: str = ""
    result: dict[str, Any] | None = None


@dataclass(slots=True)
class BrainTaskState:
    task_id: str
    text: str
    status: TaskStatus
    created_at: str = field(default_factory=utc_now_iso)
    updated_at: str = field(default_factory=utc_now_iso)
    steps: list[BrainStep] = field(default_factory=list)
    message: str = ""


class SkillRegistry:
    def __init__(self, specs: list[SkillSpec]) -> None:
        self._by_intent = {spec.intent: spec for spec in specs}

    @classmethod
    def default(cls, require_motion_confirmation: bool = True) -> "SkillRegistry":
        """??? Skill Package ???? Brain ????"""
        packages = default_catalog(require_motion_confirmation).list()
        return cls([
            SkillSpec(
                name=package.skill_id,
                intent=package.intent,
                permission=package.permission,
                description=package.description,
                requires_confirmation=package.requires_confirmation,
                safety_override=package.safety_override,
            )
            for package in packages
        ])

    def get(self, intent: str) -> SkillSpec | None:
        return self._by_intent.get(intent)

    def as_dicts(self) -> list[dict[str, Any]]:
        return [asdict(spec) for spec in sorted(self._by_intent.values(), key=lambda item: item.name)]


class BrainOrchestrator:
    """任务大脑调度器

    支持两种后端：
    - framework="langgraph"  → 使用 LangGraph StateGraph 工作流
    - framework="langgraph_ready"（或其他） → 使用传统过程式逻辑
    """

    _split_pattern = re.compile(r"(?:\s*(?:\u7136\u540e|\u518d|\u63a5\u7740|\u968f\u540e)\s*)|[\uff1b\n\u3001\uff0c]+")

    def __init__(
        self,
        config: AgentConfig,
        parser: IntentParser,
        executor: AgentExecutor,
        registry: SkillRegistry | None = None,
        llm_planner: LLMTaskPlanner | None = None,
    ) -> None:
        self.config = config
        self.parser = parser
        self.executor = executor
        self.registry = registry or SkillRegistry.default(config.brain.require_confirmation_for_motion)
        self.skill_catalog = default_catalog(config.brain.require_confirmation_for_motion)
        self.llm_planner = llm_planner

        # LangGraph 后端（仅在 framework=langgraph 且已安装 langgraph 时启用）
        self._lg_brain: LangGraphBrain | None = None
        if config.brain.framework == "langgraph":
            if not _LANGGRAPH_AVAILABLE:
                import warnings
                import logging
                warnings.warn(
                    "framework=langgraph 但 langgraph 未安装。将退回到过程式逻辑。"
                    "安装 langgraph: pip install langgraph",
                    stacklevel=2,
                )
                logging.getLogger(__name__).warning(
                    "BrainOrchestrator: langgraph not installed, falling back to procedural mode"
                )
            else:
                self._lg_brain = LangGraphBrain(
                config=config,
                parser=parser,
                executor=executor,
                registry=self.registry,
                llm_planner=llm_planner,
            )

        self._tasks: dict[str, BrainTaskState] = {}

    def status(self) -> dict[str, Any]:
        if self._lg_brain is not None:
            return self._lg_brain.status()
        return {
            "enabled": self.config.brain.enabled,
            "framework": self.config.brain.framework,
            "state_backend": self.config.brain.state_backend,
            "robot_profile": self.config.brain.robot_profile,
            "require_confirmation_for_motion": self.config.brain.require_confirmation_for_motion,
            "skills": self.registry.as_dicts(),
            "skill_packages": self.skill_catalog.as_agent_tools(),
            "tasks_in_memory": len(self._tasks),
            "llm_enabled": self.llm_planner is not None and self.config.llm.enabled,
        }

    def run_text_task(self, text: str, confirmed: bool = False) -> dict[str, Any]:
        if self._lg_brain is not None:
            return self._lg_brain.run_text_task(text=text, confirmed=confirmed)

        state = BrainTaskState(task_id=str(uuid.uuid4()), text=text, status="failed")
        if not self.config.brain.enabled:
            state.status = "blocked"
            state.message = "Brain orchestrator is disabled"
            self._store(state)
            return self._serialize(state)

        # LLM 规划路径
        if self.llm_planner is not None and self.config.llm.enabled:
            plan = self.llm_planner.plan(text)
            if not plan["steps"]:
                state.status = "blocked"
                state.message = plan.get("message", "LLM 无法理解该指令")
                self._store(state)
                return self._serialize(state)

            if len(plan["steps"]) > self.config.brain.max_steps:
                state.status = "blocked"
                state.message = f"Task has {len(plan['steps'])} steps; max_steps is {self.config.brain.max_steps}"
                self._store(state)
                return self._serialize(state)

            state.steps = self._build_steps_from_llm(plan["steps"])
        else:
            # 正则规则规划路径（原有逻辑）
            segments = self._split_text(text)
            if len(segments) > self.config.brain.max_steps:
                state.status = "blocked"
                state.message = f"Task has {len(segments)} steps; max_steps is {self.config.brain.max_steps}"
                self._store(state)
                return self._serialize(state)

            state.steps = self._plan(segments)

        blocked = self._first_blocked_step(state.steps, confirmed=confirmed)
        if blocked is not None:
            blocked.status = "blocked"
            blocked.message = "Motion action requires explicit confirmation"
            state.status = "blocked"
            state.message = blocked.message
            self._store(state)
            return self._serialize(state)

        for step in state.steps:
            if step.intent == "unsupported":
                step.status = "blocked"
                step.message = "Unsupported command in brain task"
                state.status = "blocked"
                state.message = step.message
                self._store(state)
                return self._serialize(state)

            result = self.executor.execute(
                ExecutionRequest(
                    intent=step.intent,
                    validated_slots=step.slots,
                    mode="dry_run" if self.config.dry_run else "live",
                )
            )
            step.result = asdict(result)
            step.status = "completed" if result.success else "failed"
            step.message = result.message
            if not result.success:
                state.status = "failed"
                state.message = result.message
                self._store(state)
                return self._serialize(state)

        state.status = "completed"
        state.message = "Task completed"
        self._store(state)
        return self._serialize(state)

    def get_task(self, task_id: str) -> dict[str, Any] | None:
        if self._lg_brain is not None:
            return self._lg_brain.get_task(task_id)
        state = self._tasks.get(task_id)
        if state is None:
            return None
        return self._serialize(state)

    def _split_text(self, text: str) -> list[str]:
        return [part.strip() for part in self._split_pattern.split(text.strip()) if part.strip()]

    def _plan(self, segments: list[str]) -> list[BrainStep]:
        steps: list[BrainStep] = []
        for index, segment in enumerate(segments, start=1):
            parsed = self.parser.parse(segment)
            spec = self.registry.get(parsed.intent)
            permission: PermissionLevel = spec.permission if spec else "safe_action"
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

    def _build_steps_from_llm(self, llm_steps: list[dict[str, Any]]) -> list[BrainStep]:
        steps: list[BrainStep] = []
        for index, step_data in enumerate(llm_steps, start=1):
            intent = step_data.get("intent", "unsupported")
            spec = self.registry.get(intent)
            permission: PermissionLevel = spec.permission if spec else "safe_action"
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

    def _first_blocked_step(self, steps: list[BrainStep], confirmed: bool) -> BrainStep | None:
        if confirmed:
            return None
        for step in steps:
            spec = self.registry.get(step.intent)
            if spec and spec.requires_confirmation and not spec.safety_override:
                return step
        return None

    def _store(self, state: BrainTaskState) -> None:
        state.updated_at = utc_now_iso()
        self._tasks[state.task_id] = state

    @staticmethod
    def _serialize(state: BrainTaskState) -> dict[str, Any]:
        return asdict(state)
