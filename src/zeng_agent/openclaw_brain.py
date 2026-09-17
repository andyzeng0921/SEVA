"""
OpenClaw 原生 Brain 实现 (openclaw_brain.py)

替代 langgraph_brain.py 的 OpenClaw 版本。
使用 OpenClaw 的自主决策模式：Agent 通过 HTTP API 接收任务，
通过 autonomy 模块的能力卡片接口执行物理技能，
每步结果反馈给 Agent 构成闭环。

架构对比：
  langgraph_brain:  StateGraph 工作流 → plan → check_confirmation → execute (循环)
  openclaw_brain:   Agent 自主决策 → 能力发现 → 参数约束内选择 → 安全层校验 → 执行

职责：
  - 管理 OpenClaw 探索会话的生命周期
  - 提供任务状态追踪
  - 统计预算使用情况
  - 暴露 Agent 决策所需的状态信息
"""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Literal

from .config import AgentConfig
from .executor import AgentExecutor
from .models import utc_now_iso

try:
    from .autonomy.orchestrator import AutonomyOrchestrator, create_orchestrator
    from .autonomy.autonomy_manager import AutonomyLevel
    _AUTONOMY_AVAILABLE = True
except ImportError:
    _AUTONOMY_AVAILABLE = False
    AutonomyOrchestrator = None  # type: ignore
    AutonomyLevel = None  # type: ignore


TaskStatus = Literal["pending", "running", "completed", "failed", "blocked"]


@dataclass
class OpenClawStep:
    """单步执行记录"""
    index: int
    skill_id: str
    params: Dict[str, Any]
    success: bool
    result: Optional[Dict[str, Any]] = None
    timestamp: str = field(default_factory=utc_now_iso)


@dataclass
class OpenClawTask:
    """OpenClaw 任务状态"""
    task_id: str
    instruction: str
    status: TaskStatus = "pending"
    steps: List[OpenClawStep] = field(default_factory=list)
    messages: List[str] = field(default_factory=list)
    created_at: str = field(default_factory=utc_now_iso)
    updated_at: str = field(default_factory=utc_now_iso)
    session_id: Optional[str] = None


class OpenClawBrain:
    """
    OpenClaw 原生机器人大脑

    与 LangGraphBrain 接口兼容，但底层使用 autonomy 模块的自主决策模式。

    关键区别：
    - 不做任务规划（由 Agent LLM 完成）
    - 不做确认检查（Agent 自主决定，安全层兜底）
    - 只管理会话生命周期和状态追踪
    """

    def __init__(
        self,
        config: AgentConfig,
        executor: AgentExecutor,
    ) -> None:
        self.config = config
        self.executor = executor
        self._tasks: Dict[str, OpenClawTask] = {}
        self._current_session_id: Optional[str] = None

        # 初始化自主决策编排器
        if _AUTONOMY_AVAILABLE and AutonomyOrchestrator is not None:
            self._orchestrator: Optional[AutonomyOrchestrator] = create_orchestrator(
                budget_config={
                    "max_actions": config.openclaw_robot.max_actions,
                    "max_motion_actions": config.openclaw_robot.max_motion_actions,
                    "max_elapsed_seconds": config.openclaw_robot.max_elapsed_seconds,
                    "max_forward_m": config.openclaw_robot.max_total_forward_m,
                    "max_rotation_deg": config.openclaw_robot.max_total_rotation_deg,
                },
                initial_level=AutonomyLevel.STANDARD if AutonomyLevel else 2,
                dry_run=config.dry_run,
            )
        else:
            self._orchestrator = None

    # ========== 公开 API（与 LangGraphBrain 兼容） ==========

    def status(self) -> Dict[str, Any]:
        return {
            "enabled": self.config.brain.enabled,
            "framework": "openclaw",
            "state_backend": "openclaw_memory",
            "robot_profile": self.config.brain.robot_profile,
            "tasks_in_memory": len(self._tasks),
            "session_active": self._current_session_id is not None,
            "autonomy_available": _AUTONOMY_AVAILABLE,
            "budget": self._orchestrator.status()["safety"]
            if self._orchestrator else {},
        }

    def run_text_task(self, text: str, confirmed: bool = False) -> Dict[str, Any]:
        """处理文本任务（OpenClaw 模式下 Agent 自主规划）"""
        task = OpenClawTask(
            task_id=str(uuid.uuid4()),
            instruction=text,
            status="running",
        )

        if not self.config.brain.enabled:
            task.status = "blocked"
            task.messages.append("Brain orchestrator is disabled")
            self._tasks[task.task_id] = task
            return self._serialize(task)

        # OpenClaw 模式：不执行任务，返回任务 ID 让 Agent 自主决策
        # Agent 通过 autonomy 模块的能力卡片接口逐步执行
        task.session_id = self._current_session_id
        task.status = "running"
        task.messages.append(f"任务已接收。Agent 可自主决定执行步骤。")

        self._tasks[task.task_id] = task
        return self._serialize(task)

    def get_task(self, task_id: str) -> Optional[Dict[str, Any]]:
        """获取任务状态"""
        task = self._tasks.get(task_id)
        if task is None:
            return None
        return self._serialize(task)

    def list_tasks(self) -> List[Dict[str, Any]]:
        """列出所有任务"""
        return [self._serialize(t) for t in self._tasks.values()]

    # ========== Agent 自主决策接口 ==========

    def get_agent_prompt(self) -> str:
        """获取注入 Agent 的完整系统提示词"""
        if self._orchestrator:
            prompt = self._orchestrator.get_agent_system_prompt()
        else:
            prompt = (
                "# 可用物理技能\n"
                "autonomy 模块未加载，无法提供能力卡片。"
                "请确保 zeng_agent.autonomy 子包可用。\n"
            )
        return prompt

    def execute_agent_action(self, skill_id: str,
                             params: Dict[str, Any]) -> Dict[str, Any]:
        """Agent 通过此方法执行一次物理技能

        这是 OpenClaw Agent 与机器人系统的唯一桥接点。
        所有安全检查和参数校验在此层下游的 autonomy 模块中完成。
        """
        if not self._orchestrator:
            return {
                "success": False,
                "error": "autonomy 模块未加载，无法执行物理技能",
                "skill_id": skill_id,
            }

        result = self._orchestrator.execute(skill_id, params)

        # 关联到当前任务
        if self._current_session_id:
            task = self._tasks.get(self._current_session_id)
            if task:
                step = OpenClawStep(
                    index=len(task.steps) + 1,
                    skill_id=skill_id,
                    params=params,
                    success=result.get("success", False),
                    result=result,
                )
                task.steps.append(step)
                task.updated_at = utc_now_iso()

        return result

    # ========== 会话管理 ==========

    def start_exploration_session(self, goal: str) -> Dict[str, Any]:
        """开始自主探索会话"""
        if not self._orchestrator:
            return {
                "success": False,
                "error": "autonomy 模块未加载",
            }

        result = self._orchestrator.start_session(goal, reset=True)
        self._current_session_id = result["session_id"]

        # 创建追踪任务
        task = OpenClawTask(
            task_id=self._current_session_id,
            instruction=goal,
            status="running",
            session_id=self._current_session_id,
        )
        task.messages.append(f"探索会话已启动。自治等级: {result['autonomy_level']}")
        self._tasks[task.task_id] = task

        return result

    def stop_exploration_session(self, reason: str = "requested") -> Dict[str, Any]:
        """停止探索会话"""
        if self._orchestrator:
            result = self._orchestrator.stop_session(reason)
            self._current_session_id = None
        else:
            result = {"success": True, "message": "autonomy 未加载，无需停止"}

        return result

    def get_session_status(self) -> Dict[str, Any]:
        """获取当前会话状态"""
        if not self._orchestrator:
            return {"session_active": False, "error": "autonomy 未加载"}

        status = self._orchestrator.status()
        status["tasks_in_memory"] = len(self._tasks)
        return status

    # ========== 内部工具 ==========

    def _serialize(self, task: OpenClawTask) -> Dict[str, Any]:
        return {
            "task_id": task.task_id,
            "instruction": task.instruction,
            "status": task.status,
            "steps": [
                {
                    "index": s.index,
                    "skill_id": s.skill_id,
                    "params": s.params,
                    "success": s.success,
                    "result": s.result,
                    "timestamp": s.timestamp,
                }
                for s in task.steps
            ],
            "messages": task.messages,
            "created_at": task.created_at,
            "updated_at": task.updated_at,
            "session_id": task.session_id,
        }
