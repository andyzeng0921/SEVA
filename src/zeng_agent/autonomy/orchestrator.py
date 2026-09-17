"""
自主决策编排器 (AutonomyOrchestrator)

将所有分层组件串联为一个可运行的整体。

这是外部代码（如 service.py / bootstrap.py）与自主决策系统交互的唯一入口。
"""
import time
import logging
from typing import Any, Dict, Optional

from .skill_provider import SkillProvider
from .safety_enforcer import SafetyEnforcer
from .decision_engine import DecisionEngine
from .autonomy_manager import AutonomyManager, AutonomyLevel

logger = logging.getLogger(__name__)


class AutonomyOrchestrator:
    """
    自主决策编排器 — 分层架构的总线
    
    使用方式:
        orch = create_orchestrator(budget_config)
        
        # Agent 获取能力描述
        prompt = orch.get_agent_system_prompt()
        
        # Agent 执行物理技能
        result = orch.execute("move_forward", {"distance_m": 0.3})
        
        # 查看状态
        status = orch.status()
    """

    def __init__(self,
                 skill_provider: SkillProvider,
                 safety_enforcer: SafetyEnforcer,
                 decision_engine: DecisionEngine,
                 autonomy_manager: AutonomyManager):
        self._provider = skill_provider
        self._safety = safety_enforcer
        self._engine = decision_engine
        self._autonomy = autonomy_manager
        
        # 会话信息
        self._session_id: Optional[str] = None
        self._session_goal: str = ""
        
        # 运行时 hook
        self._pre_execute_hooks = []
        self._post_execute_hooks = []

    # ========== 会话管理 ==========

    def start_session(self, goal: str, reset: bool = True) -> Dict[str, Any]:
        """开始新的自主探索会话"""
        self._session_id = f"auto_{int(time.time())}"
        self._session_goal = goal
        
        if reset:
            # 重置到默认位置
            self._provider.move_arm_preset("reset", speed_pct=40)
        
        return {
            "session_id": self._session_id,
            "goal": goal,
            "autonomy_level": self._autonomy.get_current_level().name,
            "available_skills": self._autonomy.get_available_skills(),
            "message": "会话已启动。Agent 可以开始自主决策。",
        }

    def stop_session(self, reason: str = "requested") -> Dict[str, Any]:
        """终止会话"""
        self._provider.emergency_stop()
        metrics = self._autonomy.get_metrics()
        
        logger.info(f"[SESSION] {self._session_id} 结束: {reason}")
        
        return {
            "session_id": self._session_id,
            "reason": reason,
            "session_metrics": metrics,
            "message": "会话已终止。",
        }

    # ========== Agent 接口 ==========

    def get_agent_system_prompt(self) -> str:
        """
        生成 Agent 完整的 System Prompt
        
        包含：
        - 能力卡片目录
        - 安全约束
        - 当前自治级别
        - 决策格式要求
        """
        parts = []
        
        # 1. 能力卡片
        parts.append(self._engine.get_capabilities_prompt())
        
        # 2. 安全约束
        parts.append(self._safety.generate_safety_context_for_agent())
        
        # 3. 自治上下文
        parts.append(self._autonomy.generate_autonomy_context())
        
        # 4. 当前任务
        if self._session_goal:
            parts.append(f"## 当前任务目标\n{self._session_goal}\n")
        
        return "\n".join(parts)

    def execute(self, skill_id: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        执行 Agent 的一次自主决策
        
        完整链路:
        Agent → AutonomyManager(级别过滤) → DecisionEngine → SafetyEnforcer → SkillProvider
        """
        # 在自治级别允许的范围内执行
        available = self._autonomy.get_available_skills()
        if skill_id not in available:
            return {
                "success": False,
                "error": f"技能 '{skill_id}' 不在当前自治级别 ({self._autonomy.get_current_level().name}) 的允许范围内",
                "available_skills": available,
            }
        
        # 应用自治级别的参数缩放
        scaled_params = {}
        for k, v in params.items():
            scale = self._autonomy.get_param_scale(skill_id, k)
            if isinstance(v, (int, float)) and scale != 1.0:
                scaled_params[k] = v * scale
                logger.debug(f"[SCALE] {skill_id}.{k}: {v} → {scaled_params[k]} (×{scale})")
            else:
                scaled_params[k] = v
        
        # 执行
        result = self._engine.execute(skill_id, scaled_params)
        
        # 记录反馈
        success = result.get("success", False)
        violation = result.get("safety_blocked", False)
        estop = result.get("emergency_stopped", False)
        exec_time = result.get("execution_time_ms", 0)
        
        self._autonomy.record_outcome(
            success=success,
            violation=violation,
            emergency_stop=estop,
            exec_time_ms=exec_time,
        )
        
        # 注入自治上下文到返回结果
        result["autonomy"] = self._autonomy.get_metrics()
        
        return result

    def execute_plan(self, plan: list) -> list:
        """
        执行 Agent 自主规划的多步骤计划
        
        plan 是一个列表，每个元素是 {"skill_id": "...", "params": {...}}
        """
        results = []
        for step in plan:
            result = self.execute(step["skill_id"], step.get("params", {}))
            results.append(result)
            
            # 急停则中断
            if result.get("emergency_stopped"):
                results.append({
                    "success": False,
                    "error": "急停触发，计划中断",
                })
                break
            
            # 将上次结果传给下一步（Agent 可根据此结果调整后续决策）
            if not result.get("success"):
                # 不自动中断 — Agent 在下一轮自行判断
                pass
        
        return results

    # ========== 管理接口 ==========

    def set_autonomy_level(self, level_name: str) -> Dict[str, Any]:
        """手动设置自治级别"""
        level_map = {
            "restricted": AutonomyLevel.RESTRICTED,
            "standard": AutonomyLevel.STANDARD,
            "enhanced": AutonomyLevel.ENHANCED,
        }
        level = level_map.get(level_name.lower())
        if level is None:
            return {"success": False, "error": f"未知级别: {level_name}"}
        return self._autonomy.set_level(level, "manual_override")

    def update_safety_state(self, **kwargs) -> None:
        """更新安全状态（由底层硬件回调）"""
        self._safety.update_system_state(**kwargs)

    def status(self) -> Dict[str, Any]:
        """获取系统完整状态"""
        return {
            "session_id": self._session_id,
            "goal": self._session_goal,
            "autonomy": self._autonomy.get_metrics(),
            "safety": self._safety.get_budget_report(),
            "execution": {
                "total_executed": len(self._engine.get_execution_context()["history"]),
                "enabled_skills": self._autonomy.get_available_skills(),
            },
            "violations": self._safety.get_violations(),
        }


def create_orchestrator(
    budget_config: Optional[Dict[str, Any]] = None,
    initial_level: AutonomyLevel = AutonomyLevel.STANDARD,
    dry_run: bool = True,
) -> AutonomyOrchestrator:
    """
    工厂函数 — 创建完整的自主决策编排器
    
    Args:
        budget_config: 安全预算配置（覆盖默认值）
        initial_level: 初始自治等级
        dry_run: 是否为干运行模式（不实际操作硬件）
    """
    # 创建各层组件
    provider = SkillProvider()
    provider.set_dry_run(dry_run)
    
    safety = SafetyEnforcer(budget_config)
    
    engine = DecisionEngine(provider, safety)
    
    autonomy = AutonomyManager(engine)
    if initial_level != AutonomyLevel.STANDARD:
        autonomy.set_level(initial_level, "initial_config")
    
    return AutonomyOrchestrator(provider, safety, engine, autonomy)


# ============================================================
# 集成示例
# ============================================================

if __name__ == "__main__":
    # 配置
    budget_config = {
        "max_actions": 24,
        "max_motion_actions": 12,
        "max_elapsed_seconds": 300.0,
        "max_forward_m": 2.0,
        "max_rotation_deg": 720.0,
    }
    
    print("=" * 60)
    print("  自主决策系统 — 集成演示")
    print("=" * 60)
    
    # 1. 创建编排器
    orch = create_orchestrator(
        budget_config=budget_config,
        initial_level=AutonomyLevel.STANDARD,
        dry_run=True,
    )
    
    # 2. 启动会话
    session = orch.start_session("在房间内搜索红色物体并靠近它", reset=True)
    print(f"\n会话: {session['session_id']}")
    print(f"自治等级: {session['autonomy_level']}")
    
    # 3. Agent System Prompt（Agent 看到的能力目录）
    prompt = orch.get_agent_system_prompt()
    print(f"\nPrompt 长度: {len(prompt)} 字符")
    
    # 4. 模拟 Agent 自主决策序列
    print("\n--- Agent 自主决策序列 ---")
    
    plan = [
        {"skill_id": "capture_frame", "params": {"source": "front_rgb"}},
        {"skill_id": "detect_objects", "params": {"target_classes": "red objects"}},
        {"skill_id": "turn", "params": {"angle_deg": 30}},
        {"skill_id": "move_forward", "params": {"distance_m": 0.3, "speed": 0.5}},
        {"skill_id": "move_arm_preset", "params": {"preset_name": "package_pos"}},
        {"skill_id": "set_gripper", "params": {"side": "both", "action": "close"}},
        {"skill_id": "emergency_stop", "params": {}},
        {"skill_id": "move_forward", "params": {"distance_m": 0.5}},  # 急停后应该被拒绝
    ]
    
    results = orch.execute_plan(plan)
    
    for i, (step, result) in enumerate(zip(plan, results)):
        status_icon = "✓" if result.get("success") else "✗"
        error = f" — {result.get('error', '')}" if not result.get("success") else ""
        print(f"  {i+1}. {status_icon} {step['skill_id']}({step.get('params', {})}){error}")
    
    # 5. 查看状态
    print("\n--- 最终状态 ---")
    status = orch.status()
    metrics = status["autonomy"]
    budget = status["safety"]
    
    print(f"  自治级别: {metrics['level']}")
    print(f"  成功率: {metrics['success_rate']:.1%}")
    print(f"  违规数: {metrics['safety_violations']}")
    print(f"  急停数: {metrics['emergency_stops']}")
    print(f"  预算剩余: {budget['remaining_actions']} 次行动, "
          f"{budget['remaining_forward_m']}m, {budget['remaining_rotation_deg']}°")
    
    print("\n" + "=" * 60)
    print("  演示完成")
    print("=" * 60)
