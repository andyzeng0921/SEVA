"""
自主决策引擎接口 (DecisionEngine)

这是 OpenClaw Agent 与机器人物理技能之间的唯一桥梁。
Agent 的每一次自主决策都通过这个入口执行，经过 SafetyEnforcer 校验后透传到 SkillProvider。

职责：
1. 暴露能力卡片目录供 Agent 发现可用技能
2. 接收 Agent 的决策指令（skill_id + params）
3. 通过 SafetyEnforcer 校验
4. 执行经过校验的技能
5. 将结果 + 安全上下文反馈给 Agent

Agent 不能绕过此引擎直接访问 SkillProvider 或硬件。
"""
import json
import time
import logging
from typing import Any, Dict, List, Optional, Callable

from .capability_cards import (
    CapabilityCard, DecisionAuthority, ALL_CAPABILITY_CARDS,
)
from .safety_enforcer import SafetyEnforcer
from .skill_provider import SkillProvider, SkillResult

logger = logging.getLogger(__name__)


class DecisionEngine:
    """
    自主决策引擎
    
    架构位:
    ┌──────────────────┐
    │  OpenClaw Agent  │  LLM 自主决策
    └────────┬─────────┘
             │ skill_id + params
    ┌────────▼─────────┐
    │  DecisionEngine  │  ← 你在这里
    └────────┬─────────┘
             │ 校验 → 执行
    ┌────────▼─────────┐
    │ SafetyEnforcer   │  不可绕过
    └────────┬─────────┘
             │ 允许
    ┌────────▼─────────┐
    │  SkillProvider   │  物理执行
    └──────────────────┘
    """

    def __init__(self, skill_provider: SkillProvider,
                 safety_enforcer: SafetyEnforcer,
                 on_confirm_needed: Optional[Callable] = None):
        self._provider = skill_provider
        self._safety = safety_enforcer
        self._on_confirm_needed = on_confirm_needed  # 外部确认回调
        
        # 技能执行历史
        self._execution_history: List[Dict[str, Any]] = []
        
        # 已注册的能力卡片（由 autonomy manager 配置）
        self._enabled_cards: Dict[str, CapabilityCard] = {}
        self._restore_default_cards()

    def _restore_default_cards(self) -> None:
        """恢复默认能力卡片集"""
        self._enabled_cards = dict(ALL_CAPABILITY_CARDS)

    def enable_skill(self, skill_id: str) -> bool:
        """启用某个技能"""
        if skill_id in ALL_CAPABILITY_CARDS:
            self._enabled_cards[skill_id] = ALL_CAPABILITY_CARDS[skill_id]
            return True
        return False

    def disable_skill(self, skill_id: str) -> bool:
        """禁用某个技能（即使 Agent 请求也会被拒绝）"""
        self._enabled_cards.pop(skill_id, None)
        return True

    def set_skill_authority(self, skill_id: str, authority: DecisionAuthority) -> bool:
        """动态调整技能的决策权限"""
        if skill_id in self._enabled_cards:
            self._enabled_cards[skill_id].authority = authority
            return True
        return False

    # ========== Agent 接口 ==========

    def get_capabilities_prompt(self) -> str:
        """
        Agent 发现可用能力
        
        返回一个注入到 Agent System Prompt 中的文本，
        描述 Agent 可以使用哪些技能及其约束。
        Agent 据此自主决定选择哪个技能、用什么参数。
        """
        lines = ["# 可用物理技能", ""]
        lines.append("你是机器人的自主决策引擎。你可以自主选择以下技能完成任务。")
        lines.append("每个技能有明确的参数约束和安全边界，你必须在约束内决策。")
        lines.append("")

        for card in self._enabled_cards.values():
            authority_tag = {
                DecisionAuthority.AUTONOMOUS: " [自主]",
                DecisionAuthority.SOFT_CONFIRM: " [自主·记录]",
                DecisionAuthority.HARD_CONFIRM: " [需确认]",
            }.get(card.authority, "")

            lines.append(f"## {card.name}{authority_tag}")
            lines.append(f"- ID: `{card.skill_id}`")
            lines.append(f"- 类别: {card.category.name}")
            lines.append(f"- 描述: {card.description}")

            if card.parameters:
                lines.append("- 参数:")
                for p in card.parameters:
                    constraint = ""
                    if p.min_value is not None and p.max_value is not None:
                        constraint = f", 范围 [{p.min_value}, {p.max_value}]"
                    elif p.allowed_values:
                        constraint = f", 可选 [{', '.join(str(v) for v in p.allowed_values)}]"
                    def_val = f", 默认 {p.default_value}" if p.default_value is not None else ""
                    lines.append(f"  - {p.name} ({p.param_type}): {p.description}{constraint}{def_val}")

            lines.append(f"- 调用限制: 最多 {card.max_invocations_per_session or '无限'} 次/session")
            if card.side_effects:
                lines.append(f"- 副作用: {card.side_effects}")
            lines.append("")
        
        lines.append("## 决策格式")
        lines.append("当你决定执行物理操作时，使用以下格式：")
        lines.append('```json')
        lines.append('{"skill_id": "move_forward", "params": {"distance_m": 0.3, "speed": 0.5}}')
        lines.append('```')
        lines.append("")

        return "\n".join(lines)

    def execute(self, skill_id: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Agent 通过此方法执行物理技能。
        """
        t_start = time.time()

        # 0. 检查技能是否启用
        if skill_id not in self._enabled_cards:
            self._safety.record_violation(skill_id, f"技能 '{skill_id}' 未启用", params)
            return {
                "success": False,
                "skill_id": skill_id,
                "error": f"技能 '{skill_id}' 未在当前自治级别启用",
                "available_skills": list(self._enabled_cards.keys()),
            }

        card = self._enabled_cards[skill_id]

        # 0.5 填充默认参数值
        filled_params = dict(params)
        for p in card.parameters:
            if p.name not in filled_params and p.default_value is not None:
                filled_params[p.name] = p.default_value

        # 1. 安全检查
        allowed, reason, audit_data = self._safety.check(skill_id, filled_params)
        if not allowed:
            self._safety.record_violation(skill_id, reason, filled_params)
            return {
                "success": False,
                "skill_id": skill_id,
                "error": reason,
                "safety_blocked": True,
            }

        # 2. 执行物理技能
        skill_method = getattr(self._provider, skill_id, None)
        
        if skill_method is None:
            return {
                "success": False,
                "skill_id": skill_id,
                "error": f"技能 '{skill_id}' 已注册但未实现",
            }

        try:
            result: SkillResult = skill_method(**filled_params)
        except Exception as e:
            logger.exception(f"技能 '{skill_id}' 执行异常")
            if card.emergency_stop_on_failure:
                self._provider.emergency_stop()
            return {
                "success": False,
                "skill_id": skill_id,
                "error": f"执行异常: {e}",
                "emergency_stopped": card.emergency_stop_on_failure,
            }

        # 3. 记录执行
        self._safety.record_invocation(skill_id, filled_params, result.success)
        self._execution_history.append({
            "skill_id": skill_id,
            "params": filled_params,
            "success": result.success,
            "timestamp": time.time(),
            "execution_time_ms": result.execution_time_ms,
        })

        # 4. 构建返回结果（附带安全上下文，帮助 Agent 决策）
        response = result.to_dict()
        response["budget"] = self._safety.get_budget_report()
        response["hint"] = self._build_hint(skill_id, result)

        elapsed = (time.time() - t_start) * 1000
        logger.info(
            f"[DECISION] {skill_id}({params}) → "
            f"{'OK' if result.success else 'FAIL'} in {elapsed:.0f}ms"
        )

        return response

    def execute_batch(self, actions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        批量执行多个技能（Agent 自主规划的步骤序列）
        
        每个步骤独立校验，任一步骤失败时可根据配置决定是否继续。
        """
        results = []
        for action in actions:
            result = self.execute(action["skill_id"], action.get("params", {}))
            results.append(result)
            # 如果触发急停，停止后续执行
            if result.get("emergency_stopped"):
                results.append({
                    "success": False,
                    "skill_id": "batch_aborted",
                    "error": "前序步骤触发急停，批次中断",
                })
                break
            if not result["success"]:
                # 失败不中断 — Agent 根据结果自主决定是否继续
                pass
        return results

    def get_execution_context(self) -> Dict[str, Any]:
        """
        获取当前执行上下文供 Agent 参考
        
        Agent 在规划时应该看到这些信息：
        - 已执行了什么
        - 还剩多少预算
        - 有什么约束变化
        """
        return {
            "history": self._execution_history[-10:],  # 最近10步
            "budget": self._safety.get_budget_report(),
            "enabled_skills": list(self._enabled_cards.keys()),
            "total_executed": len(self._execution_history),
        }

    def _build_hint(self, skill_id: str, result: SkillResult) -> str:
        """为 Agent 生成下一步提示"""
        if not result.success:
            return f"'{skill_id}' 执行失败: {result.error}。考虑替代方案或重试。"
        
        hints = {
            "capture_frame": "已获取新画面，可进行目标检测或分析场景。",
            "detect_objects": f"检测到 {result.data.get('count', 0)} 个物体。如有目标，可考虑导航。",
            "move_forward": f"已前进 {result.data.get('distance_m', 0)}m。检查是否需要对齐或再次观察。",
            "turn": f"已转向 {result.data.get('angle_deg', 0)}°。检查目标是否在视野中心。",
            "set_gripper": "夹爪操作完成。如需抓取，确认物体位置。",
            "move_arm_preset": f"手臂已移动到 {result.data.get('preset_name', '?')} 姿态。",
        }
        return hints.get(skill_id, f"'{skill_id}' 执行成功。根据任务目标决定下一步。")
