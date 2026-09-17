"""
安全执行层 (SafetyEnforcer)

运行时守护 — 位于决策引擎和物理技能之间，不可绕过。

职责：
1. 参数边界强制校验（即使 Agent 失误也不会执行危险操作）
2. 调用频率/预算控制（防止 Agent 无限制调用）
3. 运行时状态检查（锁定、急停、碰撞）
4. 权限等级执行（AUTONOMOUS/SOFT_CONFIRM/HARD_CONFIRM/FORBIDDEN）
5. 审计日志（记录每一次 Agent 触发的物理操作）

安全层不参与决策 — 它只做"允许/拒绝"的二元判断。
"""
import time
import logging
import threading
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field

from .capability_cards import (
    CapabilityCard, DecisionAuthority, ALL_CAPABILITY_CARDS,
)

logger = logging.getLogger(__name__)


@dataclass
class ViolationRecord:
    """违规记录"""
    timestamp: float
    skill_id: str
    reason: str
    params: Dict[str, Any]
    authority_level: str


@dataclass
class SessionBudget:
    """会话预算追踪"""
    max_actions: int = 36
    max_motion_actions: int = 24
    max_elapsed_seconds: float = 600.0
    max_forward_m: float = 3.0
    max_rotation_deg: float = 1080.0
    
    # 已使用量
    total_actions: int = 0
    motion_actions: int = 0
    start_time: float = field(default_factory=time.time)
    forward_m: float = 0.0
    rotation_deg: float = 0.0


class SafetyEnforcer:
    """
    安全执行层
    
    每个物理技能执行前，此层执行以下检查：
    1. 技能是否存在
    2. 参数是否合法
    3. 预算是否充足
    4. 前置条件是否满足
    5. 权限是否匹配
    6. 冷却时间是否满足
    """

    def __init__(self, budget_config: Optional[Dict[str, Any]] = None):
        self._budget = SessionBudget(
            **(budget_config or {})
        )
        self._violations: List[ViolationRecord] = []
        self._last_invocation: Dict[str, float] = {}  # skill_id -> timestamp
        self._invocation_counts: Dict[str, int] = {}   # skill_id -> count
        self._lock = threading.Lock()
        
        # 外部状态引用（由 orchestrator 更新）
        self._is_system_locked = False
        self._is_estopped = False
        self._collision_detected = False
        self._battery_level = 100.0
        
        # 需要人工确认的待处理请求
        self._pending_confirmations: Dict[str, dict] = {}

    # ========== 状态更新接口 ==========

    def update_system_state(self, *, locked: bool = None, estopped: bool = None,
                            collision: bool = None, battery: float = None) -> None:
        """更新系统状态（由底层硬件回调触发）"""
        if locked is not None:
            self._is_system_locked = locked
        if estopped is not None:
            self._is_estopped = estopped
        if collision is not None:
            self._collision_detected = collision
        if battery is not None:
            self._battery_level = battery

    # ========== 核心校验方法 ==========

    def check(self, skill_id: str, params: Dict[str, Any],
              force_confirm: bool = False) -> Tuple[bool, str, Dict[str, Any]]:
        """
        执行完整的安全校验
        
        Returns:
            (allowed, reason, audit_data)
        """
        with self._lock:
            # 1. 技能存在性检查
            card = ALL_CAPABILITY_CARDS.get(skill_id)
            if card is None:
                return False, f"未知技能: {skill_id}", {}

            # 2. 系统级安全状态检查（优先级最高）
            if skill_id != "emergency_stop":
                if self._is_estopped:
                    return False, "系统处于急停状态，仅允许 emergency_stop", {}
                if self._is_system_locked and skill_id not in ("get_status", "emergency_stop"):
                    return False, "系统已锁定，仅允许状态查询和急停", {}
                if self._collision_detected and card.category.value >= 1:  # 运动/操作类
                    return False, "检测到碰撞，运动类技能已封锁", {}

            # 3. 运行时前置条件检查
            for precond in card.preconditions:
                if precond.condition_type == "not_locked" and self._is_system_locked:
                    return False, f"前置条件不满足: {precond.description}", {}
                if precond.condition_type == "not_estopped" and self._is_estopped:
                    return False, f"前置条件不满足: {precond.description}", {}
                if precond.condition_type == "battery_ok" and self._battery_level < 20:
                    return False, f"电池电量不足 ({self._battery_level}%)", {}

            # 4. 参数合法性检查
            param_ok, param_errors = card.validate_params(params)
            if not param_ok:
                return False, f"参数非法: {'; '.join(param_errors)}", {}

            # 5. 预算检查
            budget_ok, budget_reason = self._check_budget(card, params)
            if not budget_ok:
                return False, budget_reason, {}

            # 6. 调用频率/冷却检查
            if skill_id in self._last_invocation:
                elapsed = time.time() - self._last_invocation[skill_id]
                if elapsed < card.cooldown_seconds:
                    return False, f"冷却中，还需等待 {card.cooldown_seconds - elapsed:.1f}s", {}

            # 7. 调用次数上限检查
            count = self._invocation_counts.get(skill_id, 0)
            if card.max_invocations_per_session > 0 and count >= card.max_invocations_per_session:
                return False, f"技能 '{skill_id}' 已达调用上限 ({card.max_invocations_per_session})", {}

            # 8. 权限等级处理
            if card.authority == DecisionAuthority.FORBIDDEN:
                return False, f"技能 '{skill_id}' 禁止 Agent 直接调用", {}
            
            if card.authority == DecisionAuthority.HARD_CONFIRM:
                return False, f"技能 '{skill_id}' 需要外部确认后才能执行", {
                    "requires_confirmation": True,
                    "skill_id": skill_id,
                    "params": params,
                }

            if card.authority == DecisionAuthority.AUTONOMOUS:
                audit_tag = "AUTONOMOUS"
            else:
                audit_tag = "SOFT_CONFIRM"
                logger.info(f"[SOFT_CONFIRM] Agent 自主调用 {skill_id} with {params}")

            # 通过所有检查
            audit_data = {
                "skill_id": skill_id,
                "params": params,
                "authority": audit_tag,
                "timestamp": time.time(),
            }
            return True, "OK", audit_data

    def record_invocation(self, skill_id: str, params: Dict[str, Any],
                          success: bool) -> None:
        """技能执行后记录（由 orchestrator 调用）"""
        with self._lock:
            self._last_invocation[skill_id] = time.time()
            self._invocation_counts[skill_id] = self._invocation_counts.get(skill_id, 0) + 1
            
            card = ALL_CAPABILITY_CARDS.get(skill_id)
            if card:
                self._budget.total_actions += 1
                if card.category.value in (1, 3, 4):  # 运动类
                    self._budget.motion_actions += 1
                
                # 追踪物理运动量
                if skill_id == "move_forward":
                    self._budget.forward_m += params.get("distance_m", 0)
                elif skill_id == "turn":
                    self._budget.rotation_deg += abs(params.get("angle_deg", 0))

    def record_violation(self, skill_id: str, reason: str, params: Dict[str, Any]) -> None:
        """记录违规"""
        violation = ViolationRecord(
            timestamp=time.time(),
            skill_id=skill_id,
            reason=reason,
            params=params,
            authority_level="VIOLATION",
        )
        self._violations.append(violation)
        logger.warning(f"[SAFETY_VIOLATION] {skill_id}: {reason}")

    def _check_budget(self, card, params: Dict[str, Any]) -> Tuple[bool, str]:
        """检查预算"""
        b = self._budget
        
        # 总行动数
        if b.total_actions >= b.max_actions:
            return False, f"总行动数已达上限 ({b.max_actions})"
        
        # 运动行动数
        if card.category.value in (1, 3, 4):
            if b.motion_actions >= b.max_motion_actions:
                return False, f"运动行动数已达上限 ({b.max_motion_actions})"
        
        # 时间预算
        elapsed = time.time() - b.start_time
        if elapsed > b.max_elapsed_seconds:
            return False, f"会话超时 ({elapsed:.0f}s > {b.max_elapsed_seconds}s)"
        
        # 前进距离预算
        d = params.get("distance_m", 0)
        if b.forward_m + d > b.max_forward_m:
            return False, f"前进总距离将超过上限 ({b.max_forward_m}m)"
        
        # 旋转角度预算
        r = abs(params.get("angle_deg", 0))
        if b.rotation_deg + r > b.max_rotation_deg:
            return False, f"旋转总角度将超过上限 ({b.max_rotation_deg}°)"
        
        return True, ""

    # ========== 状态报告 ==========

    def get_budget_report(self) -> Dict[str, Any]:
        """获取预算使用报告"""
        b = self._budget
        elapsed = time.time() - b.start_time
        return {
            "total_actions": f"{b.total_actions}/{b.max_actions}",
            "motion_actions": f"{b.motion_actions}/{b.max_motion_actions}",
            "elapsed_seconds": f"{elapsed:.1f}/{b.max_elapsed_seconds}",
            "forward_m": f"{b.forward_m:.2f}/{b.max_forward_m}",
            "rotation_deg": f"{b.rotation_deg:.0f}/{b.max_rotation_deg}",
            "violations": len(self._violations),
            "remaining_actions": b.max_actions - b.total_actions,
            "remaining_forward_m": round(b.max_forward_m - b.forward_m, 2),
            "remaining_rotation_deg": round(b.max_rotation_deg - b.rotation_deg, 0),
            "is_locked": self._is_system_locked,
            "is_estopped": self._is_estopped,
        }

    def get_violations(self) -> List[Dict[str, Any]]:
        """获取违规记录"""
        return [{
            "timestamp": v.timestamp,
            "skill_id": v.skill_id,
            "reason": v.reason,
            "params": v.params,
        } for v in self._violations]

    def generate_safety_context_for_agent(self) -> str:
        """
        生成注入到 Agent 提示词中的安全上下文
        
        这告诉 Agent 它当前的约束边界，帮助它做更好的自主决策。
        """
        b = self._budget
        elapsed = time.time() - b.start_time
        
        lines = [
            "## 当前安全边界与预算",
            "",
            "你可以在以下约束内自主决策：",
            "",
            f"- 剩余行动次数: {b.max_actions - b.total_actions} / {b.max_actions}",
            f"- 剩余运动次数: {b.max_motion_actions - b.motion_actions} / {b.max_motion_actions}",
            f"- 剩余时间: {max(0, b.max_elapsed_seconds - elapsed):.0f}s / {b.max_elapsed_seconds}s",
            f"- 剩余前进距离: {b.max_forward_m - b.forward_m:.2f}m / {b.max_forward_m}m",
            f"- 剩余旋转角度: {b.max_rotation_deg - b.rotation_deg:.0f}° / {b.max_rotation_deg}°",
            "",
            "安全规则（不可违反）:",
            "- 参数必须在能力卡片定义的范围内",
            "- 系统锁定时只能查询状态或急停",
            "- 检测到碰撞时自动封锁运动",
            "- 总步数/距离/时间达到上限时自动终止",
            "",
        ]

        if self._violations:
            lines.append("### 历史违规记录")
            for v in self._violations[-3:]:  # 最近3条
                lines.append(f"- {v.skill_id}: {v.reason}")
            lines.append("")

        return "\n".join(lines)
