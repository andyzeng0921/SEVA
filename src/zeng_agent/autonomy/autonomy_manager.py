"""
渐进自治管理器 (AutonomyManager)

管理 Agent 的自主权级别，平衡「自主性」与「稳定性」。

核心机制：

1. 自治等级 — 三级渐进体系
   LEVEL 1: 受限模式 — 新手/高风险环境
   LEVEL 2: 标准模式 — 常规运行
   LEVEL 3: 增强模式 — 高信任度环境

2. 反馈回路 — 基于行为质量自动调整自治级别
   - 成功率追踪
   - 违规计数
   - 效率评估

3. 降级策略 — 出现安全问题时的优雅降级
   - 违规 → 缩小参数范围
   - 频繁失败 → 降自治级别
   - 碰撞/急停 → 立即回到 LEVEL 1

4. 授权范围 — 每个级别对应不同的技能开放度
"""
import time
import logging
from enum import IntEnum
from typing import Any, Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, field

from .capability_cards import (
    CapabilityCard, DecisionAuthority,
    SkillCategory, ALL_CAPABILITY_CARDS,
)
from .decision_engine import DecisionEngine

logger = logging.getLogger(__name__)


class AutonomyLevel(IntEnum):
    """自治等级"""
    RESTRICTED = 1    # 受限：只开放感知 + 急停
    STANDARD = 2      # 标准：开放所有 AUTONOMOUS 技能
    ENHANCED = 3      # 增强：额外开放 SOFT_CONFIRM 技能转为 AUTONOMOUS


@dataclass
class LevelPolicy:
    """各级别的技能开放策略"""
    level: AutonomyLevel
    name: str
    # 允许的技能 ID 集合
    allowed_categories: Set[SkillCategory]
    # 将哪些权限升级为 AUTONOMOUS
    upgrade_to_autonomous: Set[str] = field(default_factory=set)
    # 参数约束缩放（如 Level 1 中前进距离缩小到 50%）
    param_scale: Dict[str, float] = field(default_factory=dict)
    # 最大行动数
    max_actions_override: Optional[int] = None
    # 描述
    description: str = ""


# 三级策略定义
LEVEL_POLICIES: Dict[AutonomyLevel, LevelPolicy] = {
    AutonomyLevel.RESTRICTED: LevelPolicy(
        level=AutonomyLevel.RESTRICTED,
        name="受限模式",
        allowed_categories={
            SkillCategory.PERCEPTION,   # 可以看
            SkillCategory.SAFETY,       # 可以急停
            SkillCategory.META,         # 可以查状态
        },
        param_scale={"distance_m": 0.5, "speed": 0.5},  # 所有前进距离和速度减半
        max_actions_override=12,
        description="仅开放感知和急停能力。适用于初始部署、故障恢复后、高风险环境。",
    ),
    AutonomyLevel.STANDARD: LevelPolicy(
        level=AutonomyLevel.STANDARD,
        name="标准模式",
        allowed_categories={
            SkillCategory.PERCEPTION,
            SkillCategory.LOCOMOTION,     # 可以移动
            SkillCategory.MANIPULATION,   # 可以操作
            SkillCategory.ARM_MOTION,     # 可以手臂
            SkillCategory.SAFETY,
            SkillCategory.META,
        },
        max_actions_override=36,
        description="标准运行模式。Agent 可自主使用大部分技能，手臂运动记录日志。",
    ),
    AutonomyLevel.ENHANCED: LevelPolicy(
        level=AutonomyLevel.ENHANCED,
        name="增强模式",
        allowed_categories={
            SkillCategory.PERCEPTION,
            SkillCategory.LOCOMOTION,
            SkillCategory.MANIPULATION,
            SkillCategory.ARM_MOTION,
            SkillCategory.SAFETY,
            SkillCategory.META,
        },
        upgrade_to_autonomous={"move_arm_joint"},  # 关节运动从 SOFT_CONFIRM 升级为 AUTONOMOUS
        max_actions_override=48,
        description="增强自主模式。手臂关节运动也由 Agent 自主决定，适用于已充分验证的环境。",
    ),
}


@dataclass
class BehaviorMetrics:
    """行为质量指标"""
    total_attempts: int = 0
    successful: int = 0
    safety_violations: int = 0
    consecutive_failures: int = 0
    emergency_stops: int = 0
    avg_execution_time_ms: float = 0.0
    
    # 时间窗口内的统计（滑动窗口）
    window_attempts: int = 0
    window_successes: int = 0
    window_size: int = 10  # 最近 N 次行动
    
    @property
    def success_rate(self) -> float:
        if self.total_attempts == 0:
            return 1.0
        return self.successful / self.total_attempts
    
    @property
    def recent_success_rate(self) -> float:
        if self.window_attempts == 0:
            return 1.0
        return self.window_successes / self.window_attempts


class AutonomyManager:
    """
    渐进自治管理器
    
    职责：
    1. 根据当前自治级别控制技能开放范围
    2. 追踪 Agent 的行为质量
    3. 基于反馈回路自动升降自治级别
    4. 产生安全降级事件
    """

    def __init__(self, decision_engine: DecisionEngine):
        self._engine = decision_engine
        self._current_level = AutonomyLevel.STANDARD
        self._metrics = BehaviorMetrics()
        
        # 动作结果窗口（用于计算近期成功率）
        self._recent_results: List[bool] = []
        
        # 降级事件日志
        self._degradation_events: List[Dict[str, Any]] = []
        
        # 升级冷却：升级后至少保持 N 秒才能再次调整
        self._last_level_change = time.time()
        self._level_cooldown_seconds = 30.0
        
        # 策略阈值
        self._upgrade_threshold = 0.85      # 成功率 > 85% 才考虑升级
        self._downgrade_threshold = 0.50    # 成功率 < 50% 考虑降级
        self._max_consecutive_failures = 5  # 连续失败 N 次强制降级
        self._max_violations = 3            # N 次违规强制降级
        
        # 应用初始策略
        self._apply_level_policy()

    # ========== 自治级别控制 ==========

    def set_level(self, level: AutonomyLevel, reason: str = "") -> Dict[str, Any]:
        """手动设置自治级别"""
        if level == self._current_level:
            return {"changed": False, "level": self._current_level.name}
        
        now = time.time()
        if now - self._last_level_change < self._level_cooldown_seconds:
            return {
                "changed": False,
                "level": self._current_level.name,
                "reason": f"冷却中 ({self._level_cooldown_seconds - (now - self._last_level_change):.0f}s)",
            }
        
        old = self._current_level
        self._current_level = level
        self._last_level_change = now
        self._apply_level_policy()
        
        logger.info(f"[AUTONOMY] 级别变更: {old.name} → {level.name} ({reason})")
        
        return {
            "changed": True,
            "from": old.name,
            "to": level.name,
            "reason": reason,
            "policy": LEVEL_POLICIES[level].description,
        }

    def _apply_level_policy(self) -> None:
        """应用当前自治级别的策略到决策引擎"""
        policy = LEVEL_POLICIES[self._current_level]
        
        # 1. 根据类别启用/禁用技能
        all_cards = ALL_CAPABILITY_CARDS
        for skill_id, card in all_cards.items():
            if card.category in policy.allowed_categories:
                self._engine.enable_skill(skill_id)
                # 恢复卡片默认权限
                self._engine.set_skill_authority(skill_id, card.authority)
            else:
                self._engine.disable_skill(skill_id)
        
        # 2. 对指定技能升级权限
        for skill_id in policy.upgrade_to_autonomous:
            self._engine.set_skill_authority(skill_id, DecisionAuthority.AUTONOMOUS)

    def get_current_level(self) -> AutonomyLevel:
        return self._current_level

    def get_level_description(self) -> str:
        policy = LEVEL_POLICIES[self._current_level]
        return policy.description

    # ========== 反馈回路 ==========

    def record_outcome(self, success: bool, violation: bool = False,
                       emergency_stop: bool = False, exec_time_ms: float = 0.0) -> None:
        """
        记录每次技能执行的结果
        
        这些数据驱动自治级别的自动调整。
        """
        m = self._metrics
        m.total_attempts += 1
        if success:
            m.successful += 1
            m.consecutive_failures = 0
        else:
            m.consecutive_failures += 1
        
        if violation:
            m.safety_violations += 1
        
        if emergency_stop:
            m.emergency_stops += 1
        
        # 更新执行时间均值（指数移动平均）
        if m.total_attempts == 1:
            m.avg_execution_time_ms = exec_time_ms
        else:
            m.avg_execution_time_ms = m.avg_execution_time_ms * 0.9 + exec_time_ms * 0.1
        
        # 滑动窗口
        self._recent_results.append(success)
        if len(self._recent_results) > m.window_size:
            self._recent_results.pop(0)
        m.window_attempts = len(self._recent_results)
        m.window_successes = sum(self._recent_results)
        
        # 自动评估是否需要调整级别
        self._evaluate_auto_adjustment()

    def _evaluate_auto_adjustment(self) -> Optional[Dict[str, Any]]:
        """评估是否自动调整自治级别"""
        m = self._metrics
        
        # 强制降级条件（优先级最高）
        if m.emergency_stops > 0:
            return self.set_level(
                AutonomyLevel.RESTRICTED,
                f"检测到 {m.emergency_stops} 次急停，强制降级到受限模式"
            )
        
        if m.safety_violations >= self._max_violations:
            return self.set_level(
                AutonomyLevel.RESTRICTED,
                f"安全违规达 {m.safety_violations} 次 (上限 {self._max_violations})，强制降级"
            )
        
        if m.consecutive_failures >= self._max_consecutive_failures:
            return self.set_level(
                AutonomyLevel.RESTRICTED,
                f"连续失败 {m.consecutive_failures} 次，降级到受限模式等待人工介入"
            )
        
        # 渐进调整
        if self._current_level == AutonomyLevel.RESTRICTED:
            # 受限模式 → 恢复条件：无违规 + 近期成功率 100%
            if m.safety_violations == 0 and m.recent_success_rate >= 1.0 and m.window_attempts >= 5:
                return self.set_level(
                    AutonomyLevel.STANDARD,
                    "近期无违规且成功率 100%，恢复到标准模式"
                )
        
        elif self._current_level == AutonomyLevel.STANDARD:
            # 标准 → 降级
            if m.recent_success_rate < self._downgrade_threshold and m.window_attempts >= 5:
                return self.set_level(
                    AutonomyLevel.RESTRICTED,
                    f"近期成功率 {m.recent_success_rate:.0%} < {self._downgrade_threshold:.0%}，降级"
                )
            # 标准 → 增强
            if m.recent_success_rate >= self._upgrade_threshold and m.window_attempts >= 10:
                return self.set_level(
                    AutonomyLevel.ENHANCED,
                    f"近期成功率 {m.recent_success_rate:.0%} ≥ {self._upgrade_threshold:.0%}，升级到增强模式"
                )
        
        elif self._current_level == AutonomyLevel.ENHANCED:
            # 增强 → 降级
            if m.recent_success_rate < self._downgrade_threshold and m.window_attempts >= 5:
                return self.set_level(
                    AutonomyLevel.STANDARD,
                    f"近期成功率 {m.recent_success_rate:.0%} < {self._downgrade_threshold:.0%}，降级到标准模式"
                )
        
        return None

    # ========== 能力卡片权限查询 ==========

    def get_available_skills(self) -> List[str]:
        """获取当前可用的技能列表（受自治级别限制）"""
        policy = LEVEL_POLICIES[self._current_level]
        return [
            sid for sid, card in ALL_CAPABILITY_CARDS.items()
            if card.category in policy.allowed_categories
        ]

    def get_param_scale(self, skill_id: str, param_name: str) -> float:
        """
        获取当前自治级别的参数缩放因子
        
        例如：RESTRICTED 模式下 distance_m 缩放 0.5
        """
        policy = LEVEL_POLICIES[self._current_level]
        return policy.param_scale.get(param_name, 1.0)

    # ========== 反馈信息生成 ==========

    def generate_autonomy_context(self) -> str:
        """
        生成注入 Agent 提示词的自治上下文
        
        这告诉 Agent:
        - 当前自治等级意味着什么
        - 它有多大自由度
        - 如果有降级，原因是什么
        """
        m = self._metrics
        policy = LEVEL_POLICIES[self._current_level]
        
        lines = [
            f"## 当前自治等级: {policy.name} (Level {self._current_level})",
            "",
            policy.description,
            "",
            f"可用技能类别: {', '.join(c.name for c in policy.allowed_categories)}",
            "",
            "### 行为质量",
            f"- 总操作次数: {m.total_attempts}",
            f"- 成功率: {m.success_rate:.1%}",
            f"- 近期成功率(最近{m.window_attempts}次): {m.recent_success_rate:.1%}",
            f"- 安全违规: {m.safety_violations}",
            f"- 连续失败: {m.consecutive_failures}",
            "",
        ]

        if self._degradation_events:
            lines.append("### 降级事件记录")
            for evt in self._degradation_events[-3:]:
                lines.append(f"- {evt['timestamp']}: {evt['from']} → {evt['to']} ({evt['reason']})")
            lines.append("")

        if self._current_level != AutonomyLevel.ENHANCED:
            lines.append("### 升级条件")
            if self._current_level == AutonomyLevel.RESTRICTED:
                lines.append(f"- 连续 5 次无违规且全部成功 → 升级到标准模式")
            elif self._current_level == AutonomyLevel.STANDARD:
                lines.append(f"- 近期成功率 ≥ {self._upgrade_threshold:.0%} 超过 10 次 → 升级到增强模式")
            lines.append("")

        return "\n".join(lines)

    def get_metrics(self) -> Dict[str, Any]:
        """获取当前行为指标"""
        m = self._metrics
        return {
            "level": self._current_level.name,
            "level_description": LEVEL_POLICIES[self._current_level].description,
            "total_attempts": m.total_attempts,
            "successful": m.successful,
            "success_rate": round(m.success_rate, 3),
            "recent_success_rate": round(m.recent_success_rate, 3),
            "safety_violations": m.safety_violations,
            "consecutive_failures": m.consecutive_failures,
            "emergency_stops": m.emergency_stops,
            "avg_execution_time_ms": round(m.avg_execution_time_ms, 1),
            "available_skills": self.get_available_skills(),
        }
