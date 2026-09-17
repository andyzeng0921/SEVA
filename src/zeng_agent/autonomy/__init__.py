"""
自主决策系统 — 包入口

导出完整的自主决策分层架构：

    ┌───────────────────────────────────────────────────┐
    │                   OpenClaw Agent                  │
    │               (LLM 自主决策引擎)                    │
    │    - 任务优先级判断                                 │
    │    - 多步骤操作规划                                 │
    │    - 环境适应性调整                                 │
    │    - 在能力卡片约束内自主选择技能和参数               │
    └───────────────┬───────────────────────────────────┘
                    │ skill_id + params
    ┌───────────────▼───────────────────────────────────┐
    │            AutonomyManager (自治管理器)            │
    │    - 渐进自治等级 (RESTRICTED/STANDARD/ENHANCED)   │
    │    - 行为质量追踪与反馈回路                         │
    │    - 自动升降级 + 安全降级                          │
    └───────────────┬───────────────────────────────────┘
                    │ 经过自治策略过滤
    ┌───────────────▼───────────────────────────────────┐
    │            DecisionEngine (决策引擎)               │
    │    - 暴露能力卡片目录                               │
    │    - 接收 Agent 决策 → 转发执行                     │
    │    - 返回结构化结果 + 上下文                        │
    └───────────────┬───────────────────────────────────┘
                    │ check() → execute()
    ┌───────────────▼───────────────────────────────────┐
    │           SafetyEnforcer (安全执行层)              │
    │    ★ 不可绕过 ★                                    │
    │    - 参数边界强制校验                               │
    │    - 预算控制 (步数/距离/时间/角度)                   │
    │    - 运行时状态检查 (锁定/急停/碰撞)                 │
    │    - 权限等级执行 (AUTONOMOUS/CONFIRM/FORBIDDEN)   │
    │    - 违规记录与审计                                 │
    └───────────────┬───────────────────────────────────┘
                    │ 允许执行
    ┌───────────────▼───────────────────────────────────┐
    │            SkillProvider (物理技能层)              │
    │    - 感知: capture_frame, detect_objects           │
    │    - 运动: move_forward, turn                      │
    │    - 操作: set_gripper, move_arm_joint             │
    │    - 安全: emergency_stop                          │
    │    - Agent 不可直接访问此层的实现                   │
    └───────────────────────────────────────────────────┘
"""
from .capability_cards import (
    CapabilityCard,
    DecisionAuthority,
    SkillCategory,
    ParamConstraint,
    RuntimeCondition,
    ALL_CAPABILITY_CARDS,
)
from .skill_provider import SkillProvider, SkillResult
from .safety_enforcer import SafetyEnforcer, SessionBudget, ViolationRecord
from .decision_engine import DecisionEngine
from .autonomy_manager import (
    AutonomyManager,
    AutonomyLevel,
    LevelPolicy,
    BehaviorMetrics,
)
from .orchestrator import AutonomyOrchestrator, create_orchestrator
