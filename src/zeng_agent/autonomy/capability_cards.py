"""
决策权限边界定义 — 能力卡片模型 (Capability Card)

定义了"物理技能"与"自主决策"之间的清晰边界：

┌─────────────────────────────────────────────────────┐
│  自主决策层 (OpenClaw Agent 自主决定)                 │
│  - 选择哪个技能、按什么顺序                            │
│  - 在安全参数范围内选择具体值                           │
│  - 多步骤规划、任务分解、重试策略                      │
└────────────┬────────────────────────────────────────┘
             │ 通过能力卡片接口调用
┌────────────▼────────────────────────────────────────┐
│  物理技能层 (系统控制，Agent 不可修改)                  │
│  - 抓取力度上限、关节运动范围                           │
│  - 碰撞检测、异常停止                                  │
│  - 传感器数据获取                                     │
└─────────────────────────────────────────────────────┘

每一张「能力卡片」定义了：
  1. 技能是什么（语义描述，供 LLM 理解）
  2. 参数约束（范围、类型、条件）
  3. 执行前置条件（什么状态下才能调用）
  4. 决策权限等级（Agent 自由决定 / 需确认 / 禁止）
"""
from enum import IntEnum
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Callable, Union


class DecisionAuthority(IntEnum):
    """决策权限等级"""
    AUTONOMOUS = 0       # Agent 自主决定，无需确认
    SOFT_CONFIRM = 1     # Agent 可自主决定，但系统会记录日志
    HARD_CONFIRM = 2     # 必须外部确认（人工或更高层系统）
    FORBIDDEN = 3        # Agent 不可直接调用，只能由系统内部触发


class SkillCategory(IntEnum):
    """技能类别"""
    PERCEPTION = 0       # 感知（相机、传感器读数）
    LOCOMOTION = 1       # 运动（前进、转向、停止）
    MANIPULATION = 2     # 操作（抓取、放置、夹爪控制）
    ARM_MOTION = 3       # 手臂运动（关节控制、姿态设定）
    NAVIGATION = 4       # 导航（路径规划、航点）
    SAFETY = 5           # 安全（急停、限位检查）
    META = 6             # 元操作（查询状态、预算）


@dataclass
class ParamConstraint:
    """参数约束定义"""
    name: str
    param_type: str                # int, float, bool, str, enum
    description: str               # 语义描述
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    allowed_values: Optional[List[Any]] = None
    default_value: Optional[Any] = None
    required: bool = True
    
    def validate(self, value: Any) -> tuple[bool, str]:
        """验证参数值是否合法"""
        if value is None:
            if self.default_value is not None:
                return True, ""
            if self.required:
                return False, f"参数 '{self.name}' 为必填项"
            return True, ""
        
        if self.param_type == "float":
            try:
                value = float(value)
            except (TypeError, ValueError):
                return False, f"参数 '{self.name}' 应为浮点数，实际: {type(value).__name__}"
            if self.min_value is not None and value < self.min_value:
                return False, f"参数 '{self.name}'={value} 小于最小值 {self.min_value}"
            if self.max_value is not None and value > self.max_value:
                return False, f"参数 '{self.name}'={value} 大于最大值 {self.max_value}"
        
        elif self.param_type == "int":
            try:
                value = int(value)
            except (TypeError, ValueError):
                return False, f"参数 '{self.name}' 应为整数，实际: {type(value).__name__}"
            if self.min_value is not None and value < self.min_value:
                return False, f"参数 '{self.name}'={value} 小于最小值 {int(self.min_value)}"
            if self.max_value is not None and value > self.max_value:
                return False, f"参数 '{self.name}'={value} 大于最大值 {int(self.max_value)}"
        
        elif self.param_type == "enum":
            if self.allowed_values and value not in self.allowed_values:
                return False, f"参数 '{self.name}'='{value}' 不在允许值: {self.allowed_values}"
        
        elif self.param_type == "bool":
            if not isinstance(value, bool):
                return False, f"参数 '{self.name}' 应为布尔值"
        
        return True, ""


@dataclass
class RuntimeCondition:
    """运行时前置条件"""
    condition_type: str           # joint_range, force_ok, battery_ok, not_locked, not_estopped
    threshold: Optional[float] = None
    joint_id: Optional[int] = None
    description: str = ""


@dataclass
class CapabilityCard:
    """
    能力卡片 — 定义一项物理技能的完整契约
    
    Agent 看到这个卡片后，自主决定：
    - 是否使用这个技能
    - 在约束范围内选择什么参数
    - 何时调用、调用多少次
    
    Agent 不能：
    - 修改参数约束
    - 绕过前置条件
    - 修改权限等级
    """
    skill_id: str                              # 唯一标识
    name: str                                  # 人类可读名称
    description: str                           # LLM 可理解的语义描述
    category: SkillCategory                    # 技能类别
    authority: DecisionAuthority               # 决策权限等级
    
    # 参数定义
    parameters: List[ParamConstraint] = field(default_factory=list)
    
    # 执行约束
    preconditions: List[RuntimeCondition] = field(default_factory=list)
    max_invocations_per_session: int = 0       # 每个 session 最大调用次数，0=无限制
    cooldown_seconds: float = 0.0              # 两次调用间最小间隔
    
    # 副作用说明
    side_effects: str = ""                     # 对物理世界的影响描述
    
    # 安全边界
    emergency_stop_on_failure: bool = False    # 失败时是否触发急停
    
    def to_llm_description(self) -> str:
        """生成供 LLM Agent 理解的能力描述"""
        params_desc = []
        for p in self.parameters:
            constraint = ""
            if p.min_value is not None and p.max_value is not None:
                constraint = f" 范围 [{p.min_value}, {p.max_value}]"
            elif p.allowed_values:
                constraint = f" 可选值: {p.allowed_values}"
            params_desc.append(f"    - {p.name} ({p.param_type}): {p.description}{constraint}")
        
        authority_desc = {
            DecisionAuthority.AUTONOMOUS: "可自主决定",
            DecisionAuthority.SOFT_CONFIRM: "系统会记录日志",
            DecisionAuthority.HARD_CONFIRM: "需要外部确认",
        }.get(self.authority, "")
        
        precond = ""
        if self.preconditions:
            precond = "\n  前置条件:\n" + "\n".join(
                f"    - {p.description}" for p in self.preconditions
            )
        
        return (
            f"技能: {self.name} ({self.skill_id})\n"
            f"类别: {self.category.name} | 权限: {authority_desc}\n"
            f"描述: {self.description}\n"
            f"参数:\n" + ("\n".join(params_desc) if params_desc else "    无参数") +
            f"{precond}\n"
            f"副作用: {self.side_effects or '无'}\n"
            f"调用限制: 最多 {self.max_invocations_per_session or '无限'} 次 / session, "
            f"冷却 {self.cooldown_seconds}s"
        )

    def validate_params(self, params: Dict[str, Any]) -> tuple[bool, List[str]]:
        """验证参数"""
        errors = []
        provided = set(params.keys())
        
        for constraint in self.parameters:
            value = params.get(constraint.name)
            ok, msg = constraint.validate(value)
            if not ok:
                errors.append(msg)
        
        # 检查未知参数
        known = {p.name for p in self.parameters}
        unknown = provided - known
        for u in unknown:
            errors.append(f"未知参数: '{u}'")
        
        return len(errors) == 0, errors


# ============================================================
#  预定义的物理技能能力卡片集
# ============================================================

# --- 感知技能 ---
CARD_CAPTURE_FRAME = CapabilityCard(
    skill_id="capture_frame",
    name="捕获实时画面",
    description="从机器人前向 RGBD 相机获取一帧实时图像和深度数据。返回图像数据及时间戳，用于目标检测和场景理解。",
    category=SkillCategory.PERCEPTION,
    authority=DecisionAuthority.AUTONOMOUS,
    parameters=[
        ParamConstraint("source", "enum", "相机来源",
                        allowed_values=["front_rgb", "front_depth", "head_stereo"],
                        default_value="front_rgb"),
        ParamConstraint("timeout_seconds", "float", "等待新帧的超时时间",
                        min_value=0.5, max_value=10.0, default_value=2.0),
    ],
    max_invocations_per_session=120,
    cooldown_seconds=0.5,
    side_effects="无（仅读取传感器）",
)

CARD_DETECT_OBJECTS = CapabilityCard(
    skill_id="detect_objects",
    name="目标检测",
    description="对当前画面进行目标检测，返回检测到的物体列表，包含类别、边界框、置信度和大致距离估计。",
    category=SkillCategory.PERCEPTION,
    authority=DecisionAuthority.AUTONOMOUS,
    parameters=[
        ParamConstraint("target_classes", "str", "要检测的目标类别描述",
                        default_value="all"),
        ParamConstraint("min_confidence", "float", "最低置信度阈值",
                        min_value=0.1, max_value=0.95, default_value=0.5),
    ],
    max_invocations_per_session=60,
    cooldown_seconds=1.0,
    side_effects="无",
)

# --- 运动技能 ---
CARD_MOVE_FORWARD = CapabilityCard(
    skill_id="move_forward",
    name="前进",
    description="控制机器人底盘直线前进指定距离。机器人会先检查前方是否有障碍物，安全时才会执行。",
    category=SkillCategory.LOCOMOTION,
    authority=DecisionAuthority.AUTONOMOUS,
    parameters=[
        ParamConstraint("distance_m", "float", "前进距离（米）",
                        min_value=0.05, max_value=0.5, default_value=0.2),
        ParamConstraint("speed", "float", "前进速度比例",
                        min_value=0.1, max_value=1.0, default_value=0.5),
    ],
    preconditions=[
        RuntimeCondition("not_locked", description="系统未锁定"),
        RuntimeCondition("not_estopped", description="未处于急停状态"),
    ],
    max_invocations_per_session=24,
    cooldown_seconds=1.0,
    side_effects="机器人物理位置改变",
    emergency_stop_on_failure=True,
)

CARD_TURN = CapabilityCard(
    skill_id="turn",
    name="转向",
    description="控制机器人原地转向。正角度表示右转，负角度表示左转。",
    category=SkillCategory.LOCOMOTION,
    authority=DecisionAuthority.AUTONOMOUS,
    parameters=[
        ParamConstraint("angle_deg", "float", "转向角度（度），正=右转，负=左转",
                        min_value=-120.0, max_value=120.0, default_value=30.0),
    ],
    preconditions=[
        RuntimeCondition("not_locked", description="系统未锁定"),
    ],
    max_invocations_per_session=60,
    cooldown_seconds=0.5,
    side_effects="机器人方向改变",
)

# --- 手臂操作技能 ---
CARD_SET_GRIPPER = CapabilityCard(
    skill_id="set_gripper",
    name="夹爪控制",
    description="控制夹爪的开合程度和夹持力度。",
    category=SkillCategory.MANIPULATION,
    authority=DecisionAuthority.AUTONOMOUS,
    parameters=[
        ParamConstraint("side", "enum", "夹爪侧",
                        allowed_values=["left", "right", "both"],
                        default_value="both"),
        ParamConstraint("action", "enum", "操作",
                        allowed_values=["open", "close", "set_width", "set_force"],
                        default_value="close"),
        ParamConstraint("width_mm", "float", "夹爪开度（毫米），仅 action=set_width 时有效",
                        min_value=0.0, max_value=80.0, default_value=30.0),
        ParamConstraint("force_pct", "float", "抓取力度百分比，仅 action=set_force 时有效",
                        min_value=10.0, max_value=100.0, default_value=50.0),
    ],
    preconditions=[
        RuntimeCondition("not_estopped", description="未处于急停状态"),
    ],
    max_invocations_per_session=30,
    cooldown_seconds=1.0,
    side_effects="夹爪物理状态改变",
)

CARD_MOVE_ARM_JOINT = CapabilityCard(
    skill_id="move_arm_joint",
    name="关节运动",
    description="控制指定手臂关节运动到目标角度。系统会自动检查关节限位和碰撞风险。",
    category=SkillCategory.ARM_MOTION,
    authority=DecisionAuthority.SOFT_CONFIRM,  # 手臂运动有碰撞风险，记录日志
    parameters=[
        ParamConstraint("joint_id", "int", "关节编号 (1-6)",
                        min_value=1, max_value=6),
        ParamConstraint("target_angle_deg", "float", "目标角度（度）",
                        min_value=-360.0, max_value=360.0),
        ParamConstraint("speed_pct", "float", "运动速度百分比",
                        min_value=5.0, max_value=100.0, default_value=50.0),
    ],
    preconditions=[
        RuntimeCondition("not_locked", description="系统未锁定"),
        RuntimeCondition("not_estopped", description="未处于急停状态"),
        RuntimeCondition("joint_range", description="目标角度在关节限位范围内"),
    ],
    max_invocations_per_session=48,
    cooldown_seconds=0.5,
    side_effects="手臂关节角度改变，存在碰撞风险",
    emergency_stop_on_failure=True,
)

CARD_MOVE_ARM_PRESET = CapabilityCard(
    skill_id="move_arm_preset",
    name="手臂预设姿态",
    description="将手臂移动到预定义的姿态。预设姿态已经过安全验证，包含完整的关节角度序列。",
    category=SkillCategory.ARM_MOTION,
    authority=DecisionAuthority.AUTONOMOUS,     # 预设姿态已验证安全
    parameters=[
        ParamConstraint("preset_name", "enum", "预设姿态名称",
                        allowed_values=[
                            "reset", "zero_position", "wave",
                            "package_pos", "standup_pos",
                        ]),
        ParamConstraint("speed_pct", "float", "运动速度百分比",
                        min_value=10.0, max_value=100.0, default_value=60.0),
    ],
    preconditions=[
        RuntimeCondition("not_locked", description="系统未锁定"),
        RuntimeCondition("not_estopped", description="未处于急停状态"),
    ],
    max_invocations_per_session=24,
    cooldown_seconds=2.0,
    side_effects="手臂整体姿态改变",
)

# --- 安全技能 ---
CARD_ESTOP = CapabilityCard(
    skill_id="emergency_stop",
    name="紧急停止",
    description="立即停止所有运动，锁定所有关节。在任何紧急情况下都可调用。",
    category=SkillCategory.SAFETY,
    authority=DecisionAuthority.AUTONOMOUS,      # Agent 必须能自主急停
    parameters=[],
    max_invocations_per_session=10,
    cooldown_seconds=0.0,
    side_effects="所有运动立即停止，系统进入锁定状态",
)

# --- 元技能 ---
CARD_GET_STATUS = CapabilityCard(
    skill_id="get_status",
    name="查询状态",
    description="查询当前探索会话的状态，包括已用/剩余预算（步数、距离、时间）、当前观察结果等。",
    category=SkillCategory.META,
    authority=DecisionAuthority.AUTONOMOUS,
    parameters=[],
    max_invocations_per_session=100,
    cooldown_seconds=0.0,
    side_effects="无",
)


# 所有已注册的能力卡片
ALL_CAPABILITY_CARDS: Dict[str, CapabilityCard] = {
    card.skill_id: card for card in [
        CARD_CAPTURE_FRAME,
        CARD_DETECT_OBJECTS,
        CARD_MOVE_FORWARD,
        CARD_TURN,
        CARD_SET_GRIPPER,
        CARD_MOVE_ARM_JOINT,
        CARD_MOVE_ARM_PRESET,
        CARD_ESTOP,
        CARD_GET_STATUS,
    ]
}
