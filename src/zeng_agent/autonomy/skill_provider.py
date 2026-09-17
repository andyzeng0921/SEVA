"""
物理技能提供层 (SkillProvider)

封装所有物理操作的底层实现。Agent 不能直接访问这些实现，
只能通过能力卡片接口调用 — 这保证了参数约束和安全检查不可绕过。

设计原则：
1. 每个技能方法只做一件事
2. 所有安全检查由 SafetyEnforcer 在调用前完成
3. 返回值结构归一化，便于 Agent 消费
"""
import time
import logging
import math
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field

# 模拟硬件接口（实际使用时对接 ROS2/硬件驱动）
try:
    from ..live_arm import LiveArm
    from ..vision_provider import VisionProvider
    from ..chassis_controller import ChassisController
    HAS_HARDWARE = True
except ImportError:
    HAS_HARDWARE = False

logger = logging.getLogger(__name__)


@dataclass
class SkillResult:
    """统一的技能执行结果"""
    success: bool
    skill_id: str
    data: Dict[str, Any] = field(default_factory=dict)
    error: str = ""
    timestamp: float = field(default_factory=time.time)
    execution_time_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "skill_id": self.skill_id,
            "data": self.data,
            "error": self.error,
            "timestamp": self.timestamp,
            "execution_time_ms": round(self.execution_time_ms, 1),
        }


class SkillProvider:
    """
    物理技能提供者
    
    Agent 通过能力卡片调用这里的方法。
    所有方法签名与 CapabilityCard 的参数定义一一对应。
    """

    def __init__(self):
        self._arm = None       # LiveArm 实例
        self._vision = None    # VisionProvider 实例
        self._chassis = None   # ChassisController 实例
        self._dry_run = True   # 默认干运行模式
        
        # 运行时约束状态
        self._joint_limits: Dict[int, Tuple[float, float]] = {
            1: (-180.0, 180.0), 2: (-90.0, 135.0), 3: (-135.0, 90.0),
            4: (-360.0, 360.0), 5: (-120.0, 120.0), 6: (-360.0, 360.0),
        }
        self._current_joint_angles: Dict[int, float] = {j: 0.0 for j in range(1, 7)}
        self._is_locked = False
        self._is_estopped = False

    def set_hardware(self, arm, vision, chassis) -> None:
        """注入硬件接口（启动时由 bootstrap 调用）"""
        self._arm = arm
        self._vision = vision
        self._chassis = chassis
        self._dry_run = False

    def set_dry_run(self, dry_run: bool) -> None:
        self._dry_run = dry_run

    # ========== 感知技能 ==========

    def capture_frame(self, source: str = "front_rgb",
                      timeout_seconds: float = 2.0) -> SkillResult:
        """捕获实时画面"""
        t0 = time.time()
        try:
            if self._vision and not self._dry_run:
                frame = self._vision.capture_frame(source, timeout_seconds)
            else:
                frame = {"width": 640, "height": 480, "format": "rgb",
                         "timestamp": time.time(), "source": source,
                         "placeholder": True}
            return SkillResult(
                success=True, skill_id="capture_frame",
                data={"frame": frame},
                execution_time_ms=(time.time() - t0) * 1000,
            )
        except Exception as e:
            return SkillResult(
                success=False, skill_id="capture_frame",
                error=str(e), execution_time_ms=(time.time() - t0) * 1000,
            )

    def detect_objects(self, target_classes: str = "all",
                       min_confidence: float = 0.5) -> SkillResult:
        """目标检测"""
        t0 = time.time()
        try:
            if self._vision and not self._dry_run:
                detections = self._vision.detect(target_classes, min_confidence)
            else:
                detections = []
            return SkillResult(
                success=True, skill_id="detect_objects",
                data={"detections": detections, "count": len(detections)},
                execution_time_ms=(time.time() - t0) * 1000,
            )
        except Exception as e:
            return SkillResult(
                success=False, skill_id="detect_objects",
                error=str(e), execution_time_ms=(time.time() - t0) * 1000,
            )

    # ========== 运动技能 ==========

    def move_forward(self, distance_m: float = 0.2,
                     speed: float = 0.5) -> SkillResult:
        """前进"""
        t0 = time.time()
        try:
            if self._chassis and not self._dry_run:
                self._chassis.move_forward(distance_m, speed)
            else:
                logger.info(f"[DRY_RUN] 前进 {distance_m}m, 速度 {speed}")
            return SkillResult(
                success=True, skill_id="move_forward",
                data={"distance_m": distance_m, "speed": speed},
                execution_time_ms=(time.time() - t0) * 1000,
            )
        except Exception as e:
            return SkillResult(
                success=False, skill_id="move_forward",
                error=str(e), execution_time_ms=(time.time() - t0) * 1000,
            )

    def turn(self, angle_deg: float = 30.0) -> SkillResult:
        """转向"""
        t0 = time.time()
        try:
            if self._chassis and not self._dry_run:
                self._chassis.turn(angle_deg)
            else:
                logger.info(f"[DRY_RUN] 转向 {angle_deg}°")
            return SkillResult(
                success=True, skill_id="turn",
                data={"angle_deg": angle_deg},
                execution_time_ms=(time.time() - t0) * 1000,
            )
        except Exception as e:
            return SkillResult(
                success=False, skill_id="turn",
                error=str(e), execution_time_ms=(time.time() - t0) * 1000,
            )

    # ========== 手臂操作技能 ==========

    def set_gripper(self, side: str = "both", action: str = "close",
                    width_mm: float = 30.0, force_pct: float = 50.0) -> SkillResult:
        """夹爪控制"""
        t0 = time.time()
        try:
            if self._arm and not self._dry_run:
                # 实际调用手臂夹爪接口
                if action == "open":
                    self._arm.open_gripper(side)
                elif action == "close":
                    self._arm.close_gripper(side)
                elif action == "set_width":
                    self._arm.set_gripper_width(side, width_mm)
                elif action == "set_force":
                    self._arm.set_gripper_force(side, force_pct)
            else:
                logger.info(f"[DRY_RUN] 夹爪 {side}: {action}")
            return SkillResult(
                success=True, skill_id="set_gripper",
                data={"side": side, "action": action},
                execution_time_ms=(time.time() - t0) * 1000,
            )
        except Exception as e:
            return SkillResult(
                success=False, skill_id="set_gripper",
                error=str(e), execution_time_ms=(time.time() - t0) * 1000,
            )

    def move_arm_joint(self, joint_id: int, target_angle_deg: float,
                       speed_pct: float = 50.0) -> SkillResult:
        """单一关节运动"""
        t0 = time.time()
        try:
            # 关节限位检查（双保险：SafetyEnforcer 也会检查）
            limits = self._joint_limits.get(joint_id)
            if limits:
                lo, hi = limits
                if target_angle_deg < lo or target_angle_deg > hi:
                    return SkillResult(
                        success=False, skill_id="move_arm_joint",
                        error=f"关节{joint_id} 目标角度 {target_angle_deg}° 超出限位 [{lo}°, {hi}°]",
                        execution_time_ms=(time.time() - t0) * 1000,
                    )

            if self._arm and not self._dry_run:
                self._arm.move_joint(joint_id, target_angle_deg, speed_pct)
                self._current_joint_angles[joint_id] = target_angle_deg
            else:
                logger.info(f"[DRY_RUN] 关节{joint_id} → {target_angle_deg}°")
            
            return SkillResult(
                success=True, skill_id="move_arm_joint",
                data={"joint_id": joint_id, "target_deg": target_angle_deg, "speed_pct": speed_pct},
                execution_time_ms=(time.time() - t0) * 1000,
            )
        except Exception as e:
            return SkillResult(
                success=False, skill_id="move_arm_joint",
                error=str(e), execution_time_ms=(time.time() - t0) * 1000,
            )

    def move_arm_preset(self, preset_name: str,
                        speed_pct: float = 60.0) -> SkillResult:
        """手臂预设姿态"""
        t0 = time.time()
        try:
            if self._arm and not self._dry_run:
                self._arm.move_to_preset(preset_name, speed_pct)
            else:
                logger.info(f"[DRY_RUN] 手臂预设 → {preset_name}")
            return SkillResult(
                success=True, skill_id="move_arm_preset",
                data={"preset_name": preset_name, "speed_pct": speed_pct},
                execution_time_ms=(time.time() - t0) * 1000,
            )
        except Exception as e:
            return SkillResult(
                success=False, skill_id="move_arm_preset",
                error=str(e), execution_time_ms=(time.time() - t0) * 1000,
            )

    # ========== 安全技能 ==========

    def emergency_stop(self) -> SkillResult:
        """紧急停止 — Agent 自主可调用的最高优先级技能"""
        t0 = time.time()
        try:
            self._is_locked = True
            self._is_estopped = True
            if self._arm and not self._dry_run:
                self._arm.emergency_stop()
            if self._chassis and not self._dry_run:
                self._chassis.stop()
            logger.warning("[ESTOP] 紧急停止已触发！")
            return SkillResult(
                success=True, skill_id="emergency_stop",
                data={"state": "estopped"},
                execution_time_ms=(time.time() - t0) * 1000,
            )
        except Exception as e:
            return SkillResult(
                success=False, skill_id="emergency_stop",
                error=str(e), execution_time_ms=(time.time() - t0) * 1000,
            )

    # ========== 查询技能 ==========

    def get_status(self) -> SkillResult:
        """查询状态"""
        return SkillResult(
            success=True, skill_id="get_status",
            data={
                "joint_angles": dict(self._current_joint_angles),
                "is_locked": self._is_locked,
                "is_estopped": self._is_estopped,
                "is_dry_run": self._dry_run,
            },
        )

    # ========== Agent 能力发现 ==========

    def get_available_skills_prompt(self) -> str:
        """
        生成 Agent 可用的技能描述文本
        
        这是 Agent 在规划时看到的内容 — 它描述了"我能做什么"，
        但不暴露物理实现的任何细节。
        """
        from .capability_cards import ALL_CAPABILITY_CARDS
        
        lines = ["## 可用物理技能", ""]
        
        for card in ALL_CAPABILITY_CARDS.values():
            authority_tag = {
                "AUTONOMOUS": "[自主]",
                "SOFT_CONFIRM": "[自主-记录]",
                "HARD_CONFIRM": "[需确认]",
            }.get(card.authority.name, "")
            
            lines.append(f"### {authority_tag} {card.name}")
            lines.append(f"ID: `{card.skill_id}`")
            lines.append(f"类别: {card.category.name}")
            lines.append(f"描述: {card.description}")
            
            if card.parameters:
                lines.append("参数:")
                for p in card.parameters:
                    constraint = ""
                    if p.min_value is not None and p.max_value is not None:
                        constraint = f" [{p.min_value}, {p.max_value}]"
                    elif p.allowed_values:
                        constraint = f" 可选: {p.allowed_values}"
                    lines.append(f"  - `{p.name}` ({p.param_type}): {p.description}{constraint}")
            
            if card.preconditions:
                lines.append(f"前置条件: {'; '.join(p.description for p in card.preconditions)}")
            
            lines.append(f"副作用: {card.side_effects or '无'}")
            lines.append(f"调用限制: 单session最多{card.max_invocations_per_session or '无限'}次")
            lines.append("")
        
        return "\n".join(lines)
