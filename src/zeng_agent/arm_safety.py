"""
arm_safety.py — 机械臂碰撞预防安全模块

碰撞根因分析:
  1. Shoulder_Inner 正值 → 左臂甩到身后 (X负) → 撞身体
  2. Shoulder_Inner 负值 → 右臂甩到身后
  3. 左臂右移(Y正方向过大) → 手臂越过身体中线 → 撞身体
  4. 右臂左移(Y负方向过大) → 手臂越过身体中线 → 撞身体
  
安全策略:
  - 检查上电状态 (joint enable + topic 活跃)
  - 关节角度限幅
  - EEF 空间位置边界
  - 运动前校验 → 拒绝/放行/修正
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Optional, Literal

# ============================================================
# 安全关节角度范围（基于实测标定）
# ============================================================

# 左臂关节安全范围
LEFT_ARM_SAFE_RANGE = {
    "Joint_Left_Shoulder_Inner":  (-50.0, 5.0),   # ≤5°: 防甩身后, ≥-50°: 防过度
    "Joint_Left_Shoulder_Outer":  (-30.0, 30.0),
    "Joint_Left_UpperArm":        (-10.0, 60.0),   # 正值向前
    "Joint_Left_Elbow":           (-20.0, 45.0),
    "Joint_Left_Forearm":         (-40.0, 40.0),
    "Joint_Left_Wrist_Upper":     (-30.0, 30.0),
    "Joint_Left_Wrist_Lower":     (-30.0, 30.0),
}

# 右臂关节安全范围
RIGHT_ARM_SAFE_RANGE = {
    "Joint_Right_Shoulder_Inner": (-5.0, 50.0),    # ≥-5°: 防甩身后
    "Joint_Right_Shoulder_Outer": (-30.0, 30.0),
    "Joint_Right_UpperArm":       (-60.0, 10.0),   # 负值向前
    "Joint_Right_Elbow":          (-45.0, 20.0),
    "Joint_Right_Forearm":        (-40.0, 40.0),
    "Joint_Right_Wrist_Upper":    (-30.0, 30.0),
    "Joint_Right_Wrist_Lower":    (-30.0, 30.0),
}

# 关节在 left_arm/right_arm 数组中的索引
LEFT_JOINT_INDEX = {
    "Joint_Left_Shoulder_Inner":  0,
    "Joint_Left_Shoulder_Outer":  1,
    "Joint_Left_UpperArm":        2,
    "Joint_Left_Elbow":           3,
    "Joint_Left_Forearm":         4,
    "Joint_Left_Wrist_Upper":     5,
    "Joint_Left_Wrist_Lower":     6,
}
RIGHT_JOINT_INDEX = {
    "Joint_Right_Shoulder_Inner":  0,
    "Joint_Right_Shoulder_Outer":  1,
    "Joint_Right_UpperArm":        2,
    "Joint_Right_Elbow":           3,
    "Joint_Right_Forearm":         4,
    "Joint_Right_Wrist_Upper":     5,
    "Joint_Right_Wrist_Lower":     6,
}

# ============================================================
# EEF 空间位置安全边界
# ============================================================

@dataclass
class EEFBoundary:
    """末端执行器 (EEF) 空间安全边界"""
    X_MIN: float = -0.05    # 禁止到身后
    X_MAX: float = 0.80     # 前伸极限
    Y_MIN_LEFT: float = -0.10   # 左臂禁止越过身体中线右侧
    Y_MAX_LEFT: float = 0.55    # 左臂左伸极限
    Y_MIN_RIGHT: float = -0.55  # 右臂右伸极限
    Y_MAX_RIGHT: float = 0.10   # 右臂禁止越过身体中线左侧
    Z_MIN: float = 0.60         # 最低点 (防撞腿)
    Z_MAX: float = 1.20         # 最高点

EEF_BOUNDARY = EEFBoundary()


# ============================================================
# 安全校验结果
# ============================================================

@dataclass
class SafetyCheckResult:
    allowed: bool
    reason: str = ""
    warnings: list[str] = field(default_factory=list)
    corrections: dict[str, float] = field(default_factory=dict)  # 修正后的值


def check_power_on(
    node,
    topic_node_id: str = "0_283",
    timeout: float = 3.0,
) -> SafetyCheckResult:
    """检查机械臂是否上电 (关节状态可用)

    通过订阅 joint status topic 判断是否有实际数据。
    """
    import rclpy
    from rclpy.node import Node as RclpyNode
    from std_msgs.msg import String as RosString

    topic = f"/topic_arm_whole_body_and_gripper_current_joints_status_{topic_node_id}"
    result = []

    def cb(msg):
        try:
            data = json.loads(msg.data)
            if "left_arm_joint_state" in data and "right_arm_joint_state" in data:
                result.append(data)
        except (json.JSONDecodeError, KeyError):
            pass

    sub = node.create_subscription(RosString, topic, cb, 1)
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline and not result:
        rclpy.spin_once(node, timeout_sec=0.05)
    node.destroy_subscription(sub)

    if not result:
        return SafetyCheckResult(
            allowed=False,
            reason="机械臂未上电: 无法获取关节状态数据。请确保 arm_control_service 已启动。",
        )
    return SafetyCheckResult(allowed=True, reason="机械臂已上电 ✅")


def validate_joint_angles(
    left_arm: list[float],
    right_arm: list[float],
) -> SafetyCheckResult:
    """校验关节角度是否在安全范围内

    返回修正后的角度（超限钳位到安全边界）。
    """
    warnings = []
    corrections = {}
    clamped_left = list(left_arm)
    clamped_right = list(right_arm)

    # 左臂检查
    for jname, (lo, hi) in LEFT_ARM_SAFE_RANGE.items():
        idx = LEFT_JOINT_INDEX[jname]
        if idx >= len(clamped_left):
            continue
        val = clamped_left[idx]
        if val < lo:
            warnings.append(f"⚠ 左臂 {jname}={val:.1f}° < 安全下限{lo}° → 钳位到{lo}°")
            clamped_left[idx] = lo
            corrections[jname] = lo
        elif val > hi:
            warnings.append(f"⚠ 左臂 {jname}={val:.1f}° > 安全上限{hi}° → 钳位到{hi}°")
            clamped_left[idx] = hi
            corrections[jname] = hi

    # 右臂检查
    for jname, (lo, hi) in RIGHT_ARM_SAFE_RANGE.items():
        idx = RIGHT_JOINT_INDEX[jname]
        if idx >= len(clamped_right):
            continue
        val = clamped_right[idx]
        if val < lo:
            warnings.append(f"⚠ 右臂 {jname}={val:.1f}° < 安全下限{lo}° → 钳位到{lo}°")
            clamped_right[idx] = lo
            corrections[jname] = lo
        elif val > hi:
            warnings.append(f"⚠ 右臂 {jname}={val:.1f}° > 安全上限{hi}° → 钳位到{hi}°")
            clamped_right[idx] = hi
            corrections[jname] = hi

    # 关键安全检查
    critical = []
    # 左臂 Shoulder_Inner > 5° → 即将甩身后
    if len(clamped_left) > 0 and clamped_left[0] > 5.0:
        critical.append(f"⛔ 左肩 Inner={clamped_left[0]:.1f}° → 手臂将甩到身后!")
    # 右臂 Shoulder_Inner < -5° → 即将甩身后
    if len(clamped_right) > 0 and clamped_right[0] < -5.0:
        critical.append(f"⛔ 右肩 Inner={clamped_right[0]:.1f}° → 手臂将甩到身后!")

    if critical:
        return SafetyCheckResult(
            allowed=False,
            reason="; ".join(critical),
            warnings=warnings,
            corrections=corrections,
        )

    return SafetyCheckResult(
        allowed=True,
        reason="关节角度在安全范围内 ✅",
        warnings=warnings,
        corrections=corrections,
    )


def validate_eef_position(
    left_pos: tuple[float, float, float],
    right_pos: tuple[float, float, float],
) -> SafetyCheckResult:
    """校验 EEF 空间位置是否在安全边界内"""
    lx, ly, lz = left_pos
    rx, ry, rz = right_pos
    issues = []

    # 左臂检查
    if lx < EEF_BOUNDARY.X_MIN:
        issues.append(f"⛔ 左臂 X={lx:.2f} 到身后!")
    if ly > EEF_BOUNDARY.Y_MAX_LEFT:
        issues.append(f"⛔ 左臂 Y={ly:.2f} 越过身体中线右侧!")
    if lz < EEF_BOUNDARY.Z_MIN:
        issues.append(f"⛔ 左臂 Z={lz:.2f} 过低可能撞腿!")

    # 右臂检查
    if rx < EEF_BOUNDARY.X_MIN:
        issues.append(f"⛔ 右臂 X={rx:.2f} 到身后!")
    if ry < EEF_BOUNDARY.Y_MIN_RIGHT:
        issues.append(f"⛔ 右臂 Y={ry:.2f} 越过身体中线左侧!")
    if rz < EEF_BOUNDARY.Z_MIN:
        issues.append(f"⛔ 右臂 Z={rz:.2f} 过低可能撞腿!")

    if issues:
        return SafetyCheckResult(allowed=False, reason="; ".join(issues))

    return SafetyCheckResult(allowed=True, reason="EEF 空间位置安全 ✅")


# ============================================================
# 前向抬起预设（安全标定得出的最佳组合）
# ============================================================

# 左臂前上: Inner=-30°, UpperArm=+30°
LEFT_ARM_FRONT_UP = [-30.0, 0.0, 30.0, 0.0, 0.0, 0.0, 0.0]
# 左臂下垂(复位): 全零
LEFT_ARM_DOWN = [0.0] * 7

# 右臂前上: Inner=+30°, UpperArm=-30°
RIGHT_ARM_FRONT_UP = [30.0, 0.0, -30.0, 0.0, 0.0, 0.0, 0.0]
# 右臂下垂(复位): 全零
RIGHT_ARM_DOWN = [0.0] * 7


def make_cross_wave_joints(phase: Literal["left_up", "right_up"]) -> tuple[list[float], list[float]]:
    """生成一手上下一手交叉的关节角度

    phase="left_up":  左臂上 + 右臂下
    phase="right_up": 左臂下 + 右臂上
    """
    if phase == "left_up":
        return (LEFT_ARM_FRONT_UP, RIGHT_ARM_DOWN)
    else:
        return (LEFT_ARM_DOWN, RIGHT_ARM_FRONT_UP)


def make_whole_body_joint_msg(left_arm: list[float], right_arm: list[float]) -> str:
    """生成 whole_body_target_joints_position 的 JSON 消息"""
    return json.dumps({
        "leg_waist_target_joints_position": [0.0] * 4,
        "left_arm_target_joints_position": left_arm,
        "right_arm_target_joints_position": right_arm,
    })
