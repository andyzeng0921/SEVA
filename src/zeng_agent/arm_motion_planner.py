from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

Limb = Literal["left", "right"]
Direction = Literal["up", "down", "left", "right", "forward", "backward"]

# ── 方位量化表 ──────────────────────────────────────────────
# 将中文方向词映射到机器人坐标系下的位移向量
#
# 坐标系约定 (机器人坐标系):
#   X: 前进方向 (forward)
#   Y: 向左为正方向 (left)
#   Z: 向上为正方向 (up)
#
# 对于左臂 (left), 其"向左"是 Y+ 方向
# 对于右臂 (right), 其"向左"是 Y- 方向（因为镜像）

ARM_DIRECTION_VECTORS: dict[Limb, dict[Direction, list[float]]] = {
    "left": {
        "up":       [0.0,  0.0,  1.0],   # Z+
        "down":     [0.0,  0.0, -1.0],   # Z-
        "left":     [0.0,  1.0,  0.0],   # Y+
        "right":    [0.0, -1.0,  0.0],   # Y-
        "forward":  [1.0,  0.0,  0.0],   # X+
        "backward": [-1.0, 0.0,  0.0],   # X-
    },
    "right": {
        "up":       [0.0,  0.0,  1.0],   # Z+
        "down":     [0.0,  0.0, -1.0],   # Z-
        "left":     [0.0, -1.0,  0.0],   # Y- (镜像对称)
        "right":    [0.0,  1.0,  0.0],   # Y+ (镜像对称)
        "forward":  [1.0,  0.0,  0.0],   # X+
        "backward": [-1.0, 0.0,  0.0],   # X-
    },
}

# ── 幅度量化表 ──────────────────────────────────────────────
# 将中文模糊量词映射为具体距离（米）

AMOUNT_TABLE: dict[str, float] = {
    "一点点":  0.03,
    "一点":    0.03,
    "点儿":   0.03,
    "稍微":   0.05,
    "少许":   0.05,
    "一些":    0.08,
    "一下":    0.08,
    "中等":   0.12,
    "一些些":  0.12,
    "很多":   0.20,
    "大量":   0.30,
    "最大":   0.50,
}

_DEFAULT_STEP = 0.05  # 默认步长 5cm


@dataclass(slots=True)
class MotionPlan:
    """运动规划结果"""
    direction: Direction
    amount_meters: float
    limb: Limb
    current_pose: dict[str, list[float]] | None = None
    target_position: list[float] | None = None
    target_rotation: list[float] | None = None
    displacement_vector: list[float] | None = None
    raw: dict[str, Any] | None = None


def parse_amount(text: str) -> float:
    """解析中文幅度量词为米

    >>> parse_amount("一点点")
    0.03
    >>> parse_amount("很多")
    0.2
    >>> parse_amount("")  # 默认
    0.05
    """
    text = text.strip().lower()
    for keyword, distance in AMOUNT_TABLE.items():
        if keyword in text:
            return distance
    return _DEFAULT_STEP


def parse_direction(text: str) -> Direction | None:
    """解析中文方向词

    >>> parse_direction("向下")
    'down'
    >>> parse_direction("往上")
    'up'
    >>> parse_direction("向左")
    'left'
    >>> parse_direction("")
    None
    """
    mapping = {
        "上": "up",
        "下": "down",
        "左": "left",
        "右": "right",
        "前": "forward",
        "后": "backward",
        "forward": "forward",
        "backward": "backward",
        "up": "up",
        "down": "down",
    }
    for zh, en in mapping.items():
        if zh in text:
            return en
    return None


def parse_limb(text: str) -> Limb:
    """解析肢体/手臂侧

    >>> parse_limb("左臂")
    'left'
    >>> parse_limb("机械臂")
    'left'
    """
    if "右" in text:
        return "right"
    return "left"


def plan_motion(
    direction_text: str,
    amount_text: str,
    limb_text: str = "",
    current_pose: dict[str, list[float]] | None = None,
) -> MotionPlan:
    """将自然语言运动指令解析为运动规划

    参数:
        direction_text: 方向描述 (如 "向下", "往上")
        amount_text:    幅度描述 (如 "一点点", "很多")
        limb_text:      手臂侧描述 (如 "左臂", "右臂")
        current_pose:   当前位姿 {position: [x,y,z], rotation: [qx,qy,qz,qw]}

    返回: MotionPlan
    """
    direction = parse_direction(direction_text)
    amount = parse_amount(amount_text)
    limb = parse_limb(limb_text)

    if direction is None:
        return MotionPlan(
            direction="down", amount_meters=0.0, limb=limb,
            raw={"error": f"无法解析方向: {direction_text}"},
        )

    vec = ARM_DIRECTION_VECTORS[limb][direction]
    displacement = [v * amount for v in vec]

    target_position = None
    target_rotation = None
    if current_pose:
        cp = current_pose["position"]
        target_position = [cp[0] + displacement[0], cp[1] + displacement[1], cp[2] + displacement[2]]
        target_rotation = list(current_pose.get("rotation", []))

    return MotionPlan(
        direction=direction,
        amount_meters=amount,
        limb=limb,
        current_pose=current_pose,
        target_position=target_position,
        target_rotation=target_rotation,
        displacement_vector=displacement,
        raw={
            "direction": direction,
            "amount_meters": amount,
            "limb": limb,
            "displacement": displacement,
            "target_position": target_position,
        },
    )
