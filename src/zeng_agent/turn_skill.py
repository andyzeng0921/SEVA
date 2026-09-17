"""
turn_skill.py — 视觉检测转身技能模块

流程:
  1. 拍照 → 视觉模型评估左右转身空间
  2. 空间足够 → 执行转身
  3. 反馈实际转动角度

集成点:
  - VisionProvider: 前方双目摄像头
  - ChassisController: 底盘转向控制
  - LangGraph: 作为标准化 skill 被调用
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from .chassis_controller import ChassisConfig, ChassisController
from .vision_provider import VisionConfig, VisionProvider


# ── 空间检测结果 ──

class SpaceCheck(str, Enum):
    """转身空间检测结果"""
    BOTH_SAFE = "BOTH_SAFE"        # 左右两侧空间都足够
    LEFT_SAFE = "LEFT_SAFE"        # 仅左侧安全
    RIGHT_SAFE = "RIGHT_SAFE"      # 仅右侧安全
    NONE_SAFE = "NONE_SAFE"        # 空间不足, 无法转身
    UNKNOWN = "UNKNOWN"            # 检测失败

    @classmethod
    def from_string(cls, s: str) -> "SpaceCheck":
        s_upper = s.upper().strip()
        for member in cls:
            if member.value in s_upper:
                return member
        # 只看最后 300 字符匹配
        tail = s_upper[-300:]
        if "BOTH" in tail or "两侧" in tail or "都可以" in tail:
            return cls.BOTH_SAFE
        if "LEFT" in tail or "左侧" in tail:
            return cls.LEFT_SAFE
        if "RIGHT" in tail or "右侧" in tail:
            return cls.RIGHT_SAFE
        if "NONE" in tail or "不足" in tail or "无法" in tail or "阻塞" in tail:
            return cls.NONE_SAFE
        return cls.UNKNOWN


@dataclass
class TurnResult:
    """转身执行结果"""
    success: bool
    direction: str                # "left" | "right"
    angle_target_deg: float
    angle_actual_deg: float
    space_check: str              # 空间检测结果
    error: str = ""
    latency_ms: float = 0.0

    def to_dict(self) -> dict:
        return {
            "success": self.success,
            "direction": self.direction,
            "angle_target_deg": self.angle_target_deg,
            "angle_actual_deg": self.angle_actual_deg,
            "space_check": self.space_check,
            "error": self.error,
            "latency_ms": round(self.latency_ms, 0),
        }


# ── 配置 ──

@dataclass
class TurnConfig:
    """转身技能配置"""
    # 视觉
    api_key: str = ""
    api_url: str = "http://127.0.0.1:8000/v1/chat/completions"
    vision_model: str = "qwen3.5-35b-a3b"
    camera_source: str = "front_left"
    vision_timeout: int = 60
    max_tokens: int = 400

    # 转身参数
    default_angle_deg: float = 30.0   # 默认转身角度
    turn_speed_deg_per_s: float = 30.0  # 转身角速度 (°/s)

    # 安全
    dry_run: bool = True
    require_space_check: bool = True   # 是否必须先检测空间
    max_angle_deg: float = 180.0       # 最大转身角度

    # 空间检测提示词
    space_check_prompt: str = (
        "你是一个机器人安全评估系统。这张照片来自机器人前方鱼眼摄像头。\n"
        "请判断机器人原位转身的空间是否足够：\n"
        "1. BOTH_SAFE - 画面左右两侧都有足够空地，左右转身都安全\n"
        "2. LEFT_SAFE - 只有画面左侧有足够空间，只能左转\n"
        "3. RIGHT_SAFE - 只有画面右侧有足够空间，只能右转\n"
        "4. NONE_SAFE - 两侧空间都不足，不能转身\n\n"
        "注意: 画面底部弧形物体是机器人自身底盘，请忽略。\n"
        "空旷的办公室走廊/开放区域通常是 BOTH_SAFE。\n"
        "请仅回复一个词: BOTH_SAFE, LEFT_SAFE, RIGHT_SAFE 或 NONE_SAFE。\n"
        "然后简要解释理由。"
    )


# ── 空间检测器 ──

class TurnSpaceDetector:
    """视觉转身空间检测器"""

    def __init__(self, config: TurnConfig):
        self.config = config
        self.vision = VisionProvider(VisionConfig(
            api_key=config.api_key,
            api_url=config.api_url,
            model=config.vision_model,
            camera_source=config.camera_source,
            max_tokens=config.max_tokens,
            timeout=config.vision_timeout,
        ))

    def check(self) -> tuple[SpaceCheck, str]:
        """检测转身空间

        Returns:
            (检测结果, 模型推理原文)
        """
        img_bytes, err = self.vision.capture_frame()
        if err:
            return SpaceCheck.UNKNOWN, f"拍照失败: {err}"

        result = self.vision.describe_image(img_bytes, self.config.space_check_prompt)
        if not result.success:
            return SpaceCheck.UNKNOWN, f"视觉模型错误: {result.error}"

        content = result.content or result.reasoning or ""
        check = SpaceCheck.from_string(content)
        return check, content[:500]


# ── 转身执行器 ──

class TurnSkill:
    """标准化转身技能

    用法:
        skill = TurnSkill(TurnConfig(api_key="...", dry_run=False))
        result = skill.turn("left", 30)   # 左转30°
        result = skill.turn("right", 90)  # 右转90°
        result = skill.safe_turn("left")  # 先检测空间→再执行
    """

    def __init__(self, config: TurnConfig):
        self.config = config
        self.detector = TurnSpaceDetector(config)
        self.chassis = ChassisController(ChassisConfig(
            default_angular_speed=config.turn_speed_deg_per_s / 57.3,
        ))

    def check_space(self) -> dict:
        """仅检测空间, 不执行"""
        check, reasoning = self.detector.check()
        return {
            "space": check.value,
            "reasoning": reasoning[:300],
            "can_turn_left": check in (SpaceCheck.BOTH_SAFE, SpaceCheck.LEFT_SAFE),
            "can_turn_right": check in (SpaceCheck.BOTH_SAFE, SpaceCheck.RIGHT_SAFE),
        }

    def turn(self, direction: str, angle_deg: float | None = None) -> TurnResult:
        """直接转身 (不做空间检测, 用于已知安全场景)

        Args:
            direction: "left" | "right"
            angle_deg: 转身角度 (°), 默认 30
        """
        t0 = time.monotonic()
        angle = angle_deg if angle_deg is not None else self.config.default_angle_deg
        angle = min(abs(angle), self.config.max_angle_deg)

        try:
            if self.config.dry_run:
                return TurnResult(
                    success=True, direction=direction,
                    angle_target_deg=angle, angle_actual_deg=angle,
                    space_check="dry_run", latency_ms=(time.monotonic() - t0) * 1000,
                )

            if direction == "left":
                result = self.chassis.turn_left(angle)
            else:
                result = self.chassis.turn_right(angle)

            return TurnResult(
                success=result.get("success", False),
                direction=direction,
                angle_target_deg=angle,
                angle_actual_deg=result.get("angle_target_deg", angle),
                space_check="not_checked",
                latency_ms=(time.monotonic() - t0) * 1000,
            )
        except Exception as e:
            return TurnResult(
                success=False, direction=direction,
                angle_target_deg=angle, angle_actual_deg=0,
                space_check="error", error=str(e),
                latency_ms=(time.monotonic() - t0) * 1000,
            )

    def safe_turn(self, direction: str, angle_deg: float | None = None) -> TurnResult:
        """安全转身 — 先视觉检测空间, 再执行

        Args:
            direction: "left" | "right" | "auto" (自动选择安全方向)
            angle_deg: 转身角度 (°)
        """
        t0 = time.monotonic()
        angle = angle_deg if angle_deg is not None else self.config.default_angle_deg
        angle = min(abs(angle), self.config.max_angle_deg)

        # 1. 空间检测
        if self.config.require_space_check:
            check, reasoning = self.detector.check()
        else:
            check, reasoning = SpaceCheck.BOTH_SAFE, "space_check_disabled"

        # 2. 根据检测结果选择方向
        if direction == "auto":
            if check in (SpaceCheck.BOTH_SAFE, SpaceCheck.LEFT_SAFE):
                direction = "left"
            elif check == SpaceCheck.RIGHT_SAFE:
                direction = "right"
            else:
                return TurnResult(
                    success=False, direction="auto",
                    angle_target_deg=angle, angle_actual_deg=0,
                    space_check=check.value, error=f"空间不足无法转身: {reasoning[:200]}",
                    latency_ms=(time.monotonic() - t0) * 1000,
                )

        # 3. 方向安全检查
        if direction == "left" and check not in (SpaceCheck.BOTH_SAFE, SpaceCheck.LEFT_SAFE):
            return TurnResult(
                success=False, direction="left",
                angle_target_deg=angle, angle_actual_deg=0,
                space_check=check.value, error=f"左侧空间不足: {reasoning[:200]}",
                latency_ms=(time.monotonic() - t0) * 1000,
            )
        if direction == "right" and check not in (SpaceCheck.BOTH_SAFE, SpaceCheck.RIGHT_SAFE):
            return TurnResult(
                success=False, direction="right",
                angle_target_deg=angle, angle_actual_deg=0,
                space_check=check.value, error=f"右侧空间不足: {reasoning[:200]}",
                latency_ms=(time.monotonic() - t0) * 1000,
            )

        # 4. 执行转身
        result = self.turn(direction, angle)
        result.space_check = check.value
        result.latency_ms = (time.monotonic() - t0) * 1000
        return result

    def shutdown(self) -> None:
        self.chassis.shutdown()
