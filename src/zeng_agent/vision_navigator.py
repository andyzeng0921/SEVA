"""
vision_navigator.py — 视觉导航融合引擎

将视觉理解与底盘控制融合，实现"看→走→看→决策"的视觉导航循环。

流程:
  1. 拍照 (前方双目 SHM)
  2. 视觉模型分析路径可行性
  3. 返回决策: STRAIGHT / LEFT / RIGHT / BLOCKED
  4. 底盘执行对应动作
  5. 重复直到到达目标或阻塞

LangGraph 集成:
  - 作为 LangGraph 的 step executor
  - state 中追踪步数、决策历史、位置链
"""
from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from .chassis_controller import ChassisConfig, ChassisController
from .vision_provider import VisionConfig, VisionProvider, VisionResult


# ── 导航决策枚举 ──

class NavDecision(str, Enum):
    STRAIGHT = "STRAIGHT"          # 前方畅通,可直行
    LEFT = "LEFT"                  # 需要左转
    RIGHT = "RIGHT"                # 需要右转
    BLOCKED = "BLOCKED"            # 阻塞,无法前进
    GOAL_REACHED = "GOAL_REACHED"  # 到达目标
    UNKNOWN = "UNKNOWN"            # 无法判断

    @classmethod
    def from_string(cls, s: str) -> "NavDecision":
        """从模型输出中提取导航决策。
        
        策略: 取最后几行中第一个匹配的决策词，避免被 prompt 中的描述干扰。
        """
        lines = s.upper().strip().split("\n")
        # 反向查找最后几行，匹配明确决策
        for line in reversed(lines[-10:]):
            line = line.strip()
            # 精确匹配
            for member in cls:
                if member.value == line or line.startswith(member.value):
                    return member
        # 回退: 在整个文本中模糊搜索（只在模型输出部分）
        s_upper = s.upper()
        if "STRAIGHT" in s_upper[-200:]:
            return cls.STRAIGHT
        if "LEFT" in s_upper[-200:]:
            return cls.LEFT
        if "RIGHT" in s_upper[-200:]:
            return cls.RIGHT
        if "BLOCKED" in s_upper[-200:]:
            return cls.BLOCKED
        if "GOAL_REACHED" in s_upper[-200:] or "GOAL" in s_upper[-200:]:
            return cls.GOAL_REACHED
        # 最终回退
        if "直行" in s[-200:] or "前进" in s[-200:]:
            return cls.STRAIGHT
        if "阻塞" in s[-200:] or "障碍" in s[-200:]:
            return cls.BLOCKED
        return cls.STRAIGHT  # 默认直行


# ── 导航状态 ──

@dataclass
class NavState:
    """视觉导航状态追踪"""
    step: int = 0
    total_distance_m: float = 0.0
    decision_history: list[dict] = field(default_factory=list)
    last_decision: NavDecision = NavDecision.UNKNOWN
    last_reasoning: str = ""
    running: bool = True
    success: bool = False
    message: str = ""

    def to_dict(self) -> dict:
        return {
            "step": self.step,
            "total_distance_m": round(self.total_distance_m, 3),
            "last_decision": self.last_decision.value,
            "last_reasoning": self.last_reasoning[:200],
            "running": self.running,
            "success": self.success,
            "message": self.message,
        }


# ── 视觉导航配置 ──

@dataclass
class VisionNavConfig:
    """视觉导航融合配置"""
    # 视觉
    api_key: str = ""
    api_url: str = "http://127.0.0.1:8000/v1/chat/completions"
    vision_model: str = "qwen3.5-35b-a3b"
    camera_source: str = "front_left"

    # 底盘
    step_forward_distance: float = 0.20  # m, 每步前进
    max_steps: int = 50                  # 最大步数
    goal_distance: float = 2.0           # m, 目标总距离
    turn_angle_deg: float = 30.0         # °, 每次转弯角度

    # 安全
    dry_run: bool = True                 # True=不实际移动底盘
    require_confirmation: bool = True    # True=每步需要确认

    # 提示词
    vision_prompt: str = (
        "你是一个机器人视觉导航系统。请分析这张前方摄像头画面（鱼眼广角镜头），" 
        "画面中央偏上是机器人前方的真实环境，最下方弧形可能是机器人自身底盘。\n"
        "请判断机器人前方路径是否可以通行：\n"
        "1. STRAIGHT - 前方有大片可通行空地或走廊，可以安全继续直行\n"
        "2. LEFT - 前方有阻碍但画面左侧有可通行的空间\n"
        "3. RIGHT - 前方有阻碍但画面右侧有可通行的空间\n"
        "4. BLOCKED - 前方严重阻塞，完全无法通行（如墙壁、紧闭的门、密集人群障碍）\n"
        "5. GOAL_REACHED - 已非常接近目标区域\n\n"
        "注意: 画面底部弧形物体是机器人自身底盘，请忽略。空的走廊/空地应该判断为STRAIGHT。\n"
        "请仅回复一个词: STRAIGHT, LEFT, RIGHT, BLOCKED 或 GOAL_REACHED。\n"
        "然后简要解释你的判断理由。"
    )


# ── 导航融合引擎 ──

class VisionNavigator:
    """视觉导航器 — 融合视觉+底盘的主控

    用法:
        nav = VisionNavigator(VisionNavConfig(
            api_key="gpustack_...",
            dry_run=False,
            goal_distance=3.0,
        ))

        # 单步: 看→决策
        decision, reasoning, image = nav.look_and_decide()

        # 完整多步循环
        state = nav.run_steps()
    """

    def __init__(self, config: VisionNavConfig) -> None:
        self.config = config
        self.state = NavState()

        # 视觉提供器
        self.vision = VisionProvider(VisionConfig(
            api_key=config.api_key,
            api_url=config.api_url,
            model=config.vision_model,
            camera_source=config.camera_source,
            max_tokens=800,
            timeout=60,
        ))

        # 底盘控制器
        self.chassis = ChassisController(ChassisConfig(
            step_forward_distance=config.step_forward_distance,
            turn_angle=math.radians(config.turn_angle_deg),
        ))

    # ── 核心: 看→决策 ──

    def look_and_decide(self) -> tuple[NavDecision, str, Optional[bytes]]:
        """拍照 → 视觉模型分析 → 返回决策

        Returns:
            (decision, reasoning, image_bytes)
        """
        # 拍照
        img_bytes, err = self.vision.capture_frame()
        if err:
            return NavDecision.UNKNOWN, f"拍照失败: {err}", None

        # 视觉模型推理
        result = self.vision.describe_image(img_bytes, self.config.vision_prompt)
        if not result.success:
            return NavDecision.UNKNOWN, f"视觉模型错误: {result.error}", img_bytes

        content = result.content or result.reasoning
        decision = NavDecision.from_string(content)
        reasoning = content[:500] if content else ""

        return decision, reasoning, img_bytes

    # ── 单步执行 ──

    def step(self) -> dict:
        """执行一个完整的导航步骤: 看→决策→走

        Returns:
            {"decision": str, "reasoning": str, "action_taken": str, "odom": dict, ...}
        """
        self.state.step += 1

        # 1. 视觉决策
        decision, reasoning, _ = self.look_and_decide()
        self.state.last_decision = decision
        self.state.last_reasoning = reasoning

        step_result = {
            "step": self.state.step,
            "decision": decision.value,
            "reasoning": reasoning[:300],
        }

        # 2. 根据决策执行底盘动作
        if self.config.dry_run:
            step_result["action_taken"] = f"DRY_RUN: would {decision.value}"
            step_result["odom"] = {"x": 0, "y": 0}
        else:
            try:
                if decision == NavDecision.STRAIGHT:
                    odom = self.chassis.move_forward(self.config.step_forward_distance)
                    step_result["action_taken"] = f"前进 {odom.get('distance_actual', 0):.2f}m"
                    step_result["odom"] = odom
                    self.state.total_distance_m += odom.get("distance_actual", 0)

                elif decision == NavDecision.LEFT:
                    odom = self.chassis.turn_left(self.config.turn_angle_deg)
                    step_result["action_taken"] = f"左转 {odom.get('angle_actual_deg', 0):.0f}°"
                    step_result["odom"] = odom

                elif decision == NavDecision.RIGHT:
                    odom = self.chassis.turn_right(self.config.turn_angle_deg)
                    step_result["action_taken"] = f"右转 {odom.get('angle_actual_deg', 0):.0f}°"
                    step_result["odom"] = odom

                elif decision == NavDecision.BLOCKED:
                    self.state.running = False
                    self.state.message = "路径阻塞"
                    step_result["action_taken"] = "停止 (阻塞)"

                elif decision == NavDecision.GOAL_REACHED:
                    self.state.running = False
                    self.state.success = True
                    self.state.message = "到达目标"
                    step_result["action_taken"] = "停止 (到达)"

                else:
                    # UNKNOWN → 试探前进
                    odom = self.chassis.move_forward(self.config.step_forward_distance * 0.5)
                    step_result["action_taken"] = f"前进(试探) {odom.get('distance_actual', 0):.2f}m"
                    step_result["odom"] = odom

            except Exception as e:
                step_result["action_taken"] = f"底盘错误: {e}"
                step_result["odom"] = None

        # 3. 检查终止条件
        if self.state.total_distance_m >= self.config.goal_distance:
            self.state.running = False
            self.state.success = True
            self.state.message = f"达到目标距离 {self.config.goal_distance}m"
            step_result["terminal"] = True

        if self.state.step >= self.config.max_steps:
            self.state.running = False
            self.state.message = f"达到最大步数 {self.config.max_steps}"
            step_result["terminal"] = True

        self.state.decision_history.append(step_result)
        return step_result

    # ── 完整多步循环 ──

    def run_steps(self) -> NavState:
        """执行完整视觉导航循环直到终止"""
        while self.state.running:
            self.step()
            if not self.state.running:
                break
            time.sleep(0.5)  # 步间暂停
        return self.state

    # ── 单步预览（不实际移动） ──

    def preview_step(self) -> dict:
        """只看不移动，返回视觉分析结果"""
        decision, reasoning, img_bytes = self.look_and_decide()
        return {
            "decision": decision.value,
            "reasoning": reasoning[:300],
            "image_bytes": len(img_bytes) if img_bytes else 0,
        }

    def shutdown(self) -> None:
        """安全关闭"""
        self.chassis.stop()
        self.chassis.shutdown()
