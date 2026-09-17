"""
explorer.py — 视觉探索引擎

自主探索循环:
  1. 拍照 (SHM DICOTA 4K 前双目)
  2. 视觉模型分析 (qwen3.5): "看到了什么? 往哪走?"
  3. 决策解析: GO / LEFT / RIGHT / STOP / BACK
  4. 底盘执行
  5. 循环

用法:
    explorer = VisualExplorer(config)
    explorer.explore(steps=5, step_distance=0.2)
"""

from __future__ import annotations

import base64
import json
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional

import requests


class Action(Enum):
    GO = "GO"           # 直行
    LEFT = "LEFT"       # 左转
    RIGHT = "RIGHT"     # 右转
    STOP = "STOP"       # 停止 (到达目标/阻塞)
    BACK = "BACK"       # 后退 (死胡同)
    GREET = "GREET"     # 看到人, 打招呼


@dataclass
class ExploreStep:
    """单步探索结果"""
    step: int
    image_path: str
    description: str         # 视觉模型描述
    action: Action
    reason: str              # 决策理由
    executed: bool
    error: Optional[str] = None


@dataclass
class ExplorerConfig:
    """探索配置"""
    # 视觉模型
    api_key: str = ""
    base_url: str = "http://127.0.0.1:8000/v1"
    model: str = "qwen3.5-35b-a3b"
    vision_timeout: int = 15

    # 摄像头
    camera_source: str = "front_left"  # front_left / front_right / rear_fisheye

    # 底盘
    step_distance: float = 0.20    # 每次前进距离 (m)
    turn_angle: float = 30.0       # 转弯角度 (度)
    move_speed: float = 0.15       # 移动速度 (m/s)
    turn_speed: float = 0.3        # 转弯速度 (rad/s)
    move_timeout: float = 5.0      # 单次移动超时

    # 探索
    max_steps: int = 10
    output_dir: str = "/tmp/zeng_explore"


class VisualExplorer:
    """视觉探索引擎"""

    def __init__(self, config: Optional[ExplorerConfig] = None):
        self.config = config or ExplorerConfig()
        self._steps: list[ExploreStep] = []
        self._chassis = None
        self._setup_output_dir()

    def _setup_output_dir(self):
        Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)

    # ── 拍照 ──

    def _capture(self, step: int) -> str:
        """拍照并保存, 返回图片路径"""
        from .vision_provider import VisionProvider, VisionConfig

        vp = VisionProvider(VisionConfig(
            api_key=self.config.api_key,
            model=self.config.model,
            camera_source=self.config.camera_source,
        ))

        jpeg_data, err = vp.capture_frame()
        if jpeg_data is None:
            raise RuntimeError(f"摄像头拍照失败: {err}")

        path = str(Path(self.config.output_dir) / f"explore_{step:03d}.jpg")
        from PIL import Image
        import io
        img = Image.open(io.BytesIO(jpeg_data))
        img.save(path, "JPEG")
        return path

    # ── 视觉分析 ──

    def _analyze(self, image_path: str) -> tuple[str, Action, str]:
        """用视觉模型分析图片, 返回 (描述, 动作, 理由)"""
        from .vision_provider import VisionProvider, VisionConfig

        with open(image_path, "rb") as f:
            img_b64 = base64.b64encode(f.read()).decode()

        prompt = """你是一个机器人视觉导航系统。请仔细分析这张前方摄像头拍摄的画面，按以下优先级做决策:

【最高优先级】人物检测:
- 如果画面中看到人 (不管远近), 优先选择 GREET

【次优先级】路径分析:
1. 描述你看到了什么 (用中文, 40字以内)
2. 前方路径评估:
   - 正前方是否畅通? 有无障碍物/墙壁/死角?
   - 左侧是否有可行通道?
   - 右侧是否有可行通道?
3. 综合决策, 从以下选择一个:
   - GREET: 画面中有人 (不管多近多远), 停下打招呼
   - GO: 正前方畅通无阻, 可直行
   - LEFT: 前方有障碍, 但左侧有通行空间
   - RIGHT: 前方有障碍, 但右侧有通行空间
   - STOP: 三面均阻塞, 无路可走
   - BACK: 后方更安全, 需要倒车

决策原则:
- 有人优先打招呼, 不要撞到人
- 不要总是GO, 注意观察左右方向是否有更好的路径
- 如果前方是墙/死路, 果断选择 LEFT 或 RIGHT

请严格用以下 JSON 格式回复 (不要加其他文字):
{"description": "...", "action": "GREET|GO|LEFT|RIGHT|STOP|BACK", "reason": "..."}
"""

        vp = VisionProvider(VisionConfig(
            api_key=self.config.api_key,
            model=self.config.model,
        ))
        api_url = f"{vp.config.api_url}"

        resp = requests.post(
            api_url,
            headers={
                "Authorization": f"Bearer {self.config.api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": self.config.model,
                "messages": [{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {
                            "url": f"data:image/jpeg;base64,{img_b64}"
                        }},
                    ],
                }],
                "max_tokens": 1024,
                "temperature": 0.1,
            },
            timeout=self.config.vision_timeout,
        )

        data = resp.json()
        choice = data["choices"][0]
        msg = choice["message"]
        content = msg.get("content") or msg.get("reasoning_content") or msg.get("reasoning") or ""
        return self._parse_response(content)

    @staticmethod
    def _parse_response(content: str) -> tuple[str, Action, str]:
        """解析视觉模型的 JSON 回复"""
        # 尝试提取 JSON
        content = content.strip()
        # 去掉可能的 markdown 代码块
        if content.startswith("```"):
            lines = content.split("\n")
            content = "\n".join(lines[1:-1]) if len(lines) > 2 else content

        try:
            obj = json.loads(content)
        except json.JSONDecodeError:
            # 尝试提取 JSON 片段
            import re
            m = re.search(r'\{.*\}', content, re.DOTALL)
            if m:
                try:
                    obj = json.loads(m.group())
                except json.JSONDecodeError:
                    return (content[:100], Action.STOP, "无法解析决策")
            else:
                return (content[:100], Action.STOP, "无法解析决策")

        desc = obj.get("description", "")[:100]
        action_str = obj.get("action", "STOP").upper().strip()
        reason = obj.get("reason", "")[:100]

        # 映射动作
        action_map = {
            "GO": Action.GO, "LEFT": Action.LEFT, "RIGHT": Action.RIGHT,
            "STOP": Action.STOP, "BACK": Action.BACK, "GREET": Action.GREET,
            "向前": Action.GO, "直行": Action.GO, "前进": Action.GO,
            "左转": Action.LEFT, "右转": Action.RIGHT,
            "停止": Action.STOP, "后退": Action.BACK,
            "打招呼": Action.GREET, "欢迎": Action.GREET, "问好": Action.GREET,
        }
        action = action_map.get(action_str, Action.STOP)

        return (desc, action, reason)

    # ── 底盘控制 ──

    def _ensure_chassis(self):
        """懒加载底盘控制器"""
        if self._chassis is not None:
            return
        from .chassis_controller import ChassisController, ChassisConfig
        self._chassis = ChassisController(ChassisConfig(
            default_linear_speed=self.config.move_speed,
            default_angular_speed=self.config.turn_speed,
            step_forward_distance=self.config.step_distance,
            turn_angle=self.config.turn_angle,
        ))

    # ── 语音输出 ──

    # 预制欢迎语库
    GREETING_PHRASES = [
        "你好！欢迎光临！",
        "你好，很高兴见到你！",
        "您好，请问需要帮助吗？",
    ]

    def _ensure_speech(self):
        """懒加载 TTS 引擎"""
        if hasattr(self, "_tts") and self._tts is not None:
            return
        from .zeng_tts import ZengTTSEngine
        self._tts = ZengTTSEngine()
        self._greet_index = 0

    def _say(self, text: str) -> bool:
        """TTS 播报: ROS2 topic + 本地 WAV 双路输出"""
        import json as _json, subprocess as _sp
        ok = False

        # 路1: 本地 Piper WAV
        self._ensure_speech()
        try:
            wav = self._tts.synthesize(text)
            if wav:
                ok = True
        except Exception as e:
            print(f"  [TTS] WAV 失败: {e}")

        # 路2: ROS2 topic → vision service Piper
        topic = "/topic_tts_0_283"
        ros_setup = "/opt/ros/jazzy/setup.bash"
        payload = _json.dumps({"status": "play", "text": text}, ensure_ascii=False)
        escaped = payload.replace('"', '\\"')
        cmd = (
            f'ros2 topic pub -1 {topic} std_msgs/msg/String '
            f'\'data: "' + escaped + '"\''
        )
        full = f". {ros_setup} 2>/dev/null && timeout 8 {cmd}"
        try:
            _sp.run(full, shell=True, executable='/bin/bash',
                    capture_output=True, text=True, timeout=10)
            ok = True
        except Exception:
            pass

        return ok

    def _greet(self) -> str:
        """轮播欢迎语, 返回使用的欢迎文本"""
        self._ensure_speech()
        phrase = self.GREETING_PHRASES[self._greet_index % len(self.GREETING_PHRASES)]
        self._greet_index += 1
        self._say(phrase)
        return phrase

    def _execute_action(self, action: Action) -> bool:
        """执行底盘动作或交互动作"""
        try:
            if action == Action.GO:
                self._ensure_chassis()
                self._chassis.move_forward(self.config.step_distance)
            elif action == Action.LEFT:
                self._ensure_chassis()
                self._chassis.turn_left(self.config.turn_angle)
            elif action == Action.RIGHT:
                self._ensure_chassis()
                self._chassis.turn_right(self.config.turn_angle)
            elif action == Action.BACK:
                self._ensure_chassis()
                self._chassis.move_backward(self.config.step_distance)
            elif action == Action.GREET:
                # 停下 + 播报欢迎语
                greeting = self._greet()
                print(f"  🎤 播报: \"{greeting}\"")
            elif action == Action.STOP:
                pass  # 不移动
            return True
        except Exception as e:
            print(f"[Explorer] 执行错误: {e}")
            return False

    # ── 探索循环 ──

    def explore(self, steps: int = 5) -> list[ExploreStep]:
        """执行探索循环

        Args:
            steps: 最大探索步数

        Returns:
            探索步骤列表
        """
        print(f"\n{'='*50}")
        print(f"  视觉探索模式")
        print(f"  摄像头: {self.config.camera_source}")
        print(f"  步长: {self.config.step_distance}m, 转角: {self.config.turn_angle}°")
        print(f"  最大步数: {steps}")
        print(f"{'='*50}\n")

        self._steps = []

        for i in range(1, steps + 1):
            print(f"\n--- Step {i}/{steps} ---")

            # 1. 拍照
            print("  📷 拍照...", end=" ", flush=True)
            try:
                img_path = self._capture(i)
                print(f"ok ({img_path})")
            except Exception as e:
                print(f"失败: {e}")
                self._steps.append(ExploreStep(
                    step=i, image_path="", description="",
                    action=Action.STOP, reason=f"摄像头错误: {e}",
                    executed=False, error=str(e),
                ))
                break

            # 2. 视觉分析
            print("  🧠 分析...", end=" ", flush=True)
            try:
                desc, action, reason = self._analyze(img_path)
            except Exception as e:
                print(f"失败: {e}")
                desc, action, reason = "", Action.STOP, f"视觉分析失败: {e}"
            print(f"→ {action.value} | {desc[:60]}")

            # 3. 执行
            if action == Action.STOP:
                print(f"  ⏹ 停止: {reason}")
                self._steps.append(ExploreStep(
                    step=i, image_path=img_path, description=desc,
                    action=action, reason=reason, executed=True,
                ))
                break

            print(f"  🚗 执行 {action.value}...", end=" ", flush=True)
            ok = self._execute_action(action)
            if action == Action.GREET:
                status = "已打招呼" if ok else "失败"
            else:
                status = "ok" if ok else "失败"
            print(status)

            self._steps.append(ExploreStep(
                step=i, image_path=img_path, description=desc,
                action=action, reason=reason, executed=ok,
            ))

            if not ok:
                print("  ❌ 底盘执行失败, 停止探索")
                break

            time.sleep(1.0)

        return self._steps

    def summary(self) -> str:
        """探索结果摘要"""
        if not self._steps:
            return "无探索步骤"

        lines = [
            f"\n{'='*40}",
            f"  探索完成: {len(self._steps)} 步",
            f"{'='*40}",
        ]
        for s in self._steps:
            icon = {Action.GO: "⬆", Action.LEFT: "⬅", Action.RIGHT: "➡",
                    Action.STOP: "⏹", Action.BACK: "⬇", Action.GREET: "👋"}.get(s.action, "?")
            lines.append(
                f"  {icon} Step{s.step}: {s.action.value} | "
                f"{s.description[:50]} | {s.reason[:40]}"
            )
        return "\n".join(lines)
