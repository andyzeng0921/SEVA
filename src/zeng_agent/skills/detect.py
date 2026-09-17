"""Visual target detection and motion-decision skill."""
from __future__ import annotations

import base64
import json
import math
import os
import re
import time
from dataclasses import dataclass
from typing import Any, Optional

import requests

from .base import Skill, SkillInput, SkillOutput


@dataclass
class DetectConfig:
    api_key: str = ""
    api_url: str = "http://127.0.0.1:8000/v1/chat/completions"
    model: str = "qwen3.5-35b-a3b"
    camera_source: str = "auto_front"
    timeout: int = 30
    max_tokens: int = 2048
    output_dir: str = "/tmp/zeng_fire_search"
    fresh_frame_timeout: float = 2.0
    max_turn_angle: float = 180.0
    min_approach: float = 0.1
    max_approach: float = 3.0
    alignment_tolerance_deg: float = 3.0


DECISION_PROMPT = """你是轮式机器人的前向视觉导航模块。图像来自机器人头部前向相机，不要假设机器人当前朝向已经正确。

任务是寻找并靠近红色消防栓。请只根据当前这张图作判断：
- turn_angle: 为使头部前向相机正对消防栓，机器人应转多少度。正数=向右转，负数=向左转。
- approach_distance_m: 正对后建议向前走的安全距离。未发现消防栓或道路不安全时必须为0。
- camera_forward_ready: 只有相机已抬头、能看到前方环境时为true；若画面主要是地板、机器人机身或严重朝下，必须为false，所有运动建议设为0。
- ready_to_approach: 只有同时满足以下条件才为true：确认看到消防栓、消防栓位于画面中央（误差约3度以内）、这是正面观察而非背面推测、正前方路径可通行。
- 若消防栓未居中，ready_to_approach必须为false，先给出转角，不能建议前进。
- 若未发现消防栓且camera_forward_ready=true，必须给出下一次非零搜索转角（绝对值30到120度），但前进距离必须为0，不能返回0度。

严格输出一个JSON对象，不要附加其他文字：
{"description":"30字以内","camera_forward_ready":true,"fire_hydrant":"YES|NO","has_person":"YES|NO","turn_angle":0.0,"approach_distance_m":0.0,"ready_to_approach":false,"confidence":0.0}
"""


class VisualDetectSkill(Skill):
    name = "visual_detect"
    description = "从前向共享内存取得新帧，识别消防栓并输出受安全门控的视觉决策"
    version = "3.0.0"

    def __init__(self, config: Optional[DetectConfig] = None):
        self.config = config or DetectConfig()
        os.makedirs(self.config.output_dir, exist_ok=True)
        self._counter = 0

    @staticmethod
    def _parse_response(content: str) -> dict[str, Any]:
        stripped = content.strip()
        if stripped.startswith("```"):
            stripped = re.sub(r"^```(?:json)?\s*|\s*```$", "", stripped, flags=re.IGNORECASE)
        try:
            value = json.loads(stripped)
            return value if isinstance(value, dict) else {}
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", stripped, re.DOTALL)
            if not match:
                return {}
            try:
                value = json.loads(match.group(0))
                return value if isinstance(value, dict) else {}
            except json.JSONDecodeError:
                return {}

    @staticmethod
    def _number(value: Any, default: float = 0.0) -> float:
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            return default
        return parsed if math.isfinite(parsed) else default

    @staticmethod
    def _yes(value: Any) -> bool:
        return str(value).strip().upper() in {"YES", "TRUE", "1"}

    def _normalize_decision(self, raw: dict[str, Any]) -> dict[str, Any]:
        camera_forward_ready = self._yes(raw.get("camera_forward_ready", False))
        found = self._yes(raw.get("fire_hydrant", "NO"))
        turn = self._number(raw.get("turn_angle"), 0.0)
        turn = max(-self.config.max_turn_angle, min(self.config.max_turn_angle, turn))
        distance = self._number(raw.get("approach_distance_m"), 0.0)
        if not found or distance <= 0:
            distance = 0.0
        else:
            distance = max(self.config.min_approach, min(self.config.max_approach, distance))
        model_ready = self._yes(raw.get("ready_to_approach", False))
        ready = camera_forward_ready and found and model_ready and abs(turn) <= self.config.alignment_tolerance_deg
        if not camera_forward_ready:
            turn = 0.0
            distance = 0.0
        if not ready:
            distance = 0.0 if not found else distance
        confidence = max(0.0, min(1.0, self._number(raw.get("confidence"), 0.0)))
        return {
            "description": str(raw.get("description", ""))[:120],
            "camera_forward_ready": camera_forward_ready,
            "fire_hydrant": "YES" if found else "NO",
            "has_person": "YES" if self._yes(raw.get("has_person", "NO")) else "NO",
            "confidence": confidence,
            "turn_angle": round(turn, 1),
            "turn_angle_raw": raw.get("turn_angle", 0),
            "approach_distance_m": round(distance, 2),
            "approach_distance_raw": raw.get("approach_distance_m", 0),
            "ready_to_approach": ready,
        }

    def execute(self, input: SkillInput) -> SkillOutput:
        camera_source = str(input.get("camera_source", self.config.camera_source))
        require_newer_than = input.get("require_newer_than")
        save_image = bool(input.get("save_image", True))

        from ..vision_provider import VisionConfig, VisionProvider

        provider = VisionProvider(VisionConfig(
            api_key=self.config.api_key,
            api_url=self.config.api_url,
            model=self.config.model,
            camera_source=camera_source,
            timeout=self.config.timeout,
            max_tokens=self.config.max_tokens,
            fresh_frame_timeout=self.config.fresh_frame_timeout,
        ))
        jpeg_data, metadata, error = provider.capture_live_frame(
            require_newer_than=str(require_newer_than) if require_newer_than else None,
        )
        if jpeg_data is None or metadata is None:
            return SkillOutput(success=False, error=error or "拍照失败", message="未取得新的前向图像")

        self._counter += 1
        image_path = ""
        if save_image:
            image_path = os.path.join(
                self.config.output_dir,
                f"detect_{self._counter:04d}_{time.time_ns()}.jpg",
            )
            with open(image_path, "wb") as image_file:
                image_file.write(jpeg_data)

        payload = {
            "model": self.config.model,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": DECISION_PROMPT},
                    {"type": "image_url", "image_url": {
                        "url": "data:image/jpeg;base64," + base64.b64encode(jpeg_data).decode("ascii")
                    }},
                ],
            }],
            "max_tokens": self.config.max_tokens,
            "temperature": 0.1,
        }
        try:
            response = requests.post(
                self.config.api_url,
                headers={
                    "Authorization": f"Bearer {self.config.api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
                timeout=self.config.timeout,
            )
            response.raise_for_status()
            body = response.json()
            message = body["choices"][0]["message"]
            content = message.get("content") or message.get("reasoning") or ""
        except (requests.RequestException, ValueError, KeyError, IndexError) as exc:
            return SkillOutput(
                success=False,
                error=f"视觉模型请求失败: {exc}",
                data={"image_path": image_path, "frame_id": metadata.frame_id},
            )

        raw = self._parse_response(content)
        if not raw:
            return SkillOutput(
                success=False,
                error="视觉模型未返回有效JSON",
                data={"image_path": image_path, "frame_id": metadata.frame_id, "raw_content": content[:300]},
            )
        result = self._normalize_decision(raw)
        result.update({
            "image_path": image_path,
            "camera_source": metadata.source,
            "camera_transport": metadata.transport,
            "frame_id": metadata.frame_id,
            "frame_fresh": True,
            "captured_at": metadata.captured_at,
            "raw_content": content[:300],
        })
        return SkillOutput(
            success=True,
            data=result,
            message=(
                f"检测: 消防栓={result['fire_hydrant']}, 转向={result['turn_angle']:.1f}°, "
                f"可前进={result['ready_to_approach']}, 距离={result['approach_distance_m']:.2f}m"
            ),
        )
