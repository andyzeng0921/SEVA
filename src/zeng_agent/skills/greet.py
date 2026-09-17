"""
greet.py — 语音打招呼 Skill

检测到人时用 TTS 播报欢迎语。
"""
from __future__ import annotations

import json as _json
import subprocess as _sp
from dataclasses import dataclass
from typing import Optional

from .base import Skill, SkillInput, SkillOutput


GREETING_PHRASES = [
    "你好！欢迎光临！",
    "你好，很高兴见到你！",
    "您好，请问需要帮助吗？",
]


@dataclass
class GreetConfig:
    """打招呼配置"""
    tts_topic: str = "/topic_tts_0_283"
    ros_setup: str = "/opt/ros/jazzy/setup.bash"
    dry_run: bool = True


class GreetSkill(Skill):
    """语音打招呼技能

    输入参数:
      - phrase: 自定义欢迎语 (可选, 默认轮播预置语库)
      - person_direction: 人物方位 (记录用)

    输出:
      - success: bool
      - data: {phrase, wav_path, ...}
    """

    name = "greet"
    description = "TTS 语音打招呼: 检测到人时播放欢迎语"
    version = "1.0.0"

    def __init__(self, config: Optional[GreetConfig] = None):
        self.config = config or GreetConfig()
        self._index = 0
        self._tts = None

    def _ensure_tts(self):
        if self._tts is not None:
            return
        from ..zeng_tts import ZengTTSEngine
        self._tts = ZengTTSEngine()

    def _say_ros2(self, text: str) -> bool:
        """发布到 ROS2 TTS topic"""
        if self.config.dry_run:
            return True
        payload = _json.dumps({"status": "play", "text": text}, ensure_ascii=False)
        escaped = payload.replace('"', '\\"')
        cmd = (
            f'ros2 topic pub -1 {self.config.tts_topic} std_msgs/msg/String '
            f'\'data: "' + escaped + '"\''
        )
        full = f". {self.config.ros_setup} 2>/dev/null && timeout 8 {cmd}"
        try:
            _sp.run(full, shell=True, executable='/bin/bash',
                    capture_output=True, text=True, timeout=10)
            return True
        except Exception:
            return False

    def execute(self, input: SkillInput) -> SkillOutput:
        phrase = input.get("phrase", "")
        if not phrase:
            phrase = GREETING_PHRASES[self._index % len(GREETING_PHRASES)]
            self._index += 1

        wav_path = None
        if not self.config.dry_run:
            self._ensure_tts()
            try:
                wav_path = self._tts.synthesize(phrase)
            except Exception as e:
                pass  # WAV 失败不阻塞

        ros2_ok = self._say_ros2(phrase)

        return SkillOutput(
            success=ros2_ok or wav_path is not None,
            data={
                "phrase": phrase,
                "wav_path": wav_path,
                "ros2_sent": ros2_ok,
            },
            message=f"播报: {phrase}",
        )
