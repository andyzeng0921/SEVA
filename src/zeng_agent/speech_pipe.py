"""
speech_pipe.py — 语音对话管道

将机器人本地 ASR/TTS 接入 LangGraph 智能体:

  ASR: ros2 topic echo /topic_chat_text_{NODE_ID}  → 解析文本
       ↓
  LangGraph: BrainOrchestrator.run_text_task(text)
       ↓
  TTS: 双路输出
    1. ros2 topic pub /topic_tts_{NODE_ID} → vision service Piper TTS
    2. zeng_tts 独立 Piper WAV 引擎 → /tmp/zeng_tts/*.wav (验证用)

支持:
  - 连续语音对话模式 (listen once or continuous)
  - 唤醒词过滤 (可选)
  - 回退到 whisper.cpp 本地 ASR (若 ROS2 topic 无数据)
"""
from __future__ import annotations

import json
import re
import subprocess
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

from .zeng_tts import ZengTTSEngine, TTSConfig


@dataclass
class SpeechConfig:
    """语音管道配置"""
    topic_node_id: str = "0_283"
    ros_setup: str = "/opt/ros/jazzy/setup.bash"

    # ASR
    asr_topic: str = ""          # 自动生成为 /topic_chat_text_{NODE_ID}
    asr_timeout_s: float = 15.0  # 单次监听超时
    asr_min_length: int = 2      # 最少字符数 (过滤噪音)

    # TTS
    tts_topic: str = ""          # 自动生成为 /topic_tts_{NODE_ID}
    tts_enabled: bool = True

    # 行为
    continuous: bool = False     # True=持续对话, False=单次
    text_mode: bool = False      # True=从 stdin 读文字(不回退TTS), False=从麦克风
    wake_words: list[str] = field(default_factory=list)
    response_max_chars: int = 200

    def __post_init__(self):
        if not self.asr_topic:
            self.asr_topic = f"/topic_chat_text_{self.topic_node_id}"
        if not self.tts_topic:
            self.tts_topic = f"/topic_tts_{self.topic_node_id}"


class SpeechPipe:
    """语音对话管道 — ASR → Agent → TTS

    用法:
        pipe = SpeechPipe(SpeechConfig(), brain_orchestrator)
        pipe.listen_once()      # 单次: 监听→对话→TTS
        pipe.listen_loop()      # 持续对话循环
        pipe.say("你好")        # 仅 TTS 输出
    """

    def __init__(self, config: SpeechConfig, brain: "BrainOrchestrator"):
        self.config = config
        self.brain = brain
        self._asr_proc: Optional[subprocess.Popen] = None
        self._running = False
        self._on_asr_text: Optional[Callable] = None
        # 独立 TTS 引擎 (生成 WAV 验证文件)
        self._zeng_tts: Optional[ZengTTSEngine] = None
        try:
            self._zeng_tts = ZengTTSEngine()
        except Exception as e:
            print(f"[SpeechPipe] zeng_tts init failed: {e}")

    # ── TTS 输出 ──

    def say(self, text: str) -> bool:
        """通过 TTS 播报文本 (双路: ROS2 topic + 本地 WAV)"""
        if not self.config.tts_enabled:
            print(f"[TTS] (muted): {text}")
            return False

        if len(text) > self.config.response_max_chars:
            text = text[:self.config.response_max_chars] + "..."

        success = False

        # 路1: ROS2 topic → vision service Piper TTS (依赖扬声器硬件)
        payload = json.dumps({"status": "play", "text": text}, ensure_ascii=False)
        escaped = payload.replace('"', '\\"')
        cmd = (
            f'ros2 topic pub -1 {self.config.tts_topic} '
            f'std_msgs/msg/String \'data: "' + escaped + '"\''
        )
        full = f". {self.config.ros_setup} 2>/dev/null && timeout 8 {cmd}"
        try:
            subprocess.run(
                full, shell=True, executable='/bin/bash',
                capture_output=True, text=True, timeout=10,
            )
            success = True
        except Exception as e:
            print(f"[TTS] ROS2 发布失败: {e}")

        # 路2: 独立 Piper WAV 引擎 (验证用, 不依赖扬声器)
        if self._zeng_tts:
            try:
                wav_path = self._zeng_tts.synthesize(text)
                if wav_path:
                    success = True
            except Exception as e:
                print(f"[TTS] WAV 生成失败: {e}")

        return success

    def stop_speaking(self) -> None:
        """停止当前 TTS 播放"""
        payload = json.dumps({"status": "stop"})
        escaped = payload.replace('"', '\\"')
        cmd = (
            f'ros2 topic pub -1 {self.config.tts_topic} '
            f'std_msgs/msg/String \'data: "' + escaped + '"\''
        )
        full = f". {self.config.ros_setup} 2>/dev/null && timeout 8 {cmd}"
        try:
            subprocess.run(
                full, shell=True, executable='/bin/bash',
                capture_output=True, text=True, timeout=10,
            )
        except Exception:
            pass

    # ── ASR 输入 ──

    def _start_asr_listener(self) -> None:
        """启动 ASR 监听子进程"""
        if self._asr_proc is not None:
            return

        cmd = f"ros2 topic echo {self.config.asr_topic}"
        full = f". {self.config.ros_setup} 2>/dev/null && {cmd}"
        self._asr_proc = subprocess.Popen(
            full, shell=True,
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            text=True, bufsize=1,
        )

    def _stop_asr_listener(self) -> None:
        if self._asr_proc:
            self._asr_proc.terminate()
            try:
                self._asr_proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self._asr_proc.kill()
            self._asr_proc = None

    @staticmethod
    def _parse_asr_line(line: str) -> Optional[str]:
        """解析 ros2 topic echo 输出行, 提取 ASR 文本

        ros2 topic echo 输出格式:
          data: '你好'
        """
        # 匹配 'data: ...' 格式
        m = re.search(r"data:\s*'(.+?)'", line)
        if m:
            return m.group(1)
        # 也尝试匹配 JSON 格式
        m = re.search(r'data:\s*"(.*?)"', line)
        if m:
            return m.group(1)
        return None

    def _wait_for_asr_text(self, timeout_s: float | None = None) -> Optional[str]:
        """等待 ASR 识别文本

        Returns:
            识别文本, 或 None (超时)
        """
        if timeout_s is None:
            timeout_s = self.config.asr_timeout_s

        self._start_asr_listener()
        deadline = time.monotonic() + timeout_s

        while time.monotonic() < deadline:
            line = self._asr_proc.stdout.readline()
            if not line:
                time.sleep(0.05)
                continue

            text = self._parse_asr_line(line)
            if text and len(text.strip()) >= self.config.asr_min_length:
                text = text.strip()
                print(f"[ASR] 用户说: {text}")
                if self._on_asr_text:
                    self._on_asr_text(text)
                return text

        return None

    # ── 对话方法 ──

    def listen_once(self) -> dict:
        """单次语音对话: 监听→Agent处理→TTS回复

        Returns:
            {"asr_text": str, "agent_result": dict, "tts_sent": bool}
        """
        # 1. 获取输入 (ASR 或 stdin)
        if self.config.text_mode:
            try:
                text = input("\n[TextMode] 输入指令: ").strip()
            except (EOFError, KeyboardInterrupt):
                return {"asr_text": None, "agent_result": None, "tts_sent": False}
            if not text:
                return {"asr_text": None, "agent_result": None, "tts_sent": False}
            print(f"[TextMode] 用户说: {text}")
        else:
            text = self._wait_for_asr_text()
            if not text:
                self.say("没有听到您说话")
                return {"asr_text": None, "agent_result": None, "tts_sent": False}

        # 2. 唤醒词过滤
        if self.config.wake_words:
            has_wake = any(w in text for w in self.config.wake_words)
            if not has_wake:
                print(f"[SpeechPipe] 未检测到唤醒词, 忽略: {text}")
                return {"asr_text": text, "agent_result": None, "tts_sent": False, "ignored": True}

        # 3. Agent 处理
        print(f"[SpeechPipe] → Agent: {text}")
        agent_result = self.brain.run_text_task(text, confirmed=True)

        # 4. TTS 回复
        tts_text = self._extract_reply(agent_result)
        tts_sent = self.say(tts_text) if tts_text else False

        return {
            "asr_text": text,
            "agent_result": agent_result,
            "tts_text": tts_text,
            "tts_sent": tts_sent,
        }

    def listen_loop(self) -> None:
        """持续语音对话循环"""
        self._running = True
        print(f"[SpeechPipe] 开始持续语音对话 (Ctrl+C 退出)")
        print(f"  ASR topic: {self.config.asr_topic}")
        print(f"  TTS topic: {self.config.tts_topic}")

        try:
            while self._running:
                print("\n[SpeechPipe] 等待语音输入...")
                result = self.listen_once()

                if result.get("ignored"):
                    continue

                if not result["asr_text"]:
                    continue

                if "停止" in result["asr_text"] or "退出" in result["asr_text"]:
                    self.say("好的，再见")
                    break

        except KeyboardInterrupt:
            print("\n[SpeechPipe] 用户中断")
        finally:
            self._running = False
            self._stop_asr_listener()

    def stop(self) -> None:
        """停止对话循环"""
        self._running = False
        self.stop_speaking()
        self._stop_asr_listener()

    @staticmethod
    def _extract_reply(agent_result: dict) -> str:
        """从 Agent 结果提取 TTS 回复文本"""
        if not agent_result:
            return "处理失败"

        # 从 steps 提取消息
        steps = agent_result.get("steps", [])
        if steps:
            last_step = steps[-1]
            msg = last_step.get("message", "")
            if msg:
                return msg

        # 从 message 字段
        msg = agent_result.get("message", "")
        if msg:
            return msg

        # 从 status
        status = agent_result.get("status", "")
        if status == "completed":
            return "已完成"
        elif status == "failed":
            return "执行失败"

        return "好的"
