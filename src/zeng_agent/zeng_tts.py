"""
zeng_tts.py — 独立 TTS 引擎

绕过 vision service 的 ALSA 硬件依赖, 直接用 piper CLI 生成 WAV 文件。
可作为 vision service TTS 的验证/回退方案。

用法:
    engine = ZengTTSEngine()
    wav_path = engine.synthesize("你好世界")
    # → /tmp/zeng_tts/zeng_tts_001.wav
"""

from __future__ import annotations

import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class TTSConfig:
    piper_bin: str = "/opt/robot/env/bin/piper"
    model_path: str = (
        "/opt/robot/env/lib/python3.12/"
        "site-packages/autolife_robot_vision/assets/tts/piper/"
        "zh_CN-huayan-medium.onnx"
    )
    output_dir: str = "/tmp/zeng_tts"
    max_files: int = 20  # 保留最近 N 个文件
    output_sample_rate: int = 22050


class ZengTTSEngine:
    """独立 TTS 引擎 — 用 piper CLI 生成 WAV 文件"""

    def __init__(self, config: Optional[TTSConfig] = None):
        self.config = config or TTSConfig()
        self._counter = 0
        os.makedirs(self.config.output_dir, exist_ok=True)
        self._verify_binary()

    def _verify_binary(self):
        """检查 piper 二进制和模型是否存在"""
        bin_path = Path(self.config.piper_bin)
        model_path = Path(self.config.model_path)

        if not bin_path.exists():
            raise FileNotFoundError(f"piper binary not found: {bin_path}")
        if not model_path.exists():
            raise FileNotFoundError(f"piper model not found: {model_path}")

    def synthesize(self, text: str) -> Optional[str]:
        """合成语音, 返回 WAV 文件路径

        Args:
            text: 要合成的文本

        Returns:
            WAV 文件路径, 或 None (合成失败)
        """
        if not text.strip():
            return None

        self._counter += 1
        out_path = os.path.join(
            self.config.output_dir,
            f"zeng_tts_{self._counter:03d}.wav",
        )

        try:
            result = subprocess.run(
                [
                    self.config.piper_bin,
                    "--model", self.config.model_path,
                    "--output_file", out_path,
                ],
                input=text.strip(),
                capture_output=True,
                text=True,
                timeout=30,
            )

            if result.returncode != 0:
                print(f"[ZengTTS] piper error: {result.stderr[:200]}")
                self._counter -= 1
                return None

            # 验证文件
            if not os.path.exists(out_path) or os.path.getsize(out_path) == 0:
                print(f"[ZengTTS] empty output file: {out_path}")
                self._counter -= 1
                return None

            # 清理旧文件
            self._cleanup()

            return out_path

        except FileNotFoundError:
            print(f"[ZengTTS] piper binary not found: {self.config.piper_bin}")
            self._counter -= 1
            return None
        except subprocess.TimeoutExpired:
            print(f"[ZengTTS] piper timeout for text: {text[:50]}")
            self._counter -= 1
            return None
        except Exception as e:
            print(f"[ZengTTS] error: {e}")
            self._counter -= 1
            return None

    def _cleanup(self):
        """删除超过 max_files 的旧文件"""
        files = sorted(
            Path(self.config.output_dir).glob("zeng_tts_*.wav"),
            key=lambda p: p.stat().st_mtime,
        )
        excess = len(files) - self.config.max_files
        for f in files[:excess]:
            try:
                f.unlink()
            except Exception:
                pass

    def say(self, text: str) -> bool:
        """合成并尝试播放 (aplay)

        Returns:
            True 如果合成成功
        """
        wav_path = self.synthesize(text)
        if not wav_path:
            return False

        # 尝试播放
        return self._try_play(wav_path)

    @staticmethod
    def _try_play(wav_path: str) -> bool:
        """尝试用 aplay 播放 WAV"""
        try:
            subprocess.run(
                ["aplay", "-q", wav_path],
                timeout=10,
                capture_output=True,
            )
            return True
        except Exception:
            # 播放失败不影响合成结果
            return True  # 合成成功就算成功

    def latest_wav(self) -> Optional[str]:
        """返回最新生成的 WAV 文件路径"""
        files = sorted(
            Path(self.config.output_dir).glob("zeng_tts_*.wav"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        return str(files[0]) if files else None
