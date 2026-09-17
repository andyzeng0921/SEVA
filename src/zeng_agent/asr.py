from __future__ import annotations

import json
import os
import shlex
import subprocess
import tempfile
from dataclasses import asdict
from pathlib import Path

import requests

from .models import AsrResult, CommandText


class LocalMicASRProvider:
    def __init__(self, device: str, sample_rate: int, transcriber_command: str | None, channels: int = 1, duration_seconds: int = 4) -> None:
        self.device = device
        self.sample_rate = sample_rate
        self.transcriber_command = transcriber_command
        self.channels = channels
        self.duration_seconds = duration_seconds

    def record_once(self, output_path: str) -> subprocess.CompletedProcess[str]:
        cmd = [
            "arecord",
            "-D",
            self.device,
            "-f",
            "S16_LE",
            "-r",
            str(self.sample_rate),
            "-c",
            str(self.channels),
            "-d",
            str(self.duration_seconds),
            output_path,
        ]
        return subprocess.run(cmd, capture_output=True, text=True, check=False)

    def transcribe_existing_file(self, wav_path: str) -> AsrResult:
        if not self.transcriber_command:
            return AsrResult(success=False, message="Local ASR needs a transcriber command")

        rendered = self.transcriber_command.replace("{wav_path}", shlex.quote(wav_path))
        result = subprocess.run(rendered, shell=True, capture_output=True, text=True, check=False)
        if result.returncode != 0:
            return AsrResult(success=False, message=result.stderr.strip() or "Local transcription command failed")

        output = result.stdout.strip()
        if not output:
            return AsrResult(success=False, message="Local transcription command returned empty output")

        try:
            payload = json.loads(output)
            command = CommandText(
                text=str(payload["text"]),
                confidence=float(payload.get("confidence", 1.0)),
                source=payload.get("source", "local_asr"),
            )
            return AsrResult(success=True, message="ok", command=command)
        except json.JSONDecodeError:
            return AsrResult(success=True, message="ok", command=CommandText(text=output, source="local_asr"))

    def capture_and_transcribe(self) -> AsrResult:
        temp_dir = Path(tempfile.mkdtemp(prefix="zeng-agent-"))
        wav_path = temp_dir / "capture.wav"
        record = self.record_once(str(wav_path))
        if record.returncode != 0:
            return AsrResult(success=False, message=record.stderr.strip() or "arecord failed")
        return self.transcribe_existing_file(str(wav_path))


class RemoteASRProvider:
    def __init__(self, base_url: str, timeout_seconds: int = 20) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds

    def parse_response(self, payload: dict) -> CommandText:
        return CommandText(
            text=str(payload["text"]),
            confidence=float(payload.get("confidence", 1.0)),
            source=payload.get("source", "remote_asr"),
        )

    def transcribe_existing_file(self, wav_path: str) -> AsrResult:
        with open(wav_path, "rb") as handle:
            response = requests.post(
                f"{self.base_url}/transcribe",
                files={"file": (os.path.basename(wav_path), handle, "audio/wav")},
                timeout=self.timeout_seconds,
            )
        response.raise_for_status()
        return AsrResult(success=True, message="ok", command=self.parse_response(response.json()))


def as_dict(result: AsrResult) -> dict:
    payload = asdict(result)
    if result.command is not None:
        payload["command"] = asdict(result.command)
    return payload
