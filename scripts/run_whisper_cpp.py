#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Run whisper.cpp and emit JSON transcript")
    parser.add_argument("--binary", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--wav", required=True)
    parser.add_argument("--language", default="zh")
    args = parser.parse_args()

    with tempfile.TemporaryDirectory(prefix="zeng-whisper-") as temp_dir:
        out_prefix = str(Path(temp_dir) / "result")
        cmd = [
            args.binary,
            "-m",
            args.model,
            "-f",
            args.wav,
            "-l",
            args.language,
            "-nt",
            "-np",
            "-of",
            out_prefix,
            "-otxt",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if result.returncode != 0:
            print(
                json.dumps(
                    {"text": "", "confidence": 0.0, "source": "whisper_cpp", "error": result.stderr.strip() or "whisper.cpp failed"},
                    ensure_ascii=False,
                )
            )
            return result.returncode

        transcript_path = Path(out_prefix + ".txt")
        text = transcript_path.read_text(encoding="utf-8").strip() if transcript_path.exists() else result.stdout.strip()
        print(json.dumps({"text": text, "confidence": 0.8, "source": "whisper_cpp"}, ensure_ascii=False))
        return 0


if __name__ == "__main__":
    sys.exit(main())
