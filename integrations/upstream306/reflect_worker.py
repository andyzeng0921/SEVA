"""Isolate REFLECT's original relative-directory API from the HTTP process."""
import contextlib
import json
import sys
from types import SimpleNamespace
from filelock import FileLock
from pathlib import Path
from .reflect_adapter import _summarize_in_process


def main():
    request = json.load(sys.stdin)
    runtime = Path(request['runtime'])
    with FileLock(str(runtime / 'reflect.lock')), contextlib.redirect_stdout(sys.stderr):
        result = _summarize_in_process(request['task'], runtime,
            SimpleNamespace(**request['llm_config']) if request['llm_config'] else None)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()
