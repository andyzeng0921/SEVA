from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(slots=True)
class CommandText:
    text: str
    source: str = "text"
    confidence: float = 1.0
    timestamp: str = field(default_factory=utc_now_iso)


@dataclass(slots=True)
class ParsedIntent:
    intent: str
    slots: dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0
    requires_confirmation: bool = False


@dataclass(slots=True)
class ExecutionRequest:
    intent: str
    validated_slots: dict[str, Any]
    mode: str


@dataclass(slots=True)
class ExecutionResult:
    success: bool
    message: str
    data: dict[str, Any] = field(default_factory=dict)
    latency_ms: float = 0.0


@dataclass(slots=True)
class AsrResult:
    success: bool
    message: str
    command: CommandText | None = None

