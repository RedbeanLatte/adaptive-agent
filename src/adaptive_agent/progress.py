from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, IO, Protocol


@dataclass(frozen=True)
class ProgressEvent:
    kind: str
    message: str
    fields: dict[str, Any] = field(default_factory=dict)


class ProgressReporter(Protocol):
    def emit(self, event: ProgressEvent) -> None: ...


class NullProgressReporter:
    def emit(self, event: ProgressEvent) -> None:
        return None


class ConsoleProgressReporter:
    def __init__(self, output: IO[str]) -> None:
        self.output = output

    def emit(self, event: ProgressEvent) -> None:
        message = event.message.strip()
        if not message:
            return
        self.output.write(f"[agent] {message}\n")
        self.output.flush()


def progress_event(kind: str, message: str, **fields: Any) -> ProgressEvent:
    return ProgressEvent(kind=kind, message=message, fields=fields)


def summarize_arguments(arguments: dict[str, Any], *, limit: int = 120) -> str:
    if not isinstance(arguments, dict) or not arguments:
        return ""
    parts: list[str] = []
    for key, value in arguments.items():
        if key == "argv" and isinstance(value, list):
            value_text = "[" + ", ".join(str(item) for item in value[:3]) + (", ..." if len(value) > 3 else "") + "]"
        else:
            value_text = str(value)
        parts.append(f"{key}={value_text}")
    text = ", ".join(parts)
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."
