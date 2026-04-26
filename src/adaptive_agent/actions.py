from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any


class ActionParseError(ValueError):
    """Raised when the model output cannot be parsed into a valid action."""


ACTION_TYPES: dict[str, tuple[str, ...]] = {
    "answer": ("final_answer",),
    "ask_user": ("question",),
    "run_builtin": ("tool_name",),
    "reuse_tool": ("tool_name",),
    "create_tool": ("tool_name", "code"),
    "request_approval": ("question",),
}

BUILTIN_ACTION_TYPE_ALIASES = frozenset(
    {
        "inspect_file",
        "read_text_file",
        "write_text_file",
        "list_files",
    }
)


@dataclass
class Action:
    action_type: str
    fields: dict[str, Any] = field(default_factory=dict)
    reasoning_summary: str = ""


_FENCE_RE = re.compile(r"^```(?:json)?\s*\n?|\n?```\s*$", re.IGNORECASE | re.MULTILINE)


def _strip_fences(raw: str) -> str:
    stripped = _FENCE_RE.sub("", raw).strip()
    return stripped or raw.strip()


def _decode_first_json_object(text: str) -> Any:
    """Decode the first complete JSON object in ``text``.

    Tolerates trailing prose or extra closing braces (a common LLM artifact).
    Returns the parsed object; raises ActionParseError on failure.
    """
    candidate = text.strip()
    start = candidate.find("{")
    if start == -1:
        raise ActionParseError("no JSON object found")
    decoder = json.JSONDecoder()
    try:
        obj, _ = decoder.raw_decode(candidate[start:])
    except json.JSONDecodeError as exc:
        raise ActionParseError(f"malformed JSON: {exc.msg}") from exc
    return obj


def parse_action(raw: str) -> Action:
    """Parse a model response string into a validated Action.

    Tolerates surrounding whitespace, ```json fences, prose wrapping the JSON
    object, and trailing characters such as a stray closing brace.
    Raises ActionParseError on any structural problem.
    """
    if not raw or not raw.strip():
        raise ActionParseError("empty model response")

    text = _strip_fences(raw)
    data = _decode_first_json_object(text)

    if not isinstance(data, dict):
        raise ActionParseError("action payload must be a JSON object")

    action_type = data.get("action_type")
    if not isinstance(action_type, str):
        raise ActionParseError("missing or non-string action_type")

    if action_type not in ACTION_TYPES:
        if action_type not in BUILTIN_ACTION_TYPE_ALIASES:
            raise ActionParseError(f"unknown action_type: {action_type!r}")
        data = {**data, "action_type": "run_builtin", "tool_name": action_type}
        action_type = "run_builtin"

    required = ACTION_TYPES[action_type]
    for key in required:
        if key not in data or data[key] in (None, ""):
            raise ActionParseError(f"{action_type} action missing required field: {key}")

    fields: dict[str, Any] = {
        k: v for k, v in data.items() if k not in ("action_type", "reasoning_summary")
    }
    # Normalize defaults used by downstream runtime
    if action_type in ("run_builtin", "reuse_tool", "create_tool"):
        fields.setdefault("arguments", {})
        if not isinstance(fields["arguments"], dict):
            raise ActionParseError("arguments must be a JSON object")

    reasoning = data.get("reasoning_summary", "") or ""
    if not isinstance(reasoning, str):
        raise ActionParseError("reasoning_summary must be a string")

    return Action(action_type=action_type, fields=fields, reasoning_summary=reasoning)


def action_type_names() -> list[str]:
    return list(ACTION_TYPES.keys())
