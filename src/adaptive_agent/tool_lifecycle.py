from __future__ import annotations

import json
import re
import shutil
import tempfile
from pathlib import Path
from typing import Any

from adaptive_agent.catalog import SavedToolSpec, ToolCatalog
from adaptive_agent.generated_tools import arguments_to_argv, execute_tool_code
from adaptive_agent.runtime_guard import RuntimeGuardResult

_REPLAY_IGNORE = shutil.ignore_patterns(
    ".agent_state",
    ".git",
    ".pytest_cache",
    ".venv",
    "__pycache__",
    "*.egg-info",
)


def execute_tool_replay_in_isolated_workspace(
    *,
    code: str,
    arguments: dict[str, Any],
    workdir: Path,
    timeout: float = 15.0,
) -> RuntimeGuardResult:
    """Replay generated code in a disposable copy of the workdir.

    The initial create_tool run may intentionally mutate the live workspace.
    Replay is verification evidence, so it must not duplicate those side effects
    in the user's live workroot.
    """
    workdir = Path(workdir)
    with tempfile.TemporaryDirectory(prefix="adaptive-agent-replay-") as td:
        replay_root = Path(td) / "work"
        shutil.copytree(workdir, replay_root, ignore=_REPLAY_IGNORE)
        return execute_tool_code(
            code=code,
            arguments=arguments,
            workdir=replay_root,
            timeout=timeout,
        )


def build_pending_save_spec(
    *,
    catalog: ToolCatalog,
    name: str,
    code: str,
    args: dict[str, Any],
    expected: str,
    initial: RuntimeGuardResult,
    replay: RuntimeGuardResult,
    current_task: str,
    model_info: dict[str, Any],
) -> SavedToolSpec:
    argv = arguments_to_argv(args)
    verification_status = "replay_verified" if replay.ok else "replay_failed"
    return SavedToolSpec(
        name=name,
        description=expected or "generated tool",
        input_summary="",
        output_summary="",
        code=code,
        version=catalog.next_version(name),
        example_args=argv,
        verification_status=verification_status,
        verification_details={
            "method": "create_tool_execution+fixture_replay",
            "replay_isolated": True,
            "argv": argv,
            "replay_ok": replay.ok,
            "initial_run": verification_run_payload(initial),
            "replay_run": verification_run_payload(replay),
        },
    )


VERIFICATION_LABEL_NOTE = (
    "참고: replay_verified는 example_args로 재실행했을 때 exit 0임을 의미하며 "
    "정답성은 검증하지 않습니다."
)


_SUSPICIOUS_KEYWORDS: tuple[str, ...] = (
    "traceback",
    "exception",
    "error:",
    "errno",
    "syntaxerror",
    "오류",
    "예외",
)


def detect_suspicious_output(stdout: str | None) -> str | None:
    """Heuristic check for tool stdout that likely indicates a logic bug.

    Returns a short Korean explanation if suspicious, else None. Designed to be
    cheap (no LLM): catches the most common silent-wrong-output failure modes
    we observed during interview testing (all-zero counters, empty results,
    error keywords leaked to stdout).
    """
    text = (stdout or "").strip()
    if not text:
        return "출력이 비어있습니다"
    lower = text.lower()
    for keyword in _SUSPICIOUS_KEYWORDS:
        if keyword in lower:
            return f"출력에 의심스러운 키워드 '{keyword}'가 포함되어 있습니다"
    if text[0] not in "{[":
        return None
    parsed = _try_parse_json(text)
    if isinstance(parsed, dict) and parsed and _all_zero_or_empty(parsed.values()):
        return "JSON 출력의 모든 값이 0/빈 값입니다"
    if isinstance(parsed, list) and not parsed:
        return "JSON 출력이 빈 리스트입니다"
    return None


def _try_parse_json(text: str) -> Any:
    try:
        return json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return None


def _all_zero_or_empty(values: Any) -> bool:
    seen_numeric = False
    for value in values:
        if isinstance(value, bool):
            if value:
                return False
            continue
        if isinstance(value, (int, float)):
            seen_numeric = True
            if value != 0:
                return False
            continue
        if value in ("", None, [], {}):
            continue
        return False
    return seen_numeric


def build_save_tool_approval_question(
    spec: SavedToolSpec, *, output_warning: str | None = None
) -> str:
    if output_warning:
        return f"출력이 의심스럽습니다({output_warning}). 생성한 도구 '{spec.name}'를 저장할까요?"
    return f"생성한 도구 '{spec.name}'를 저장할까요?"


def tail(text: str | None, limit: int) -> str:
    if text is None:
        return ""
    if len(text) <= limit:
        return text
    return text[-limit:]


def head(text: str | None, limit: int) -> str:
    if text is None:
        return ""
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def summarize_output(stdout: str | None, limit: int = 240) -> str:
    text = (stdout or "").strip()
    if not text:
        return ""
    artifact = _artifact_summary(text)
    if artifact:
        return head(artifact, limit)
    return head(text, limit)


def _artifact_summary(text: str) -> str:
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return ""
    if not isinstance(payload, dict):
        return ""
    artifact_type = payload.get("artifact_type")
    if not isinstance(artifact_type, str) or not artifact_type.strip():
        return ""
    parts = [f"artifact_type={artifact_type.strip()}"]
    title = payload.get("title")
    if isinstance(title, str) and title.strip():
        parts.append(f"title={title.strip()}")
    items = payload.get("items")
    if isinstance(items, list):
        parts.append(f"items={len(items)}")
        if items:
            first_item = str(items[0]).strip()
            if first_item:
                parts.append(f"first_item={first_item}")
    return " | ".join(parts)


def summarize_input(arguments: dict[str, Any]) -> str:
    argv = arguments_to_argv(arguments)
    if not argv:
        return "(no args)"
    return " ".join(argv)


def input_schema(arguments: dict[str, Any]) -> dict[str, Any]:
    argv = arguments.get("argv") if isinstance(arguments, dict) else None
    if isinstance(argv, list):
        return {"argv": [f"arg{i + 1}" for i in range(len(argv))]}
    if not isinstance(arguments, dict):
        return {}
    return {key: type(value).__name__ for key, value in arguments.items()}


def default_tags(tool_name: str) -> list[str]:
    return [part for part in tool_name.split("_") if part]


def verification_run_payload(result: RuntimeGuardResult) -> dict[str, Any]:
    return {
        "ok": result.ok,
        "exit_status": result.exit_status,
        "timed_out": result.timed_out,
        "duration_sec": round(result.duration_sec, 3),
        "stdout_tail": tail(result.stdout, 500),
        "stderr_tail": tail(result.stderr, 300),
    }


def model_info(llm: Any) -> dict[str, Any]:
    config = getattr(llm, "config", None)
    model = getattr(config, "model", None)
    base_url = getattr(config, "base_url", None)
    info: dict[str, Any] = {}
    if model:
        info["model"] = str(model)
    if base_url:
        info["base_url"] = str(base_url)
    return info
