from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from adaptive_agent.catalog import SavedToolSpec, ToolCatalog
from adaptive_agent.runtime_guard import RuntimeGuardResult, run_guarded_python


def arguments_to_argv(arguments: dict[str, Any]) -> list[str]:
    """Convert an LLM-produced `arguments` mapping into a positional argv list.

    Convention (kept intentionally small):
    - if `arguments["argv"]` is a list, use it as-is (stringified)
    - otherwise, use each value in insertion order, stringified
    """
    if not arguments:
        return []
    argv_value = arguments.get("argv")
    if isinstance(argv_value, list):
        return [str(x) for x in argv_value]
    return [str(v) for v in arguments.values()]


def execute_tool_code(
    *,
    code: str,
    arguments: dict[str, Any],
    workdir: Path,
    timeout: float = 15.0,
) -> RuntimeGuardResult:
    """Execute generated tool code as a one-off subprocess with positional args."""
    return run_guarded_python(
        code,
        workdir=workdir,
        args=arguments_to_argv(arguments),
        timeout=timeout,
    )


def save_if_approved(
    catalog: ToolCatalog,
    spec: SavedToolSpec,
    *,
    approval_fn: Callable[[str], bool],
) -> bool:
    """Persist a generated tool only if the user approves via ``approval_fn``.

    Returns True if the tool was saved, False otherwise.
    """
    question = f"생성한 도구 '{spec.name}'를 저장할까요?"
    if not approval_fn(question):
        return False
    catalog.save_generated(spec)
    return True
