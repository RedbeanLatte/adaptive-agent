from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from adaptive_agent.catalog import SavedToolSpec, ToolNotFoundError
from adaptive_agent.generated_tools import execute_tool_code
from adaptive_agent.tool_lifecycle import VERIFICATION_LABEL_NOTE


@dataclass(frozen=True)
class SlashCommandSpec:
    name: str
    description: str
    args_hint: str = ""
    aliases: tuple[str, ...] = ()


COMMAND_REGISTRY: list[SlashCommandSpec] = [
    SlashCommandSpec("help", "지원하는 슬래시 명령을 표시합니다"),
    SlashCommandSpec("tools", "저장된 도구를 관리합니다", args_hint="<list|inspect|verify|remove> [name]", aliases=("t",)),
]


_COMMAND_LOOKUP: dict[str, SlashCommandSpec] = {}
for _cmd in COMMAND_REGISTRY:
    _COMMAND_LOOKUP[_cmd.name] = _cmd
    for _alias in _cmd.aliases:
        _COMMAND_LOOKUP[_alias] = _cmd


def resolve_command(name: str) -> SlashCommandSpec | None:
    return _COMMAND_LOOKUP.get(name.lower().lstrip("/"))


def dispatch_slash_command(runtime: Any, command: str) -> bool:
    text = (command or "").strip()
    if not text.startswith("/"):
        return False

    spec = resolve_command(text.split()[0])
    if spec is None:
        _write(runtime, f"알 수 없는 명령입니다: {text}")
        _write(runtime, "/help 로 지원 명령을 확인하세요.")
        _trace(runtime, "slash_command_unknown", command=text)
        return True

    _trace(runtime, "slash_command", command=text, name=spec.name)
    if spec.name == "help":
        _handle_help(runtime)
        return True
    if spec.name == "tools":
        _handle_tools(runtime, text)
        return True

    _write(runtime, f"알 수 없는 명령입니다: {text}")
    _write(runtime, "/help 로 지원 명령을 확인하세요.")
    _trace(runtime, "slash_command_unknown", command=text)
    return True


def _handle_help(runtime: Any) -> None:
    _write(runtime, "지원 명령:")
    _write(runtime, "/help")
    _write(runtime, "/tools list")
    _write(runtime, "/tools inspect <name>")
    _write(runtime, "/tools verify <name>")
    _write(runtime, "/tools remove <name>")


def _handle_tools(runtime: Any, text: str) -> None:
    parts = text.split()
    if len(parts) < 2:
        _write(runtime, "사용법: /tools <list|inspect|verify|remove> [name]")
        return

    subcommand = parts[1].lower()
    if subcommand == "list":
        _handle_tools_list(runtime)
        return
    if subcommand == "inspect" and len(parts) >= 3:
        _handle_tools_inspect(runtime, parts[2])
        return
    if subcommand == "verify" and len(parts) >= 3:
        _handle_tools_verify(runtime, parts[2])
        return
    if subcommand == "remove" and len(parts) >= 3:
        _handle_tools_remove(runtime, parts[2])
        return

    _write(runtime, f"알 수 없는 명령입니다: {text}")
    _write(runtime, "/help 로 지원 명령을 확인하세요.")
    _trace(runtime, "slash_command_unknown", command=text)


def _handle_tools_list(runtime: Any) -> None:
    tools = runtime.catalog.list_saved()
    if not tools:
        _write(runtime, "저장된 도구가 없습니다.")
        return

    _write(runtime, "저장된 도구:")
    for spec in tools:
        _write(
            runtime,
            f"- {spec.name} v{spec.version} | 검증: {spec.verification_status} | {spec.description}",
        )


def _handle_tools_inspect(runtime: Any, name: str) -> None:
    spec = _get_saved_tool(runtime, name)
    if spec is None:
        return

    example = " ".join(spec.example_args) if spec.example_args else "(없음)"
    execution_hints = ", ".join(
        f"{key}={value}" for key, value in sorted(spec.execution_hints.items())
    ) or "(없음)"
    safety = ", ".join(f"{key}={value}" for key, value in sorted(spec.safety.items())) or "(없음)"
    lines = [
        f"이름: {spec.name}",
        f"버전: {spec.version}",
        f"설명: {spec.description}",
        f"입력: {spec.input_summary}",
        f"출력: {spec.output_summary}",
        f"검증: {spec.verification_status}",
        f"예시 인자: {example}",
        f"위험도: {spec.risk_level}",
        f"실행 힌트: {execution_hints}",
        f"안전 설정: {safety}",
        f"원본 작업: {spec.source_task_summary or '(없음)'}",
    ]
    for line in lines:
        _write(runtime, line)

    last_run = runtime.catalog.load_last_run(name)
    if last_run:
        recorded_at = last_run.get("recorded_at")
        if recorded_at:
            _write(runtime, f"마지막 실행 시각: {recorded_at}")
        _write(runtime, f"마지막 실행 성공: {last_run.get('ok')}")
        _write(runtime, f"마지막 실행 exit_status: {last_run.get('exit_status')}")
        stdout_tail = last_run.get("stdout_tail")
        if stdout_tail:
            _write(runtime, f"마지막 실행 stdout: {stdout_tail}")
    _write(runtime, VERIFICATION_LABEL_NOTE)


def _handle_tools_verify(runtime: Any, name: str) -> None:
    spec = _get_saved_tool(runtime, name)
    if spec is None:
        return

    _write(runtime, f"{name} 검증을 시작합니다...")
    result = execute_tool_code(
        code=spec.code,
        arguments={"argv": list(spec.example_args)},
        workdir=runtime.config.workroot,
    )
    runtime.catalog.verify_saved(
        name,
        ok=result.ok,
        exit_status=result.exit_status,
        timed_out=result.timed_out,
        duration_sec=round(result.duration_sec, 3),
        stdout_tail=_tail(result.stdout, 500),
        stderr_tail=_tail(result.stderr, 300),
        argv=list(spec.example_args),
        task_summary="slash verify",
    )
    _trace(runtime, "tool_verified", name=name, ok=result.ok, exit_status=result.exit_status)
    if result.ok:
        _write(runtime, "example_args 재실행 성공")
        _write(runtime, "검증 상태: replay_verified")
        _write(runtime, "last_run.json / verification metadata를 갱신했습니다.")
        _write(runtime, VERIFICATION_LABEL_NOTE)
        return

    _write(runtime, f"example_args 재실행 실패: exit_status={result.exit_status}")
    _write(runtime, "검증 상태: replay_failed")
    if result.stderr:
        _write(runtime, _tail(result.stderr, 300))
    _write(runtime, VERIFICATION_LABEL_NOTE)


def _handle_tools_remove(runtime: Any, name: str) -> None:
    spec = _get_saved_tool(runtime, name)
    if spec is None:
        return

    question = f"저장된 도구 '{name}'을 삭제할까요?"
    if not runtime.approval_fn(question):
        _write(runtime, "삭제를 취소했습니다.")
        _trace(runtime, "tool_remove_cancelled", name=name)
        return

    runtime.catalog.remove_saved(name)
    _trace(runtime, "tool_removed", name=name)
    _write(runtime, f"{name} 삭제 완료")


def _get_saved_tool(runtime: Any, name: str) -> SavedToolSpec | None:
    try:
        spec = runtime.catalog.lookup(name)
    except ToolNotFoundError:
        _write(runtime, f"저장된 도구를 찾을 수 없습니다: {name}")
        return None
    if not isinstance(spec, SavedToolSpec):
        _write(runtime, f"저장된 생성 도구만 관리할 수 있습니다: {name}")
        return None
    return spec


def _write(runtime: Any, text: str) -> None:
    writer = getattr(runtime, "_write_line", None)
    if callable(writer):
        writer(text)
        return
    runtime.output.write(text + "\n")
    runtime.output.flush()


def _trace(runtime: Any, kind: str, **fields: Any) -> None:
    trace = getattr(runtime, "trace", None)
    if trace is None:
        return
    event = getattr(trace, "event", None)
    if callable(event):
        event(kind, **fields)


def _tail(text: str, limit: int) -> str:
    if text is None:
        return ""
    if len(text) <= limit:
        return text
    return text[-limit:]
