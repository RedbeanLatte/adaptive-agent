from __future__ import annotations

import io
import json
from pathlib import Path

import pytest

from adaptive_agent.builtins import register_builtins
from adaptive_agent.catalog import SavedToolSpec, ToolCatalog
from adaptive_agent.runtime import Runtime, RuntimeConfig, run_repl
from adaptive_agent.session import Session
from adaptive_agent.slash_commands import dispatch_slash_command


class ScriptedLLM:
    def __init__(self, responses: list[str] | None = None) -> None:
        self._responses = list(responses or [])
        self.calls: list[list[dict[str, str]]] = []

    def complete(self, messages: list[dict[str, str]]) -> str:
        self.calls.append(messages)
        if not self._responses:
            raise AssertionError("ScriptedLLM should not be called")
        return self._responses.pop(0)


def _mk_runtime(tmp_path: Path) -> tuple[Runtime, ScriptedLLM, io.StringIO]:
    state_dir = tmp_path / "state"
    workroot = tmp_path / "work"
    state_dir.mkdir(exist_ok=True)
    workroot.mkdir(exist_ok=True)

    catalog = ToolCatalog(state_dir=state_dir)
    register_builtins(catalog)
    session = Session(session_id="slash")
    llm = ScriptedLLM()
    out = io.StringIO()

    rt = Runtime(
        llm=llm,
        catalog=catalog,
        session=session,
        config=RuntimeConfig(
            workroot=workroot,
            state_dir=state_dir,
            session_id="slash",
            max_iterations=4,
        ),
        approval_fn=lambda question: True,
        user_input_fn=lambda prompt: "",
        output_stream=out,
    )
    return rt, llm, out


def _save_tool(
    rt: Runtime,
    *,
    name: str = "csv_cleanup",
    code: str = "print('ok')",
    description: str = "dedupe + sort",
    last_used_at: str = "",
    success_count: int = 2,
    failure_count: int = 0,
) -> SavedToolSpec:
    spec = SavedToolSpec(
        name=name,
        description=description,
        input_summary="fixtures/example3_dirty_rows.csv",
        output_summary="cleaned rows",
        code=code,
        version=1,
        risk_level="low",
        execution_hints={"read_only": True, "idempotent": True},
        example_args=["fixtures/example3_dirty_rows.csv"],
        verification_status="replay_verified",
        verification_details={
            "method": "create_tool_execution+fixture_replay",
            "argv": ["fixtures/example3_dirty_rows.csv"],
            "replay_ok": True,
            "initial_run": {"ok": True, "exit_status": 0, "stdout_tail": "ok"},
            "replay_run": {"ok": True, "exit_status": 0, "stdout_tail": "ok"},
        },
        success_count=success_count,
        failure_count=failure_count,
        last_used_at=last_used_at,
        source_task_summary="중복 제거 후 날짜순 정렬",
    )
    rt.catalog.save_generated(spec)
    return spec


def test_help_command_prints_supported_commands_and_bypasses_llm(tmp_path):
    rt, llm, out = _mk_runtime(tmp_path)

    handled = dispatch_slash_command(rt, "/help")

    assert handled is True
    text = out.getvalue()
    assert "/help" in text
    assert "/tools list" in text
    assert "/tools inspect <name>" in text
    assert "/tools verify <name>" in text
    assert "/tools remove <name>" in text
    assert llm.calls == []


def test_tools_list_shows_saved_tools_and_bypasses_llm(tmp_path):
    rt, llm, out = _mk_runtime(tmp_path)
    _save_tool(rt)

    handled = dispatch_slash_command(rt, "/tools list")

    assert handled is True
    text = out.getvalue()
    assert "csv_cleanup" in text
    assert "replay_verified" in text
    # Reuse counters are intentionally not persisted in the minimal manifest
    # (commit 1162a38), so the list view must not display always-zero rows.
    assert "성공:" not in text
    assert "실패:" not in text
    assert "마지막 사용:" not in text
    assert llm.calls == []


def test_tools_list_includes_description(tmp_path):
    rt, _, out = _mk_runtime(tmp_path)
    _save_tool(rt, description="csv dedupe and sort helper")

    handled = dispatch_slash_command(rt, "/tools list")

    assert handled is True
    text = out.getvalue()
    assert "csv dedupe and sort helper" in text


def test_tools_inspect_shows_metadata_for_named_tool(tmp_path):
    rt, _, out = _mk_runtime(tmp_path)
    _save_tool(rt)

    handled = dispatch_slash_command(rt, "/tools inspect csv_cleanup")

    assert handled is True
    text = out.getvalue()
    assert "이름: csv_cleanup" in text
    assert "설명: dedupe + sort" in text
    assert "검증: replay_verified" in text
    assert "예시 인자: fixtures/example3_dirty_rows.csv" in text
    assert "실행 힌트:" in text
    assert "read_only=True" in text
    assert "안전 설정:" in text
    assert "approval_required=True" in text


def test_tools_inspect_drops_unpersisted_usage_counters(tmp_path):
    # The minimal manifest (commit 1162a38) does not persist success/failure
    # counts or last_used_at across CLI invocations, so the inspect view must
    # not display rows that would always show zero in real usage.
    rt, _, out = _mk_runtime(tmp_path)
    _save_tool(
        rt,
        success_count=4,
        failure_count=1,
        last_used_at="2026-04-23T12:00:00Z",
    )

    handled = dispatch_slash_command(rt, "/tools inspect csv_cleanup")

    assert handled is True
    text = out.getvalue()
    assert "성공 횟수" not in text
    assert "실패 횟수" not in text
    assert "마지막 사용:" not in text


def test_tools_inspect_shows_last_run_recorded_at_when_available(tmp_path):
    rt, _, out = _mk_runtime(tmp_path)
    _save_tool(rt)
    # Seed last_run.json with a recorded_at timestamp the inspect view should surface.
    last_run_path = tmp_path / "state" / "tools" / "csv_cleanup" / "last_run.json"
    last_run_path.write_text(
        json.dumps(
            {
                "ok": True,
                "exit_status": 0,
                "stdout_tail": "ok",
                "stderr_tail": "",
                "recorded_at": "2026-04-25T09:30:00Z",
            }
        ),
        encoding="utf-8",
    )

    dispatch_slash_command(rt, "/tools inspect csv_cleanup")

    text = out.getvalue()
    assert "마지막 실행 시각: 2026-04-25T09:30:00Z" in text
    assert "마지막 실행 stdout: ok" in text


def test_tools_verify_reexecutes_example_args_and_updates_last_run(tmp_path):
    rt, _, out = _mk_runtime(tmp_path)
    _save_tool(rt, code="import sys; print('verified:' + sys.argv[1])")

    handled = dispatch_slash_command(rt, "/tools verify csv_cleanup")

    assert handled is True
    text = out.getvalue()
    assert "검증을 시작합니다" in text
    assert "재실행 성공" in text

    tool_dir = tmp_path / "state" / "tools" / "csv_cleanup"
    last_run = json.loads((tool_dir / "last_run.json").read_text(encoding="utf-8"))
    assert last_run["ok"] is True
    assert last_run["exit_status"] == 0
    assert "verified:fixtures/example3_dirty_rows.csv" in last_run["stdout_tail"]

    loaded = rt.catalog.lookup("csv_cleanup")
    assert loaded.success_count >= 3
    assert loaded.last_used_at


def test_tools_verify_failure_prints_replay_failed_status_and_error_tail(tmp_path):
    rt, _, out = _mk_runtime(tmp_path)
    _save_tool(rt, code="raise RuntimeError('verify boom')")

    handled = dispatch_slash_command(rt, "/tools verify csv_cleanup")

    assert handled is True
    text = out.getvalue()
    assert "example_args 재실행 실패" in text
    assert "검증 상태: replay_failed" in text
    assert "verify boom" in text

    loaded = rt.catalog.lookup("csv_cleanup")
    assert loaded.verification_status == "replay_failed"
    assert loaded.failure_count >= 1


def test_tools_remove_deletes_saved_tool_after_confirmation(tmp_path):
    approvals = iter([True])
    rt, _, out = _mk_runtime(tmp_path)
    rt.approval_fn = lambda question: next(approvals)
    _save_tool(rt)

    handled = dispatch_slash_command(rt, "/tools remove csv_cleanup")

    assert handled is True
    assert "삭제 완료" in out.getvalue()
    assert not rt.catalog.has("csv_cleanup")
    assert not (tmp_path / "state" / "tools" / "csv_cleanup").exists()


def test_unknown_slash_command_prints_helpful_error(tmp_path):
    rt, _, out = _mk_runtime(tmp_path)

    handled = dispatch_slash_command(rt, "/tools prune")

    assert handled is True
    text = out.getvalue()
    assert "알 수 없는 명령입니다: /tools prune" in text
    assert "/help 로 지원 명령을 확인하세요." in text


def test_run_repl_routes_slash_commands_without_calling_llm(tmp_path, monkeypatch):
    rt, llm, out = _mk_runtime(tmp_path)
    user_inputs = iter(["/help"])

    def fake_input(prompt: str) -> str:
        try:
            return next(user_inputs)
        except StopIteration:
            raise EOFError

    monkeypatch.setattr("adaptive_agent.runtime._build_runtime", lambda session_id: rt)
    monkeypatch.setattr("adaptive_agent.runtime._read_repl_input", fake_input)

    exit_code = run_repl(session_id="slash")

    assert exit_code == 0
    assert llm.calls == []
    assert "/tools list" in out.getvalue()
    assert "Adaptive Agent REPL입니다" in out.getvalue()


@pytest.mark.parametrize("command", ["exit", "/exit", "quit", "/quit"])
def test_run_repl_exit_commands_bypass_llm(tmp_path, monkeypatch, command):
    rt, llm, out = _mk_runtime(tmp_path)
    user_inputs = iter([command])

    def fake_input(prompt: str) -> str:
        try:
            return next(user_inputs)
        except StopIteration:
            raise EOFError

    monkeypatch.setattr("adaptive_agent.runtime._build_runtime", lambda session_id: rt)
    monkeypatch.setattr("adaptive_agent.runtime._read_repl_input", fake_input)

    exit_code = run_repl(session_id="slash")

    assert exit_code == 0
    assert llm.calls == []
    assert "종료합니다." in out.getvalue()


def test_run_repl_passes_korean_input_to_runtime(tmp_path, monkeypatch):
    rt, llm, out = _mk_runtime(tmp_path)
    user_inputs = iter(["한글 입력 테스트", "/exit"])
    seen_tasks: list[str] = []

    def fake_input(prompt: str) -> str:
        try:
            return next(user_inputs)
        except StopIteration:
            raise EOFError

    def fake_run_task(task: str):
        seen_tasks.append(task)
        return None

    rt.run_task = fake_run_task  # type: ignore[method-assign]
    monkeypatch.setattr("adaptive_agent.runtime._build_runtime", lambda session_id: rt)
    monkeypatch.setattr("adaptive_agent.runtime._read_repl_input", fake_input)

    exit_code = run_repl(session_id="slash")

    assert exit_code == 0
    assert seen_tasks == ["한글 입력 테스트"]
    assert llm.calls == []
