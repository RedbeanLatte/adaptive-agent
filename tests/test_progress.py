from __future__ import annotations

import io
import json
from pathlib import Path

from adaptive_agent.builtins import register_builtins
from adaptive_agent.catalog import SavedToolSpec, ToolCatalog
from adaptive_agent.progress import ProgressEvent
from adaptive_agent.runtime import Runtime, RuntimeConfig
from adaptive_agent.session import Session


class ScriptedLLM:
    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)

    def complete(self, messages: list[dict[str, str]]) -> str:
        if not self._responses:
            raise AssertionError("ScriptedLLM out of responses")
        return self._responses.pop(0)


class ListProgressReporter:
    def __init__(self) -> None:
        self.events: list[ProgressEvent] = []

    def emit(self, event: ProgressEvent) -> None:
        self.events.append(event)


def _A(payload: dict) -> str:
    return json.dumps(payload)


def _mk_runtime(
    tmp_path: Path,
    responses: list[str],
    *,
    progress: ListProgressReporter | None = None,
    show_progress: bool = True,
    approvals: list[bool] | None = None,
) -> tuple[Runtime, io.StringIO, ListProgressReporter | None]:
    state_dir = tmp_path / "state"
    workroot = tmp_path / "work"
    state_dir.mkdir(exist_ok=True)
    workroot.mkdir(exist_ok=True)
    catalog = ToolCatalog(state_dir=state_dir)
    register_builtins(catalog)
    out = io.StringIO()
    approvals_iter = iter(approvals or [])

    rt = Runtime(
        llm=ScriptedLLM(responses),
        catalog=catalog,
        session=Session(session_id="progress"),
        config=RuntimeConfig(
            workroot=workroot,
            state_dir=state_dir,
            session_id="progress",
            max_iterations=6,
            show_progress=show_progress,
        ),
        approval_fn=lambda question: next(approvals_iter, False),
        user_input_fn=lambda prompt: "",
        output_stream=out,
        progress_reporter=progress,
    )
    return rt, out, progress


def _event_kinds(events: list[ProgressEvent]) -> list[str]:
    return [event.kind for event in events]


def test_create_tool_progress_events_include_execution_replay_approval_and_answer(tmp_path: Path):
    progress = ListProgressReporter()
    rt, _, _ = _mk_runtime(
        tmp_path,
        [
            _A(
                {
                    "action_type": "create_tool",
                    "reasoning_summary": "need generated helper",
                    "tool_name": "csv_cleanup",
                    "code": "print('ok')",
                    "arguments": {},
                    "expected_outcome": "clean csv rows",
                }
            ),
            _A(
                {
                    "action_type": "request_approval",
                    "question": "save?",
                    "approval_kind": "save_tool",
                }
            ),
            _A({"action_type": "answer", "final_answer": "done"}),
        ],
        progress=progress,
        approvals=[True],
    )

    rt.run_task("csv cleanup")

    kinds = _event_kinds(progress.events)
    expected_order = [
        "task_started",
        "llm_start",
        "action_received",
        "tool_create_start",
        "tool_create_complete",
        "tool_replay_start",
        "tool_replay_complete",
        "answer_start",
        "approval_wait",
        "approval_result",
    ]
    positions = [kinds.index(kind) for kind in expected_order]
    assert positions == sorted(positions)

    create_event = next(event for event in progress.events if event.kind == "tool_create_start")
    assert create_event.fields["tool_name"] == "csv_cleanup"
    assert create_event.fields["reasoning_summary"] == "need generated helper"

    approval_event = next(event for event in progress.events if event.kind == "approval_wait")
    assert approval_event.fields["tool_name"] == "csv_cleanup"
    assert approval_event.fields["verification_status"] == "replay_verified"
    assert "code_sha256" in approval_event.fields


def test_builtin_and_reuse_progress_events_include_tool_names(tmp_path: Path):
    progress = ListProgressReporter()
    rt, _, _ = _mk_runtime(
        tmp_path,
        [
            _A(
                {
                    "action_type": "run_builtin",
                    "tool_name": "read_text_file",
                    "arguments": {"path": "data.txt"},
                }
            ),
            _A(
                {
                    "action_type": "reuse_tool",
                    "tool_name": "saved_echo",
                    "arguments": {"argv": ["hello"]},
                }
            ),
            _A({"action_type": "answer", "final_answer": "done"}),
        ],
        progress=progress,
    )
    (rt.config.workroot / "data.txt").write_text("payload", encoding="utf-8")
    rt.catalog.save_generated(
        SavedToolSpec(
            name="saved_echo",
            description="echo text",
            input_summary="text",
            output_summary="echo text",
            code="import sys; print('echo:' + sys.argv[1])",
            example_args=["hello"],
            verification_status="replay_verified",
        )
    )

    rt.run_task("read then reuse")

    kinds = _event_kinds(progress.events)
    assert "builtin_start" in kinds
    assert "builtin_complete" in kinds
    assert "reuse_tool_start" in kinds
    assert "reuse_tool_complete" in kinds
    assert next(event for event in progress.events if event.kind == "builtin_start").fields["tool_name"] == "read_text_file"
    assert next(event for event in progress.events if event.kind == "reuse_tool_start").fields["tool_name"] == "saved_echo"


def test_default_console_progress_writes_compact_agent_lines(tmp_path: Path):
    rt, out, _ = _mk_runtime(
        tmp_path,
        [
            _A(
                {
                    "action_type": "answer",
                    "reasoning_summary": "simple direct answer",
                    "final_answer": "done",
                }
            )
        ],
    )

    rt.run_task("say done")

    text = out.getvalue()
    assert "[agent] 작업 시작" in text
    assert "[agent] 요청 분석 중" in text
    assert "[agent] 다음 작업 선택: answer" in text
    assert "[agent] 문제 분석: simple direct answer" in text
    assert text.count("[agent] 문제 분석:") == 1
    assert "[agent] 최종 답변 생성 중" in text
    assert "done" in text


def test_console_progress_shows_problem_analysis_once_for_multi_step_run(tmp_path: Path):
    rt, out, _ = _mk_runtime(
        tmp_path,
        [
            _A(
                {
                    "action_type": "run_builtin",
                    "reasoning_summary": "read the file before answering",
                    "tool_name": "read_text_file",
                    "arguments": {"path": "data.txt"},
                }
            ),
            _A(
                {
                    "action_type": "answer",
                    "reasoning_summary": "summarize observed tool output",
                    "final_answer": "done",
                }
            ),
        ],
    )
    (rt.config.workroot / "data.txt").write_text("payload", encoding="utf-8")

    rt.run_task("read then answer")

    text = out.getvalue()
    assert text.count("[agent] 요청 분석 중") == 1
    assert "[agent] 다음 단계 판단 중" in text
    assert "[agent] 다음 작업 선택: run_builtin" in text
    assert "[agent] 다음 작업 선택: answer" in text
    assert "[agent] 문제 분석: read the file before answering" in text
    assert "summarize observed tool output" not in text
    assert text.count("[agent] 문제 분석:") == 1
    assert "done" in text


def test_console_progress_omits_problem_analysis_when_summary_is_empty(tmp_path: Path):
    rt, out, _ = _mk_runtime(
        tmp_path,
        [_A({"action_type": "answer", "reasoning_summary": "", "final_answer": "done"})],
    )

    rt.run_task("say done")

    text = out.getvalue()
    assert "[agent] 문제 분석:" not in text
    assert "[agent] 요청 분석 중" in text
    assert "[agent] 다음 작업 선택: answer" in text
    assert "done" in text


def test_console_progress_does_not_fall_back_to_later_summary(tmp_path: Path):
    rt, out, _ = _mk_runtime(
        tmp_path,
        [
            _A(
                {
                    "action_type": "run_builtin",
                    "reasoning_summary": "",
                    "tool_name": "read_text_file",
                    "arguments": {"path": "data.txt"},
                }
            ),
            _A(
                {
                    "action_type": "answer",
                    "reasoning_summary": "summarize observed tool output",
                    "final_answer": "done",
                }
            ),
        ],
    )
    (rt.config.workroot / "data.txt").write_text("payload", encoding="utf-8")

    rt.run_task("read then answer")

    text = out.getvalue()
    assert "[agent] 문제 분석:" not in text
    assert "summarize observed tool output" not in text
    assert "[agent] 요청 분석 중" in text
    assert "[agent] 다음 단계 판단 중" in text
    assert "done" in text


def test_progress_can_be_disabled_for_runtime(tmp_path: Path):
    rt, out, _ = _mk_runtime(
        tmp_path,
        [
            _A(
                {
                    "action_type": "answer",
                    "reasoning_summary": "simple direct answer",
                    "final_answer": "done",
                }
            )
        ],
        show_progress=False,
    )

    rt.run_task("say done")

    text = out.getvalue()
    assert "[agent] 요청 분석 중" not in text
    assert "[agent] 문제 분석:" not in text
    assert "[agent] 최종 답변 생성 중" not in text
    assert "done" in text
