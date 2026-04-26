from __future__ import annotations

import io
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import pytest

from adaptive_agent.builtins import register_builtins
from adaptive_agent.catalog import ToolCatalog
from adaptive_agent.runtime import (
    InteractiveInputUnavailable,
    Runtime,
    RuntimeConfig,
    _build_runtime,
    run_one_shot,
)
from adaptive_agent.session import Session


class ScriptedLLM:
    """Emits pre-scripted strings in order; records received messages."""

    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)
        self.calls: list[list[dict[str, str]]] = []

    def complete(self, messages: list[dict[str, str]]) -> str:
        self.calls.append(messages)
        if not self._responses:
            raise AssertionError("ScriptedLLM out of responses")
        return self._responses.pop(0)


def _mk_runtime(
    tmp_path: Path,
    responses: list[str],
    *,
    user_replies: list[str] | None = None,
    approvals: list[bool] | None = None,
    output: io.StringIO | None = None,
) -> tuple[Runtime, ScriptedLLM, io.StringIO]:
    state_dir = tmp_path / "state"
    workroot = tmp_path / "work"
    state_dir.mkdir(exist_ok=True)
    workroot.mkdir(exist_ok=True)

    catalog = ToolCatalog(state_dir=state_dir)
    register_builtins(catalog)
    session = Session(session_id="t")
    llm = ScriptedLLM(responses)

    out = output if output is not None else io.StringIO()
    user_iter = iter(user_replies or [])
    approvals_iter = iter(approvals or [])

    def user_input_fn(prompt: str) -> str:
        try:
            return next(user_iter)
        except StopIteration:
            return ""

    def approval_fn(question: str) -> bool:
        try:
            return next(approvals_iter)
        except StopIteration:
            return False

    rt = Runtime(
        llm=llm,
        catalog=catalog,
        session=session,
        config=RuntimeConfig(
            workroot=workroot,
            state_dir=state_dir,
            session_id="t",
            max_iterations=8,
        ),
        approval_fn=approval_fn,
        user_input_fn=user_input_fn,
        output_stream=out,
    )
    return rt, llm, out


def _A(d: dict) -> str:
    return json.dumps(d)



def _repo_fixture_text(name: str) -> str:
    root = Path(__file__).resolve().parents[1]
    return (root / "fixtures" / name).read_text(encoding="utf-8")



def test_plain_answer(tmp_path):
    rt, _, out = _mk_runtime(
        tmp_path,
        [_A({"action_type": "answer", "final_answer": "hi there"})],
    )
    result = rt.run_task("say hi")
    assert result.final_answer == "hi there"
    assert "hi there" in out.getvalue()


def test_system_prompt_describes_llm_routing_policy(tmp_path):
    rt, llm, _ = _mk_runtime(
        tmp_path,
        [_A({"action_type": "answer", "final_answer": "done"})],
    )

    rt.run_task("오늘 날씨는 어때?")

    system_prompt = llm.calls[0][0]["content"]
    assert "Routing policy" in system_prompt
    assert "LLM chooses the action" in system_prompt
    assert "Tool-first policy" in system_prompt
    assert "Direct answer is only for pure explanation" in system_prompt
    assert "file-backed data analysis" in system_prompt
    assert "answer" in system_prompt
    assert "create_tool" in system_prompt
    assert "realtime external data" in system_prompt


def test_general_realtime_question_can_finish_with_answer_without_tool(tmp_path):
    rt, _, out = _mk_runtime(
        tmp_path,
        [
            _A(
                {
                    "action_type": "answer",
                    "final_answer": "실시간 날씨 조회 capability가 없어 현재 날씨는 확인할 수 없습니다.",
                }
            )
        ],
    )

    result = rt.run_task("오늘 날씨는 어때?")

    assert result.final_answer == "실시간 날씨 조회 capability가 없어 현재 날씨는 확인할 수 없습니다."
    assert result.ok is True
    assert "실시간 날씨 조회 capability" in out.getvalue()
    assert [turn for turn in rt.session.turns if turn.role == "tool"] == []


def test_ask_user_then_answer(tmp_path):
    rt, llm, out = _mk_runtime(
        tmp_path,
        [
            _A({"action_type": "ask_user", "question": "what name?"}),
            _A({"action_type": "answer", "final_answer": "hello you"}),
        ],
        user_replies=["Kim"],
    )
    rt.run_task("greet someone")
    # Runtime should have a user turn for "Kim" in the second LLM call
    second_call = llm.calls[1]
    assert any(m["content"] == "Kim" for m in second_call)
    assert out.getvalue().count("what name?") == 1


def test_run_builtin_read_text_file(tmp_path):
    (tmp_path / "work").mkdir(exist_ok=True)
    (tmp_path / "work" / "data.txt").write_text("payload")
    rt, _, out = _mk_runtime(
        tmp_path,
        [
            _A(
                {
                    "action_type": "run_builtin",
                    "tool_name": "read_text_file",
                    "arguments": {"path": "data.txt"},
                }
            ),
            _A({"action_type": "answer", "final_answer": "read it"}),
        ],
    )
    rt.run_task("read file")
    assert "read it" in out.getvalue()


def test_missing_file_observation_surfaces_not_found_error(tmp_path):
    # The "do not silently substitute another file" prompt rule depends on the
    # runtime emitting a structured `{"ok": false, "error": "file not found: ..."}`
    # observation that the LLM can see. Lock the contract here so prompt-only
    # fixes for silent-substitution remain meaningful.
    (tmp_path / "work").mkdir(exist_ok=True)
    rt, _, _ = _mk_runtime(
        tmp_path,
        [
            _A(
                {
                    "action_type": "run_builtin",
                    "tool_name": "read_text_file",
                    "arguments": {"path": "does_not_exist.json"},
                }
            ),
            _A(
                {
                    "action_type": "answer",
                    "final_answer": "요청하신 파일이 존재하지 않습니다.",
                }
            ),
        ],
    )

    rt.run_task("does_not_exist.json 의 평균을 알려줘")

    tool_turns = [turn.content for turn in rt.session.turns if turn.role == "tool"]
    assert tool_turns, "runtime should emit a tool observation for the failed builtin"
    assert any("file not found" in turn for turn in tool_turns)
    assert any('"ok": false' in turn.lower() or '"ok":false' in turn.lower() for turn in tool_turns)


def test_system_prompt_blocks_silent_file_substitution(tmp_path):
    rt, llm, _ = _mk_runtime(
        tmp_path,
        [_A({"action_type": "answer", "final_answer": "ok"})],
    )

    rt.run_task("noop")

    system_prompt = llm.calls[0][0]["content"]
    assert "do not silently substitute another file" in system_prompt
    assert '"file not found"' in system_prompt or "file not found:" in system_prompt


def test_system_prompt_blocks_audit_checklist_direct_answer(tmp_path):
    rt, llm, _ = _mk_runtime(
        tmp_path,
        [_A({"action_type": "answer", "final_answer": "ok"})],
    )

    rt.run_task("noop")

    system_prompt = llm.calls[0][0]["content"]
    assert "Code audits, checklists, scaffolds, and prompt packs" in system_prompt
    assert "not exceptions to the tool-first policy" in system_prompt.lower()


def test_run_builtin_read_text_file_non_utf8_becomes_tool_error(tmp_path):
    (tmp_path / "work").mkdir(exist_ok=True)
    (tmp_path / "work" / "bad.bin").write_bytes(b"\xff\xfe\x00\x01")
    rt, _, out = _mk_runtime(
        tmp_path,
        [
            _A(
                {
                    "action_type": "run_builtin",
                    "tool_name": "read_text_file",
                    "arguments": {"path": "bad.bin"},
                }
            ),
            _A({"action_type": "answer", "final_answer": "fallback"}),
        ],
    )
    result = rt.run_task("read binary as text")
    assert result.final_answer == "fallback"
    assert "fallback" in out.getvalue()
    tool_turns = [turn.content for turn in rt.session.turns if turn.role == "tool"]
    assert any('"ok": false' in turn.lower() or '"ok":false' in turn.lower() for turn in tool_turns)
    assert any("UTF-8" in turn for turn in tool_turns)


def test_create_tool_success_and_save_flow(tmp_path):
    code = "print('Orc, Dragon / 225')"
    rt, _, out = _mk_runtime(
        tmp_path,
        [
            _A(
                {
                    "action_type": "create_tool",
                    "tool_name": "monsters_filter",
                    "code": code,
                    "arguments": {"argv": ["fixtures/example1_monsters.json"]},
                    "expected_outcome": "filter and average",
                }
            ),
            _A(
                {
                    "action_type": "request_approval",
                    "question": "save monsters_filter?",
                    "approval_kind": "save_tool",
                }
            ),
            _A(
                {
                    "action_type": "answer",
                    "final_answer": "Orc, Dragon / avg 225",
                }
            ),
        ],
        approvals=[True],
    )
    rt.run_task("monster task")
    # catalog should now contain the saved tool
    saved = rt.catalog.lookup("monsters_filter")
    assert saved.code == code
    assert saved.verification_status == "replay_verified"
    assert saved.example_args == ["fixtures/example1_monsters.json"]
    assert saved.source_task_summary == ""
    assert saved.success_count == 0
    assert saved.failure_count == 0

    tool_dir = tmp_path / "state" / "tools" / "monsters_filter"
    manifest = json.loads((tool_dir / "manifest.json").read_text())
    verification = json.loads((tool_dir / "verification.json").read_text())
    last_run = json.loads((tool_dir / "last_run.json").read_text())
    assert manifest == {
        "schema_version": 2,
        "name": "monsters_filter",
        "version": 1,
        "description": "filter and average",
        "example_args": ["fixtures/example1_monsters.json"],
        "verification_status": "replay_verified",
        "created_at": manifest["created_at"],
    }
    assert verification["method"] == "create_tool_execution+fixture_replay"
    assert verification["initial_run"]["exit_status"] == 0
    assert verification["replay_run"]["exit_status"] == 0
    assert verification["replay_ok"] is True
    assert last_run["exit_status"] == 0


def test_create_tool_success_prompts_save_after_final_answer(tmp_path):
    code = "print('saved automatically')"
    rt, _, out = _mk_runtime(
        tmp_path,
        [
            _A(
                {
                    "action_type": "create_tool",
                    "tool_name": "auto_saved_tool",
                    "code": code,
                    "arguments": {},
                    "expected_outcome": "prove automatic approval gate",
                }
            ),
            _A({"action_type": "answer", "final_answer": "done"}),
        ],
    )
    approval_questions: list[str] = []
    output_seen_at_prompt: list[str] = []

    def approve(question: str) -> bool:
        approval_questions.append(question)
        output_seen_at_prompt.append(out.getvalue())
        return True

    rt.approval_fn = approve

    result = rt.run_task("make a reusable helper")

    assert result.final_answer == "done"
    assert len(approval_questions) == 1
    assert "auto_saved_tool" in approval_questions[0]
    assert "done" in output_seen_at_prompt[0]
    saved = rt.catalog.lookup("auto_saved_tool")
    assert saved.code == code
    assert saved.verification_status == "replay_verified"


def test_create_tool_success_denied_without_model_approval_action_does_not_persist(tmp_path):
    rt, _, _ = _mk_runtime(
        tmp_path,
        [
            _A(
                {
                    "action_type": "create_tool",
                    "tool_name": "ephemeral_tool",
                    "code": "print('temporary')",
                    "arguments": {},
                }
            ),
            _A({"action_type": "answer", "final_answer": "done"}),
        ],
    )
    approval_questions: list[str] = []

    def deny(question: str) -> bool:
        approval_questions.append(question)
        return False

    rt.approval_fn = deny

    result = rt.run_task("make a temporary helper")

    assert result.final_answer == "done"
    assert len(approval_questions) == 1
    assert not rt.catalog.has("ephemeral_tool")


def test_model_request_approval_before_answer_is_deferred_and_not_prompted_twice(tmp_path):
    code = "print('ok')"
    rt, _, _ = _mk_runtime(
        tmp_path,
        [
            _A(
                {
                    "action_type": "create_tool",
                    "tool_name": "single_prompt_tool",
                    "code": code,
                    "arguments": {},
                }
            ),
            _A(
                {
                    "action_type": "request_approval",
                    "question": "save again?",
                    "approval_kind": "save_tool",
                }
            ),
            _A({"action_type": "answer", "final_answer": "done"}),
        ],
    )
    approval_questions: list[str] = []

    def approve(question: str) -> bool:
        approval_questions.append(question)
        return True

    rt.approval_fn = approve

    result = rt.run_task("make one saved helper")

    assert result.final_answer == "done"
    assert len(approval_questions) == 1
    saved = rt.catalog.lookup("single_prompt_tool")
    assert saved.version == 1
    assert saved.code == code
    tool_turns = [turn.content for turn in rt.session.turns if turn.role == "tool"]
    assert any("deferred_until_answer" in turn for turn in tool_turns)


def test_create_tool_replay_failure_is_recorded(tmp_path):
    work = tmp_path / "work"
    work.mkdir(exist_ok=True)
    (work / "payload.txt").write_text("hello", encoding="utf-8")
    code = "from pathlib import Path; import sys; p = Path(sys.argv[1]); print(p.read_text().upper()); p.unlink()"
    rt, _, _ = _mk_runtime(
        tmp_path,
        [
            _A(
                {
                    "action_type": "create_tool",
                    "tool_name": "one_shot_file_reader",
                    "code": code,
                    "arguments": {"argv": ["payload.txt"]},
                    "expected_outcome": "read once then remove file",
                }
            ),
            _A(
                {
                    "action_type": "request_approval",
                    "question": "save one_shot_file_reader?",
                    "approval_kind": "save_tool",
                }
            ),
            _A(
                {
                    "action_type": "answer",
                    "final_answer": "done",
                }
            ),
        ],
        approvals=[True],
    )

    rt.run_task("payload 파일을 한 번 읽고 삭제해줘")
    saved = rt.catalog.lookup("one_shot_file_reader")
    assert saved.verification_status == "replay_failed"
    assert saved.success_count == 0
    assert saved.failure_count == 0
    assert saved.last_failure_at == ""

    tool_dir = tmp_path / "state" / "tools" / "one_shot_file_reader"
    verification = json.loads((tool_dir / "verification.json").read_text())
    assert verification["replay_ok"] is False
    assert verification["replay_run"]["ok"] is False
    assert "FileNotFoundError" in verification["replay_run"]["stderr_tail"]


def test_save_tool_denied_does_not_persist(tmp_path):
    code = "print('x')"
    rt, _, out = _mk_runtime(
        tmp_path,
        [
            _A(
                {
                    "action_type": "create_tool",
                    "tool_name": "keep_me_ephemeral",
                    "code": code,
                    "arguments": {},
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
        approvals=[False],
    )
    rt.run_task("temp tool")
    assert not rt.catalog.has("keep_me_ephemeral")


def test_save_tool_approval_without_pending_tool_is_skipped(tmp_path):
    rt, _, out = _mk_runtime(
        tmp_path,
        [
            _A(
                {
                    "action_type": "request_approval",
                    "question": "save a hallucinated tool?",
                    "approval_kind": "save_tool",
                }
            ),
            _A({"action_type": "answer", "final_answer": "done"}),
        ],
    )

    def fail_if_called(question: str) -> bool:
        raise AssertionError("approval_fn should not be called without a pending tool")

    rt.approval_fn = fail_if_called
    result = rt.run_task("bad approval")

    assert result.final_answer == "done"
    assert "pending 생성 도구 없음" in out.getvalue()
    tool_turns = [turn.content for turn in rt.session.turns if turn.role == "tool"]
    assert any("저장할 pending 생성 도구가 없습니다" in turn for turn in tool_turns)


def test_ask_user_eof_is_treated_as_empty_reply(tmp_path):
    rt, _, _ = _mk_runtime(
        tmp_path,
        [
            _A({"action_type": "ask_user", "question": "more?"}),
            _A({"action_type": "answer", "final_answer": "done"}),
        ],
    )

    def raise_eof(prompt: str) -> str:
        raise EOFError

    rt.user_input_fn = raise_eof
    result = rt.run_task("need clarification")

    assert result.final_answer == "done"


def test_ask_user_in_single_shot_mode_fails_fast(tmp_path):
    rt, llm, out = _mk_runtime(
        tmp_path,
        [
            _A({"action_type": "ask_user", "question": "어떤 데이터를 정리할까요?"}),
            # The runtime must NOT consume this second response — fail-fast
            # has to break out of the loop before another LLM call.
            _A({"action_type": "answer", "final_answer": "should-not-reach"}),
        ],
    )

    def raise_unavailable(prompt: str) -> str:
        raise InteractiveInputUnavailable

    rt.user_input_fn = raise_unavailable
    result = rt.run_task("데이터 정리해줘.")

    assert result.ok is False
    assert "단일 실행 모드" in result.final_answer
    # The agent's question is surfaced on its own line so the runtime notice
    # stays focused on the mode limitation rather than embedding the question.
    assert "어떤 데이터를 정리할까요?" in out.getvalue()
    assert len(llm.calls) == 1, "fail-fast should not trigger a second LLM call"



def test_successful_create_tool_is_saved_before_later_failed_attempt(tmp_path):
    # Save approval is runtime-managed after the final answer/failure. A later
    # failed create_tool in the same task must not discard the last successful
    # reusable candidate.
    good = "print('ok')"
    bad = "raise RuntimeError('regression')"
    rt, _, _ = _mk_runtime(
        tmp_path,
        [
            _A(
                {
                    "action_type": "create_tool",
                    "tool_name": "winner",
                    "code": good,
                    "arguments": {},
                }
            ),
            _A(
                {
                    "action_type": "create_tool",
                    "tool_name": "loser",
                    "code": bad,
                    "arguments": {},
                }
            ),
            _A(
                {
                    "action_type": "create_tool",
                    "tool_name": "loser",
                    "code": bad,
                    "arguments": {},
                }
            ),
        ],
        approvals=[True],
    )
    rt.run_task("compose two tools")
    assert rt.catalog.has("winner")
    assert not rt.catalog.has("loser")


def test_save_tool_denial_does_not_leak_to_later_task(tmp_path):
    code = "print('x')"
    rt, _, _ = _mk_runtime(
        tmp_path,
        [
            _A(
                {
                    "action_type": "create_tool",
                    "tool_name": "first_tool",
                    "code": code,
                    "arguments": {},
                }
            ),
            _A(
                {
                    "action_type": "request_approval",
                    "question": "save first?",
                    "approval_kind": "save_tool",
                }
            ),
            _A({"action_type": "answer", "final_answer": "done1"}),
            _A(
                {
                    "action_type": "request_approval",
                    "question": "save later?",
                    "approval_kind": "save_tool",
                }
            ),
            _A({"action_type": "answer", "final_answer": "done2"}),
        ],
        approvals=[False, True],
    )
    rt.run_task("task1")
    rt.run_task("task2")
    assert not rt.catalog.has("first_tool")


def test_create_tool_failure_then_repair(tmp_path):
    # first code fails, second code succeeds
    bad = "raise RuntimeError('bug')"
    good = "print('ok')"
    rt, _, out = _mk_runtime(
        tmp_path,
        [
            _A(
                {
                    "action_type": "create_tool",
                    "tool_name": "flaky",
                    "code": bad,
                    "arguments": {},
                }
            ),
            _A(
                {
                    "action_type": "create_tool",
                    "tool_name": "flaky",
                    "code": good,
                    "arguments": {},
                }
            ),
            _A({"action_type": "answer", "final_answer": "fixed"}),
        ],
    )
    result = rt.run_task("buggy task")
    assert result.final_answer == "fixed"


def test_create_tool_repair_exhausted(tmp_path):
    # With max_repair_per_tool=1 the runtime allows exactly one retry after the
    # initial failure and must give up on the repair attempt rather than run a
    # third execution.
    bad = "raise RuntimeError('bug')"
    rt, llm, _ = _mk_runtime(
        tmp_path,
        [
            _A(
                {
                    "action_type": "create_tool",
                    "tool_name": "always_fail",
                    "code": bad,
                    "arguments": {},
                }
            ),
            _A(
                {
                    "action_type": "create_tool",
                    "tool_name": "always_fail",
                    "code": bad,
                    "arguments": {},
                }
            ),
        ],
    )
    result = rt.run_task("hopeless")
    assert "중단합니다" in result.final_answer
    assert len(llm.calls) == 2


def test_renamed_create_tool_failures_do_not_bypass_repair_budget(tmp_path):
    bad = "raise RuntimeError('bug')"
    rt, llm, _ = _mk_runtime(
        tmp_path,
        [
            _A({"action_type": "create_tool", "tool_name": "first_name", "code": bad, "arguments": {}}),
            _A({"action_type": "create_tool", "tool_name": "second_name", "code": bad, "arguments": {}}),
            _A({"action_type": "answer", "final_answer": "should-not-reach"}),
        ],
    )

    result = rt.run_task("hopeless renamed repairs")

    assert result.ok is False
    assert "중단합니다" in result.final_answer
    assert "second_name" in result.final_answer
    assert len(llm.calls) == 2


def test_create_tool_success_resets_consecutive_failure_budget(tmp_path):
    bad = "raise RuntimeError('bug')"
    good = "print('ok')"
    rt, _, _ = _mk_runtime(
        tmp_path,
        [
            _A({"action_type": "create_tool", "tool_name": "first_bad", "code": bad, "arguments": {}}),
            _A({"action_type": "create_tool", "tool_name": "fixed", "code": good, "arguments": {}}),
            _A({"action_type": "create_tool", "tool_name": "later_bad", "code": bad, "arguments": {}}),
            _A({"action_type": "create_tool", "tool_name": "later_fixed", "code": good, "arguments": {}}),
            _A({"action_type": "answer", "final_answer": "recovered twice"}),
        ],
    )

    result = rt.run_task("recover from two separate failures")

    assert result.ok is True
    assert result.final_answer == "recovered twice"



def test_repair_budget_resets_between_tasks(tmp_path):
    bad = "raise RuntimeError('bug')"
    good = "print('ok')"
    rt, _, _ = _mk_runtime(
        tmp_path,
        [
            # task1: two bad attempts, give up.
            _A({"action_type": "create_tool", "tool_name": "generated", "code": bad, "arguments": {}}),
            _A({"action_type": "create_tool", "tool_name": "generated", "code": bad, "arguments": {}}),
            # task2 reuses the budget: one bad attempt, one repair that succeeds.
            _A({"action_type": "create_tool", "tool_name": "generated", "code": bad, "arguments": {}}),
            _A({"action_type": "create_tool", "tool_name": "generated", "code": good, "arguments": {}}),
            _A({"action_type": "answer", "final_answer": "fixed on second task"}),
        ],
    )
    first = rt.run_task("task1")
    assert "중단합니다" in first.final_answer

    second = rt.run_task("task2")
    assert second.final_answer == "fixed on second task"


def test_reuse_tool_runs_saved_code(tmp_path):
    rt, _, out = _mk_runtime(
        tmp_path,
        [
            _A(
                {
                    "action_type": "reuse_tool",
                    "tool_name": "saved_echo",
                    "arguments": {"argv": ["hello"]},
                }
            ),
            _A({"action_type": "answer", "final_answer": "done"}),
        ],
    )
    from adaptive_agent.catalog import SavedToolSpec
    rt.catalog.save_generated(
        SavedToolSpec(
            name="saved_echo",
            description="",
            input_summary="",
            output_summary="",
            code="import sys; print('echo:' + sys.argv[1])",
            verification_status="runtime_verified",
            verification_details={"method": "create_tool_execution", "exit_status": 0},
        )
    )
    rt.run_task("use saved tool")
    # session should contain the tool observation with ok=true
    last_obs = [t for t in rt.session.turns if t.role == "tool"][-1]
    assert '"ok": true' in last_obs.content.lower() or '"ok":true' in last_obs.content.lower()

    reloaded = ToolCatalog(state_dir=tmp_path / "state").lookup("saved_echo")
    assert reloaded.success_count == 0
    assert reloaded.failure_count == 0


def test_reuse_tool_strips_version_suffix_from_tool_name(tmp_path):
    rt, _, _ = _mk_runtime(
        tmp_path,
        [
            _A(
                {
                    "action_type": "reuse_tool",
                    "tool_name": "saved_echo v1",
                    "arguments": {"argv": ["hi"]},
                }
            ),
            _A({"action_type": "answer", "final_answer": "done"}),
        ],
    )
    from adaptive_agent.catalog import SavedToolSpec
    rt.catalog.save_generated(
        SavedToolSpec(
            name="saved_echo",
            description="",
            input_summary="",
            output_summary="",
            code="import sys; print('echo:' + sys.argv[1])",
            verification_status="runtime_verified",
            verification_details={"method": "create_tool_execution", "exit_status": 0},
        )
    )
    rt.run_task("reuse with v-suffix")
    last_obs = [t for t in rt.session.turns if t.role == "tool"][-1]
    assert '"ok": true' in last_obs.content.lower() or '"ok":true' in last_obs.content.lower()


def test_runtime_system_prompt_includes_best_match_hint(tmp_path):
    rt, llm, _ = _mk_runtime(
        tmp_path,
        [_A({"action_type": "answer", "final_answer": "done"})],
    )

    from adaptive_agent.catalog import SavedToolSpec

    rt.catalog.save_generated(
        SavedToolSpec(
            name="csv_cleanup",
            description="csv rows dedupe and sort by date",
            input_summary="csv file",
            output_summary="cleaned csv",
            code="print('ok')",
            tags=["csv", "dedupe", "sort"],
            verification_status="runtime_verified",
            verification_details={"method": "create_tool_execution", "exit_status": 0},
        )
    )

    rt.run_task("csv 파일 중복 제거해줘")
    system_prompt = llm.calls[0][0]["content"]
    assert "Recommended reusable tool for current task" in system_prompt
    assert "csv_cleanup" in system_prompt


def test_malformed_action_is_tolerated(tmp_path):
    rt, _, out = _mk_runtime(
        tmp_path,
        [
            "{not json",
            _A({"action_type": "answer", "final_answer": "recovered"}),
        ],
    )
    result = rt.run_task("be robust")
    assert result.final_answer == "recovered"


def test_max_iterations_fallback(tmp_path):
    # LLM keeps asking questions; runtime should exit after max iterations
    responses = [
        _A({"action_type": "ask_user", "question": "more?"}) for _ in range(20)
    ]
    rt, _, out = _mk_runtime(
        tmp_path,
        responses,
        user_replies=["yes"] * 20,
    )
    result = rt.run_task("infinite")
    assert "최대 반복 횟수" in result.final_answer
    assert "최대 반복 횟수" in out.getvalue()


def test_reuse_tool_missing_observation_instructs_create_tool_fallback(tmp_path):
    rt, _, _ = _mk_runtime(
        tmp_path,
        [
            _A({"action_type": "reuse_tool", "tool_name": "missing_saved_tool", "arguments": {}}),
            _A({"action_type": "answer", "final_answer": "fallback"}),
        ],
    )

    result = rt.run_task("use missing saved tool")

    assert result.final_answer == "fallback"
    tool_turns = [turn.content for turn in rt.session.turns if turn.role == "tool"]
    assert any("create_tool" in turn for turn in tool_turns)


def test_double_escaped_generated_code_is_normalized_before_execution(tmp_path):
    code = "import sys\\nprint('ok')"
    rt, _, _ = _mk_runtime(
        tmp_path,
        [
            _A(
                {
                    "action_type": "create_tool",
                    "tool_name": "escaped_newline_tool",
                    "code": code,
                    "arguments": {},
                }
            ),
            _A({"action_type": "answer", "final_answer": "done"}),
        ],
    )

    result = rt.run_task("run escaped source")

    assert result.ok is True
    assert result.final_answer == "done"
    tool_turns = [turn.content for turn in rt.session.turns if turn.role == "tool"]
    assert any('"code_normalized": true' in turn for turn in tool_turns)


def test_valid_generated_code_with_escaped_string_is_not_normalized(tmp_path):
    code = "print('a\\\\nb')"
    rt, _, _ = _mk_runtime(
        tmp_path,
        [
            _A(
                {
                    "action_type": "create_tool",
                    "tool_name": "valid_escape_string_tool",
                    "code": code,
                    "arguments": {},
                }
            ),
            _A({"action_type": "answer", "final_answer": "done"}),
        ],
    )

    result = rt.run_task("run valid source")

    assert result.ok is True
    tool_turns = [turn.content for turn in rt.session.turns if turn.role == "tool"]
    assert not any('"code_normalized": true' in turn for turn in tool_turns)



def test_run_one_shot_returns_nonzero_on_failure(monkeypatch):
    class FakeRuntime:
        def run_task(self, task: str):
            return type("R", (), {"final_answer": "[LLM error] boom", "iterations_used": 1, "ok": False})()

    monkeypatch.setattr(
        "adaptive_agent.runtime._build_runtime",
        lambda **kwargs: FakeRuntime(),
    )
    assert run_one_shot("hello") == 1


def test_build_runtime_wires_embedding_client_when_configured(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("AGENT_EMBEDDING_MODEL", "embed-test")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://embedding.example")
    monkeypatch.setenv("OPENAI_API_KEY", "token")

    rt = _build_runtime(
        session_id="embed",
        llm=ScriptedLLM([_A({"action_type": "answer", "final_answer": "done"})]),
        approval_fn=lambda question: False,
        user_input_fn=lambda prompt: "",
    )

    assert rt.catalog._embedding_client is not None
    assert rt.catalog._embedding_client.model == "embed-test"


def test_run_one_shot_returns_zero_on_success(monkeypatch):
    class FakeRuntime:
        def run_task(self, task: str):
            return type("R", (), {"final_answer": "ok", "iterations_used": 1, "ok": True})()

    monkeypatch.setattr(
        "adaptive_agent.runtime._build_runtime",
        lambda **kwargs: FakeRuntime(),
    )
    assert run_one_shot("hello") == 0


def test_run_one_shot_propagates_repair_exhausted_failure(tmp_path, monkeypatch):
    # Regression: until LoopResult.ok reflected repair-exhaustion, run_one_shot
    # reported 0 even after the runtime gave up on a tool.
    bad = "raise RuntimeError('bug')"
    rt, _, _ = _mk_runtime(
        tmp_path,
        [
            _A({"action_type": "create_tool", "tool_name": "t", "code": bad, "arguments": {}}),
            _A({"action_type": "create_tool", "tool_name": "t", "code": bad, "arguments": {}}),
        ],
    )
    monkeypatch.setattr(
        "adaptive_agent.runtime._build_runtime",
        lambda **kwargs: rt,
    )
    assert run_one_shot("hopeless") == 1



def test_roblox_creator_audit_flow_reads_script_and_saves_tool(tmp_path):
    work = tmp_path / "work"
    work.mkdir(exist_ok=True)
    (work / "roblox_guard_npc.lua").write_text(
        _repo_fixture_text("roblox_guard_npc.lua"),
        encoding="utf-8",
    )

    audit_code = """
from pathlib import Path
import sys

text = Path(sys.argv[1]).read_text(encoding='utf-8')
issues = []
if 'wait(' in text:
    issues.append('wait() 사용')
if 'FindFirstChild("Humanoid")' in text and 'if not humanoid' not in text and 'if humanoid then' not in text:
    issues.append('Humanoid nil 체크 없음')
print(' | '.join(issues) or '문제 없음')
""".strip()

    rt, llm, _ = _mk_runtime(
        tmp_path,
        [
            _A(
                {
                    "action_type": "run_builtin",
                    "tool_name": "read_text_file",
                    "arguments": {"path": "roblox_guard_npc.lua"},
                }
            ),
            _A(
                {
                    "action_type": "create_tool",
                    "tool_name": "luau_script_audit",
                    "code": audit_code,
                    "arguments": {"argv": ["roblox_guard_npc.lua"]},
                    "expected_outcome": "audit Luau gameplay script",
                }
            ),
            _A(
                {
                    "action_type": "request_approval",
                    "question": "save Luau audit tool?",
                    "approval_kind": "save_tool",
                }
            ),
            _A(
                {
                    "action_type": "answer",
                    "final_answer": "Roblox creator audit complete: wait() 사용과 Humanoid nil 체크 누락을 찾았습니다.",
                }
            ),
        ],
        approvals=[True],
    )

    result = rt.run_task("이 Luau 스크립트를 리뷰해줘")
    assert result.final_answer.startswith("Roblox creator audit complete")
    assert rt.catalog.has("luau_script_audit")

    second_call = llm.calls[1]
    assert any("guard:FindFirstChild" in m["content"] for m in second_call)
    assert any("wait() 사용" in t.content for t in rt.session.turns if t.role == "tool")



def test_roblox_creator_saved_tool_reuse_on_second_script(tmp_path):
    work = tmp_path / "work"
    work.mkdir(exist_ok=True)
    (work / "roblox_damage_pad.lua").write_text(
        _repo_fixture_text("roblox_damage_pad.lua"),
        encoding="utf-8",
    )

    audit_code = """
from pathlib import Path
import sys

text = Path(sys.argv[1]).read_text(encoding='utf-8')
issues = []
if 'wait(' in text:
    issues.append('wait() 사용')
if 'FindFirstChild("Humanoid")' in text and 'if not humanoid' not in text and 'if humanoid then' not in text:
    issues.append('Humanoid nil 체크 없음')
print(' | '.join(issues) or '문제 없음')
""".strip()

    rt, _, _ = _mk_runtime(
        tmp_path,
        [
            _A(
                {
                    "action_type": "reuse_tool",
                    "tool_name": "luau_script_audit",
                    "arguments": {"argv": ["roblox_damage_pad.lua"]},
                }
            ),
            _A(
                {
                    "action_type": "answer",
                    "final_answer": "Roblox creator reuse audit complete.",
                }
            ),
        ],
    )

    from adaptive_agent.catalog import SavedToolSpec

    rt.catalog.save_generated(
        SavedToolSpec(
            name="luau_script_audit",
            description="audit Luau gameplay script",
            input_summary="script path",
            output_summary="issue summary",
            code=audit_code,
        )
    )

    result = rt.run_task("저장된 Luau 감사 툴로 다시 검사해줘")
    assert result.final_answer == "Roblox creator reuse audit complete."
    assert any("wait() 사용" in t.content for t in rt.session.turns if t.role == "tool")


def test_create_tool_with_suspicious_output_emits_warning_observation(tmp_path):
    """Avatar regression: tool exits 0 but every counter is 0.

    The runtime must surface this as an observation (so the LLM can decide to
    retry) and as a warning in the save approval prompt (so a human reviewer
    notices before approving).
    """
    bug_code = "import json; print(json.dumps({'mesh_count': 0, 'texture_count': 0}))"
    rt, _, out = _mk_runtime(
        tmp_path,
        [
            _A(
                {
                    "action_type": "create_tool",
                    "tool_name": "count_meshes",
                    "code": bug_code,
                    "arguments": {"argv": []},
                    "expected_outcome": "mesh and texture counts",
                }
            ),
            _A({"action_type": "answer", "final_answer": "(answer)"}),
        ],
    )

    rt.run_task("avatar 메타데이터 카운트")

    warning_turns = [
        t.content for t in rt.session.turns if t.role == "tool" and "output_warning" in t.content
    ]
    assert warning_turns, "expected an output_warning observation"
    parsed = json.loads(warning_turns[0].removeprefix("[tool_result] "))
    assert "0" in parsed["result"]["output_warning"]
    assert "의심" in out.getvalue()


def test_repl_approval_uses_repl_input_channel(monkeypatch):
    """REPL approval must read from the same channel as REPL commands so a
    follow-up command typed during a [y/n] prompt is not silently consumed by
    the 3-attempt retry loop on stdin.
    """
    from adaptive_agent.runtime import _make_repl_approval_fn

    monkeypatch.setattr("sys.stdin.isatty", lambda: False)
    inputs = iter(["y", "should-not-be-consumed"])

    def fake_read(prompt: str) -> str:
        try:
            return next(inputs)
        except StopIteration:
            raise EOFError

    monkeypatch.setattr("adaptive_agent.runtime._read_repl_input", fake_read)
    approval = _make_repl_approval_fn()

    # First approval: reads "y", succeeds without retry.
    assert approval("save tool A?") is True
    # Subsequent REPL command "should-not-be-consumed" must remain in the
    # channel — i.e., next() should still produce it for the REPL loop.
    assert next(inputs) == "should-not-be-consumed"


def test_single_run_system_prompt_includes_no_ask_user_note(tmp_path):
    rt, llm, _ = _mk_runtime(
        tmp_path,
        [_A({"action_type": "answer", "final_answer": "ok"})],
    )
    rt.config.interactive = False
    rt.run_task("hello")
    system_prompt = llm.calls[0][0]["content"]
    assert "Single-run mode" in system_prompt
    assert "ask_user" in system_prompt
