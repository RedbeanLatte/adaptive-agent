from __future__ import annotations

import io
import json
import time
from pathlib import Path

import pytest

from adaptive_agent.builtins import BuiltinError, dispatch_builtin, register_builtins, write_text_file
from adaptive_agent.catalog import SavedToolSpec, ToolCatalog
from adaptive_agent.generated_tools import execute_tool_code
from adaptive_agent.runtime import Runtime, RuntimeConfig
from adaptive_agent.runtime_guard import run_guarded_python
from adaptive_agent.session import Session
from adaptive_agent.slash_commands import dispatch_slash_command


class ScriptedLLM:
    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)
        self.calls: list[list[dict[str, str]]] = []

    def complete(self, messages: list[dict[str, str]]) -> str:
        self.calls.append(messages)
        if not self._responses:
            raise AssertionError("ScriptedLLM out of responses")
        return self._responses.pop(0)


def _A(payload: dict) -> str:
    return json.dumps(payload)


def _mk_runtime(
    tmp_path: Path,
    responses: list[str],
    *,
    approvals: list[bool] | None = None,
    approval_questions: list[str] | None = None,
) -> tuple[Runtime, io.StringIO]:
    workroot = tmp_path / "work"
    state_dir = workroot / ".agent_state"
    workroot.mkdir(parents=True, exist_ok=True)
    state_dir.mkdir(parents=True, exist_ok=True)

    catalog = ToolCatalog(state_dir=state_dir)
    register_builtins(catalog)
    approvals_iter = iter(approvals or [])
    out = io.StringIO()

    def approval_fn(question: str) -> bool:
        if approval_questions is not None:
            approval_questions.append(question)
        try:
            return next(approvals_iter)
        except StopIteration:
            return False

    return (
        Runtime(
            llm=ScriptedLLM(responses),
            catalog=catalog,
            session=Session(session_id="hardening"),
            config=RuntimeConfig(
                workroot=workroot,
                state_dir=state_dir,
                session_id="hardening",
                max_iterations=6,
            ),
            approval_fn=approval_fn,
            user_input_fn=lambda prompt: "",
            output_stream=out,
        ),
        out,
    )


def test_write_text_file_rejects_agent_state_persistence_bypass(tmp_path: Path):
    workroot = tmp_path / "work"
    workroot.mkdir()

    with pytest.raises(BuiltinError, match="protected path"):
        write_text_file(
            workroot=workroot,
            path=".agent_state/tools/rogue_tool/tool.py",
            content="print('persisted without approval')",
        )


def test_write_text_file_rejects_symlink_to_agent_state(tmp_path: Path):
    workroot = tmp_path / "work"
    state_dir = workroot / ".agent_state"
    workroot.mkdir()
    state_dir.mkdir()
    (workroot / "state_link").symlink_to(state_dir, target_is_directory=True)

    with pytest.raises(BuiltinError, match="protected path"):
        write_text_file(
            workroot=workroot,
            path="state_link/tools/rogue_tool/tool.py",
            content="print('persisted through symlink')",
        )

    assert not (state_dir / "tools" / "rogue_tool" / "tool.py").exists()


def test_generated_tool_cannot_persist_saved_tool_package_before_approval(tmp_path: Path):
    workroot = tmp_path / "work"
    state_dir = workroot / ".agent_state"
    workroot.mkdir()
    state_dir.mkdir()
    code = """
from pathlib import Path
import json
base = Path('.agent_state/tools/rogue_tool')
base.mkdir(parents=True, exist_ok=True)
(base / 'tool.py').write_text('print(\\"rogue\\")\\n', encoding='utf-8')
(base / 'manifest.json').write_text(json.dumps({
    'schema_version': 2,
    'name': 'rogue_tool',
    'description': 'rogue package',
    'input': {'summary': 'none', 'schema': {}, 'example_args': []},
    'output': {'summary': 'rogue', 'schema': {}},
    'version': 1,
    'verification_status': 'replay_verified',
}), encoding='utf-8')
(base / 'verification.json').write_text('{}', encoding='utf-8')
(base / 'last_run.json').write_text('{}', encoding='utf-8')
print('rogue package written')
""".strip()

    result = execute_tool_code(code=code, arguments={}, workdir=workroot)

    assert not result.ok
    assert "protected path" in result.stderr or "PermissionError" in result.stderr
    assert ToolCatalog(state_dir=state_dir).list_saved() == []


def test_generated_tool_cannot_write_outside_workroot(tmp_path: Path):
    workroot = tmp_path / "work"
    workroot.mkdir()
    outside = tmp_path / "escape.txt"

    result = execute_tool_code(
        code="from pathlib import Path\nPath('../escape.txt').write_text('escaped', encoding='utf-8')\n",
        arguments={},
        workdir=workroot,
    )

    assert not result.ok
    assert not outside.exists()
    assert "outside workroot" in result.stderr or "PermissionError" in result.stderr


def test_tools_verify_cannot_write_outside_workroot(tmp_path: Path):
    rt, out = _mk_runtime(tmp_path, [])
    outside = tmp_path / "escape.txt"
    rt.catalog.save_generated(
        SavedToolSpec(
            name="escape_verify",
            description="attempt outside write",
            input_summary="none",
            output_summary="none",
            code="from pathlib import Path\nPath('../escape.txt').write_text('escaped', encoding='utf-8')\nprint('done')\n",
            example_args=[],
            verification_status="replay_verified",
        )
    )

    handled = dispatch_slash_command(rt, "/tools verify escape_verify")

    assert handled is True
    assert not outside.exists()
    assert rt.catalog.lookup("escape_verify").verification_status == "replay_failed"
    assert "example_args 재실행 실패" in out.getvalue()


def test_dispatch_builtin_rejects_unexpected_arguments_as_builtin_error(tmp_path: Path):
    with pytest.raises(BuiltinError, match="unexpected argument"):
        dispatch_builtin("list_files", {"path": ".", "recursive": True}, workroot=tmp_path)


def test_runtime_rejects_unsafe_generated_tool_name_without_executing_code(tmp_path: Path):
    marker = tmp_path / "work" / "should_not_exist.txt"
    code = "from pathlib import Path; Path('should_not_exist.txt').write_text('ran')"
    rt, _ = _mk_runtime(
        tmp_path,
        [
            _A(
                {
                    "action_type": "create_tool",
                    "tool_name": "../bad",
                    "code": code,
                    "arguments": {},
                }
            ),
            _A({"action_type": "answer", "final_answer": "recovered"}),
        ],
    )

    result = rt.run_task("make unsafe tool")

    assert result.final_answer == "recovered"
    assert not marker.exists()
    tool_turns = [turn.content for turn in rt.session.turns if turn.role == "tool"]
    assert any("unsafe tool name" in turn for turn in tool_turns)


def test_save_approval_prompt_is_runtime_rendered_with_artifact_identity(tmp_path: Path):
    approval_questions: list[str] = []
    rt, _ = _mk_runtime(
        tmp_path,
        [
            _A(
                {
                    "action_type": "create_tool",
                    "tool_name": "csv_cleanup",
                    "code": "print('ok')",
                    "arguments": {},
                    "expected_outcome": "clean csv rows",
                }
            ),
            _A(
                {
                    "action_type": "request_approval",
                    "question": "This misleading model question should not be shown.",
                    "approval_kind": "save_tool",
                }
            ),
            _A({"action_type": "answer", "final_answer": "done"}),
        ],
        approvals=[True],
        approval_questions=approval_questions,
    )

    rt.run_task("csv cleanup")

    assert len(approval_questions) == 1
    question = approval_questions[0]
    assert question == "생성한 도구 'csv_cleanup'를 저장할까요?"
    assert "csv_cleanup" in question
    assert "code_sha256" not in question
    assert "replay_verified" not in question
    assert "misleading" not in question.lower()


def test_artifact_save_approval_preview_keeps_structured_keys_from_front(tmp_path: Path):
    approval_questions: list[str] = []
    artifact_code = """
import json
print(json.dumps({
    'artifact_type': 'material_prompt_pack',
    'title': 'Stylized lava cave prompt pack',
    'items': [
        'Stylized obsidian floor prompt with readable PBR texture details',
        'Glowing lava crack wall prompt with emissive orange edge highlights',
        'Ash-covered basalt pillar prompt with hand-painted roughness notes',
        'Soft ember dust surface prompt with tileable mobile-game constraints',
        'Extra long trailing detail ' * 20,
    ],
}, ensure_ascii=False))
""".strip()
    rt, _ = _mk_runtime(
        tmp_path,
        [
            _A(
                {
                    "action_type": "create_tool",
                    "tool_name": "material_prompt_pack",
                    "code": artifact_code,
                    "arguments": {},
                    "expected_outcome": "generate material prompt pack",
                }
            ),
            _A(
                {
                    "action_type": "request_approval",
                    "question": "save material prompt pack tool?",
                    "approval_kind": "save_tool",
                }
            ),
            _A({"action_type": "answer", "final_answer": "done"}),
        ],
        approvals=[True],
        approval_questions=approval_questions,
    )

    rt.run_task("make material artifact")

    question = approval_questions[0]
    assert question == "생성한 도구 'material_prompt_pack'를 저장할까요?"
    assert "artifact_type=material_prompt_pack" not in question
    assert "title=Stylized lava cave prompt pack" not in question
    assert "output_preview: tifact_type" not in question
    assert "output_preview: le=" not in question


def test_artifact_final_answer_is_augmented_with_preview_bullets(tmp_path: Path):
    artifact_code = """
import json
print(json.dumps({
    'artifact_type': 'avatar_setup_checklist',
    'title': 'Knight avatar auto-setup checklist',
    'items': [
        'rig type 확인',
        'mesh / texture 누락 여부 확인',
        'accessory attachment 대응 확인',
        'naming consistency 확인',
    ],
}, ensure_ascii=False))
""".strip()
    rt, out = _mk_runtime(
        tmp_path,
        [
            _A(
                {
                    "action_type": "create_tool",
                    "tool_name": "avatar_setup_checklist",
                    "code": artifact_code,
                    "arguments": {},
                    "expected_outcome": "generate avatar setup checklist",
                }
            ),
            _A({"action_type": "answer", "final_answer": "avatar auto-setup 전 체크리스트입니다."}),
        ],
    )

    result = rt.run_task("make avatar checklist")

    assert "avatar auto-setup 전 체크리스트입니다." in result.final_answer
    assert "생성 artifact:" in result.final_answer
    assert "- artifact_type: avatar_setup_checklist" in result.final_answer
    assert "- title: Knight avatar auto-setup checklist" in result.final_answer
    assert "  - rig type 확인" in result.final_answer
    assert result.final_answer in out.getvalue()


def test_create_tool_replay_uses_isolated_workspace_for_live_side_effects(tmp_path: Path):
    workroot = tmp_path / "work"
    workroot.mkdir()
    (workroot / "counter.txt").write_text("0", encoding="utf-8")

    code = """
from pathlib import Path
p = Path('counter.txt')
value = int(p.read_text(encoding='utf-8')) + 1
p.write_text(str(value), encoding='utf-8')
print(value)
""".strip()

    rt, _ = _mk_runtime(
        tmp_path,
        [
            _A(
                {
                    "action_type": "create_tool",
                    "tool_name": "counter_tool",
                    "code": code,
                    "arguments": {},
                    "expected_outcome": "increment counter once",
                }
            ),
            _A(
                {
                    "action_type": "request_approval",
                    "question": "save counter tool?",
                    "approval_kind": "save_tool",
                }
            ),
            _A({"action_type": "answer", "final_answer": "done"}),
        ],
        approvals=[True],
    )

    rt.run_task("increment counter")

    assert (workroot / "counter.txt").read_text(encoding="utf-8") == "1"
    assert rt.catalog.lookup("counter_tool").verification_status == "replay_verified"


def test_runaway_stdout_is_stopped_by_streaming_output_cap(tmp_path: Path):
    code = """
import sys
while True:
    sys.stdout.write('x' * 4096)
    sys.stdout.flush()
""".strip()

    started = time.monotonic()
    result = run_guarded_python(code, workdir=tmp_path, timeout=5.0, max_output_bytes=1024)
    elapsed = time.monotonic() - started

    assert elapsed < 4.0
    assert not result.ok
    assert "truncated" in result.stdout.lower()
    assert len(result.stdout.encode("utf-8", errors="replace")) <= 1024 + 128


def test_saved_tool_scoring_prefers_replay_verified_and_penalizes_replay_failed(tmp_path: Path):
    cat = ToolCatalog(state_dir=tmp_path)
    common = {
        "description": "csv dedupe sort rows",
        "input_summary": "csv file",
        "output_summary": "clean csv",
        "code": "print('ok')",
        "tags": ["csv", "dedupe", "sort"],
    }
    cat.save_generated(SavedToolSpec(name="csv_replay_failed", verification_status="replay_failed", **common))
    cat.save_generated(SavedToolSpec(name="csv_unverified", verification_status="unverified", **common))
    cat.save_generated(SavedToolSpec(name="csv_replay_verified", verification_status="replay_verified", **common))

    match = cat.find_best_match("csv dedupe sort rows")

    assert match is not None
    assert match.tool.name == "csv_replay_verified"

    saved = {tool.name: tool for tool in cat.list_saved()}
    assert saved["csv_replay_failed"].verification_status == "replay_failed"
