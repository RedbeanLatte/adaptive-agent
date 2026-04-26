from __future__ import annotations

import io
import json
from pathlib import Path

from adaptive_agent.builtins import register_builtins
from adaptive_agent.catalog import ToolCatalog
from adaptive_agent.runtime import Runtime, RuntimeConfig
from adaptive_agent.session import Session


class ScriptedLLM:
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
    approvals: list[bool] | None = None,
) -> tuple[Runtime, ScriptedLLM, io.StringIO]:
    state_dir = tmp_path / "state"
    workroot = tmp_path / "work"
    state_dir.mkdir(exist_ok=True)
    workroot.mkdir(exist_ok=True)

    catalog = ToolCatalog(state_dir=state_dir)
    register_builtins(catalog)
    session = Session(session_id="demo")
    llm = ScriptedLLM(responses)
    out = io.StringIO()
    approvals_iter = iter(approvals or [])

    rt = Runtime(
        llm=llm,
        catalog=catalog,
        session=session,
        config=RuntimeConfig(
            workroot=workroot,
            state_dir=state_dir,
            session_id="demo",
            max_iterations=8,
        ),
        approval_fn=lambda question: next(approvals_iter, False),
        user_input_fn=lambda prompt: "",
        output_stream=out,
    )
    return rt, llm, out


def _A(payload: dict) -> str:
    return json.dumps(payload, ensure_ascii=False)


def _repo_fixture_text(name: str) -> str:
    root = Path(__file__).resolve().parents[1]
    return (root / "fixtures" / name).read_text(encoding="utf-8")


def test_roblox_code_scaffold_flow_reads_request_and_saves_tool(tmp_path):
    work = tmp_path / "work"
    work.mkdir(exist_ok=True)
    (work / "roblox_coin_pickup_request.txt").write_text(
        _repo_fixture_text("roblox_coin_pickup_request.txt"),
        encoding="utf-8",
    )

    scaffold_code = """
from pathlib import Path
import json
import sys

brief = Path(sys.argv[1]).read_text(encoding='utf-8')
assert 'Touched' in brief
assert 'leaderstats' in brief
print(json.dumps({
    'artifact_type': 'roblox_code_scaffold',
    'title': 'Coin pickup scaffold',
    'items': [
        'Touched event',
        'duplicate guard',
        'leaderstats Coins +1',
        'disable pickup',
    ],
}))
""".strip()

    rt, llm, _ = _mk_runtime(
        tmp_path,
        [
            _A(
                {
                    "action_type": "run_builtin",
                    "tool_name": "read_text_file",
                    "arguments": {"path": "roblox_coin_pickup_request.txt"},
                }
            ),
            _A(
                {
                    "action_type": "create_tool",
                    "tool_name": "roblox_coin_pickup_scaffold",
                    "code": scaffold_code,
                    "arguments": {"argv": ["roblox_coin_pickup_request.txt"]},
                    "expected_outcome": "generate Luau coin pickup scaffold",
                }
            ),
            _A(
                {
                    "action_type": "request_approval",
                    "question": "save Roblox coin scaffold tool?",
                    "approval_kind": "save_tool",
                }
            ),
            _A(
                {
                    "action_type": "answer",
                    "final_answer": "Luau 코인 pickup 초안을 생성했습니다.",
                }
            ),
        ],
        approvals=[True],
    )

    result = rt.run_task("코인 pickup 스크립트 초안을 만들어줘")

    assert result.final_answer.startswith("Luau 코인 pickup 초안을 생성했습니다.")
    assert "생성 artifact:" in result.final_answer
    assert "- artifact_type: roblox_code_scaffold" in result.final_answer
    assert "  - Touched event" in result.final_answer
    assert rt.catalog.has("roblox_coin_pickup_scaffold")
    second_call = llm.calls[1]
    assert any("leaderstats" in m["content"] for m in second_call)
    tool_turns = [t.content for t in rt.session.turns if t.role == "tool"]
    assert any('"artifact_preview"' in content for content in tool_turns)
    assert any('"artifact_type": "roblox_code_scaffold"' in content for content in tool_turns)
    assert any('"item_count": 4' in content for content in tool_turns)


def test_material_prompt_pack_flow_reads_brief_and_saves_tool(tmp_path):
    work = tmp_path / "work"
    work.mkdir(exist_ok=True)
    (work / "material_lava_cave_brief.txt").write_text(
        _repo_fixture_text("material_lava_cave_brief.txt"),
        encoding="utf-8",
    )

    prompt_code = """
from pathlib import Path
import json
import sys

brief = Path(sys.argv[1]).read_text(encoding='utf-8')
assert 'stylized lava cave' in brief.lower()
assert 'material' in brief.lower()
print(json.dumps({
    'artifact_type': 'material_prompt_pack',
    'title': 'Stylized lava cave prompt pack',
    'items': [
        'Stylized obsidian floor',
        'Glowing lava crack wall',
        'Ash-covered basalt pillar',
        'Soft ember dust surface',
    ],
}))
""".strip()

    rt, llm, _ = _mk_runtime(
        tmp_path,
        [
            _A(
                {
                    "action_type": "run_builtin",
                    "tool_name": "read_text_file",
                    "arguments": {"path": "material_lava_cave_brief.txt"},
                }
            ),
            _A(
                {
                    "action_type": "create_tool",
                    "tool_name": "material_prompt_pack",
                    "code": prompt_code,
                    "arguments": {"argv": ["material_lava_cave_brief.txt"]},
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
            _A(
                {
                    "action_type": "answer",
                    "final_answer": "material/texture prompt pack을 만들었습니다.",
                }
            ),
        ],
        approvals=[True],
    )

    result = rt.run_task("lava cave용 material prompt pack을 만들어줘")

    assert result.final_answer.startswith("material/texture prompt pack을 만들었습니다.")
    assert "생성 artifact:" in result.final_answer
    assert "- artifact_type: material_prompt_pack" in result.final_answer
    assert "  - Stylized obsidian floor" in result.final_answer
    assert rt.catalog.has("material_prompt_pack")
    second_call = llm.calls[1]
    assert any("stylized lava cave" in m["content"].lower() for m in second_call)
    tool_turns = [t.content for t in rt.session.turns if t.role == "tool"]
    assert any('"artifact_preview"' in content for content in tool_turns)
    assert any('"artifact_type": "material_prompt_pack"' in content for content in tool_turns)
    assert any('"title": "Stylized lava cave prompt pack"' in content for content in tool_turns)


def test_avatar_checklist_flow_reads_metadata_and_saves_tool(tmp_path):
    work = tmp_path / "work"
    work.mkdir(exist_ok=True)
    (work / "avatar_knight_metadata.json").write_text(
        _repo_fixture_text("avatar_knight_metadata.json"),
        encoding="utf-8",
    )

    checklist_code = """
from pathlib import Path
import json
import sys

data = json.loads(Path(sys.argv[1]).read_text(encoding='utf-8'))
assert data['asset_name'] == 'knight_avatar'
assert data['rig_type'] == 'R15'
print(json.dumps({
    'artifact_type': 'avatar_setup_checklist',
    'title': 'Knight avatar auto-setup checklist',
    'items': [
        'rig type 확인',
        'mesh / texture 누락 여부 확인',
        'accessory attachment 대응 확인',
        'naming consistency 확인',
    ],
}))
""".strip()

    rt, llm, _ = _mk_runtime(
        tmp_path,
        [
            _A(
                {
                    "action_type": "run_builtin",
                    "tool_name": "read_text_file",
                    "arguments": {"path": "avatar_knight_metadata.json"},
                }
            ),
            _A(
                {
                    "action_type": "create_tool",
                    "tool_name": "avatar_setup_checklist",
                    "code": checklist_code,
                    "arguments": {"argv": ["avatar_knight_metadata.json"]},
                    "expected_outcome": "generate avatar setup checklist",
                }
            ),
            _A(
                {
                    "action_type": "request_approval",
                    "question": "save avatar checklist tool?",
                    "approval_kind": "save_tool",
                }
            ),
            _A(
                {
                    "action_type": "answer",
                    "final_answer": "avatar auto-setup 전 체크리스트입니다.",
                }
            ),
        ],
        approvals=[True],
    )

    result = rt.run_task("avatar auto-setup 체크리스트를 만들어줘")

    assert result.final_answer.startswith("avatar auto-setup 전 체크리스트입니다.")
    assert "생성 artifact:" in result.final_answer
    assert "- artifact_type: avatar_setup_checklist" in result.final_answer
    assert "  - rig type 확인" in result.final_answer
    assert rt.catalog.has("avatar_setup_checklist")
    second_call = llm.calls[1]
    assert any("knight_avatar" in m["content"] for m in second_call)
    tool_turns = [t.content for t in rt.session.turns if t.role == "tool"]
    assert any('"artifact_preview"' in content for content in tool_turns)
    assert any('"artifact_type": "avatar_setup_checklist"' in content for content in tool_turns)
    assert any('"first_item": "rig type 확인"' in content for content in tool_turns)

