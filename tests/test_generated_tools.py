from __future__ import annotations

from pathlib import Path

import pytest

from adaptive_agent.catalog import SavedToolSpec, ToolCatalog
from adaptive_agent.generated_tools import (
    arguments_to_argv,
    execute_tool_code,
    save_if_approved,
)


def test_arguments_to_argv_prefers_argv_list():
    out = arguments_to_argv({"argv": ["a", 1, "b"]})
    assert out == ["a", "1", "b"]


def test_arguments_to_argv_flattens_values_in_order():
    out = arguments_to_argv({"input": "fixtures/a.csv", "mode": "dedupe"})
    assert out == ["fixtures/a.csv", "dedupe"]


def test_arguments_to_argv_empty():
    assert arguments_to_argv({}) == []


def test_execute_tool_code_runs_snippet(tmp_path):
    result = execute_tool_code(
        code="import sys; print(sys.argv[1])",
        arguments={"argv": ["hello"]},
        workdir=tmp_path,
    )
    assert result.ok
    assert "hello" in result.stdout


def test_execute_tool_code_captures_failure(tmp_path):
    result = execute_tool_code(
        code="raise ValueError('bad')",
        arguments={},
        workdir=tmp_path,
    )
    assert not result.ok
    assert "ValueError" in result.stderr


def test_save_if_approved_saves_only_on_yes(tmp_path):
    cat = ToolCatalog(state_dir=tmp_path)
    questions: list[str] = []
    spec = SavedToolSpec(
        name="my_tool",
        description="x",
        input_summary="",
        output_summary="",
        code="print('x')",
    )
    approved = save_if_approved(
        cat,
        spec,
        approval_fn=lambda question: questions.append(question) or True,
    )
    assert approved is True
    assert questions == ["생성한 도구 'my_tool'를 저장할까요?"]
    assert cat.lookup("my_tool").name == "my_tool"


def test_save_if_approved_skips_on_no(tmp_path):
    cat = ToolCatalog(state_dir=tmp_path)
    spec = SavedToolSpec(
        name="my_tool",
        description="x",
        input_summary="",
        output_summary="",
        code="print('x')",
    )
    approved = save_if_approved(cat, spec, approval_fn=lambda _: False)
    assert approved is False
    assert not cat.has("my_tool")
    # no file written either
    assert not (tmp_path / "tools" / "my_tool.py").exists()
