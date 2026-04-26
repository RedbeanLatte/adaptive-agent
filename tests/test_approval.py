from __future__ import annotations

import io

import pytest

from adaptive_agent.approval import (
    ApprovalResult,
    approval_prompt_heading,
    auto_no,
    auto_yes,
    interpret_response,
    prompt_yes_no,
)


def test_interpret_y():
    r = interpret_response("y")
    assert r == ApprovalResult.YES


def test_interpret_yes_uppercase():
    assert interpret_response("YES") == ApprovalResult.YES


def test_interpret_n():
    assert interpret_response("n") == ApprovalResult.NO


def test_interpret_no_trailing_whitespace():
    assert interpret_response("  no  \n") == ApprovalResult.NO


def test_interpret_unknown():
    assert interpret_response("maybe") == ApprovalResult.UNKNOWN


def test_interpret_empty():
    assert interpret_response("") == ApprovalResult.UNKNOWN


def test_prompt_returns_yes_on_y(monkeypatch):
    inp = io.StringIO("y\n")
    out = io.StringIO()
    assert prompt_yes_no("save?", input_stream=inp, output_stream=out) is True
    assert "save?" in out.getvalue()
    assert out.getvalue() == "\n[승인 요청]\nsave? [y/n]: "


def test_save_tool_prompt_is_visually_separated_from_result_output():
    inp = io.StringIO("y\n")
    out = io.StringIO()

    assert prompt_yes_no("생성한 도구 'filter_monsters'를 저장할까요?", input_stream=inp, output_stream=out) is True

    assert out.getvalue() == "\n[도구 저장]\n생성한 도구 'filter_monsters'를 저장할까요? [y/n]: "


def test_heading_uses_delete_for_remove_prompt():
    # Regression: the substring "저장" inside "저장된 도구..." used to false-match
    # the save heading and label delete prompts as "[도구 저장]".
    assert approval_prompt_heading("저장된 도구 'foo'을 삭제할까요?") == "도구 삭제"


def test_heading_keeps_save_for_save_prompt():
    assert approval_prompt_heading("생성한 도구 'foo'를 저장할까요?") == "도구 저장"


def test_heading_falls_back_for_unrelated_question():
    assert approval_prompt_heading("계속 진행할까요?") == "승인 요청"


def test_prompt_returns_no_on_n():
    inp = io.StringIO("n\n")
    out = io.StringIO()
    assert prompt_yes_no("save?", input_stream=inp, output_stream=out) is False


def test_prompt_reprompts_on_unknown():
    inp = io.StringIO("huh\ny\n")
    out = io.StringIO()
    assert prompt_yes_no("save?", input_stream=inp, output_stream=out) is True
    assert "y 또는 n으로 답해주세요." in out.getvalue()


def test_prompt_defaults_to_no_on_eof():
    inp = io.StringIO("")  # EOF
    out = io.StringIO()
    assert prompt_yes_no("save?", input_stream=inp, output_stream=out) is False


def test_auto_yes_and_no():
    assert auto_yes("anything?") is True
    assert auto_no("anything?") is False
