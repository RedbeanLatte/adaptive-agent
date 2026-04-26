from __future__ import annotations

from adaptive_agent.tool_lifecycle import (
    build_save_tool_approval_question,
    detect_suspicious_output,
    summarize_output,
)
from adaptive_agent.catalog import SavedToolSpec


def test_summarize_output_keeps_head_when_truncating():
    text = "제거된 행 수: 1\n정렬된 데이터:\n" + "{'date': '2026-04-01'}\n" * 200
    out = summarize_output(text, limit=80)

    assert out.startswith("제거된 행 수: 1")
    assert out.endswith("...")
    assert len(out) <= 80


def test_summarize_output_returns_short_text_unchanged():
    assert summarize_output("hello", limit=240) == "hello"


def test_summarize_output_empty_input():
    assert summarize_output("", limit=240) == ""
    assert summarize_output(None, limit=240) == ""


# ---------- detect_suspicious_output ----------


def test_detect_suspicious_output_passes_legitimate_results():
    assert detect_suspicious_output('{"names": ["Orc"], "average_hp": 225.0}') is None
    assert detect_suspicious_output("Orc, Dragon / average 225") is None
    assert detect_suspicious_output("3개의 메시 발견") is None


def test_detect_suspicious_output_flags_empty_stdout():
    reason = detect_suspicious_output("")
    assert reason is not None and "비어" in reason


def test_detect_suspicious_output_flags_all_zero_dict():
    # The "avatar" regression: tool returned counters but key names were wrong
    # so every count came back as 0 with exit 0.
    reason = detect_suspicious_output(
        '{"mesh_count": 0, "texture_count": 0, "accessory_count": 0}'
    )
    assert reason is not None and "0" in reason


def test_detect_suspicious_output_flags_empty_list():
    reason = detect_suspicious_output("[]")
    assert reason is not None


def test_detect_suspicious_output_flags_traceback_keyword():
    reason = detect_suspicious_output("Traceback (most recent call last):\n  File ...")
    assert reason is not None and "traceback" in reason.lower()


def test_detect_suspicious_output_does_not_flag_dict_with_some_zeros():
    # Some zeros are legitimate when other values are non-zero.
    assert detect_suspicious_output('{"hits": 5, "misses": 0}') is None


# ---------- build_save_tool_approval_question with output_warning ----------


def _spec() -> SavedToolSpec:
    return SavedToolSpec(
        name="filter_monsters",
        description="filter monsters by hp",
        input_summary="(no args)",
        output_summary="ok",
        code="print('ok')",
    )


def test_save_tool_question_without_warning():
    question = build_save_tool_approval_question(_spec())
    assert "filter_monsters" in question
    assert "의심" not in question


def test_save_tool_question_includes_warning():
    question = build_save_tool_approval_question(
        _spec(), output_warning="JSON 출력의 모든 값이 0/빈 값입니다"
    )
    assert "의심" in question
    assert "0" in question
    assert "filter_monsters" in question
