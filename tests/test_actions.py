from __future__ import annotations

import json

import pytest

from adaptive_agent.actions import Action, ActionParseError, parse_action


# ---------- happy paths ----------

def test_answer_action():
    raw = json.dumps(
        {"action_type": "answer", "reasoning_summary": "done", "final_answer": "hi"}
    )
    action = parse_action(raw)
    assert isinstance(action, Action)
    assert action.action_type == "answer"
    assert action.fields["final_answer"] == "hi"
    assert action.reasoning_summary == "done"


def test_ask_user_action():
    raw = json.dumps(
        {"action_type": "ask_user", "question": "what date format?"}
    )
    action = parse_action(raw)
    assert action.action_type == "ask_user"
    assert action.fields["question"] == "what date format?"


def test_run_builtin_action():
    raw = json.dumps(
        {
            "action_type": "run_builtin",
            "tool_name": "read_text_file",
            "arguments": {"path": "a.csv"},
        }
    )
    action = parse_action(raw)
    assert action.action_type == "run_builtin"
    assert action.fields["tool_name"] == "read_text_file"
    assert action.fields["arguments"] == {"path": "a.csv"}


def test_builtin_name_action_type_is_normalized_to_run_builtin():
    raw = json.dumps(
        {
            "action_type": "read_text_file",
            "arguments": {"path": "a.csv"},
        }
    )
    action = parse_action(raw)
    assert action.action_type == "run_builtin"
    assert action.fields["tool_name"] == "read_text_file"
    assert action.fields["arguments"] == {"path": "a.csv"}


def test_builtin_name_action_type_defaults_arguments():
    raw = json.dumps({"action_type": "list_files"})
    action = parse_action(raw)
    assert action.action_type == "run_builtin"
    assert action.fields["tool_name"] == "list_files"
    assert action.fields["arguments"] == {}


def test_reuse_tool_action():
    raw = json.dumps(
        {
            "action_type": "reuse_tool",
            "tool_name": "csv_dedupe_sort",
            "arguments": {"input": "a.csv"},
        }
    )
    action = parse_action(raw)
    assert action.action_type == "reuse_tool"
    assert action.fields["tool_name"] == "csv_dedupe_sort"


def test_create_tool_action():
    raw = json.dumps(
        {
            "action_type": "create_tool",
            "tool_name": "csv_cleanup",
            "code": "print('hi')",
            "arguments": {"input": "a.csv"},
            "expected_outcome": "cleaned csv",
        }
    )
    action = parse_action(raw)
    assert action.action_type == "create_tool"
    assert action.fields["code"] == "print('hi')"
    assert action.fields["tool_name"] == "csv_cleanup"


def test_request_approval_action():
    raw = json.dumps(
        {
            "action_type": "request_approval",
            "question": "save this tool?",
            "approval_kind": "save_tool",
        }
    )
    action = parse_action(raw)
    assert action.action_type == "request_approval"
    assert action.fields["approval_kind"] == "save_tool"


# ---------- fenced code blocks (LLMs often wrap JSON in ```json ... ```) ----------

def test_parser_strips_code_fences():
    raw = """```json
{"action_type": "answer", "final_answer": "ok"}
```"""
    action = parse_action(raw)
    assert action.action_type == "answer"


def test_parser_strips_bare_fences():
    raw = "```\n{\"action_type\": \"answer\", \"final_answer\": \"ok\"}\n```"
    action = parse_action(raw)
    assert action.action_type == "answer"


def test_parser_finds_first_json_object_in_mixed_text():
    raw = 'The next step is:\n{"action_type": "answer", "final_answer": "ok"}\nThanks.'
    action = parse_action(raw)
    assert action.action_type == "answer"


# ---------- defensive errors ----------

def test_empty_input_raises():
    with pytest.raises(ActionParseError):
        parse_action("")


def test_malformed_json_raises():
    with pytest.raises(ActionParseError):
        parse_action("{not json")


def test_unknown_action_type_raises():
    raw = json.dumps({"action_type": "teleport"})
    with pytest.raises(ActionParseError):
        parse_action(raw)


def test_missing_action_type_raises():
    raw = json.dumps({"final_answer": "oops"})
    with pytest.raises(ActionParseError):
        parse_action(raw)


def test_missing_required_field_raises_run_builtin():
    raw = json.dumps({"action_type": "run_builtin", "arguments": {}})
    with pytest.raises(ActionParseError):
        parse_action(raw)


def test_missing_required_field_raises_create_tool():
    raw = json.dumps({"action_type": "create_tool", "tool_name": "x"})
    # missing "code"
    with pytest.raises(ActionParseError):
        parse_action(raw)


def test_reasoning_summary_is_optional():
    raw = json.dumps({"action_type": "answer", "final_answer": "ok"})
    action = parse_action(raw)
    assert action.reasoning_summary == ""


def test_arguments_defaults_to_empty_dict():
    raw = json.dumps({"action_type": "run_builtin", "tool_name": "list_files"})
    action = parse_action(raw)
    assert action.fields["arguments"] == {}


def test_non_object_json_raises():
    raw = json.dumps(["answer"])
    with pytest.raises(ActionParseError):
        parse_action(raw)


# ---------- LLM-emitted artifacts (gpt-4o-mini occasionally appends a stray '}') ----------


def test_parser_tolerates_trailing_extra_brace():
    raw = '{"action_type":"answer","final_answer":"ok"}}'
    action = parse_action(raw)
    assert action.action_type == "answer"
    assert action.fields["final_answer"] == "ok"


def test_parser_tolerates_trailing_prose_after_object():
    raw = '{"action_type":"answer","final_answer":"ok"} -- end of message'
    action = parse_action(raw)
    assert action.action_type == "answer"


def test_parser_handles_braces_inside_string_fields():
    raw = json.dumps(
        {
            "action_type": "create_tool",
            "tool_name": "noisy",
            "code": "if x: { print('a') }",
            "arguments": {"argv": []},
        }
    )
    action = parse_action(raw)
    assert action.action_type == "create_tool"
    assert "{ print('a') }" in action.fields["code"]
