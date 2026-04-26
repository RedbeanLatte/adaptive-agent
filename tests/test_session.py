from __future__ import annotations

import pytest

from adaptive_agent.session import Session


def test_new_session_starts_empty():
    s = Session(session_id="abc")
    assert s.session_id == "abc"
    assert s.turns == []
    assert s.messages_for_llm() == []


def test_add_user_turn():
    s = Session(session_id="s1")
    s.add_user("hello")
    assert len(s.turns) == 1
    msgs = s.messages_for_llm()
    assert msgs == [{"role": "user", "content": "hello"}]


def test_add_assistant_turn():
    s = Session(session_id="s1")
    s.add_user("hi")
    s.add_assistant('{"action_type":"answer","final_answer":"yo"}')
    msgs = s.messages_for_llm()
    assert msgs[-1] == {
        "role": "assistant",
        "content": '{"action_type":"answer","final_answer":"yo"}',
    }


def test_add_tool_observation_turn():
    s = Session(session_id="s1")
    s.add_tool_observation("read_text_file", {"ok": True, "content": "abc"})
    msgs = s.messages_for_llm()
    assert msgs[-1]["role"] == "user"
    assert "read_text_file" in msgs[-1]["content"]
    assert "abc" in msgs[-1]["content"]


def test_recent_window_limits_messages():
    s = Session(session_id="s1", max_turns=3)
    for i in range(10):
        s.add_user(f"u{i}")
        s.add_assistant(f"a{i}")
    msgs = s.messages_for_llm()
    # 3 turns => 6 messages max
    assert len(msgs) <= 6
    # most recent user message preserved
    assert msgs[-2]["content"] == "u9"
    assert msgs[-1]["content"] == "a9"


def test_system_prompt_prepended_when_supplied():
    s = Session(session_id="s1", system_prompt="you are X")
    s.add_user("hi")
    msgs = s.messages_for_llm()
    assert msgs[0] == {"role": "system", "content": "you are X"}
    assert msgs[-1] == {"role": "user", "content": "hi"}


def test_sessions_are_isolated():
    a = Session(session_id="a")
    b = Session(session_id="b")
    a.add_user("for a")
    b.add_user("for b")
    assert a.messages_for_llm()[-1]["content"] == "for a"
    assert b.messages_for_llm()[-1]["content"] == "for b"


def test_empty_message_rejected():
    s = Session(session_id="s1")
    with pytest.raises(ValueError):
        s.add_user("")

