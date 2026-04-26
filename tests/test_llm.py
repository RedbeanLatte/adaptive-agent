from __future__ import annotations

import types

import pytest
import requests

from adaptive_agent.llm import (
    LLMClient,
    LLMError,
    LLMConfig,
    strip_think_blocks,
)


def test_strip_think_removes_block():
    raw = "before <think>hidden\nchain</think> after"
    assert strip_think_blocks(raw) == "before  after"


def test_strip_think_handles_no_block():
    raw = "plain output"
    assert strip_think_blocks(raw) == "plain output"


def test_strip_think_multiple_blocks():
    raw = "<think>a</think>real<think>b</think>"
    assert strip_think_blocks(raw) == "real"


def _mock_response(*, status=200, content="hello"):
    resp = types.SimpleNamespace()
    resp.status_code = status
    resp.ok = 200 <= status < 300

    def json_fn():
        return {
            "choices": [
                {"message": {"role": "assistant", "content": content}}
            ]
        }

    resp.json = json_fn
    resp.text = "mock body"
    return resp


def test_complete_returns_content(monkeypatch):
    calls: list[dict] = []

    def fake_post(url, headers=None, json=None, timeout=None):
        calls.append({"url": url, "headers": headers, "json": json, "timeout": timeout})
        return _mock_response(content='{"action_type":"answer","final_answer":"hi"}')

    monkeypatch.setattr(requests, "post", fake_post)
    client = LLMClient(LLMConfig(model="m", base_url="http://x", token="t", timeout=10.0))
    out = client.complete([{"role": "user", "content": "hi"}])
    assert out == '{"action_type":"answer","final_answer":"hi"}'
    assert calls[0]["url"] == "http://x/v1/chat/completions"
    assert calls[0]["headers"]["Authorization"] == "Bearer t"
    assert calls[0]["json"]["model"] == "m"


def test_complete_without_token_omits_authorization(monkeypatch):
    def fake_post(url, headers=None, json=None, timeout=None):
        assert "Authorization" not in headers
        return _mock_response(content="ok")

    monkeypatch.setattr(requests, "post", fake_post)
    client = LLMClient(LLMConfig(model="m", base_url="http://x", token=None, timeout=5.0))
    assert client.complete([{"role": "user", "content": "hi"}]) == "ok"


def test_complete_strips_think(monkeypatch):
    def fake_post(url, headers=None, json=None, timeout=None):
        return _mock_response(content="<think>inner</think>visible")

    monkeypatch.setattr(requests, "post", fake_post)
    client = LLMClient(LLMConfig(model="m", base_url="http://x"))
    assert client.complete([{"role": "user", "content": "hi"}]) == "visible"


def test_complete_401_mapped(monkeypatch):
    def fake_post(url, headers=None, json=None, timeout=None):
        return _mock_response(status=401, content="")

    monkeypatch.setattr(requests, "post", fake_post)
    client = LLMClient(LLMConfig(model="m", base_url="http://x"))
    with pytest.raises(LLMError) as e:
        client.complete([{"role": "user", "content": "hi"}])
    assert "401" in str(e.value) or "auth" in str(e.value).lower() or "OPENAI_API_KEY" in str(e.value)


def test_complete_timeout_mapped(monkeypatch):
    def fake_post(url, headers=None, json=None, timeout=None):
        raise requests.Timeout("slow")

    monkeypatch.setattr(requests, "post", fake_post)
    client = LLMClient(LLMConfig(model="m", base_url="http://x"))
    with pytest.raises(LLMError) as e:
        client.complete([{"role": "user", "content": "hi"}])
    assert "timeout" in str(e.value).lower()


def test_complete_connection_error_mapped(monkeypatch):
    def fake_post(url, headers=None, json=None, timeout=None):
        raise requests.ConnectionError("down")

    monkeypatch.setattr(requests, "post", fake_post)
    client = LLMClient(LLMConfig(model="m", base_url="http://x"))
    with pytest.raises(LLMError) as e:
        client.complete([{"role": "user", "content": "hi"}])
    assert "connect" in str(e.value).lower()


def test_config_from_env(monkeypatch):
    monkeypatch.setenv("AGENT_MODEL", "gpt-4o-mini")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://example")
    monkeypatch.setenv("OPENAI_API_KEY", "abc")
    monkeypatch.setenv("AGENT_HTTP_TIMEOUT", "30")
    cfg = LLMConfig.from_env()
    assert cfg.model == "gpt-4o-mini"
    assert cfg.base_url == "https://example"
    assert cfg.token == "abc"
    assert cfg.timeout == 30.0


def test_config_ignores_env_local_from_current_directory(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("AGENT_MODEL", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("AGENT_HTTP_TIMEOUT", raising=False)
    (tmp_path / ".env.local").write_text(
        "\n".join(
            [
                "AGENT_MODEL=local-model",
                "OPENAI_BASE_URL=https://local-llm.example",
                "OPENAI_API_KEY=local-token",
                "AGENT_HTTP_TIMEOUT=45",
            ]
        ),
        encoding="utf-8",
    )

    cfg = LLMConfig.from_env()

    assert cfg.model == "qwen3.6:27b"
    assert cfg.base_url == "https://ollama-wsl.tail71f338.ts.net:8443"
    assert cfg.token is None
    assert cfg.timeout == 120.0


def test_config_from_environment(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("AGENT_MODEL", "env-model")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://env.example")
    monkeypatch.setenv("OPENAI_API_KEY", "env-token")
    monkeypatch.setenv("AGENT_HTTP_TIMEOUT", "12")
    (tmp_path / ".env.local").write_text(
        "\n".join(
            [
                "AGENT_MODEL=local-model",
                "OPENAI_BASE_URL=https://local-llm.example",
                "OPENAI_API_KEY=local-token",
                "AGENT_HTTP_TIMEOUT=45",
            ]
        ),
        encoding="utf-8",
    )

    cfg = LLMConfig.from_env()

    assert cfg.model == "env-model"
    assert cfg.base_url == "https://env.example"
    assert cfg.token == "env-token"
    assert cfg.timeout == 12.0


def test_config_defaults(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("AGENT_MODEL", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("AGENT_HTTP_TIMEOUT", raising=False)
    cfg = LLMConfig.from_env()
    assert cfg.model == "qwen3.6:27b"
    assert cfg.base_url == "https://ollama-wsl.tail71f338.ts.net:8443"
    assert cfg.token is None
    assert cfg.timeout == 120.0
