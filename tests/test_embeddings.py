from __future__ import annotations

import types

import pytest
import requests

from adaptive_agent.embeddings import EmbeddingClient, EmbeddingConfig, EmbeddingError


def _mock_response(*, status=200, embedding=None):
    resp = types.SimpleNamespace()
    resp.status_code = status
    resp.ok = 200 <= status < 300
    resp.text = "mock body"

    def json_fn():
        return {"data": [{"embedding": embedding if embedding is not None else [0.1, 0.2]}]}

    resp.json = json_fn
    return resp


def test_embedding_config_uses_default_model_without_env(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("AGENT_EMBEDDING_MODEL", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("AGENT_HTTP_TIMEOUT", raising=False)

    cfg = EmbeddingConfig.from_env()

    assert cfg.model == "nomic-embed-text"
    assert cfg.enabled is True
    assert cfg.base_url == "https://ollama-wsl.tail71f338.ts.net:8443"


def test_embedding_config_ignores_env_file(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("AGENT_EMBEDDING_MODEL", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("AGENT_HTTP_TIMEOUT", raising=False)
    env_file = tmp_path / ".env.local"
    env_file.write_text(
        "\n".join(
            [
                "AGENT_EMBEDDING_MODEL=embed-local",
                "OPENAI_BASE_URL=https://local.example",
                "OPENAI_API_KEY=token",
                "AGENT_HTTP_TIMEOUT=9",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("AGENT_ENV_FILE", str(env_file))

    cfg = EmbeddingConfig.from_env()

    assert cfg.model == "nomic-embed-text"
    assert cfg.enabled is True
    assert cfg.base_url == "https://ollama-wsl.tail71f338.ts.net:8443"
    assert cfg.token is None
    assert cfg.timeout == 120.0


def test_embedding_config_model_env_overrides_default(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("AGENT_EMBEDDING_MODEL", "embed-test")
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("AGENT_HTTP_TIMEOUT", raising=False)

    cfg = EmbeddingConfig.from_env()

    assert cfg.model == "embed-test"
    assert cfg.enabled is True


def test_embed_posts_to_openai_compatible_embeddings_endpoint(monkeypatch):
    calls: list[dict] = []

    def fake_post(url, headers=None, json=None, timeout=None):
        calls.append({"url": url, "headers": headers, "json": json, "timeout": timeout})
        return _mock_response(embedding=[0.25, -0.5, 1.0])

    monkeypatch.setattr(requests, "post", fake_post)
    client = EmbeddingClient(
        EmbeddingConfig(model="embed-test", base_url="https://api.example", token="secret", timeout=12)
    )

    vector = client.embed("tool text")

    assert vector == [0.25, -0.5, 1.0]
    assert calls[0]["url"] == "https://api.example/v1/embeddings"
    assert calls[0]["headers"]["Authorization"] == "Bearer secret"
    assert calls[0]["json"] == {"model": "embed-test", "input": "tool text"}
    assert calls[0]["timeout"] == 12


def test_embed_without_token_omits_authorization(monkeypatch):
    def fake_post(url, headers=None, json=None, timeout=None):
        assert "Authorization" not in headers
        return _mock_response()

    monkeypatch.setattr(requests, "post", fake_post)
    client = EmbeddingClient(EmbeddingConfig(model="embed-test", base_url="https://api.example"))

    assert client.embed("x") == [0.1, 0.2]


def test_embed_http_failure_raises_embedding_error(monkeypatch):
    def fake_post(url, headers=None, json=None, timeout=None):
        return _mock_response(status=500)

    monkeypatch.setattr(requests, "post", fake_post)
    client = EmbeddingClient(EmbeddingConfig(model="embed-test", base_url="https://api.example"))

    with pytest.raises(EmbeddingError, match="HTTP 500"):
        client.embed("x")


def test_embed_disabled_raises_embedding_error():
    client = EmbeddingClient(EmbeddingConfig(model=""))

    with pytest.raises(EmbeddingError, match="disabled"):
        client.embed("x")
