from __future__ import annotations

from dataclasses import dataclass

import requests

from adaptive_agent.llm import DEFAULT_BASE_URL, _config_value


class EmbeddingError(RuntimeError):
    """Raised for embedding provider failures."""


@dataclass
class EmbeddingConfig:
    model: str = ""
    base_url: str = DEFAULT_BASE_URL
    token: str | None = None
    timeout: float = 120.0

    @property
    def enabled(self) -> bool:
        return bool(self.model.strip())

    @classmethod
    def from_env(cls) -> "EmbeddingConfig":
        return cls(
            model=_config_value("AGENT_EMBEDDING_MODEL", ""),
            base_url=_config_value("OPENAI_BASE_URL", DEFAULT_BASE_URL),
            token=_config_value("OPENAI_API_KEY", "") or None,
            timeout=float(_config_value("AGENT_HTTP_TIMEOUT", "120")),
        )


class EmbeddingClient:
    """Direct HTTP client for OpenAI-compatible /v1/embeddings endpoints."""

    def __init__(self, config: EmbeddingConfig | None = None) -> None:
        self.config = config or EmbeddingConfig.from_env()

    @property
    def model(self) -> str:
        return self.config.model

    def embed(self, text: str) -> list[float]:
        if not self.config.enabled:
            raise EmbeddingError("embedding disabled: AGENT_EMBEDDING_MODEL is not set")

        url = self.config.base_url.rstrip("/") + "/v1/embeddings"
        headers = {"Content-Type": "application/json"}
        if self.config.token:
            headers["Authorization"] = f"Bearer {self.config.token}"
        payload = {"model": self.config.model, "input": text}

        try:
            resp = requests.post(
                url,
                headers=headers,
                json=payload,
                timeout=self.config.timeout,
            )
        except requests.Timeout as e:
            raise EmbeddingError(f"embedding timeout after {self.config.timeout}s") from e
        except requests.ConnectionError as e:
            raise EmbeddingError(f"embedding connection error: {e}") from e
        except requests.RequestException as e:
            raise EmbeddingError(f"embedding request failed: {e}") from e

        if not getattr(resp, "ok", False):
            status = getattr(resp, "status_code", "?")
            body = (getattr(resp, "text", "") or "")[:200]
            raise EmbeddingError(f"embedding HTTP {status}: {body}")

        try:
            data = resp.json()
            vector = data["data"][0]["embedding"]
        except (KeyError, IndexError, TypeError) as e:
            raise EmbeddingError(f"embedding response missing data[0].embedding: {data!r}") from e
        except Exception as e:  # pragma: no cover - defensive JSON parser guard
            raise EmbeddingError(f"embedding returned non-JSON: {e}") from e

        if not isinstance(vector, list) or not vector:
            raise EmbeddingError(f"embedding vector is empty or invalid: {vector!r}")
        try:
            return [float(item) for item in vector]
        except (TypeError, ValueError) as e:
            raise EmbeddingError(f"embedding vector contains non-numeric values: {vector!r}") from e
