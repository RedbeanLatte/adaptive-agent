from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Any

import requests


class LLMError(RuntimeError):
    """Raised for any failure while talking to the LLM provider."""


_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
DEFAULT_MODEL = "qwen3:30b"
DEFAULT_BASE_URL = "https://ollama-wsl.tail71f338.ts.net:8443"


def _config_value(key: str, default: str) -> str:
    return os.environ.get(key, default)


def strip_think_blocks(text: str) -> str:
    return _THINK_RE.sub("", text)


@dataclass
class LLMConfig:
    model: str = DEFAULT_MODEL
    base_url: str = DEFAULT_BASE_URL
    token: str | None = None
    timeout: float = 120.0

    @classmethod
    def from_env(cls) -> "LLMConfig":
        return cls(
            model=_config_value("AGENT_MODEL", DEFAULT_MODEL),
            base_url=_config_value("OPENAI_BASE_URL", DEFAULT_BASE_URL),
            token=_config_value("OPENAI_API_KEY", "") or None,
            timeout=float(_config_value("AGENT_HTTP_TIMEOUT", "120")),
        )


class LLMClient:
    """Direct HTTP client for any OpenAI-compatible /v1/chat/completions endpoint.

    No agent framework, no SDK wrappers — a single requests.post call.
    """

    def __init__(self, config: LLMConfig | None = None) -> None:
        self.config = config or LLMConfig.from_env()

    def complete(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.2,
        extra: dict[str, Any] | None = None,
    ) -> str:
        url = self.config.base_url.rstrip("/") + "/v1/chat/completions"
        headers = {"Content-Type": "application/json"}
        if self.config.token:
            headers["Authorization"] = f"Bearer {self.config.token}"

        payload: dict[str, Any] = {
            "model": self.config.model,
            "messages": messages,
            "temperature": temperature,
            "stream": False,
        }
        if extra:
            payload.update(extra)

        try:
            resp = requests.post(
                url,
                headers=headers,
                json=payload,
                timeout=self.config.timeout,
            )
        except requests.Timeout as e:
            raise LLMError(f"LLM timeout after {self.config.timeout}s") from e
        except requests.ConnectionError as e:
            raise LLMError(f"LLM connection error: {e}") from e
        except requests.RequestException as e:
            raise LLMError(f"LLM request failed: {e}") from e

        if not getattr(resp, "ok", False):
            status = getattr(resp, "status_code", "?")
            body = (getattr(resp, "text", "") or "")[:200]
            if status == 401:
                raise LLMError(f"LLM auth failed (HTTP 401). Check provider token settings. body={body}")
            if status in (502, 503, 504):
                raise LLMError(
                    f"LLM upstream unavailable (HTTP {status}). Model may be loading. body={body}"
                )
            raise LLMError(f"LLM HTTP {status}: {body}")

        try:
            data = resp.json()
        except Exception as e:  # pragma: no cover - defensive
            raise LLMError(f"LLM returned non-JSON: {e}") from e

        try:
            content = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as e:
            raise LLMError(f"LLM response missing choices/message/content: {data!r}") from e

        if not isinstance(content, str):
            raise LLMError(f"LLM content is not a string: {content!r}")

        return strip_think_blocks(content).strip()
