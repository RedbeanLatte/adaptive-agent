from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Literal

Role = Literal["user", "assistant", "tool"]


@dataclass
class Turn:
    role: Role
    content: str


@dataclass
class Session:
    session_id: str
    system_prompt: str = ""
    max_turns: int = 12
    turns: list[Turn] = field(default_factory=list)

    def add_user(self, content: str) -> None:
        if not content or not content.strip():
            raise ValueError("user turn content must not be empty")
        self.turns.append(Turn(role="user", content=content))

    def add_assistant(self, content: str) -> None:
        if content is None:
            raise ValueError("assistant turn content must not be None")
        self.turns.append(Turn(role="assistant", content=content))

    def add_tool_observation(self, tool_name: str, result: Any) -> None:
        payload = json.dumps({"tool": tool_name, "result": result}, ensure_ascii=False)
        self.turns.append(
            Turn(role="tool", content=f"[tool_result] {payload}")
        )

    def messages_for_llm(self) -> list[dict[str, str]]:
        """Emit OpenAI chat-completion shaped messages with a recent-turn window.

        "tool" turns are folded into the `user` role since some OpenAI-compatible
        providers (including Ollama's /v1/chat/completions) don't expose a
        first-class tool role when we're not using function-calling.
        """
        window = self._recent_pairs_window()

        msgs: list[dict[str, str]] = []
        if self.system_prompt:
            msgs.append({"role": "system", "content": self.system_prompt})
        for turn in window:
            role = "assistant" if turn.role == "assistant" else "user"
            msgs.append({"role": role, "content": turn.content})
        return msgs

    def _recent_pairs_window(self) -> list[Turn]:
        """Keep at most `max_turns` user/assistant pairs plus any trailing tool/user turns.

        Simpler: just keep the last 2*max_turns entries. Given max_turns=12 this
        keeps context bounded without over-engineering.
        """
        if self.max_turns <= 0:
            return []
        cap = self.max_turns * 2
        return list(self.turns[-cap:])
