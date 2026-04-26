from __future__ import annotations

import sys
from enum import Enum
from typing import IO, Callable


class ApprovalResult(Enum):
    YES = "yes"
    NO = "no"
    UNKNOWN = "unknown"


YES_TOKENS = {"y", "yes"}
NO_TOKENS = {"n", "no"}


def approval_prompt_heading(question: str) -> str:
    if "도구" in question and "삭제" in question:
        return "도구 삭제"
    if "도구" in question and "저장" in question:
        return "도구 저장"
    return "승인 요청"


def format_approval_prompt(question: str) -> str:
    return f"\n[{approval_prompt_heading(question)}]\n{question} [y/n]: "


def interpret_response(text: str) -> ApprovalResult:
    normalized = (text or "").strip().lower()
    if normalized in YES_TOKENS:
        return ApprovalResult.YES
    if normalized in NO_TOKENS:
        return ApprovalResult.NO
    return ApprovalResult.UNKNOWN


def yes_no_loop(
    question: str,
    read_input: Callable[[str], str],
    *,
    max_attempts: int = 3,
    on_invalid: Callable[[], None] | None = None,
) -> bool:
    """Generic yes/no loop driven by a caller-supplied input function.

    ``read_input`` receives the formatted prompt and returns the user response;
    raising EOFError/KeyboardInterrupt or returning "" defaults to NO. Allows
    callers (REPL) to share the prompt formatting + retry logic while reading
    from a non-stdin channel (e.g. prompt_toolkit).
    """
    prompt = format_approval_prompt(question)
    for _ in range(max_attempts):
        try:
            line = read_input(prompt)
        except (EOFError, KeyboardInterrupt):
            return False
        if line == "":
            return False
        result = interpret_response(line)
        if result is ApprovalResult.YES:
            return True
        if result is ApprovalResult.NO:
            return False
        if on_invalid is not None:
            on_invalid()
    return False


def prompt_yes_no(
    question: str,
    *,
    input_stream: IO[str] | None = None,
    output_stream: IO[str] | None = None,
    max_attempts: int = 3,
) -> bool:
    """Ask a yes/no question and return True on yes, False on no or EOF."""
    out = output_stream if output_stream is not None else sys.stdout
    inp = input_stream if input_stream is not None else sys.stdin

    def _read(prompt: str) -> str:
        out.write(prompt)
        out.flush()
        return inp.readline()

    def _on_invalid() -> None:
        out.write("y 또는 n으로 답해주세요.\n")
        out.flush()

    return yes_no_loop(question, _read, max_attempts=max_attempts, on_invalid=_on_invalid)


# Non-interactive helpers useful for tests and one-shot runs
def auto_yes(question: str) -> bool:
    return True


def auto_no(question: str) -> bool:
    return False
