from __future__ import annotations

import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, IO, Protocol

from adaptive_agent.actions import Action, ActionParseError, parse_action
from adaptive_agent.approval import prompt_yes_no, yes_no_loop
from adaptive_agent.builtins import BuiltinError, dispatch_builtin, register_builtins
from adaptive_agent.catalog import SavedToolSpec, ToolCatalog, ToolNotFoundError
from adaptive_agent.embeddings import EmbeddingClient, EmbeddingConfig
from adaptive_agent.generated_tools import execute_tool_code
from adaptive_agent.llm import LLMClient, LLMConfig, LLMError
from adaptive_agent.progress import (
    ConsoleProgressReporter,
    NullProgressReporter,
    ProgressReporter,
    progress_event,
    summarize_arguments,
)
from adaptive_agent.runtime_guard import RuntimeGuardResult
from adaptive_agent.session import Session
from adaptive_agent.slash_commands import dispatch_slash_command
from adaptive_agent.tool_lifecycle import (
    build_pending_save_spec,
    build_save_tool_approval_question,
    detect_suspicious_output,
    execute_tool_replay_in_isolated_workspace,
    model_info,
    verification_run_payload,
)
from adaptive_agent.policy import PolicyError, validate_tool_name
from adaptive_agent.trace import TraceWriter


class LLM(Protocol):
    def complete(self, messages: list[dict[str, str]]) -> str: ...


class InteractiveInputUnavailable(Exception):
    """Raised by user_input_fn when no human is available to answer ask_user.

    Triggers a fail-fast in the runtime instead of looping on EOFError.
    """


@dataclass
class RuntimeConfig:
    workroot: Path
    state_dir: Path
    session_id: str = "default"
    max_iterations: int = 12
    max_repair_per_tool: int = 1  # Phase 9 repair policy
    show_progress: bool = True
    interactive: bool = True  # False for one-shot mode without an attached human


SYSTEM_PROMPT_HEAD = """\
You are Adaptive Agent, a coding assistant that plans and executes Python tools to solve user tasks.

Respond with ONE JSON object per turn, matching one of these action shapes (no markdown fences, no prose outside JSON):

- answer: {"action_type":"answer","reasoning_summary":"short","final_answer":"natural-language answer"}
- ask_user: {"action_type":"ask_user","reasoning_summary":"short","question":"one clarifying question"}
- run_builtin: {"action_type":"run_builtin","reasoning_summary":"short","tool_name":"<name>","arguments":{...}}
- reuse_tool: {"action_type":"reuse_tool","reasoning_summary":"short","tool_name":"<saved name>","arguments":{"argv":["..."]}}
- create_tool: {"action_type":"create_tool","reasoning_summary":"short","tool_name":"snake_case_name","code":"<python source>","arguments":{"argv":["..."]},"expected_outcome":"short"}
- request_approval: {"action_type":"request_approval","reasoning_summary":"short","question":"...","approval_kind":"save_tool|other"}

Rules:
- Emit exactly one JSON object per turn, no surrounding text.
- Routing policy: the LLM chooses the action; the runtime validates and executes it.
- Tool-first policy: prefer solving executable tasks by running a saved or generated tool, then use `answer` only to summarize the observed result.
- Direct answer is only for pure explanation, conversation, missing-capability limitations, or the final summary after tool execution.
- Use reuse_tool or create_tool for executable work that can actually run in the current Python/workspace environment.
- Prefer reuse_tool over create_tool when a saved tool clearly matches the request.
- Prefer create_tool for file-backed data analysis, calculations, repeatable data processing, file transformation, code audits, scaffold generation, prompt packs, or checklists where executable verification is useful.
- Builtin tools are helpers for inspecting, reading, listing, or writing workspace files; do not use builtin file reads as a substitute for create_tool when the user asks for analysis, calculation, transformation, cleanup, or extraction.
- Code audits, checklists, scaffolds, and prompt packs are NOT exceptions to the tool-first policy. After a builtin read, the next action for these tasks MUST be `create_tool` (or `reuse_tool`) that emits the result. It is FORBIDDEN to follow `run_builtin` immediately with `answer` for audit/checklist/scaffold/prompt-pack output — the file content alone is not the deliverable, the generated tool's stdout is.
  - BAD: run_builtin(read_text_file) → answer(markdown checklist composed by you).
  - GOOD: run_builtin(read_text_file) → create_tool(reads the file via argv, prints the checklist/audit/scaffold to stdout) → answer(short summary referring to the tool output).
  - Even when the user wants markdown output, you must emit it from a generated tool, not from the `answer` body.
- For unsupported realtime external data requests (weather today, current exchange rates, latest news), do not invent facts or blindly create a tool. If no explicit location/API/network capability is available, use `answer` to explain the limitation.
- For dynamic computations involving the current date, time, or locale (e.g., "X days from today"), generate a tool that uses the Python standard library (`datetime`, `time`). Do not refuse based on lack of inherent knowledge.
- Use ask_user when required information is missing and the user can reasonably provide it.
- Never embed questions for the user inside an `answer` action. If you need input, emit `ask_user`; otherwise commit to a course of action.
- If the user references a file path that may not exist, attempt read_text_file or list_files (always passing the path argument) before deciding the file is missing. Do not refuse based on assumption alone.
- If a builtin returns `{"ok": false, "error": "file not found: ..."}` for a path the user explicitly named, do not silently substitute another file. Emit `ask_user` to confirm the intended path, or `answer` stating the file does not exist. Do not run `create_tool`/`reuse_tool` against a different file than the user named.
- Every run_builtin action must include all required arguments under `arguments` (e.g., `{"action_type":"run_builtin","tool_name":"read_text_file","arguments":{"path":"fixtures/x.json"}}`). Never emit a builtin call with empty arguments.
- Only emit reuse_tool for tools listed under "Saved (reusable) tools". If reuse_tool reports "tool not found", create a new tool unless the task is not executable.
- create_tool code must be self-contained and may read positional args from sys.argv; print the final answer or a JSON payload to stdout.
- create_tool code must use the Python standard library only unless the user explicitly asks for a third-party package. If a generated tool fails because of a missing third-party import, repair it with the standard library.
- create_tool stdout is shown to the user verbatim. Print human-readable text (markdown table, plain text, or JSON), not Python `repr` of dict/tuple/list.
- When a tool needs file content, pass the file path as argv and let the tool read it. Do not embed entire file contents in argv.
- Ask clarifying questions (ask_user) when the request is ambiguous.
- The runtime asks the user whether to save successful generated tools; do not emit request_approval just to save a tool.
- After a successful tool run, sanity-check that stdout matches your `expected_outcome` before emitting `answer`. If the result looks unexpectedly empty, all-zero, or mismatched, prefer to retry with a corrected create_tool rather than reporting it as the final answer.
- Match the language of the user's most recent task message in the `final_answer` field. If they wrote in Korean, answer in Korean.
- End every task with an `answer` action.

Available tools:
"""


SYSTEM_PROMPT_SINGLE_RUN_NOTE = (
    "Single-run mode: there is no human available to answer ask_user. Prefer to "
    "make a reasonable assumption and proceed; if input is truly required, end "
    "with `answer` explaining what is missing instead of emitting ask_user."
)


@dataclass
class LoopResult:
    final_answer: str
    iterations_used: int
    ok: bool = True


class Runtime:
    def __init__(
        self,
        *,
        llm: LLM,
        catalog: ToolCatalog,
        session: Session,
        config: RuntimeConfig,
        approval_fn: Callable[[str], bool],
        user_input_fn: Callable[[str], str],
        output_stream: IO[str] | None = None,
        progress_reporter: ProgressReporter | None = None,
    ) -> None:
        self.llm = llm
        self.catalog = catalog
        self.session = session
        self.config = config
        self.approval_fn = approval_fn
        self.user_input_fn = user_input_fn
        self.output = output_stream if output_stream is not None else sys.stdout
        self.progress = self._build_progress_reporter(progress_reporter)
        self.trace = TraceWriter(config.state_dir, config.session_id)
        self._attempts_per_tool: dict[str, int] = {}
        self._consecutive_create_failures = 0
        self._pending_save: SavedToolSpec | None = None
        self._pending_save_warning: str | None = None
        self._save_approval_handled = False
        self._last_artifact_preview: dict[str, Any] | None = None
        self._task_failed = False
        self._current_task = ""
        self._analysis_summary_emitted = False

    def run_task(self, task: str) -> LoopResult:
        self._reset_task_state()
        self._current_task = task
        self.session.add_user(task)
        self.trace.event("user_task", content=task)
        self._progress("task_started", "작업 시작...", task_summary=_tail(task, 120))
        return self._agent_loop()

    def _reset_task_state(self) -> None:
        self._attempts_per_tool = {}
        self._consecutive_create_failures = 0
        self._pending_save = None
        self._pending_save_warning = None
        self._save_approval_handled = False
        self._last_artifact_preview = None
        self._task_failed = False
        self._current_task = ""
        self._analysis_summary_emitted = False

    def _agent_loop(self) -> LoopResult:
        self.session.system_prompt = self._system_prompt()

        for i in range(1, self.config.max_iterations + 1):
            message = "요청 분석 중..." if i == 1 else "다음 단계 판단 중..."
            self._progress("llm_start", message, iteration=i)
            try:
                raw = self.llm.complete(self.session.messages_for_llm())
            except LLMError as e:
                msg = f"[LLM error] {e}"
                self._task_failed = True
                self.trace.event("llm_error", error=str(e))
                self._write_line(msg)
                return LoopResult(final_answer=msg, iterations_used=i, ok=False)

            self.session.add_assistant(raw)
            try:
                action = parse_action(raw)
            except ActionParseError as e:
                self.trace.event(
                    "action_parse_error",
                    error=str(e),
                    raw_head=_head(raw, 500),
                    iteration=i,
                )
                self.session.add_tool_observation(
                    "schema_error",
                    {
                        "error": str(e),
                        "hint": "respond with exactly one valid JSON action object",
                    },
                )
                self._progress(
                    "action_parse_error",
                    f"응답 형식 오류 — 재시도 ({e})",
                    error=str(e),
                    iteration=i,
                )
                continue

            self.trace.event(
                "action",
                action_type=action.action_type,
                reasoning=action.reasoning_summary,
                iteration=i,
            )
            self._emit_analysis_summary_once(action)
            self._progress(
                "action_received",
                f"다음 작업 선택: {action.action_type}",
                action_type=action.action_type,
                reasoning_summary=action.reasoning_summary,
                iteration=i,
            )

            final = self._dispatch(action)
            if final is not None:
                return LoopResult(
                    final_answer=final,
                    iterations_used=i,
                    ok=not self._task_failed,
                )

        msg = "[runtime] 최대 반복 횟수에 도달했지만 최종 답변을 만들지 못했습니다."
        self._task_failed = True
        self.trace.event("max_iterations", iterations=self.config.max_iterations)
        self._write_line(msg)
        return LoopResult(
            final_answer=msg,
            iterations_used=self.config.max_iterations,
            ok=False,
        )

    def _dispatch(self, action: Action) -> str | None:
        at = action.action_type
        if at == "answer":
            text = str(action.fields.get("final_answer", ""))
            text = self._augment_final_answer_with_artifact_preview(text)
            self._progress("answer_start", "최종 답변 생성 중...")
            self._write_line(text)
            self.trace.event("answer", content=text)
            self._handle_deferred_save_tool_approval()
            return text

        if at == "ask_user":
            question = str(action.fields.get("question", ""))
            self._progress("ask_user", _tail(question, 120), question=_tail(question, 200))
            try:
                reply = self.user_input_fn(f"[you] ")
            except InteractiveInputUnavailable:
                short_question = _tail(question, 200)
                self._write_line(f"❓ Agent 질문: {short_question}")
                msg = (
                    "[runtime] 단일 실행 모드에서는 추가 질문에 답할 수 없습니다. "
                    "./run.sh 로 REPL을 시작한 뒤 같은 작업을 다시 입력하세요."
                )
                self._write_line(msg)
                self.trace.event("ask_user_unavailable", question=question)
                self.trace.event("answer", content=msg)
                self._task_failed = True
                return msg
            except EOFError:
                reply = ""
            self.trace.event("ask_user", question=question, reply_len=len(reply))
            if reply:
                self.session.add_user(reply)
            return None

        if at == "run_builtin":
            name = str(action.fields.get("tool_name", ""))
            args = action.fields.get("arguments") or {}
            args_summary = summarize_arguments(args)
            message = f"내장 도구 실행: {name}"
            if args_summary:
                message += f"({args_summary})"
            self._progress(
                "builtin_start",
                message,
                tool_name=name,
                arguments=args,
                arguments_summary=args_summary,
                reasoning_summary=action.reasoning_summary,
            )
            try:
                result = dispatch_builtin(name, args, workroot=self.config.workroot)
                self.session.add_tool_observation(name, {"ok": True, "result": result})
                self.trace.event("builtin_ok", tool=name)
                self._progress("builtin_complete", f"내장 도구 완료: {name}", tool_name=name, ok=True)
            except BuiltinError as e:
                err = str(e)
                observation: dict[str, Any] = {"ok": False, "error": err}
                # Surface a concrete recovery hint for the most common LLM
                # mistake (calling a builtin without its required argument). The
                # error already names the missing argument; without this hint
                # gpt-4o-mini tends to bail to ask_user instead of retrying.
                if "missing required argument" in err:
                    observation["hint"] = (
                        "Retry the same builtin with the missing argument provided. "
                        "Do not switch to ask_user just because the previous call omitted it."
                    )
                self.session.add_tool_observation(name, observation)
                # The error message names the missing argument; logging the full
                # `args` dict could leak unbounded LLM-supplied content (paths,
                # snippets) into the trace — keep the trace metadata only.
                self.trace.event(
                    "builtin_error", tool=name, error=err, arg_keys=sorted(args.keys()) if isinstance(args, dict) else []
                )
                self._progress("builtin_complete", f"내장 도구 실패: {name}", tool_name=name, ok=False, error=err)
            return None

        if at == "reuse_tool":
            name = str(action.fields.get("tool_name", ""))
            # Catalog renders saved tools as "name vN — desc" in the prompt; the
            # model occasionally copies the v-suffix into tool_name. Strip it
            # so lookup matches the catalog key.
            name = re.sub(r"\s+v\d+$", "", name).strip()
            args = action.fields.get("arguments") or {}
            try:
                spec = self.catalog.lookup(name)
            except ToolNotFoundError:
                self.session.add_tool_observation(
                    name,
                    {
                        "ok": False,
                        "error": f"catalog에서 저장된 도구를 찾을 수 없습니다: {name}",
                        "hint": "실행 가능한 작업이라면 바로 답변하지 말고 create_tool로 계속 진행하세요.",
                    },
                )
                self.trace.event("reuse_missing", tool=name)
                return None
            if not isinstance(spec, SavedToolSpec):
                self.session.add_tool_observation(
                    name,
                    {
                        "ok": False,
                        "error": f"'{name}'는 내장 도구입니다. run_builtin을 사용하세요.",
                    },
                )
                return None
            self._progress(
                "reuse_tool_start",
                f"저장 도구 실행: {name}",
                tool_name=name,
                arguments=args,
                arguments_summary=summarize_arguments(args),
                reasoning_summary=action.reasoning_summary,
            )
            result = execute_tool_code(
                code=spec.code, arguments=args, workdir=self.config.workroot
            )
            self._record_exec(tool=name, kind="reuse", result=result)
            self._progress(
                "reuse_tool_complete",
                f"저장 도구 {'완료' if result.ok else '실패'}: {name}",
                tool_name=name,
                ok=result.ok,
                exit_status=result.exit_status,
            )
            return None

        if at == "create_tool":
            return self._handle_create_tool(action)

        if at == "request_approval":
            model_question = str(action.fields.get("question", ""))
            approval_kind = str(action.fields.get("approval_kind", "other"))
            if approval_kind == "save_tool":
                if self._pending_save is not None and not self._save_approval_handled:
                    self.session.add_tool_observation(
                        "approval",
                        {
                            "kind": "save_tool",
                            "approved": False,
                            "deferred_until_answer": True,
                            "model_question": model_question,
                        },
                    )
                    self.trace.event(
                        "approval_deferred",
                        approval_kind="save_tool",
                        model_question=model_question,
                        reason="answer_first_save_prompt",
                    )
                    self._progress(
                        "approval_deferred",
                        "저장 승인은 최종 답변 이후로 미룹니다",
                        approval_kind="save_tool",
                        model_question=model_question,
                        tool_name=self._pending_save.name,
                    )
                else:
                    self._handle_save_tool_approval(
                        model_question=model_question,
                        output_warning=self._pending_save_warning,
                    )
                return None

            question = model_question
            approval_fields: dict[str, Any] = {
                "approval_kind": approval_kind,
                "model_question": model_question,
            }
            self._progress("approval_wait", f"승인 대기: {approval_kind}", **approval_fields)
            ok = self.approval_fn(question)
            self.trace.event(
                "approval",
                approval_kind=approval_kind,
                question=question,
                model_question=model_question,
                approved=ok,
            )
            self.session.add_tool_observation(
                "approval",
                {
                    "kind": approval_kind,
                    "approved": ok,
                    "question": question,
                    "model_question": model_question,
                },
            )
            self._progress("approval_result", f"승인 결과: {'승인' if ok else '거절'}", approved=ok, **approval_fields)
            return None

        # unknown action_type can't happen - parse_action rejects it
        return None

    def _handle_create_tool(self, action: Action) -> str | None:
        name = str(action.fields.get("tool_name", "generated"))
        code = str(action.fields.get("code", ""))
        args = action.fields.get("arguments") or {}
        expected = str(action.fields.get("expected_outcome", ""))

        try:
            validate_tool_name(name)
        except PolicyError as e:
            self.session.add_tool_observation(name, {"ok": False, "error": str(e)})
            self.trace.event("create_tool_rejected", tool=name, error=str(e))
            self._progress("tool_create_rejected", f"생성 도구 거부: {name}", tool_name=name, error=str(e))
            return None

        attempts = self._attempts_per_tool.get(name, 0) + 1
        self._attempts_per_tool[name] = attempts

        # Keep any prior successful pending save while this attempt runs. A
        # later successful create_tool will replace it below; a failed attempt
        # must not discard the last reusable candidate.
        code, code_normalized = _normalize_generated_code(code)

        self._progress(
            "tool_create_start",
            f"생성 도구 실행: {name}",
            tool_name=name,
            arguments=args,
            arguments_summary=summarize_arguments(args),
            reasoning_summary=action.reasoning_summary,
            attempt=attempts,
        )
        result = execute_tool_code(
            code=code, arguments=args, workdir=self.config.workroot
        )
        self._record_exec(
            tool=name,
            kind="create",
            result=result,
            extra_observation={"code_normalized": True} if code_normalized else None,
        )
        self._progress(
            "tool_create_complete",
            f"생성 도구 {'완료' if result.ok else '실패'}: {name}",
            tool_name=name,
            ok=result.ok,
            exit_status=result.exit_status,
            attempt=attempts,
        )

        if result.ok:
            self._consecutive_create_failures = 0
            self._progress("tool_replay_start", f"replay 검증 중: {name}", tool_name=name)
            replay_result = execute_tool_replay_in_isolated_workspace(
                code=code, arguments=args, workdir=self.config.workroot
            )
            self._record_replay(name, replay_result)
            self._progress(
                "tool_replay_complete",
                f"replay 검증 {'완료' if replay_result.ok else '실패'}: {name}",
                tool_name=name,
                ok=replay_result.ok,
                exit_status=replay_result.exit_status,
            )
            output_warning = detect_suspicious_output(result.stdout)
            if output_warning:
                self.session.add_tool_observation(
                    name,
                    {
                        "ok": True,
                        "output_warning": output_warning,
                        "hint": (
                            "도구는 exit 0으로 끝났지만 출력이 의심스럽습니다. "
                            "expected_outcome과 비교해 다른 도구를 만들 수 있다면 "
                            "create_tool로 다시 시도하세요. 그대로 진행하려면 answer로 결과와 함께 한계를 설명하세요."
                        ),
                    },
                )
                self.trace.event("output_suspicious", tool=name, reason=output_warning)
                self._progress(
                    "output_suspicious",
                    f"⚠️ 출력 의심: {output_warning}",
                    tool_name=name,
                    reason=output_warning,
                )
            self._pending_save = build_pending_save_spec(
                catalog=self.catalog,
                name=name,
                code=code,
                args=args,
                expected=expected,
                initial=result,
                replay=replay_result,
                current_task=self._current_task,
                model_info=model_info(self.llm),
            )
            self._pending_save_warning = output_warning
            return None

        self._consecutive_create_failures += 1
        # `max_repair_per_tool` counts retries *after* the initial attempt,
        # so total allowed executions = 1 + max_repair_per_tool. Give up on
        # the attempt that exhausts the budget to avoid a forbidden retry.
        if self._consecutive_create_failures > self.config.max_repair_per_tool:
            err = (
                f"[runtime] 도구 '{name}'가 repair 재시도 후에도 실패해 중단합니다. "
                f"요약: {result.compact_summary()}"
            )
            self._task_failed = True
            self._write_line(err)
            self.trace.event(
                "repair_exhausted",
                tool=name,
                attempts=attempts,
                consecutive_failures=self._consecutive_create_failures,
            )
            self._progress(
                "repair_exhausted",
                f"repair 소진: {name}",
                tool_name=name,
                attempts=attempts,
                consecutive_failures=self._consecutive_create_failures,
            )
            self._handle_deferred_save_tool_approval()
            return err
        self.trace.event(
            "repair_requested",
            tool=name,
            attempts=attempts,
            consecutive_failures=self._consecutive_create_failures,
            summary=result.compact_summary(),
        )
        self._progress(
            "repair_requested",
            f"repair 요청: {name}",
            tool_name=name,
            attempts=attempts,
            consecutive_failures=self._consecutive_create_failures,
        )
        return None

    def _handle_deferred_save_tool_approval(self) -> None:
        if self._pending_save is None or self._save_approval_handled:
            return
        self._handle_save_tool_approval(
            model_question="runtime deferred save approval",
            output_warning=self._pending_save_warning,
        )

    def _handle_save_tool_approval(
        self, *, model_question: str, output_warning: str | None = None
    ) -> None:
        pending = self._pending_save
        if pending is None:
            if self._save_approval_handled:
                self.session.add_tool_observation(
                    "approval",
                    {
                        "kind": "save_tool",
                        "approved": False,
                        "already_handled": True,
                        "model_question": model_question,
                    },
                )
                self.trace.event(
                    "approval_skipped",
                    approval_kind="save_tool",
                    model_question=model_question,
                    reason="already_handled",
                )
                self._progress(
                    "approval_skipped",
                    "저장 승인 생략: 이미 처리됨",
                    approval_kind="save_tool",
                    model_question=model_question,
                )
                return

            self.session.add_tool_observation(
                "approval",
                {
                    "kind": "save_tool",
                    "approved": False,
                    "error": "저장할 pending 생성 도구가 없습니다",
                    "model_question": model_question,
                },
            )
            self.trace.event(
                "approval_skipped",
                approval_kind="save_tool",
                model_question=model_question,
                reason="no_pending_save",
            )
            self._progress(
                "approval_skipped",
                "저장 승인 생략: pending 생성 도구 없음",
                approval_kind="save_tool",
                model_question=model_question,
            )
            return

        import hashlib

        question = build_save_tool_approval_question(
            pending, output_warning=output_warning
        )
        approval_fields: dict[str, Any] = {
            "approval_kind": "save_tool",
            "model_question": model_question,
            "tool_name": pending.name,
            "verification_status": pending.verification_status,
            "code_sha256": hashlib.sha256(pending.code.encode("utf-8")).hexdigest()[:16],
        }
        if output_warning:
            approval_fields["output_warning"] = output_warning
        self._progress("approval_wait", f"저장 승인 대기: {pending.name}", **approval_fields)
        if not self.config.interactive:
            # Surface the auto-deny path so the user knows saves require REPL or
            # an explicit y on stdin. Without this hint the [y/n]: prompt scrolls
            # past in non-interactive runs and the user thinks save is unsupported.
            self._write_line(
                "💡 단일 실행 모드 — 도구는 자동 저장되지 않을 수 있습니다. "
                "./run.sh REPL에서 같은 작업을 다시 실행하면 저장 여부를 물어봅니다."
            )
        ok = self.approval_fn(question)
        self.trace.event(
            "approval",
            approval_kind="save_tool",
            question=question,
            model_question=model_question,
            approved=ok,
        )
        self.session.add_tool_observation(
            "approval",
            {
                "kind": "save_tool",
                "approved": ok,
                "question": question,
                "model_question": model_question,
            },
        )
        self._progress("approval_result", f"승인 결과: {'승인' if ok else '거절'}", approved=ok, **approval_fields)
        if ok:
            try:
                self.catalog.save_generated(pending)
                self.trace.event("saved_tool", name=pending.name)
            except (ValueError, RuntimeError) as e:
                self.session.add_tool_observation(
                    "save_tool",
                    {"ok": False, "error": str(e), "tool_name": pending.name},
                )
                self.trace.event("save_tool_error", name=pending.name, error=str(e))
        self._pending_save = None
        self._save_approval_handled = True

    def _record_replay(self, name: str, replay: RuntimeGuardResult) -> None:
        payload = verification_run_payload(replay)
        self.session.add_tool_observation(
            name,
            {"verification": {"kind": "fixture_replay", **payload}},
        )
        self.trace.event(
            "create_replay",
            tool=name,
            ok=replay.ok,
            exit_status=replay.exit_status,
            duration_sec=round(replay.duration_sec, 3),
        )

    def _record_exec(
        self,
        *,
        tool: str,
        kind: str,
        result: RuntimeGuardResult,
        extra_observation: dict[str, Any] | None = None,
    ) -> None:
        obs = {
            "ok": result.ok,
            "exit_status": result.exit_status,
            "timed_out": result.timed_out,
            "stdout_tail": _tail(result.stdout, 2000),
            "stderr_tail": _tail(result.stderr, 1000),
        }
        if extra_observation:
            obs.update(extra_observation)
        artifact_preview = _artifact_preview_from_stdout(result.stdout)
        if artifact_preview is not None:
            obs["artifact_preview"] = artifact_preview
            if result.ok:
                self._last_artifact_preview = artifact_preview
        self.session.add_tool_observation(tool, obs)
        if kind == "reuse" and self.catalog.has(tool):
            try:
                self.catalog.record_execution(
                    tool,
                    ok=result.ok,
                    exit_status=result.exit_status,
                    timed_out=result.timed_out,
                    duration_sec=round(result.duration_sec, 3),
                    stdout_tail=_tail(result.stdout, 500),
                    stderr_tail=_tail(result.stderr, 300),
                    task_summary=_tail(self._current_task, 200),
                )
            except ToolNotFoundError:
                pass
        self.trace.event(
            f"{kind}_exec",
            tool=tool,
            ok=result.ok,
            exit_status=result.exit_status,
            duration_sec=round(result.duration_sec, 3),
            **(extra_observation or {}),
        )

    def _augment_final_answer_with_artifact_preview(self, text: str) -> str:
        preview = self._last_artifact_preview
        if not preview or "생성 artifact:" in text:
            return text
        bullets = _format_artifact_preview_bullets(preview)
        if not bullets:
            return text
        return f"{text.rstrip()}\n\n생성 artifact:\n{bullets}"

    def _system_prompt(self) -> str:
        summary = self.catalog.summary_for_prompt(current_task=self._current_task) or "(no tools yet)"
        prompt = SYSTEM_PROMPT_HEAD + summary
        if not self.config.interactive:
            prompt = f"{prompt}\n\n{SYSTEM_PROMPT_SINGLE_RUN_NOTE}"
        return prompt

    def _build_progress_reporter(
        self, progress_reporter: ProgressReporter | None
    ) -> ProgressReporter:
        if not self.config.show_progress:
            return NullProgressReporter()
        return progress_reporter or ConsoleProgressReporter(self.output)

    def _progress(self, kind: str, message: str, **fields: Any) -> None:
        self.progress.emit(progress_event(kind, message, **fields))

    def _emit_analysis_summary_once(self, action: Action) -> None:
        # Mark the opportunity used on the first action even when its summary is
        # empty — the test contract is "no fall-back to a later action's
        # summary". Suppressing the flag bump would leak later reasoning into
        # the analysis line.
        if self._analysis_summary_emitted:
            return
        self._analysis_summary_emitted = True
        summary = action.reasoning_summary.strip()
        if not summary:
            return
        self._progress(
            "analysis_summary",
            f"문제 분석: {_tail(summary, 160)}",
            action_type=action.action_type,
            reasoning_summary=summary,
        )

    def _write_line(self, text: str) -> None:
        self.output.write(text + "\n")
        self.output.flush()


def _tail(text: str, limit: int) -> str:
    if text is None:
        return ""
    if len(text) <= limit:
        return text
    return text[-limit:]


def _head(text: str, limit: int) -> str:
    if text is None:
        return ""
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def _normalize_generated_code(code: str) -> tuple[str, bool]:
    """Repair double-escaped source only when the original cannot compile.

    Some OpenAI-compatible models return Python source as one physical line
    containing literal ``\n`` separators. We only unescape those separators when
    the original source is invalid and the normalized source compiles, so valid
    Python strings such as ``"a\\nb"`` are left untouched.
    """
    if _python_source_compiles(code):
        return code, False
    if "\\n" not in code and "\\t" not in code:
        return code, False

    candidate = code.replace("\\r\\n", "\n").replace("\\n", "\n").replace("\\t", "\t")
    if candidate != code and _python_source_compiles(candidate):
        return candidate, True
    return code, False


def _python_source_compiles(source: str) -> bool:
    try:
        compile(source, "<generated_tool>", "exec")
    except SyntaxError:
        return False
    return True


def _format_artifact_preview_bullets(preview: dict[str, Any]) -> str:
    lines: list[str] = []
    artifact_type = preview.get("artifact_type")
    if artifact_type:
        lines.append(f"- artifact_type: {artifact_type}")
    title = preview.get("title")
    if title:
        lines.append(f"- title: {title}")
    item_count = preview.get("item_count")
    items = preview.get("items") if isinstance(preview.get("items"), list) else []
    if item_count:
        lines.append(f"- items ({item_count}):")
    for item in items[:5]:
        item_text = _head(str(item).strip(), 140)
        if item_text:
            lines.append(f"  - {item_text}")
    if not items and preview.get("first_item"):
        lines.append(f"  - {_head(str(preview['first_item']).strip(), 140)}")
    return "\n".join(lines)


def _artifact_preview_from_stdout(stdout: str) -> dict[str, Any] | None:
    text = (stdout or "").strip()
    if not text:
        return None
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    artifact_type = payload.get("artifact_type")
    if not isinstance(artifact_type, str) or not artifact_type.strip():
        return None
    title = payload.get("title") if isinstance(payload.get("title"), str) else ""
    raw_items = payload.get("items") if isinstance(payload.get("items"), list) else []
    items = [str(item) for item in raw_items[:5]]
    first_item = items[0] if items else ""
    return {
        "artifact_type": artifact_type,
        "title": title,
        "item_count": len(raw_items),
        "first_item": first_item,
        "items": items,
    }


# ---------- CLI entrypoints ----------

def _default_state_dir() -> Path:
    return Path(os.environ.get("AGENT_STATE_DIR", ".agent_state")).resolve()


def _default_workroot() -> Path:
    return Path.cwd()


def _build_runtime(
    *,
    session_id: str,
    llm: LLM | None = None,
    approval_fn: Callable[[str], bool] | None = None,
    user_input_fn: Callable[[str], str] | None = None,
    output_stream: IO[str] | None = None,
    interactive: bool = True,
) -> Runtime:
    state_dir = _default_state_dir()
    state_dir.mkdir(parents=True, exist_ok=True)

    embedding_config = EmbeddingConfig.from_env()
    embedding_client = EmbeddingClient(embedding_config) if embedding_config.enabled else None
    catalog = ToolCatalog(state_dir=state_dir, embedding_client=embedding_client)
    register_builtins(catalog)

    session = Session(session_id=session_id)
    config = RuntimeConfig(
        workroot=_default_workroot(),
        state_dir=state_dir,
        session_id=session_id,
        interactive=interactive,
    )

    def _read_user_input(prompt: str) -> str:
        try:
            return input(prompt)
        except EOFError:
            return ""

    return Runtime(
        llm=llm or LLMClient(LLMConfig.from_env()),
        catalog=catalog,
        session=session,
        config=config,
        approval_fn=approval_fn or (lambda q: prompt_yes_no(q)),
        user_input_fn=user_input_fn or _read_user_input,
        output_stream=output_stream,
    )


def _raise_interactive_unavailable(prompt: str) -> str:
    raise InteractiveInputUnavailable()


def run_one_shot(task: str, session_id: str = "default") -> int:
    rt = _build_runtime(
        session_id=session_id,
        user_input_fn=_raise_interactive_unavailable,
        interactive=False,
    )
    result = rt.run_task(task)
    return 0 if getattr(result, "ok", True) else 1


def run_repl(session_id: str = "default") -> int:
    rt = _build_runtime(session_id=session_id)
    # REPL approval reads through the same channel as REPL commands so a
    # follow-up command typed during a [y/n] prompt is not silently consumed.
    rt.approval_fn = _make_repl_approval_fn()
    output = rt.output
    output.write(
        "Adaptive Agent REPL입니다. 작업을 입력하고 Enter를 누르세요. "
        "종료하려면 Ctrl-D 또는 /exit를 입력하세요.\n"
    )
    output.flush()
    while True:
        try:
            line = _read_repl_input("> ").strip()
        except EOFError:
            output.write("\n")
            output.flush()
            return 0
        if not line:
            continue
        if _is_exit_command(line):
            output.write("종료합니다.\n")
            output.flush()
            return 0
        if line.startswith("/"):
            if dispatch_slash_command(rt, line):
                continue
        rt.run_task(line)


def _read_repl_input(prompt: str) -> str:
    if not sys.stdin.isatty():
        return input(prompt)
    from prompt_toolkit import prompt as prompt_toolkit_prompt

    return prompt_toolkit_prompt(prompt)


def _make_repl_approval_fn() -> Callable[[str], bool]:
    """REPL approval reads from the same channel as REPL commands.

    In TTY mode prompt_toolkit owns input so the user cannot accidentally type
    a follow-up command while a [y/n] prompt is active. In non-TTY (piped
    stdin) we accept a single response so a follow-up REPL command is not
    silently consumed by the retry loop.
    """

    def _on_invalid() -> None:
        sys.stdout.write("y 또는 n으로 답해주세요.\n")
        sys.stdout.flush()

    def _ask(question: str) -> bool:
        max_attempts = 1 if not sys.stdin.isatty() else 3
        return yes_no_loop(
            question,
            _read_repl_input,
            max_attempts=max_attempts,
            on_invalid=_on_invalid,
        )

    return _ask


def _is_exit_command(line: str) -> bool:
    return line.strip().lower() in {"exit", "/exit", "quit", "/quit"}
