from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from adaptive_agent.policy import is_safe_tool_name, validate_tool_name
from adaptive_agent.tool_matching import (
    ToolMatch,
    score_saved_tool,
    score_semantic_tool,
    tokenize,
    tool_embedding_text,
)


class ToolNotFoundError(KeyError):
    """Raised when a tool name is not found in the catalog."""


_TOOL_CODE_FILENAME = "tool.py"
_TOOL_MANIFEST_FILENAME = "manifest.json"
_TOOL_VERIFICATION_FILENAME = "verification.json"
_TOOL_LAST_RUN_FILENAME = "last_run.json"
_TOOL_EMBEDDING_FILENAME = "embedding.json"
_MANIFEST_SCHEMA_VERSION = 2


@dataclass
class BuiltinSpec:
    name: str
    description: str
    args_schema: dict[str, str] = field(default_factory=dict)
    kind: str = "builtin"


@dataclass
class SavedToolSpec:
    name: str
    description: str
    input_summary: str
    output_summary: str
    code: str
    kind: str = "saved"
    version: int = 1
    created_at: str = ""
    tags: list[str] = field(default_factory=list)
    input_schema: dict[str, Any] = field(default_factory=dict)
    output_schema: dict[str, Any] = field(default_factory=dict)
    risk_level: str = "medium"
    execution_hints: dict[str, Any] = field(default_factory=dict)
    safety: dict[str, Any] = field(default_factory=dict)
    example_args: list[str] = field(default_factory=list)
    verification_status: str = "unverified"
    verification_details: dict[str, Any] = field(default_factory=dict)
    source_task_summary: str = ""
    model_info: dict[str, Any] = field(default_factory=dict)
    success_count: int = 0
    failure_count: int = 0
    last_used_at: str = ""
    last_failure_at: str = ""
    embedding_model: str = ""
    embedding_text_sha256: str = ""
    embedding_vector: list[float] = field(default_factory=list)

    @property
    def package_relpath(self) -> str:
        return f"tools/{self.name}"



class ToolCatalog:
    """Tool registry covering built-in tools (in-memory) and generated saved tools (on disk).

    On-disk layout (when ``state_dir`` is provided):
        <state_dir>/tools/<name>/tool.py                  - saved tool source
        <state_dir>/tools/<name>/manifest.json            - saved tool contract and metadata
        <state_dir>/tools/<name>/verification.json        - verification evidence
        <state_dir>/tools/<name>/last_run.json            - last execution snapshot

    lookup() resolves saved tools first, then built-ins, so reuse always wins
    over a shadow builtin of the same name.
    """

    def __init__(self, state_dir: Path | str | None, embedding_client: Any | None = None) -> None:
        self._state_dir: Path | None = Path(state_dir) if state_dir else None
        self._embedding_client = embedding_client
        self._builtins: dict[str, BuiltinSpec] = {}
        self._saved: dict[str, SavedToolSpec] = {}
        if self._state_dir is not None:
            self._load_from_disk()

    # -- builtin --

    def register_builtin(self, spec: BuiltinSpec) -> None:
        if spec.name in self._builtins:
            raise ValueError(f"builtin already registered: {spec.name}")
        self._builtins[spec.name] = spec

    def list_builtins(self) -> list[BuiltinSpec]:
        return list(self._builtins.values())

    # -- saved (generated) --

    def next_version(self, name: str) -> int:
        existing = self._saved.get(name)
        if isinstance(existing, SavedToolSpec) and existing.version >= 1:
            return existing.version + 1
        return 1

    def save_generated(self, spec: SavedToolSpec) -> None:
        validate_tool_name(spec.name)
        if self._state_dir is None:
            raise RuntimeError("ToolCatalog has no state_dir; cannot persist saved tools")

        normalized = self._normalize_saved_spec(spec)
        normalized = self._with_embedding_cache(normalized)
        self._saved[normalized.name] = normalized
        self._write_saved_package(normalized, update_verification=True)

    def list_saved(self) -> list[SavedToolSpec]:
        return [self._saved[name] for name in sorted(self._saved)]

    def load_last_run(self, name: str) -> dict[str, Any]:
        spec = self.lookup(name)
        if not isinstance(spec, SavedToolSpec):
            raise ToolNotFoundError(name)
        return self._read_json_dict(self._package_dir(name) / _TOOL_LAST_RUN_FILENAME) or {}

    def remove_saved(self, name: str) -> SavedToolSpec:
        spec = self.lookup(name)
        if not isinstance(spec, SavedToolSpec):
            raise ToolNotFoundError(name)
        self._saved.pop(name, None)
        package_dir = self._package_dir(name)
        shutil.rmtree(package_dir, ignore_errors=True)
        return spec

    def verify_saved(
        self,
        name: str,
        *,
        ok: bool,
        exit_status: int,
        timed_out: bool,
        duration_sec: float,
        stdout_tail: str,
        stderr_tail: str,
        argv: list[str],
        task_summary: str = "",
    ) -> SavedToolSpec:
        spec = self.lookup(name)
        if not isinstance(spec, SavedToolSpec):
            raise ToolNotFoundError(name)

        now = _utc_now()
        if ok:
            spec.success_count += 1
        else:
            spec.failure_count += 1
            spec.last_failure_at = now
        spec.last_used_at = now
        spec.verification_status = "replay_verified" if ok else "replay_failed"
        spec.verification_details = {
            "method": "manual_verify_example_args",
            "argv": [str(item) for item in argv],
            "ok": ok,
            "exit_status": exit_status,
            "timed_out": timed_out,
            "duration_sec": duration_sec,
            "stdout_tail": stdout_tail,
            "stderr_tail": stderr_tail,
            "task_summary": task_summary,
            "verified_at": now,
        }
        last_run_payload = {
            "ok": ok,
            "exit_status": exit_status,
            "timed_out": timed_out,
            "duration_sec": duration_sec,
            "stdout_tail": stdout_tail,
            "stderr_tail": stderr_tail,
            "task_summary": task_summary,
            "recorded_at": now,
        }
        self._saved[name] = spec
        self._write_saved_package(spec, update_verification=True, last_run_payload=last_run_payload)
        return spec

    def record_execution(
        self,
        name: str,
        *,
        ok: bool,
        exit_status: int,
        timed_out: bool,
        duration_sec: float,
        stdout_tail: str,
        stderr_tail: str,
        task_summary: str = "",
    ) -> None:
        spec = self.lookup(name)
        if not isinstance(spec, SavedToolSpec):
            raise ToolNotFoundError(name)

        now = _utc_now()
        if ok:
            spec.success_count += 1
        else:
            spec.failure_count += 1
            spec.last_failure_at = now
        spec.last_used_at = now

        last_run_payload = {
            "ok": ok,
            "exit_status": exit_status,
            "timed_out": timed_out,
            "duration_sec": duration_sec,
            "stdout_tail": stdout_tail,
            "stderr_tail": stderr_tail,
            "task_summary": task_summary,
            "recorded_at": now,
        }
        self._saved[name] = spec
        self._write_saved_package(spec, update_verification=False, last_run_payload=last_run_payload)

    def find_best_match(self, task: str) -> ToolMatch | None:
        semantic = self._find_semantic_match(task)
        if semantic is not None:
            return semantic

        task_tokens = tokenize(task)
        if not task_tokens:
            return None

        best: ToolMatch | None = None
        for spec in self._saved.values():
            candidate = score_saved_tool(spec, task_tokens)
            if candidate is None:
                continue
            if best is None or candidate.score > best.score:
                best = candidate
        return best

    # -- lookup --

    def lookup(self, name: str) -> BuiltinSpec | SavedToolSpec:
        if name in self._saved:
            return self._saved[name]
        if name in self._builtins:
            return self._builtins[name]
        raise ToolNotFoundError(name)

    def has(self, name: str) -> bool:
        return name in self._saved or name in self._builtins

    def summary_for_prompt(self, current_task: str | None = None) -> str:
        lines: list[str] = []
        if self._builtins:
            lines.append("Builtin tools:")
            for b in self._builtins.values():
                args = ", ".join(f"{k}: {v}" for k, v in b.args_schema.items())
                lines.append(f"  - {b.name}({args}) — {b.description}")
        if self._saved:
            lines.append("Saved reusable tools:")
            for s in self._saved.values():
                args = " ".join(s.example_args[:3]) if s.example_args else "none"
                lines.append(
                    f"  - {s.name} — {s.description} "
                    f"| verified: {s.verification_status} "
                    f"| args: {args}"
                )
        if current_task:
            match = self.find_best_match(current_task)
            if match is not None:
                matched_terms = ", ".join(match.matched_terms) if match.matched_terms else "heuristic"
                lines.append("Recommended reusable tool for current task:")
                if match.match_kind == "semantic":
                    lines.append(
                        f"  - {match.tool.name} (semantic_similarity: {match.score}; matched: {matched_terms})"
                    )
                else:
                    lines.append(
                        f"  - {match.tool.name} (score: {match.score}; matched: {matched_terms})"
                    )
        return "\n".join(lines)

    # -- internals --

    def _package_dir(self, name: str) -> Path:
        assert self._state_dir is not None
        return self._state_dir / "tools" / name

    def _load_from_disk(self) -> None:
        assert self._state_dir is not None
        self._saved = {}

        tools_dir = self._state_dir / "tools"
        self._remove_legacy_flat_catalog_artifacts(tools_dir)
        if tools_dir.is_dir():
            for entry in sorted(tools_dir.iterdir()):
                if not entry.is_dir():
                    continue
                spec = self._load_saved_package(entry)
                if spec is not None:
                    self._saved[spec.name] = spec

    def _load_saved_package(self, package_dir: Path) -> SavedToolSpec | None:
        manifest_path = package_dir / _TOOL_MANIFEST_FILENAME
        if not manifest_path.is_file():
            self._remove_legacy_package_dir(package_dir)
            return None

        manifest = self._read_json_dict(manifest_path)
        if not isinstance(manifest, dict):
            return None
        if _coerce_positive_int(manifest.get("schema_version"), default=0) != _MANIFEST_SCHEMA_VERSION:
            return None

        name = str(manifest.get("name") or package_dir.name)
        if not is_safe_tool_name(name):
            return None

        code_path = package_dir / _TOOL_CODE_FILENAME
        if not code_path.is_file():
            return None
        try:
            code = code_path.read_text(encoding="utf-8")
        except OSError:
            return None
        if not code.strip():
            return None

        verification_details = self._read_json_dict(package_dir / _TOOL_VERIFICATION_FILENAME) or {}
        embedding_details = self._read_json_dict(package_dir / _TOOL_EMBEDDING_FILENAME) or {}
        try:
            return self._spec_from_manifest(
                name=name,
                manifest=manifest,
                code=code,
                verification_details=verification_details,
                embedding_details=embedding_details,
            )
        except Exception:
            return None

    def _remove_legacy_flat_catalog_artifacts(self, tools_dir: Path) -> None:
        assert self._state_dir is not None
        legacy_catalog = self._state_dir / "catalog.json"
        if legacy_catalog.is_file():
            try:
                legacy_catalog.unlink()
            except OSError:
                pass
        if not tools_dir.is_dir():
            return
        for entry in tools_dir.iterdir():
            if entry.is_file() and entry.suffix == ".py":
                try:
                    entry.unlink()
                except OSError:
                    pass

    def _remove_legacy_package_dir(self, package_dir: Path) -> None:
        if (package_dir / "meta.json").exists():
            shutil.rmtree(package_dir, ignore_errors=True)

    def _normalize_saved_spec(self, spec: SavedToolSpec) -> SavedToolSpec:
        existing = self._saved.get(spec.name)
        version = spec.version if isinstance(spec.version, int) and spec.version >= 1 else 1
        if isinstance(existing, SavedToolSpec) and version <= existing.version:
            version = existing.version + 1

        created_at = spec.created_at or _utc_now()
        verification_details = _coerce_dict(spec.verification_details)
        verification_status = spec.verification_status or (
            "runtime_verified" if verification_details else "unverified"
        )
        success_count = _coerce_nonnegative_int(spec.success_count)
        failure_count = _coerce_nonnegative_int(spec.failure_count)
        execution_hints = _coerce_execution_hints(spec.execution_hints)
        safety = _coerce_safety(spec.safety, risk_level=spec.risk_level or "medium")
        risk_level = str(safety.get("risk_level") or "medium")
        if success_count == 0 and verification_status == "runtime_verified":
            success_count = 1
        last_used_at = spec.last_used_at or (created_at if success_count > 0 else "")
        last_failure_at = spec.last_failure_at or (created_at if failure_count > 0 else "")

        return SavedToolSpec(
            name=spec.name,
            description=spec.description,
            input_summary=spec.input_summary,
            output_summary=spec.output_summary,
            code=spec.code,
            version=version,
            created_at=created_at,
            tags=_coerce_list_of_str(spec.tags),
            input_schema=_coerce_dict(spec.input_schema),
            output_schema=_coerce_dict(spec.output_schema),
            risk_level=risk_level,
            execution_hints=execution_hints,
            safety=safety,
            example_args=_coerce_list_of_str(spec.example_args),
            verification_status=verification_status,
            verification_details=verification_details,
            source_task_summary=spec.source_task_summary or "",
            model_info=_coerce_dict(spec.model_info),
            success_count=success_count,
            failure_count=failure_count,
            last_used_at=last_used_at,
            last_failure_at=last_failure_at,
            embedding_model=spec.embedding_model or "",
            embedding_text_sha256=spec.embedding_text_sha256 or "",
            embedding_vector=_coerce_float_list(spec.embedding_vector),
        )

    def _spec_from_manifest(
        self,
        *,
        name: str,
        manifest: dict[str, Any],
        code: str,
        verification_details: dict[str, Any],
        embedding_details: dict[str, Any],
    ) -> SavedToolSpec:
        input_info = _coerce_dict(manifest.get("input"))
        output_info = _coerce_dict(manifest.get("output"))
        provenance = _coerce_dict(manifest.get("provenance"))
        usage = _coerce_dict(manifest.get("usage"))
        schema_version = _coerce_positive_int(manifest.get("schema_version"), default=0)
        if schema_version == _MANIFEST_SCHEMA_VERSION:
            return SavedToolSpec(
                name=name,
                description=str(manifest.get("description", "")),
                input_summary="",
                output_summary="",
                code=code,
                version=_coerce_positive_int(manifest.get("version"), default=1),
                created_at=str(manifest.get("created_at", "") or ""),
                example_args=_coerce_list_of_str(manifest.get("example_args")),
                verification_status=str(manifest.get("verification_status", "unverified") or "unverified"),
                verification_details=_coerce_dict(verification_details),
                embedding_model=str(embedding_details.get("model", "") or ""),
                embedding_text_sha256=str(embedding_details.get("text_sha256", "") or ""),
                embedding_vector=_coerce_float_list(embedding_details.get("vector")),
            )
        safety = _coerce_safety(
            manifest.get("safety"),
            risk_level=str(manifest.get("risk_level", "medium") or "medium"),
        )
        risk_level = str(safety.get("risk_level") or "medium")
        return SavedToolSpec(
            name=name,
            description=str(manifest.get("description", "")),
            input_summary=str(input_info.get("summary", "")),
            output_summary=str(output_info.get("summary", "")),
            code=code,
            version=_coerce_positive_int(manifest.get("version"), default=1),
            created_at=str(provenance.get("created_at", "") or ""),
            tags=_coerce_list_of_str(manifest.get("tags")),
            input_schema=_coerce_dict(input_info.get("schema")),
            output_schema=_coerce_dict(output_info.get("schema")),
            risk_level=risk_level,
            execution_hints=_coerce_execution_hints(manifest.get("execution_hints")),
            safety=safety,
            example_args=_coerce_list_of_str(input_info.get("example_args")),
            verification_status=str(manifest.get("verification_status", "unverified") or "unverified"),
            verification_details=_coerce_dict(verification_details),
            source_task_summary=str(provenance.get("source_task_summary", "") or ""),
            model_info=_coerce_dict(provenance.get("model_info")),
            success_count=_coerce_nonnegative_int(usage.get("success_count")),
            failure_count=_coerce_nonnegative_int(usage.get("failure_count")),
            last_used_at=str(usage.get("last_used_at", "") or ""),
            last_failure_at=str(usage.get("last_failure_at", "") or ""),
            embedding_model=str(embedding_details.get("model", "") or ""),
            embedding_text_sha256=str(embedding_details.get("text_sha256", "") or ""),
            embedding_vector=_coerce_float_list(embedding_details.get("vector")),
        )

    def _find_semantic_match(self, task: str) -> ToolMatch | None:
        client = self._embedding_client
        if client is None:
            return None
        try:
            task_embedding = client.embed(task)
        except Exception:
            return None

        best: ToolMatch | None = None
        for spec in self._saved.values():
            candidate = score_semantic_tool(spec, task_embedding)
            if candidate is None:
                continue
            if best is None or candidate.score > best.score:
                best = candidate
        return best

    def _with_embedding_cache(self, spec: SavedToolSpec) -> SavedToolSpec:
        client = self._embedding_client
        if client is None:
            return spec
        text = tool_embedding_text(spec)
        if not text:
            return spec
        try:
            vector = client.embed(text)
        except Exception:
            return spec
        spec.embedding_model = str(getattr(client, "model", "") or "")
        spec.embedding_text_sha256 = hashlib.sha256(text.encode("utf-8")).hexdigest()
        spec.embedding_vector = _coerce_float_list(vector)
        return spec

    def _manifest_payload(self, spec: SavedToolSpec) -> dict[str, Any]:
        return {
            "schema_version": _MANIFEST_SCHEMA_VERSION,
            "name": spec.name,
            "version": spec.version,
            "description": _shorten(spec.description, 240),
            "example_args": _coerce_list_of_str(spec.example_args)[:12],
            "verification_status": spec.verification_status or "unverified",
            "created_at": spec.created_at,
        }

    def _verification_payload(self, spec: SavedToolSpec) -> dict[str, Any]:
        payload = dict(spec.verification_details)
        payload.setdefault("verification_status", spec.verification_status)
        payload.setdefault("recorded_at", spec.created_at)
        return payload

    def _last_run_payload_from_spec(self, spec: SavedToolSpec) -> dict[str, Any]:
        details = spec.verification_details
        if not details:
            return {"status": "not_run", "recorded_at": spec.created_at}

        # Prefer the replay run (the save-time verification), fall back to the
        # initial run, then finally to the raw verification_details blob.
        source = details.get("replay_run") or details.get("initial_run") or details
        payload = dict(source) if isinstance(source, dict) else {}
        payload.setdefault("ok", payload.get("exit_status", 0) == 0)
        payload.setdefault("verification_status", spec.verification_status)
        payload.setdefault("recorded_at", spec.created_at)
        return payload

    def _write_saved_package(
        self,
        spec: SavedToolSpec,
        *,
        update_verification: bool,
        last_run_payload: dict[str, Any] | None = None,
    ) -> None:
        package_dir = self._package_dir(spec.name)
        package_dir.mkdir(parents=True, exist_ok=True)
        (package_dir / _TOOL_CODE_FILENAME).write_text(spec.code, encoding="utf-8")
        (package_dir / _TOOL_MANIFEST_FILENAME).write_text(
            json.dumps(self._manifest_payload(spec), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        if spec.embedding_model and spec.embedding_text_sha256 and spec.embedding_vector:
            (package_dir / _TOOL_EMBEDDING_FILENAME).write_text(
                json.dumps(self._embedding_payload(spec), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        else:
            try:
                (package_dir / _TOOL_EMBEDDING_FILENAME).unlink(missing_ok=True)
            except OSError:
                pass
        if update_verification:
            (package_dir / _TOOL_VERIFICATION_FILENAME).write_text(
                json.dumps(self._verification_payload(spec), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        last_run = last_run_payload or self._last_run_payload_from_spec(spec)
        (package_dir / _TOOL_LAST_RUN_FILENAME).write_text(
            json.dumps(last_run, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _embedding_payload(self, spec: SavedToolSpec) -> dict[str, Any]:
        return {
            "model": spec.embedding_model,
            "text_sha256": spec.embedding_text_sha256,
            "vector": spec.embedding_vector,
            "created_at": spec.created_at,
        }

    @staticmethod
    def _read_json_dict(path: Path) -> dict[str, Any] | None:
        if not path.is_file():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        return data if isinstance(data, dict) else None


def _shorten(value: str | None, limit: int) -> str:
    text = str(value or "").strip()
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 3)] + "..."


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _coerce_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _coerce_list_of_str(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value]


def _coerce_float_list(value: Any) -> list[float]:
    if not isinstance(value, list):
        return []
    try:
        return [float(item) for item in value]
    except (TypeError, ValueError):
        return []


def _coerce_execution_hints(value: Any) -> dict[str, bool]:
    raw = _coerce_dict(value)
    return {
        "read_only": bool(raw.get("read_only", False)),
        "destructive": bool(raw.get("destructive", False)),
        "idempotent": bool(raw.get("idempotent", False)),
        "open_world": bool(raw.get("open_world", False)),
    }


def _coerce_safety(value: Any, *, risk_level: str) -> dict[str, Any]:
    raw = _coerce_dict(value)
    return {
        "risk_level": str(raw.get("risk_level") or risk_level or "medium"),
        "approval_required": bool(raw.get("approval_required", True)),
        "guarded_subprocess": bool(raw.get("guarded_subprocess", True)),
    }


def _coerce_positive_int(value: Any, *, default: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed >= 1 else default


def _coerce_nonnegative_int(value: Any) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return 0
    return parsed if parsed >= 0 else 0
