from __future__ import annotations

import codecs
from pathlib import Path
from typing import Any, Callable

from adaptive_agent.catalog import BuiltinSpec, ToolCatalog
from adaptive_agent.policy import PolicyError, resolve_user_path


class BuiltinError(RuntimeError):
    """Raised when a builtin tool invocation fails (missing args, bad path, etc.)."""


_INSPECT_PREVIEW_CHARS = 200
_INSPECT_CHUNK_BYTES = 8192


def _resolve_safe(workroot: Path, path: str, *, for_write: bool = False) -> Path:
    try:
        return resolve_user_path(workroot, path, for_write=for_write)
    except PolicyError as e:
        raise BuiltinError(str(e)) from e


def _utf8_preview(target: Path, *, preview_chars: int = _INSPECT_PREVIEW_CHARS) -> tuple[bool, str]:
    decoder = codecs.getincrementaldecoder("utf-8")(errors="strict")
    preview_parts: list[str] = []
    remaining = preview_chars
    try:
        with target.open("rb") as handle:
            while True:
                chunk = handle.read(_INSPECT_CHUNK_BYTES)
                if not chunk:
                    tail = decoder.decode(b"", final=True)
                    if tail and remaining > 0:
                        preview_parts.append(tail[:remaining])
                    break
                text = decoder.decode(chunk, final=False)
                if text and remaining > 0:
                    clipped = text[:remaining]
                    preview_parts.append(clipped)
                    remaining -= len(clipped)
        return True, "".join(preview_parts)
    except UnicodeDecodeError:
        return False, ""


def inspect_file(*, workroot: Path, path: str) -> dict[str, Any]:
    target = _resolve_safe(workroot, path)
    if not target.is_file():
        raise BuiltinError(f"file not found: {path}")
    try:
        stat = target.stat()
        is_utf8_text, preview = _utf8_preview(target)
    except OSError as e:
        raise BuiltinError(f"failed to inspect file: {path}: {e}") from e
    return {
        "path": path,
        "size_bytes": stat.st_size,
        "suffix": target.suffix,
        "is_utf8_text": is_utf8_text,
        "preview": preview,
    }


def read_text_file(*, workroot: Path, path: str, encoding: str = "utf-8") -> dict[str, Any]:
    target = _resolve_safe(workroot, path)
    if not target.is_file():
        raise BuiltinError(f"file not found: {path}")
    try:
        content = target.read_text(encoding=encoding)
    except UnicodeDecodeError as e:
        raise BuiltinError(f"file is not valid {encoding.upper()} text: {path}") from e
    except OSError as e:
        raise BuiltinError(f"failed to read text file: {path}: {e}") from e
    return {"path": path, "content": content}


def write_text_file(
    *, workroot: Path, path: str, content: str, encoding: str = "utf-8"
) -> dict[str, Any]:
    target = _resolve_safe(workroot, path, for_write=True)
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        data = content.encode(encoding)
        target.write_bytes(data)
    except OSError as e:
        raise BuiltinError(f"failed to write text file: {path}: {e}") from e
    return {"path": path, "bytes_written": len(data)}


def list_files(*, workroot: Path, path: str = ".") -> dict[str, Any]:
    target = _resolve_safe(workroot, path)
    if not target.is_dir():
        raise BuiltinError(f"not a directory: {path}")
    entries = []
    for entry in sorted(target.iterdir()):
        entries.append(
            {
                "name": entry.name,
                "kind": "dir" if entry.is_dir() else "file",
            }
        )
    return {"path": path, "entries": entries}


BUILTIN_TOOLS: dict[str, Callable[..., dict[str, Any]]] = {
    "inspect_file": inspect_file,
    "read_text_file": read_text_file,
    "write_text_file": write_text_file,
    "list_files": list_files,
}


BUILTIN_SPECS: list[BuiltinSpec] = [
    BuiltinSpec(
        name="inspect_file",
        description="Inspect a file relative to the working directory and report whether it is valid UTF-8 text.",
        args_schema={"path": "str (relative)"},
    ),
    BuiltinSpec(
        name="read_text_file",
        description="Read the contents of a UTF-8 text file relative to the working directory.",
        args_schema={"path": "str (relative)"},
    ),
    BuiltinSpec(
        name="write_text_file",
        description="Write a UTF-8 text file relative to the working directory.",
        args_schema={"path": "str (relative)", "content": "str"},
    ),
    BuiltinSpec(
        name="list_files",
        description="List entries in a directory relative to the working directory.",
        args_schema={"path": "str (relative, default '.')"},
    ),
]


def register_builtins(catalog: ToolCatalog) -> None:
    for spec in BUILTIN_SPECS:
        catalog.register_builtin(spec)


_REQUIRED: dict[str, tuple[str, ...]] = {
    "inspect_file": ("path",),
    "read_text_file": ("path",),
    "write_text_file": ("path", "content"),
    "list_files": (),
}

_ALLOWED: dict[str, frozenset[str]] = {
    "inspect_file": frozenset({"path"}),
    "read_text_file": frozenset({"path", "encoding"}),
    "write_text_file": frozenset({"path", "content", "encoding"}),
    "list_files": frozenset({"path"}),
}


def dispatch_builtin(
    name: str, arguments: dict[str, Any], *, workroot: Path
) -> dict[str, Any]:
    fn = BUILTIN_TOOLS.get(name)
    if fn is None:
        raise BuiltinError(f"unknown builtin: {name}")
    if not isinstance(arguments, dict):
        raise BuiltinError("builtin arguments must be a JSON object")
    for required in _REQUIRED[name]:
        if required not in arguments:
            raise BuiltinError(f"{name} missing required argument: {required}")
    unexpected = sorted(set(arguments) - _ALLOWED[name])
    if unexpected:
        raise BuiltinError(f"{name} unexpected argument(s): {', '.join(unexpected)}")
    try:
        return fn(workroot=workroot, **arguments)
    except TypeError as e:
        raise BuiltinError(f"{name} argument error: {e}") from e
