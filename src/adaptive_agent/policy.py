from __future__ import annotations

import re
from pathlib import Path


class PolicyError(ValueError):
    """Raised when model/user-controlled input violates runtime policy."""


_SAFE_TOOL_NAME_RE = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]*$")
_PROTECTED_WRITE_PARTS = frozenset({".agent_state", ".git"})


def is_safe_tool_name(name: str) -> bool:
    return isinstance(name, str) and bool(_SAFE_TOOL_NAME_RE.fullmatch(name))


def validate_tool_name(name: str) -> str:
    if not is_safe_tool_name(name):
        raise PolicyError(f"unsafe tool name: {name!r}")
    return name


def resolve_user_path(workroot: Path, path: str, *, for_write: bool = False) -> Path:
    if not isinstance(path, str) or not path:
        raise PolicyError("path must be a non-empty string")
    p = Path(path)
    if p.is_absolute():
        raise PolicyError(f"absolute paths are not allowed: {path!r}")
    if for_write:
        _assert_not_protected_write_path(p, path)

    root = Path(workroot).resolve()
    target = (root / p).resolve()
    try:
        relative_target = target.relative_to(root)
    except ValueError:
        raise PolicyError(f"path escapes workroot: {path!r}")
    if for_write:
        _assert_not_protected_write_path(relative_target, path)
    return target


def _assert_not_protected_write_path(path: Path, original: str) -> None:
    if any(part in _PROTECTED_WRITE_PARTS for part in path.parts):
        raise PolicyError(f"protected path cannot be written by builtin: {original!r}")
