from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_TOKEN_RE = re.compile(r"[A-Za-z0-9_가-힣]+")


@dataclass
class ToolMatch:
    tool: Any
    score: float
    matched_terms: list[str] = field(default_factory=list)
    match_kind: str = "heuristic"


SEMANTIC_MATCH_THRESHOLD = 0.74


def score_saved_tool(spec: Any, task_tokens: set[str]) -> ToolMatch | None:
    spec_tokens = tool_tokens(spec)
    matched_terms = sorted(task_tokens & spec_tokens)
    name_terms = {part.lower() for part in str(spec.name).split("_") if part}
    name_overlap = len(task_tokens & name_terms)
    if not matched_terms and name_overlap == 0:
        return None

    score = len(matched_terms) * 12
    score += name_overlap * 4

    status = str(getattr(spec, "verification_status", "") or "unverified")
    if status in {"runtime_verified", "replay_verified"}:
        score += 6
    elif status == "replay_failed":
        score -= 10
    elif status == "unverified":
        score -= 1

    score += min(int(getattr(spec, "success_count", 0) or 0), 5) * 2
    score -= min(int(getattr(spec, "failure_count", 0) or 0), 5) * 4

    risk_level = str(getattr(spec, "risk_level", "medium") or "medium")
    if risk_level == "low":
        score += 1
    elif risk_level == "high":
        score -= 4

    stale_days = stale_days_for_spec(spec)
    if stale_days > 3650:
        score -= 12
    elif stale_days > 365:
        score -= 6
    elif stale_days > 90:
        score -= 3

    # Suppress weak recommendations: require either a substantive description
    # match (one matched_term + verified bonus pushes score past 12) or strong
    # name overlap. Below this floor the prompt's "Recommended" hint is more
    # noise than signal.
    if score < 12:
        return None
    if not matched_terms and name_overlap < 1:
        return None
    return ToolMatch(tool=spec, score=score, matched_terms=matched_terms)


def score_semantic_tool(
    spec: Any,
    task_embedding: list[float],
    *,
    threshold: float = SEMANTIC_MATCH_THRESHOLD,
) -> ToolMatch | None:
    if str(getattr(spec, "verification_status", "") or "") == "replay_failed":
        return None
    similarity = cosine_similarity(task_embedding, getattr(spec, "embedding_vector", []) or [])
    if similarity < threshold:
        return None
    return ToolMatch(
        tool=spec,
        score=round(similarity, 4),
        matched_terms=["embedding"],
        match_kind="semantic",
    )


def cosine_similarity(left: list[float], right: list[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return dot / (left_norm * right_norm)


def tool_tokens(spec: Any) -> set[str]:
    parts = [
        getattr(spec, "name", ""),
        getattr(spec, "description", ""),
        " ".join(Path(str(arg)).name for arg in (getattr(spec, "example_args", []) or [])),
    ]
    tokens: set[str] = set()
    for part in parts:
        tokens.update(tokenize(str(part)))
    return tokens


def tool_embedding_text(spec: Any) -> str:
    parts = [
        getattr(spec, "name", ""),
        getattr(spec, "description", ""),
        " ".join(Path(str(arg)).name for arg in (getattr(spec, "example_args", []) or [])),
    ]
    return "\n".join(str(part).strip() for part in parts if str(part).strip())


def stale_days_for_spec(spec: Any) -> int:
    reference = getattr(spec, "last_used_at", "") or getattr(spec, "created_at", "")
    dt = parse_iso_datetime(str(reference or ""))
    if dt is None:
        return 0
    now = datetime.now(timezone.utc)
    return max(0, (now - dt).days)


def parse_iso_datetime(value: str) -> datetime | None:
    text = (value or "").strip()
    if not text:
        return None
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00")).astimezone(timezone.utc)
    except ValueError:
        return None


def tokenize(text: str) -> set[str]:
    if not text:
        return set()
    return {match.lower() for match in _TOKEN_RE.findall(str(text))}
