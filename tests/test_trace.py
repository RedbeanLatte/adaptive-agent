from __future__ import annotations

from pathlib import Path

from adaptive_agent.trace import TraceWriter


def test_trace_writer_sanitizes_session_id_and_stays_under_trace_dir(tmp_path: Path):
    writer = TraceWriter(tmp_path, "../../../escape_global")
    writer.event("hello", ok=True)

    assert writer.path is not None
    assert writer.path.exists()
    traces_dir = (tmp_path / "traces").resolve()
    writer.path.resolve().relative_to(traces_dir)
    assert writer.path.name != "../../../escape_global.jsonl"
