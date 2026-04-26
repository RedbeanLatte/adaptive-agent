from __future__ import annotations

import json

from adaptive_agent.catalog import SavedToolSpec, ToolCatalog


def _save_verbose_spec(cat: ToolCatalog) -> None:
    cat.save_generated(
        SavedToolSpec(
            name="csv_cleanup",
            description="Clean CSV rows by sorting and removing duplicates",
            input_summary="csv file with duplicate rows and date column",
            output_summary="clean sorted csv rows",
            code="print('ok')",
            tags=["csv", "dedupe", "sort"],
            input_schema={"argv": ["input_file"]},
            output_schema={"stdout": "text"},
            risk_level="medium",
            execution_hints={"read_only": True},
            safety={"approval_required": True},
            example_args=["fixtures/example3_dirty_rows.csv"],
            verification_status="replay_verified",
            verification_details={"method": "fixture_replay", "stdout_tail": "ok"},
            source_task_summary="User asked to clean CSV rows and remove duplicate dates",
            model_info={"model": "gpt-test"},
            success_count=5,
            failure_count=1,
        )
    )


def test_save_generated_writes_minimal_manifest(tmp_path):
    cat = ToolCatalog(state_dir=tmp_path)

    _save_verbose_spec(cat)

    manifest = json.loads((tmp_path / "tools" / "csv_cleanup" / "manifest.json").read_text())
    assert manifest == {
        "schema_version": 2,
        "name": "csv_cleanup",
        "version": 1,
        "description": "Clean CSV rows by sorting and removing duplicates",
        "example_args": ["fixtures/example3_dirty_rows.csv"],
        "verification_status": "replay_verified",
        "created_at": manifest["created_at"],
    }
    assert manifest["created_at"]
    for removed_key in ["input", "output", "tags", "execution_hints", "safety", "provenance", "usage", "risk_level"]:
        assert removed_key not in manifest


def test_summary_for_prompt_omits_verbose_saved_tool_metadata(tmp_path):
    cat = ToolCatalog(state_dir=tmp_path)
    _save_verbose_spec(cat)

    summary = cat.summary_for_prompt("clean duplicate csv rows")

    assert "csv_cleanup" in summary
    assert "Clean CSV rows by sorting and removing duplicates" in summary
    assert "verified: replay_verified" in summary
    assert "args: fixtures/example3_dirty_rows.csv" in summary
    for noisy in ["risk:", "hints:", "tags:", "success:", "failure:", "input:", "output:"]:
        assert noisy not in summary
