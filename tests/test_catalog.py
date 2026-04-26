from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest

from adaptive_agent.catalog import (
    BuiltinSpec,
    SavedToolSpec,
    ToolCatalog,
    ToolMatch,
    ToolNotFoundError,
)


# ---------- builtin registration ----------

def test_register_and_list_builtins():
    cat = ToolCatalog(state_dir=None)
    cat.register_builtin(
        BuiltinSpec(name="read_text_file", description="read a file", args_schema={"path": "str"})
    )
    cat.register_builtin(
        BuiltinSpec(name="write_text_file", description="write a file", args_schema={"path": "str", "content": "str"})
    )
    names = [t.name for t in cat.list_builtins()]
    assert names == ["read_text_file", "write_text_file"]


def test_duplicate_builtin_rejected():
    cat = ToolCatalog(state_dir=None)
    cat.register_builtin(BuiltinSpec(name="read_text_file", description="", args_schema={}))
    with pytest.raises(ValueError):
        cat.register_builtin(BuiltinSpec(name="read_text_file", description="", args_schema={}))


def test_lookup_missing_raises():
    cat = ToolCatalog(state_dir=None)
    with pytest.raises(ToolNotFoundError):
        cat.lookup("nope")


# ---------- saved tool persistence ----------

def test_save_and_load_generated_tool(tmp_path):
    cat = ToolCatalog(state_dir=tmp_path)
    spec = SavedToolSpec(
        name="csv_cleanup",
        description="dedupe + sort by date",
        input_summary="csv path",
        output_summary="cleaned csv rows",
        code="print('hi')",
        version=2,
        created_at="2026-04-23T00:00:00Z",
        tags=["csv", "cleanup"],
        input_schema={"argv": ["csv_path"]},
        output_schema={"stdout": "cleaned csv rows"},
        risk_level="low",
        example_args=["fixtures/example3_dirty_rows.csv"],
        verification_status="runtime_verified",
        verification_details={
            "method": "create_tool_execution",
            "exit_status": 0,
            "argv": ["fixtures/example3_dirty_rows.csv"],
        },
        source_task_summary="중복 제거 후 날짜순 정렬",
        model_info={"model": "gpt-test"},
    )
    cat.save_generated(spec)

    # reload from disk in a fresh catalog
    fresh = ToolCatalog(state_dir=tmp_path)
    loaded = fresh.lookup("csv_cleanup")
    assert isinstance(loaded, SavedToolSpec)
    assert loaded.description == "dedupe + sort by date"
    assert loaded.code == "print('hi')"
    assert loaded.version == 2
    assert loaded.tags == []
    assert loaded.risk_level == "medium"
    assert loaded.execution_hints == {}
    assert loaded.safety == {}
    assert loaded.example_args == ["fixtures/example3_dirty_rows.csv"]
    assert loaded.verification_status == "runtime_verified"
    assert loaded.source_task_summary == ""
    assert loaded.model_info == {}
    assert loaded.verification_details["method"] == "create_tool_execution"
    assert loaded.verification_details["exit_status"] == 0
    assert loaded.verification_details["argv"] == ["fixtures/example3_dirty_rows.csv"]
    assert loaded.verification_details["verification_status"] == "runtime_verified"


def test_save_writes_expected_paths(tmp_path):
    cat = ToolCatalog(state_dir=tmp_path)
    cat.save_generated(
        SavedToolSpec(
            name="csv_cleanup",
            description="",
            input_summary="",
            output_summary="",
            code="print('ok')",
            verification_status="runtime_verified",
            verification_details={"method": "create_tool_execution", "exit_status": 0},
        )
    )
    tool_dir = tmp_path / "tools" / "csv_cleanup"
    tool_file = tool_dir / "tool.py"
    manifest_file = tool_dir / "manifest.json"
    verification_file = tool_dir / "verification.json"
    last_run_file = tool_dir / "last_run.json"
    assert tool_file.is_file()
    assert tool_file.read_text() == "print('ok')"
    assert manifest_file.is_file()
    manifest = json.loads(manifest_file.read_text())
    assert manifest["schema_version"] == 2
    assert manifest["name"] == "csv_cleanup"
    assert manifest["verification_status"] == "runtime_verified"
    assert set(manifest) == {
        "schema_version",
        "name",
        "version",
        "description",
        "example_args",
        "verification_status",
        "created_at",
    }
    assert verification_file.is_file()
    verification = json.loads(verification_file.read_text())
    assert verification["method"] == "create_tool_execution"
    assert last_run_file.is_file()
    assert not (tmp_path / "catalog.json").exists()


def test_unsafe_tool_name_rejected(tmp_path):
    cat = ToolCatalog(state_dir=tmp_path)
    with pytest.raises(ValueError):
        cat.save_generated(
            SavedToolSpec(
                name="../etc/passwd",
                description="",
                input_summary="",
                output_summary="",
                code="x",
            )
        )


# ---------- lookup priority: saved > builtin when names match; both co-exist otherwise ----------

def test_find_reusable_prefers_saved_tool(tmp_path):
    cat = ToolCatalog(state_dir=tmp_path)
    cat.register_builtin(
        BuiltinSpec(name="csv_cleanup", description="builtin ver", args_schema={})
    )
    cat.save_generated(
        SavedToolSpec(
            name="csv_cleanup",
            description="saved ver",
            input_summary="",
            output_summary="",
            code="x",
        )
    )
    result = cat.lookup("csv_cleanup")
    assert isinstance(result, SavedToolSpec)


def test_catalog_summary_for_prompt_includes_both(tmp_path):
    cat = ToolCatalog(state_dir=tmp_path)
    cat.register_builtin(
        BuiltinSpec(name="read_text_file", description="read a file", args_schema={"path": "str"})
    )
    cat.save_generated(
        SavedToolSpec(
            name="csv_cleanup",
            description="dedupe+sort",
            input_summary="csv path",
            output_summary="cleaned csv",
            code="x",
            tags=["csv"],
            risk_level="low",
            verification_status="runtime_verified",
        )
    )
    summary = cat.summary_for_prompt()
    assert "read_text_file" in summary
    assert "csv_cleanup" in summary
    assert "dedupe+sort" in summary
    assert "runtime_verified" in summary
    assert "risk:" not in summary
    assert "tags:" not in summary


def test_catalog_survives_corrupt_file(tmp_path):
    tool_dir = tmp_path / "tools" / "bad_tool"
    tool_dir.mkdir(parents=True)
    (tool_dir / "tool.py").write_text("print('bad')")
    (tool_dir / "manifest.json").write_text("{not json")
    cat = ToolCatalog(state_dir=tmp_path)
    # should not crash; simply empty
    assert cat.list_saved() == []
    assert tool_dir.exists()


def test_catalog_skips_legacy_flat_catalog_index(tmp_path):
    tools_dir = tmp_path / "tools"
    tools_dir.mkdir()
    (tools_dir / "ghost_tool.py").write_text("print('legacy')")
    (tmp_path / "catalog.json").write_text(
        json.dumps(
            {
                "ghost_tool": {
                    "description": "missing source",
                    "input_summary": "x",
                    "output_summary": "y",
                }
            }
        )
    )
    cat = ToolCatalog(state_dir=tmp_path)
    assert cat.list_saved() == []
    assert not cat.has("ghost_tool")
    assert not (tools_dir / "ghost_tool.py").exists()
    assert not (tmp_path / "catalog.json").exists()


def test_catalog_skips_saved_tool_when_source_file_is_missing(tmp_path):
    tool_dir = tmp_path / "tools" / "ghost_tool"
    tool_dir.mkdir(parents=True)
    (tool_dir / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 2,
                "name": "ghost_tool",
                "description": "missing source",
                "input": {"summary": "x", "schema": {}, "example_args": []},
                "output": {"summary": "y", "schema": {}},
            }
        )
    )
    cat = ToolCatalog(state_dir=tmp_path)
    assert cat.list_saved() == []
    assert not cat.has("ghost_tool")


def test_catalog_skips_legacy_flat_saved_tool_from_catalog_index(tmp_path):
    tools_dir = tmp_path / "tools"
    tools_dir.mkdir()
    (tools_dir / "legacy_helper.py").write_text("print('legacy')")
    (tmp_path / "catalog.json").write_text(
        json.dumps(
            {
                "legacy_helper": {
                    "description": "legacy flat layout",
                    "input_summary": "csv path",
                    "output_summary": "clean csv",
                    "version": 2,
                    "created_at": "2026-04-23T00:00:00Z",
                    "tags": ["csv", "legacy"],
                    "input_schema": {"argv": ["csv_path"]},
                    "output_schema": {"stdout": "clean csv"},
                    "risk_level": "low",
                    "example_args": ["fixtures/example.csv"],
                    "verification_status": "runtime_verified",
                    "source_task_summary": "legacy save",
                    "model_info": {"model": "gpt-test"},
                }
            }
        )
    )

    cat = ToolCatalog(state_dir=tmp_path)

    assert cat.list_saved() == []
    assert not cat.has("legacy_helper")
    assert not (tools_dir / "legacy_helper.py").exists()
    assert not (tmp_path / "catalog.json").exists()


def test_catalog_skips_legacy_meta_package_without_manifest(tmp_path):
    tool_dir = tmp_path / "tools" / "luau_script_audit"
    tool_dir.mkdir(parents=True)
    (tool_dir / "tool.py").write_text("print('ok')")
    (tool_dir / "meta.json").write_text(
        json.dumps(
            {
                "name": "luau_script_audit",
                "description": "audit Luau gameplay script",
                "input_summary": "script path",
                "output_summary": "issue summary",
                "version": 3,
                "created_at": "2026-04-23T00:00:00Z",
                "tags": ["luau", "audit"],
                "input_schema": {"argv": ["script_path"]},
                "output_schema": {"stdout": "issue summary"},
                "risk_level": "low",
                "example_args": ["roblox_guard_npc.lua"],
                "verification_status": "runtime_verified",
                "source_task_summary": "Luau 스크립트 점검",
                "model_info": {"model": "gpt-test"},
            }
        )
    )
    (tool_dir / "verification.json").write_text(
        json.dumps({"method": "create_tool_execution", "exit_status": 0})
    )

    cat = ToolCatalog(state_dir=tmp_path)
    assert cat.list_saved() == []
    assert not cat.has("luau_script_audit")
    assert not tool_dir.exists()


def test_catalog_loads_saved_tool_from_manifest_package(tmp_path):
    tool_dir = tmp_path / "tools" / "luau_script_audit"
    tool_dir.mkdir(parents=True)
    (tool_dir / "tool.py").write_text("print('ok')")
    (tool_dir / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 2,
                "name": "luau_script_audit",
                "version": 3,
                "description": "audit Luau gameplay script",
                "example_args": ["roblox_guard_npc.lua"],
                "verification_status": "runtime_verified",
                "created_at": "2026-04-23T00:00:00Z",
            }
        )
    )
    (tool_dir / "verification.json").write_text(
        json.dumps({"method": "create_tool_execution", "exit_status": 0})
    )

    cat = ToolCatalog(state_dir=tmp_path)
    loaded = cat.lookup("luau_script_audit")
    assert isinstance(loaded, SavedToolSpec)
    assert loaded.version == 3
    assert loaded.description == "audit Luau gameplay script"
    assert loaded.example_args == ["roblox_guard_npc.lua"]
    assert loaded.tags == []
    assert loaded.verification_status == "runtime_verified"
    assert loaded.model_info == {}
    assert loaded.success_count == 0
    assert loaded.failure_count == 0
    assert loaded.execution_hints == {}
    assert loaded.safety == {}


def test_find_best_match_suppresses_weakly_related_tool(tmp_path):
    # An unrelated saved tool should not be recommended for a task that doesn't
    # share substantive vocabulary with it.
    cat = ToolCatalog(state_dir=tmp_path)
    recent = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    cat.save_generated(
        SavedToolSpec(
            name="calculate_sha256",
            description="SHA256 hash of a file",
            input_summary="file path",
            output_summary="hex digest",
            code="print('ok')",
            tags=["hash", "sha256"],
            verification_status="runtime_verified",
            success_count=2,
            last_used_at=recent,
        )
    )

    match = cat.find_best_match("monsters JSON 데이터를 markdown 표로 정리해줘")
    assert match is None, "loose token overlap should not surface an unrelated tool"


def test_find_best_match_prefers_verified_relevant_tool(tmp_path):
    cat = ToolCatalog(state_dir=tmp_path)
    recent = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    cat.save_generated(
        SavedToolSpec(
            name="csv_cleanup",
            description="csv rows dedupe and sort by date",
            input_summary="csv file",
            output_summary="cleaned csv",
            code="print('ok')",
            tags=["csv", "dedupe", "sort"],
            verification_status="runtime_verified",
            success_count=4,
            last_used_at=recent,
        )
    )
    cat.save_generated(
        SavedToolSpec(
            name="echo_helper",
            description="echo plain text",
            input_summary="text",
            output_summary="text",
            code="print('ok')",
            tags=["echo"],
            verification_status="runtime_verified",
            success_count=4,
            last_used_at=recent,
        )
    )

    match = cat.find_best_match("csv 파일 중복 제거하고 날짜순으로 정렬해줘")
    assert isinstance(match, ToolMatch)
    assert match.tool.name == "csv_cleanup"
    assert "csv" in match.matched_terms


def test_find_best_match_penalizes_failures_and_staleness(tmp_path):
    cat = ToolCatalog(state_dir=tmp_path)
    recent = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    old = "2000-01-01T00:00:00Z"
    cat.save_generated(
        SavedToolSpec(
            name="csv_cleanup_old",
            description="csv dedupe sort rows",
            input_summary="csv file",
            output_summary="cleaned csv",
            code="print('ok')",
            tags=["csv", "dedupe", "sort"],
            verification_status="runtime_verified",
            success_count=0,
            failure_count=5,
            last_used_at=old,
        )
    )
    cat.save_generated(
        SavedToolSpec(
            name="csv_cleanup_new",
            description="csv dedupe sort rows",
            input_summary="csv file",
            output_summary="cleaned csv",
            code="print('ok')",
            tags=["csv", "dedupe", "sort"],
            verification_status="runtime_verified",
            success_count=2,
            failure_count=0,
            last_used_at=recent,
        )
    )

    match = cat.find_best_match("csv dedupe sort rows")
    assert isinstance(match, ToolMatch)
    assert match.tool.name == "csv_cleanup_new"


def test_record_execution_updates_saved_tool_counters_and_last_run(tmp_path):
    cat = ToolCatalog(state_dir=tmp_path)
    cat.save_generated(
        SavedToolSpec(
            name="csv_cleanup",
            description="csv dedupe sort rows",
            input_summary="csv file",
            output_summary="cleaned csv",
            code="print('ok')",
            verification_status="runtime_verified",
            verification_details={"method": "create_tool_execution", "exit_status": 0},
        )
    )

    cat.record_execution(
        "csv_cleanup",
        ok=True,
        exit_status=0,
        timed_out=False,
        duration_sec=0.12,
        stdout_tail="ok",
        stderr_tail="",
        task_summary="csv cleanup run",
    )
    cat.record_execution(
        "csv_cleanup",
        ok=False,
        exit_status=1,
        timed_out=False,
        duration_sec=0.34,
        stdout_tail="",
        stderr_tail="ValueError",
        task_summary="csv cleanup run failed",
    )

    fresh = ToolCatalog(state_dir=tmp_path)
    loaded = fresh.lookup("csv_cleanup")
    assert loaded.success_count == 0
    assert loaded.failure_count == 0
    assert loaded.last_used_at == ""
    assert loaded.last_failure_at == ""

    manifest = json.loads((tmp_path / "tools" / "csv_cleanup" / "manifest.json").read_text())
    last_run = json.loads((tmp_path / "tools" / "csv_cleanup" / "last_run.json").read_text())
    assert "usage" not in manifest
    assert last_run["ok"] is False
    assert last_run["exit_status"] == 1


def test_summary_for_prompt_includes_best_match_hint(tmp_path):
    cat = ToolCatalog(state_dir=tmp_path)
    cat.save_generated(
        SavedToolSpec(
            name="csv_cleanup",
            description="csv rows dedupe and sort by date",
            input_summary="csv file",
            output_summary="cleaned csv",
            code="print('ok')",
            tags=["csv", "dedupe", "sort"],
            verification_status="runtime_verified",
        )
    )

    summary = cat.summary_for_prompt(current_task="csv 파일 중복 제거해줘")
    assert "Recommended reusable tool for current task" in summary
    assert "csv_cleanup" in summary


class FakeEmbeddingClient:
    model = "fake-embedding"

    def __init__(self, vectors: dict[str, list[float]], *, fail: bool = False) -> None:
        self.vectors = vectors
        self.fail = fail
        self.calls: list[str] = []

    def embed(self, text: str) -> list[float]:
        self.calls.append(text)
        if self.fail:
            raise RuntimeError("embedding unavailable")
        lowered = text.lower()
        for marker, vector in self.vectors.items():
            if marker.lower() in lowered:
                return vector
        return [0.0, 1.0]


def test_save_generated_writes_and_loads_embedding_cache(tmp_path):
    client = FakeEmbeddingClient({"csv_cleanup": [1.0, 0.0]})
    cat = ToolCatalog(state_dir=tmp_path, embedding_client=client)

    cat.save_generated(
        SavedToolSpec(
            name="csv_cleanup",
            description="dedupe and sort rows",
            input_summary="csv path",
            output_summary="cleaned csv",
            code="print('ok')",
            verification_status="runtime_verified",
        )
    )

    embedding_file = tmp_path / "tools" / "csv_cleanup" / "embedding.json"
    assert embedding_file.is_file()
    payload = json.loads(embedding_file.read_text())
    assert payload["model"] == "fake-embedding"
    assert payload["vector"] == [1.0, 0.0]
    assert payload["text_sha256"]

    fresh = ToolCatalog(state_dir=tmp_path)
    loaded = fresh.lookup("csv_cleanup")
    assert isinstance(loaded, SavedToolSpec)
    assert loaded.embedding_model == "fake-embedding"
    assert loaded.embedding_vector == [1.0, 0.0]
    assert loaded.embedding_text_sha256 == payload["text_sha256"]


def test_semantic_match_can_recommend_tool_without_token_overlap(tmp_path):
    client = FakeEmbeddingClient(
        {
            "avatar rig validation": [1.0, 0.0],
            "캐릭터 장비 점검": [1.0, 0.0],
            "csv cleanup": [0.0, 1.0],
        }
    )
    cat = ToolCatalog(state_dir=tmp_path, embedding_client=client)
    cat.save_generated(
        SavedToolSpec(
            name="avatar_setup_checklist",
            description="avatar rig validation",
            input_summary="metadata json",
            output_summary="setup checklist",
            code="print('ok')",
            verification_status="runtime_verified",
        )
    )
    cat.save_generated(
        SavedToolSpec(
            name="csv_cleanup",
            description="csv cleanup",
            input_summary="csv path",
            output_summary="cleaned rows",
            code="print('ok')",
            verification_status="runtime_verified",
        )
    )

    match = cat.find_best_match("캐릭터 장비 점검해줘")

    assert isinstance(match, ToolMatch)
    assert match.tool.name == "avatar_setup_checklist"
    assert match.match_kind == "semantic"
    assert match.matched_terms == ["embedding"]

    summary = cat.summary_for_prompt(current_task="캐릭터 장비 점검해줘")
    assert "Recommended reusable tool for current task" in summary
    assert "semantic_similarity" in summary


def test_embedding_failure_falls_back_to_token_heuristic(tmp_path):
    client = FakeEmbeddingClient({}, fail=True)
    cat = ToolCatalog(state_dir=tmp_path, embedding_client=client)
    cat.save_generated(
        SavedToolSpec(
            name="csv_cleanup",
            description="csv dedupe sort rows",
            input_summary="csv file",
            output_summary="cleaned csv",
            code="print('ok')",
            tags=["csv", "dedupe", "sort"],
            verification_status="runtime_verified",
            success_count=2,
        )
    )

    match = cat.find_best_match("csv dedupe sort rows")

    assert isinstance(match, ToolMatch)
    assert match.tool.name == "csv_cleanup"
    assert match.match_kind == "heuristic"


def test_corrupt_embedding_cache_is_ignored(tmp_path):
    cat = ToolCatalog(state_dir=tmp_path)
    cat.save_generated(
        SavedToolSpec(
            name="csv_cleanup",
            description="csv dedupe sort rows",
            input_summary="csv file",
            output_summary="cleaned csv",
            code="print('ok')",
        )
    )
    embedding_file = tmp_path / "tools" / "csv_cleanup" / "embedding.json"
    embedding_file.write_text("{not json", encoding="utf-8")

    fresh = ToolCatalog(state_dir=tmp_path)
    loaded = fresh.lookup("csv_cleanup")

    assert isinstance(loaded, SavedToolSpec)
    assert loaded.embedding_vector == []


def test_save_without_embedding_removes_stale_embedding_cache(tmp_path):
    client = FakeEmbeddingClient({"csv_cleanup": [1.0, 0.0]})
    cat = ToolCatalog(state_dir=tmp_path, embedding_client=client)
    cat.save_generated(
        SavedToolSpec(
            name="csv_cleanup",
            description="csv dedupe sort rows",
            input_summary="csv file",
            output_summary="cleaned csv",
            code="print('ok')",
        )
    )
    embedding_file = tmp_path / "tools" / "csv_cleanup" / "embedding.json"
    assert embedding_file.is_file()

    no_embedding_catalog = ToolCatalog(state_dir=tmp_path)
    no_embedding_catalog.save_generated(
        SavedToolSpec(
            name="csv_cleanup",
            description="csv dedupe sort rows updated",
            input_summary="csv file",
            output_summary="cleaned csv",
            code="print('ok')",
        )
    )

    assert not embedding_file.exists()
