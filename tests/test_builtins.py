from __future__ import annotations

from pathlib import Path

import pytest

from adaptive_agent.builtins import (
    BUILTIN_TOOLS,
    BuiltinError,
    dispatch_builtin,
    inspect_file,
    list_files,
    read_text_file,
    register_builtins,
    write_text_file,
)
from adaptive_agent.catalog import ToolCatalog


def test_read_text_file(tmp_path: Path):
    p = tmp_path / "a.txt"
    p.write_text("hi", encoding="utf-8")
    out = read_text_file(workroot=tmp_path, path="a.txt")
    assert out["content"] == "hi"
    assert out["path"] == "a.txt"


def test_inspect_file_reports_utf8_text(tmp_path: Path):
    p = tmp_path / "a.txt"
    p.write_text("hello world", encoding="utf-8")
    out = inspect_file(workroot=tmp_path, path="a.txt")
    assert out["path"] == "a.txt"
    assert out["size_bytes"] == len("hello world".encode("utf-8"))
    assert out["suffix"] == ".txt"
    assert out["is_utf8_text"] is True
    assert out["preview"] == "hello world"


def test_inspect_file_reports_non_utf8_file(tmp_path: Path):
    p = tmp_path / "bad.bin"
    p.write_bytes(b"\xff\xfe\x00\x01")
    out = inspect_file(workroot=tmp_path, path="bad.bin")
    assert out["path"] == "bad.bin"
    assert out["size_bytes"] == 4
    assert out["suffix"] == ".bin"
    assert out["is_utf8_text"] is False
    assert out["preview"] == ""


def test_read_text_file_rejects_traversal(tmp_path: Path):
    with pytest.raises(BuiltinError):
        read_text_file(workroot=tmp_path, path="../etc/passwd")


def test_read_text_file_rejects_absolute(tmp_path: Path):
    with pytest.raises(BuiltinError):
        read_text_file(workroot=tmp_path, path="/etc/hosts")


def test_read_text_file_missing(tmp_path: Path):
    with pytest.raises(BuiltinError):
        read_text_file(workroot=tmp_path, path="missing.txt")


def test_read_text_file_rejects_non_utf8_file(tmp_path: Path):
    p = tmp_path / "bad.bin"
    p.write_bytes(b"\xff\xfe\x00\x01")
    with pytest.raises(BuiltinError, match="UTF-8"):
        read_text_file(workroot=tmp_path, path="bad.bin")


def test_write_text_file(tmp_path: Path):
    result = write_text_file(workroot=tmp_path, path="out.txt", content="yo")
    assert (tmp_path / "out.txt").read_text() == "yo"
    assert result["bytes_written"] == 2


def test_write_text_file_rejects_traversal(tmp_path: Path):
    with pytest.raises(BuiltinError):
        write_text_file(workroot=tmp_path, path="../escape", content="x")


def test_list_files(tmp_path: Path):
    (tmp_path / "a.txt").write_text("x")
    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / "b.txt").write_text("y")
    result = list_files(workroot=tmp_path, path=".")
    names = {e["name"] for e in result["entries"]}
    assert "a.txt" in names
    assert "sub" in names


def test_list_files_subpath(tmp_path: Path):
    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / "b.txt").write_text("y")
    result = list_files(workroot=tmp_path, path="sub")
    names = {e["name"] for e in result["entries"]}
    assert names == {"b.txt"}


def test_register_builtins_adds_to_catalog(tmp_path: Path):
    cat = ToolCatalog(state_dir=None)
    register_builtins(cat)
    names = {b.name for b in cat.list_builtins()}
    assert names == {"inspect_file", "read_text_file", "write_text_file", "list_files"}


def test_dispatch_unknown_raises(tmp_path: Path):
    with pytest.raises(BuiltinError):
        dispatch_builtin("nope", {}, workroot=tmp_path)


def test_dispatch_read_text_file(tmp_path: Path):
    (tmp_path / "a.txt").write_text("hi")
    out = dispatch_builtin("read_text_file", {"path": "a.txt"}, workroot=tmp_path)
    assert out["content"] == "hi"


def test_dispatch_inspect_file(tmp_path: Path):
    (tmp_path / "a.txt").write_text("hi", encoding="utf-8")
    out = dispatch_builtin("inspect_file", {"path": "a.txt"}, workroot=tmp_path)
    assert out["is_utf8_text"] is True
    assert out["preview"] == "hi"


def test_dispatch_missing_argument_raises(tmp_path: Path):
    with pytest.raises(BuiltinError):
        dispatch_builtin("read_text_file", {}, workroot=tmp_path)

