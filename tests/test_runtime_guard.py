from __future__ import annotations

from pathlib import Path

import pytest

from adaptive_agent.runtime_guard import run_guarded_python


def test_successful_execution_captures_stdout(tmp_path):
    result = run_guarded_python("print('hi')", workdir=tmp_path)
    assert result.exit_status == 0
    assert "hi" in result.stdout
    assert result.stderr == ""
    assert result.timed_out is False


def test_nonzero_exit_captured(tmp_path):
    code = "import sys; sys.stderr.write('oops'); sys.exit(2)"
    result = run_guarded_python(code, workdir=tmp_path)
    assert result.exit_status == 2
    assert "oops" in result.stderr


def test_exception_surfaces_in_stderr(tmp_path):
    code = "raise RuntimeError('nope')"
    result = run_guarded_python(code, workdir=tmp_path)
    assert result.exit_status != 0
    assert "RuntimeError" in result.stderr
    assert "nope" in result.stderr


def test_timeout_flag_set(tmp_path):
    code = "import time; time.sleep(5)"
    result = run_guarded_python(code, workdir=tmp_path, timeout=0.5)
    assert result.timed_out is True
    assert result.exit_status != 0


def test_stdout_is_capped(tmp_path):
    # Write far more than the cap, confirm we truncate
    code = "import sys; sys.stdout.write('x' * 1_000_000)"
    result = run_guarded_python(code, workdir=tmp_path, max_output_bytes=1024)
    assert len(result.stdout.encode("utf-8", errors="replace")) <= 1024 + 64  # cap + marker slack
    assert "truncated" in result.stdout.lower()


def test_workdir_is_cwd(tmp_path):
    code = "import os, sys; sys.stdout.write(os.getcwd())"
    result = run_guarded_python(code, workdir=tmp_path)
    assert result.exit_status == 0
    assert str(tmp_path.resolve()) in result.stdout


def test_args_are_available_on_sys_argv(tmp_path):
    code = "import sys, json; print(json.dumps(sys.argv[1:]))"
    result = run_guarded_python(code, workdir=tmp_path, args=["a", "b"])
    assert result.exit_status == 0
    assert '"a"' in result.stdout and '"b"' in result.stdout


def test_result_summary_is_compact(tmp_path):
    code = "import sys; sys.stderr.write('E' * 5000); sys.exit(1)"
    result = run_guarded_python(code, workdir=tmp_path, max_output_bytes=1024)
    summary = result.compact_summary()
    assert "exit_status=1" in summary
    assert len(summary) < 2048  # compact


def test_reject_when_workdir_missing():
    with pytest.raises(FileNotFoundError):
        run_guarded_python("print('x')", workdir=Path("/nonexistent/path/xyz"))


def test_syntax_error_returns_nonzero(tmp_path):
    result = run_guarded_python("def (", workdir=tmp_path)
    assert result.exit_status != 0
    assert "SyntaxError" in result.stderr
