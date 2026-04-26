from __future__ import annotations

import subprocess
import sys


def test_help_exits_zero():
    result = subprocess.run(
        [sys.executable, "-m", "adaptive_agent", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "Adaptive AI Agent CLI" in result.stdout


def test_run_subcommand_listed_in_help():
    result = subprocess.run(
        [sys.executable, "-m", "adaptive_agent", "--help"],
        capture_output=True,
        text=True,
    )
    assert "run" in result.stdout
    assert "repl" in result.stdout


def test_console_script_help_exits_zero():
    result = subprocess.run(
        ["adaptive-agent", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "Adaptive AI Agent CLI" in result.stdout
