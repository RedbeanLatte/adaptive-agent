from __future__ import annotations

import os
import selectors
import signal
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

from .policy import _PROTECTED_WRITE_PARTS

DEFAULT_TIMEOUT_SEC = 15.0
DEFAULT_MAX_OUTPUT_BYTES = 64 * 1024  # 64 KiB per stream
_READ_CHUNK_BYTES = 8192
_OUTPUT_CAP_GRACE_SEC = 0.1


def _runtime_guard_source(workdir: Path) -> str:
    """Return Python source that blocks generated-tool writes outside workdir.

    This is a runtime guard, not container or VM isolation. It installs a
    Python audit-hook guard for the generated-tool threat model in this
    prototype: deny filesystem mutation outside the workroot and deny mutation
    of runtime state directories.
    """
    root = str(Path(workdir).resolve())
    protected = repr(tuple(sorted(_PROTECTED_WRITE_PARTS)))
    return f'''
# --- adaptive-agent generated-tool filesystem guard ---
def _adaptive_agent_install_fs_guard():
    import os
    import sys
    from pathlib import Path

    _ROOT = Path({root!r}).resolve()
    _PROTECTED = set({protected})

    def _resolve_path(value):
        if value is None or isinstance(value, int):
            return None
        try:
            return Path(os.fsdecode(value)).resolve()
        except Exception:
            return None

    def _check_path(value, operation):
        target = _resolve_path(value)
        if target is None:
            return
        try:
            relative = target.relative_to(_ROOT)
        except ValueError as exc:
            raise PermissionError(f"{{operation}} outside workroot blocked: {{target}}") from exc
        if any(part in _PROTECTED for part in relative.parts):
            raise PermissionError(f"{{operation}} protected path blocked: {{target}}")

    def _is_write_open(mode, flags):
        mode_text = "" if mode is None else str(mode)
        if any(flag in mode_text for flag in ("w", "a", "x", "+")):
            return True
        try:
            flags_int = int(flags)
        except Exception:
            return False
        write_flags = os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND
        return bool(flags_int & write_flags)

    def _hook(event, args):
        if event == "open":
            path = args[0] if len(args) > 0 else None
            mode = args[1] if len(args) > 1 else None
            flags = args[2] if len(args) > 2 else None
            if _is_write_open(mode, flags):
                _check_path(path, "write")
            return

        if event in {{"os.mkdir", "os.rmdir", "os.remove", "os.unlink", "os.chmod", "os.chown", "os.utime", "os.truncate"}}:
            if args:
                _check_path(args[0], event)
            return

        if event in {{"os.rename", "os.replace", "os.link"}}:
            if len(args) > 0:
                _check_path(args[0], event)
            if len(args) > 1:
                _check_path(args[1], event)
            return

    sys.addaudithook(_hook)

_adaptive_agent_install_fs_guard()
# --- end adaptive-agent generated-tool filesystem guard ---
'''.lstrip()


@dataclass
class RuntimeGuardResult:
    exit_status: int
    stdout: str
    stderr: str
    timed_out: bool
    duration_sec: float
    output_truncated: bool = False

    @property
    def ok(self) -> bool:
        return self.exit_status == 0 and not self.timed_out and not self.output_truncated

    def compact_summary(self, max_chars: int = 600) -> str:
        tail = (
            "[timed_out=true] "
            if self.timed_out
            else f"exit_status={self.exit_status} "
        )
        if self.output_truncated:
            tail = "[output_truncated=true] " + tail
        err = self.stderr.strip().splitlines()
        # keep only the last few stderr lines - those are usually the traceback tail
        err_tail = "\n".join(err[-8:])
        body = err_tail if err_tail else self.stdout.strip()[-max_chars:]
        if len(body) > max_chars:
            body = body[-max_chars:]
        return f"{tail}{body}"


def _truncate(data: bytes, limit: int) -> str:
    text = data.decode("utf-8", errors="replace")
    if len(data) <= limit:
        return text
    head = data[:limit].decode("utf-8", errors="replace")
    return head + f"\n...[truncated to {limit} bytes]"


def _decode_limited(data: bytearray, *, truncated: bool, limit: int) -> str:
    text = bytes(data).decode("utf-8", errors="replace")
    if truncated:
        text += f"\n...[truncated to {limit} bytes]"
    return text


def _append_limited(target: bytearray, chunk: bytes, limit: int) -> bool:
    remaining = limit - len(target)
    if remaining > 0:
        target.extend(chunk[:remaining])
    return len(chunk) > max(remaining, 0)


def _terminate_process_tree(proc: subprocess.Popen[bytes]) -> None:
    if proc.poll() is not None:
        return
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except Exception:
        try:
            proc.kill()
        except OSError:
            pass


def run_guarded_python(
    code: str,
    *,
    workdir: Path,
    args: list[str] | None = None,
    timeout: float = DEFAULT_TIMEOUT_SEC,
    max_output_bytes: int = DEFAULT_MAX_OUTPUT_BYTES,
) -> RuntimeGuardResult:
    """Execute a Python snippet in a subprocess with timeout and streaming output caps.

    The snippet is written to a temp file inside ``workdir`` and run with the
    host interpreter. The child process's cwd is ``workdir``.
    """
    workdir = Path(workdir)
    if not workdir.is_dir():
        raise FileNotFoundError(f"workdir does not exist: {workdir}")

    args = list(args or [])
    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".py",
        dir=str(workdir),
        delete=False,
        encoding="utf-8",
    ) as tf:
        tf.write(_runtime_guard_source(workdir))
        tf.write("\n")
        tf.write(code)
        script_path = Path(tf.name)

    start = time.monotonic()
    timed_out = False
    stdout_truncated = False
    stderr_truncated = False
    output_limit_exceeded = False
    killed_for_output_cap = False
    output_cap_deadline: float | None = None
    stdout_buffer = bytearray()
    stderr_buffer = bytearray()

    proc: subprocess.Popen[bytes] | None = None
    selector: selectors.BaseSelector | None = None
    try:
        proc = subprocess.Popen(
            [sys.executable, str(script_path), *args],
            cwd=str(workdir),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
        )
        selector = selectors.DefaultSelector()
        assert proc.stdout is not None
        assert proc.stderr is not None
        selector.register(proc.stdout, selectors.EVENT_READ, "stdout")
        selector.register(proc.stderr, selectors.EVENT_READ, "stderr")

        deadline = start + timeout
        while selector.get_map():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                timed_out = True
                _terminate_process_tree(proc)
                break
            if output_cap_deadline is not None and time.monotonic() >= output_cap_deadline:
                if proc.poll() is None:
                    killed_for_output_cap = True
                    _terminate_process_tree(proc)
                break

            select_timeout = min(0.05, remaining)
            if output_cap_deadline is not None:
                select_timeout = min(select_timeout, max(0.0, output_cap_deadline - time.monotonic()))
            events = selector.select(timeout=select_timeout)
            if not events:
                if proc.poll() is not None:
                    # Pipes should become readable at EOF shortly after process exit.
                    # Keep looping until they unregister, but do not block here.
                    continue
                continue

            for key, _ in events:
                stream = key.fileobj
                name = key.data
                try:
                    chunk = os.read(stream.fileno(), _READ_CHUNK_BYTES)
                except OSError:
                    chunk = b""
                if not chunk:
                    try:
                        selector.unregister(stream)
                    except Exception:
                        pass
                    try:
                        stream.close()
                    except OSError:
                        pass
                    continue

                if name == "stdout":
                    if _append_limited(stdout_buffer, chunk, max_output_bytes):
                        stdout_truncated = True
                        output_limit_exceeded = True
                else:
                    if _append_limited(stderr_buffer, chunk, max_output_bytes):
                        stderr_truncated = True
                        output_limit_exceeded = True

                if output_limit_exceeded and output_cap_deadline is None:
                    output_cap_deadline = time.monotonic() + _OUTPUT_CAP_GRACE_SEC

        if proc is not None:
            try:
                proc.wait(timeout=1.0)
            except subprocess.TimeoutExpired:
                _terminate_process_tree(proc)
                try:
                    proc.wait(timeout=1.0)
                except subprocess.TimeoutExpired:
                    pass
        exit_status = proc.returncode if proc is not None and proc.returncode is not None else -1
        if killed_for_output_cap and not timed_out:
            exit_status = -2
    finally:
        if selector is not None:
            selector.close()
        try:
            script_path.unlink(missing_ok=True)
        except OSError:
            pass

    duration = time.monotonic() - start
    return RuntimeGuardResult(
        exit_status=exit_status,
        stdout=_decode_limited(
            stdout_buffer,
            truncated=stdout_truncated,
            limit=max_output_bytes,
        ),
        stderr=_decode_limited(
            stderr_buffer,
            truncated=stderr_truncated,
            limit=max_output_bytes,
        ),
        timed_out=timed_out,
        duration_sec=duration,
        output_truncated=stdout_truncated or stderr_truncated,
    )
