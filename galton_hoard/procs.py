"""Child processes that must be killable as a tree and must not open a console window (Windows) or outlive the app."""

from __future__ import annotations

import os
import signal
import subprocess
import sys
from pathlib import Path
from typing import Any, Optional

IS_WINDOWS = sys.platform.startswith("win")


def popen_kwargs() -> dict[str, Any]:
    """Keyword arguments for ``subprocess.Popen`` so the child has its own process group and, on Windows, no window."""
    if IS_WINDOWS:
        return {"creationflags": getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0) | getattr(subprocess, "CREATE_NO_WINDOW", 0)}
    return {"start_new_session": True}


def kill_tree(proc: Optional[subprocess.Popen], *, wait_s: float = 5.0) -> None:
    """Kill the process and everything it started. Never raises."""
    if proc is None:
        return
    pid = proc.pid
    if proc.poll() is None:
        try:
            if IS_WINDOWS:
                subprocess.run(["taskkill", "/F", "/T", "/PID", str(pid)], capture_output=True, timeout=15,
                               creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0), stdin=subprocess.DEVNULL)
            else:
                os.killpg(os.getpgid(pid), signal.SIGKILL)
        except (OSError, subprocess.SubprocessError, ProcessLookupError):
            pass
        try:
            proc.kill()
        except OSError:
            pass
    try:
        proc.wait(wait_s)
    except (subprocess.TimeoutExpired, OSError):
        pass


def kill_pid_tree(pid: int) -> None:
    """Kill a process we only know by pid (a leftover of a crashed run)."""
    try:
        if IS_WINDOWS:
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(pid)], capture_output=True, timeout=15,
                           creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0), stdin=subprocess.DEVNULL)
        else:
            os.killpg(os.getpgid(pid), signal.SIGKILL)
    except (OSError, subprocess.SubprocessError, ProcessLookupError):
        pass


def process_name(pid: int) -> str:
    """The executable name of a running pid, lowercase; empty when it does not exist or cannot be read."""
    try:
        if IS_WINDOWS:
            out = subprocess.run(["tasklist", "/FI", f"PID eq {pid}", "/FO", "CSV", "/NH"], capture_output=True, text=True, timeout=15,
                                 creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0), stdin=subprocess.DEVNULL).stdout
            first = out.strip().splitlines()[0] if out.strip() else ""
            return first.split('","')[0].strip('"').lower() if first and not first.startswith("INFO") else ""
        cmdline = Path(f"/proc/{pid}/cmdline")
        if cmdline.exists():
            return cmdline.read_bytes().replace(b"\0", b" ").decode("utf-8", "replace").lower()
    except (OSError, subprocess.SubprocessError):
        pass
    return ""
