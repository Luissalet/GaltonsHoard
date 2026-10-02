"""Killing and naming child processes. The process tree handling itself (own process group, no console window on Windows, ``taskkill /T``,
``SIGKILL`` of the group) is the shared ``hoard_link.proc``; this module keeps the two short names the rest of the app and its tests use."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Optional

from .hoard_link import proc as shared

IS_WINDOWS = sys.platform.startswith("win")


def kill_tree(proc: Optional[subprocess.Popen], *, wait_s: float = 5.0) -> None:
    """Kill the process and everything it started, at once (a model server or a checker has nothing to save). Never raises."""
    if proc is None:
        return
    try:
        shared.kill_tree(proc, grace_s=0)
    except Exception:  # noqa: BLE001 - stopping a child must never break the caller
        pass
    try:
        proc.wait(wait_s)
    except (subprocess.TimeoutExpired, OSError):
        pass


def kill_pid_tree(pid: int) -> None:
    """Kill a process we only know by pid (a leftover of a crashed run)."""
    try:
        shared.kill_tree(int(pid), grace_s=0)
    except Exception:  # noqa: BLE001
        pass


def process_name(pid: int) -> str:
    """The executable name of a running pid, lowercase (its command line on Linux); empty when it does not exist or cannot be read."""
    try:
        if IS_WINDOWS:
            out = shared.run(["tasklist", "/FI", f"PID eq {pid}", "/FO", "CSV", "/NH"], timeout=15).stdout or ""
            first = out.strip().splitlines()[0] if out.strip() else ""
            return first.split('","')[0].strip('"').lower() if first and not first.startswith("INFO") else ""
        cmdline = Path(f"/proc/{pid}/cmdline")
        if cmdline.exists():
            return cmdline.read_bytes().replace(b"\0", b" ").decode("utf-8", "replace").lower()
    except (OSError, subprocess.SubprocessError):
        pass
    return ""
