"""The processor and memory of this computer, for running a small model without a GPU: physical cores, threads to give llama-server, free RAM.

Every function answers ``None`` (or a safe default) when the system does not say; nothing here raises.
"""

from __future__ import annotations

import ctypes
import functools
import os
import subprocess
import sys
from typing import Optional

#: cores left to the owner's other programs while a model runs on the CPU
RESERVED_CORES = 2
MIN_THREADS = 2


@functools.lru_cache(maxsize=1)
def physical_cores() -> int:
    """Physical cores (not hardware threads): psutil when it is installed, else the system's own report, else half of the logical processors."""
    try:
        import psutil  # optional: not a requirement of the app
        found = psutil.cpu_count(logical=False)
        if found:
            return int(found)
    except Exception:  # noqa: BLE001
        pass
    found = _cores_windows() if sys.platform == "win32" else _cores_proc()
    if found:
        return found
    logical = os.cpu_count() or 1
    return max(1, logical // 2) if logical > 2 else logical


def _cores_proc() -> int:
    try:
        text = open("/proc/cpuinfo", encoding="utf-8", errors="replace").read()
    except OSError:
        return 0
    cores: set[tuple[str, str]] = set()
    package = ""
    for line in text.splitlines():
        key, _, value = line.partition(":")
        key, value = key.strip(), value.strip()
        if key == "physical id":
            package = value
        elif key == "core id":
            cores.add((package, value))
    return len(cores)


def _cores_windows() -> int:
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    try:
        proc = subprocess.run(["powershell", "-NoProfile", "-Command", "(Get-CimInstance Win32_Processor | Measure-Object -Property NumberOfCores -Sum).Sum"],
                              capture_output=True, text=True, stdin=subprocess.DEVNULL, timeout=8, creationflags=flags)
        return int((proc.stdout or "").strip() or 0)
    except (OSError, subprocess.SubprocessError, ValueError):
        return 0


def cpu_threads(cores: Optional[int] = None) -> int:
    """Threads for a model on the CPU: physical cores minus two, at least two."""
    return max(MIN_THREADS, int(cores if cores is not None else physical_cores()) - RESERVED_CORES)


def ram_free_mb() -> Optional[int]:
    """MiB of memory a program can still use, or ``None`` when the system does not say."""
    try:
        import psutil
        return int(psutil.virtual_memory().available // (1024 * 1024))
    except Exception:  # noqa: BLE001
        pass
    if sys.platform == "win32":
        return _ram_windows()
    try:
        for line in open("/proc/meminfo", encoding="utf-8", errors="replace"):
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) // 1024
    except (OSError, ValueError, IndexError):
        pass
    return None


def _ram_windows() -> Optional[int]:
    class Status(ctypes.Structure):
        _fields_ = [("length", ctypes.c_ulong), ("load", ctypes.c_ulong), ("total", ctypes.c_ulonglong), ("avail", ctypes.c_ulonglong),
                    ("total_page", ctypes.c_ulonglong), ("avail_page", ctypes.c_ulonglong), ("total_virtual", ctypes.c_ulonglong),
                    ("avail_virtual", ctypes.c_ulonglong), ("avail_extended", ctypes.c_ulonglong)]
    try:
        status = Status()
        status.length = ctypes.sizeof(Status)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):  # type: ignore[attr-defined]
            return int(status.avail // (1024 * 1024))
    except (AttributeError, OSError):
        pass
    return None
