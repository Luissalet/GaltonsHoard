"""Canonical job events for runs, for the family hub: ``galton.job.queued|started|progress|done|failed|cancelled``.

Data of every event: ``{job_id (the run id), title (the run label), kind: "run", progress (0..1), gpu (first GPU index or None),
eta_s, url, error}``. The hub's Work tab and its rules read these; the old ``galton.run.done`` is mapped onto ``galton.job.done`` by the hub,
so nothing else is emitted under the old name. Progress events are throttled (``min_interval_s``) so a long run does not flood the bus.
Events are hints: a failing ``emit`` never reaches the run.
"""

from __future__ import annotations

import threading
import time
from typing import Any, Callable, Optional

KIND = "run"


class RunJobEvents:
    def __init__(self, emit: Callable[[str, dict[str, Any]], None], *, clock: Callable[[], float] = time.time,
                 base_url: Callable[[], str] = lambda: "", min_interval_s: float = 5.0):
        self._emit, self._clock, self._base_url, self.min_interval_s = emit, clock, base_url, float(min_interval_s)
        self._lock = threading.Lock()
        self._started: dict[str, float] = {}
        self._last_progress: dict[str, float] = {}
        self._gpu: dict[str, Optional[int]] = {}

    # ------------------------------------------------------------------ helpers
    def url(self, run_id: str) -> str:
        base = (self._base_url() or "").rstrip("/")
        return f"{base}/#/ejecutar/{run_id}" if base else ""

    def _data(self, run: dict[str, Any], **extra: Any) -> dict[str, Any]:
        rid = run["id"]
        data: dict[str, Any] = {"job_id": rid, "title": str(run.get("label") or rid)[:120], "kind": KIND, "url": self.url(rid)}
        if run.get("source"):
            data["source"] = run["source"]
        gpu = self._gpu.get(rid)
        if gpu is not None:
            data["gpu"] = gpu
        data.update({k: v for k, v in extra.items() if v is not None})
        return data

    def _send(self, name: str, data: dict[str, Any]) -> None:
        try:
            self._emit(f"galton.job.{name}", data)
        except Exception:  # noqa: BLE001 — events are hints; the database is the truth
            pass

    # ------------------------------------------------------------------ lifecycle
    def queued(self, run: dict[str, Any]) -> None:
        self._send("queued", self._data(run, progress=0.0))

    def started(self, run: dict[str, Any]) -> None:
        with self._lock:
            self._started[run["id"]] = self._clock()
            self._last_progress[run["id"]] = self._clock()
        self._send("started", self._data(run, progress=0.0))

    def set_gpu(self, run_id: str, gpus: list[int]) -> None:
        with self._lock:
            self._gpu[run_id] = int(gpus[0]) if gpus else None

    def progress(self, run: dict[str, Any], totals: Callable[[], tuple[int, int]]) -> None:
        """Emit a progress event unless one went out less than ``min_interval_s`` ago. ``totals()`` returns ``(done, total)``
        and is only called when an event will really be sent (it reads the database)."""
        rid, now = run["id"], self._clock()
        with self._lock:
            if now - self._last_progress.get(rid, 0.0) < self.min_interval_s:
                return
            self._last_progress[rid] = now
            began = self._started.get(rid, now)
        try:
            done, total = totals()
        except Exception:  # noqa: BLE001
            return
        if total <= 0:
            return
        fraction = max(0.0, min(1.0, done / total))
        eta = None
        if 0.02 < fraction < 1.0 and now > began:
            eta = int((now - began) * (1 - fraction) / fraction)
        self._send("progress", self._data(run, progress=round(fraction, 3), eta_s=eta))

    def finished(self, run: dict[str, Any], state: str, error: str = "") -> None:
        name = {"done": "done", "failed": "failed", "cancelled": "cancelled"}.get(state)
        if name is None:
            return
        data = self._data(run, progress=1.0 if state == "done" else None, error=(error[:300] or None) if state == "failed" else None)
        with self._lock:
            self._started.pop(run["id"], None)
            self._last_progress.pop(run["id"], None)
            self._gpu.pop(run["id"], None)
        self._send(name, data)
