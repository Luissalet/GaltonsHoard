"""Background work in two lanes, one job at a time each.

* ``gpu``: runs (``run``, ref = run id) and the judge phase (``judge``). One at a time because a run owns the GPUs it leased.
* ``io``: ``refresh`` (rediscover models, every ``watch.interval_h`` hours), ``watch`` (probe shared servers, queue a quick run when it disturbs nobody,
  every minute) and ``housekeeping`` (hourly).

Runs started by a person or an app are never held back by ``scheduler.paused``; only the automatic jobs are. A job that raises is logged and never
stops its lane.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

log = logging.getLogger("galton.scheduler")

TICK_S = 15.0
WATCH_S = 60.0
HOUSEKEEPING_S = 3600.0
FIRST_REFRESH_S = 5.0
LANES = ("gpu", "io")
JOB_LANE = {"run": "gpu", "judge": "gpu", "refresh": "io", "watch": "io", "housekeeping": "io"}
AUTO = ("refresh", "watch", "housekeeping")


@dataclass
class Job:
    kind: str
    ref: str = ""
    reason: str = "schedule"
    done: threading.Event = field(default_factory=threading.Event)
    result: Any = None
    error: str = ""


class Scheduler:
    def __init__(self, handlers: dict[str, Callable[[str], Any]], store: Any, *, clock: Callable[[], float] = time.time, enabled: bool = True,
                 paused: Callable[[], bool] = lambda: False, refresh_every_h: Callable[[], float] = lambda: 6.0, watch_enabled: Callable[[], bool] = lambda: True):
        self.handlers, self.store, self.clock, self.enabled, self.paused = handlers, store, clock, enabled, paused
        self.refresh_every_h, self.watch_enabled = refresh_every_h, watch_enabled
        self._queues: dict[str, "queue.Queue[Job]"] = {lane: queue.Queue() for lane in LANES}
        self._pending: set[tuple[str, str]] = set()
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._threads: dict[str, threading.Thread] = {}
        self._current: dict[str, Optional[Job]] = {lane: None for lane in LANES}
        self._started_at = 0.0
        self.last_tick_ts: Optional[float] = None
        self.last: dict[str, Optional[float]] = {k: None for k in AUTO}
        self.jobs_done = 0

    def _alive(self) -> bool:
        return any(t.is_alive() for t in self._threads.values())

    def start(self) -> None:
        if self._alive():
            return
        self._stop.clear()
        self._started_at = self.clock()
        for lane in LANES:
            thread = threading.Thread(target=self._loop, args=(lane,), name=f"galton-{lane}", daemon=True)
            self._threads[lane] = thread
            thread.start()

    def stop(self, timeout: float = 5.0) -> None:
        self._stop.set()
        for thread in self._threads.values():
            thread.join(timeout)

    def status(self) -> dict[str, Any]:
        def view(job: Optional[Job]) -> Optional[dict[str, str]]:
            return {"kind": job.kind, "ref": job.ref, "reason": job.reason} if job else None
        return {"enabled": self.enabled, "running": self._alive(), "paused": bool(self.paused()),
                "lanes": {lane: {"queue": self._queues[lane].qsize(), "current": view(self._current[lane])} for lane in LANES},
                "last_tick_ts": self.last_tick_ts, **{f"last_{k}_ts": v for k, v in self.last.items()}, "jobs_done": self.jobs_done}

    def submit(self, kind: str, ref: str = "", reason: str = "manual") -> Optional[Job]:
        key = (kind, ref)
        with self._lock:
            if key in self._pending:
                return None
            self._pending.add(key)
        job = Job(kind, ref, reason)
        self._queues[JOB_LANE[kind]].put(job)
        return job

    def run_now(self, kind: str, ref: str = "", timeout: float = 240.0) -> Any:
        """Run a job and wait for its result. Without running lanes (tests, MCP-only mode) it runs inline."""
        if not self._alive():
            return self._execute(Job(kind, ref, "inline"))
        job = self.submit(kind, ref)
        if job is None:
            return {"queued": True, "note": "already queued"}
        if not job.done.wait(timeout):
            return {"queued": True, "note": "still running; ask again shortly"}
        if job.error:
            raise RuntimeError(job.error)
        return job.result

    def _execute(self, job: Job) -> Any:
        if job.kind in self.last:
            self.last[job.kind] = self.clock()
        handler = self.handlers.get(job.kind)
        if handler is None:
            raise ValueError(f"no handler for job kind {job.kind}")
        return handler(job.ref)

    def _loop(self, lane: str) -> None:
        while not self._stop.is_set():
            try:
                job = self._queues[lane].get(timeout=0.2)
            except queue.Empty:
                job = None
            if job is not None:
                self._run(job, lane)
                continue
            if lane != "io":
                continue
            now = self.clock()
            if self.last_tick_ts is None or now - self.last_tick_ts >= TICK_S:
                self.last_tick_ts = now
                if self.enabled and not self.paused():
                    try:
                        self.enqueue_due(now)
                    except Exception:  # noqa: BLE001
                        log.exception("scheduler tick failed")

    def _run(self, job: Job, lane: str) -> None:
        self._current[lane] = job
        try:
            job.result = self._execute(job)
        except Exception as error:  # noqa: BLE001
            job.error = f"{type(error).__name__}: {error}"
            log.warning("job %s %s failed: %s", job.kind, job.ref, job.error)
            try:
                self.store.add_activity(job.kind, job.ref, False, 0, job.error)
            except Exception:  # noqa: BLE001
                pass
        finally:
            with self._lock:
                self._pending.discard((job.kind, job.ref))
            self._current[lane] = None
            self.jobs_done += 1
            job.done.set()

    def _due(self, kind: str, now: float, every_s: float, first_after_s: float = 0.0) -> bool:
        last = self.last[kind]
        if last is None:
            return now - self._started_at >= first_after_s
        return now - last >= every_s

    def enqueue_due(self, now: float) -> int:
        n = 0
        if self._due("refresh", now, max(0.05, self.refresh_every_h()) * 3600, FIRST_REFRESH_S):
            n += bool(self.submit("refresh", "", "schedule"))
            self.last["refresh"] = now
        if self.watch_enabled() and self._due("watch", now, WATCH_S, FIRST_REFRESH_S + 10):
            n += bool(self.submit("watch", "", "schedule"))
            self.last["watch"] = now
        if self._due("housekeeping", now, HOUSEKEEPING_S, 60):
            n += bool(self.submit("housekeeping", "", "schedule"))
            self.last["housekeeping"] = now
        return n
