"""Background work in two lanes, one job at a time each: Galton's job kinds on top of the shared Hoard Link ``LaneScheduler``.

* ``gpu``: runs (``run``, ref = run id) and the judge phase (``judge``). One at a time because a run owns the GPUs it leased.
* ``io``: ``refresh`` (rediscover models, every ``watch.interval_h`` hours), ``watch`` (probe shared servers, queue a quick run when it disturbs nobody,
  every minute) and ``housekeeping`` (hourly).

Runs started by a person or an app are never held back by ``scheduler.paused``; only the automatic jobs are. A job that raises is logged, recorded as
an activity and never stops its lane. The lanes, the de-duplication, ``run_now`` (inline without running lanes) and the periodic due-checks are the
shared ones; this class only maps a job kind and a ref to a shared ``Job``.
"""

from __future__ import annotations

import time
from functools import partial
from typing import Any, Callable, Optional

from .hoard_link.lanes import Job, LaneScheduler

TICK_S = 15.0
WATCH_S = 60.0
HOUSEKEEPING_S = 3600.0
FIRST_REFRESH_S = 5.0
LANES = ("gpu", "io")
JOB_LANE = {"run": "gpu", "judge": "gpu", "refresh": "io", "watch": "io", "housekeeping": "io"}
AUTO = ("refresh", "watch", "housekeeping")


def _key(kind: str, ref: str) -> str:
    """The de-duplication key: the kind alone, or ``kind:ref`` (two runs may queue, the same run once)."""
    return f"{kind}:{ref}" if ref else kind


def _ref_of(kind: str, key: str) -> str:
    return key[len(kind) + 1:] if key.startswith(kind + ":") else ""


class Scheduler:
    def __init__(self, handlers: dict[str, Callable[[str], Any]], store: Any, *, clock: Callable[[], float] = time.time, enabled: bool = True,
                 paused: Callable[[], bool] = lambda: False, refresh_every_h: Callable[[], float] = lambda: 6.0, watch_enabled: Callable[[], bool] = lambda: True):
        self.handlers, self.store, self.clock, self.enabled, self.paused = handlers, store, clock, enabled, paused
        self.refresh_every_h, self.watch_enabled = refresh_every_h, watch_enabled
        self._lane = LaneScheduler({lane: 1 for lane in LANES}, enabled=enabled, paused=paused, on_error=self._failed, clock=clock, tick_s=TICK_S,
                                   name="galton")
        # the automatic jobs: refresh every watch.interval_h hours (first one 5 s after the start), the watch every minute, housekeeping hourly
        for kind, every, first, only_if in (("refresh", lambda: max(0.05, self.refresh_every_h()) * 3600, FIRST_REFRESH_S, None),
                                            ("watch", lambda: WATCH_S, FIRST_REFRESH_S + 10, self.watch_enabled),
                                            ("housekeeping", lambda: HOUSEKEEPING_S, 60.0, None)):
            self._lane.register(Job(kind, fn=partial(self._call, kind, ""), lane=JOB_LANE[kind], every_s=every, first_delay_s=first, enabled=only_if))

    # ------------------------------------------------------------------ lifecycle
    def _alive(self) -> bool:
        return self._lane._alive()

    def start(self) -> None:
        self._lane.start()

    def stop(self, timeout: float = 5.0) -> None:
        self._lane.stop(timeout)

    # ------------------------------------------------------------------ jobs
    def _call(self, kind: str, ref: str) -> Any:
        handler = self.handlers.get(kind)
        if handler is None:
            raise ValueError(f"no handler for job kind {kind}")
        return handler(ref)

    def _job(self, kind: str, ref: str, reason: str) -> Job:
        if kind not in JOB_LANE:
            raise ValueError(f"unknown job kind {kind!r}")
        return Job(kind, key=_key(kind, ref), fn=partial(self._call, kind, ref), lane=JOB_LANE[kind], reason=reason)

    def _failed(self, job: Job, error: BaseException) -> None:
        try:
            self.store.add_activity(job.kind, _ref_of(job.kind, job.dedupe_key), False, 0, f"{type(error).__name__}: {error}")
        except Exception:  # noqa: BLE001 - recording the failure must not break the lane
            pass

    def submit(self, kind: str, ref: str = "", reason: str = "manual") -> Optional[Job]:
        """Queue a job; ``None`` when the same kind and ref is already queued or running."""
        return self._lane.submit(self._job(kind, ref, reason))

    def run_now(self, kind: str, ref: str = "", timeout: float = 240.0) -> Any:
        """Run a job and wait for its result. Without running lanes (tests, MCP-only mode) it runs inline."""
        return self._lane.run_now(self._job(kind, ref, "manual"), timeout)

    def enqueue_due(self, now: float) -> int:
        return self._lane.enqueue_due(now)

    # ------------------------------------------------------------------ view
    @property
    def last(self) -> dict[str, float]:
        return self._lane.last

    def status(self) -> dict[str, Any]:
        shared = self._lane.status()

        def current(lane: str) -> Optional[dict[str, str]]:
            jobs = shared["lanes"][lane]["current"]
            return {"kind": jobs[0]["kind"], "ref": _ref_of(jobs[0]["kind"], jobs[0]["key"]), "reason": jobs[0]["reason"]} if jobs else None

        return {"enabled": self.enabled, "running": shared["running"], "paused": bool(self.paused()),
                "lanes": {lane: {"queue": shared["lanes"][lane]["queue"], "current": current(lane)} for lane in LANES},
                "last_tick_ts": shared["last_tick_ts"], **{f"last_{k}_ts": self.last.get(k) for k in AUTO}, "jobs_done": shared["jobs_done"]}


