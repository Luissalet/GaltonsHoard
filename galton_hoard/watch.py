"""Regression watch: notice new and changed models, measure them with the quick suite when that disturbs nobody, and say so when one got worse.

* A model that is new, or whose digest changed, or that was never measured on the quick suite, is a candidate.
* A candidate is measured automatically only when (a) it is already loaded on a shared server that has been idle for ``watch.idle_min`` minutes, or
  (b) it is a GGUF and an allowed GPU has room right now. The watch never queues for a GPU and never runs in quiet hours (default 01:00-08:00).
* After a run, each model's results are compared with the ones it had before its last digest change; a significantly worse result raises a notice
  and the family event ``galton.regression``.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Optional

import httpx

from . import placement, stats
from .errors import GaltonError
from .messages import text
from .servers import probe_openai
from .store import ACTIVE_RUN_STATES
from .util import in_quiet_hours

log = logging.getLogger("galton.watch")
SMOKE_SUITE = "s_rapida"


class Watch:
    def __init__(self, store: Any, settings: Any, runner: Any, gpus: Any, *, clock: Callable[[], float], submit_run: Callable[[str], Any],
                 emit: Optional[Callable[[str, dict[str, Any]], None]] = None, client_factory: Optional[Callable[[], httpx.Client]] = None, offline: bool = False):
        self.store, self.settings, self.runner, self.gpus, self.clock = store, settings, runner, gpus, clock
        self.submit_run = submit_run
        self.emit = emit or (lambda *_: None)
        self.client_factory = client_factory or (lambda: httpx.Client(trust_env=False, timeout=3.0))
        self.offline = offline
        self._seen: dict[str, float] = {}       # contestant -> first time seen up
        self._active: dict[str, float] = {}     # contestant -> last time it looked busy
        self._expires: dict[str, str] = {}      # Ollama: last expires_at value (it moves whenever the model is used)
        self.last_tick: Optional[float] = None
        self.last_decision: dict[str, Any] = {}

    # ------------------------------------------------------------------ idle tracking
    def idle_for(self, contestant_id: str) -> Optional[float]:
        """Seconds since the server holding this model was last seen busy (``None`` when it has not been seen up)."""
        seen = self._seen.get(contestant_id)
        if seen is None:
            return None
        return self.clock() - max(seen, self._active.get(contestant_id, 0.0))

    def probe(self) -> dict[str, Any]:
        """One cheap pass over the shared servers: who is up, who is busy. Feeds ``idle_for``."""
        if self.offline:
            return {"offline": True}
        now = self.clock()
        up = 0
        with self.client_factory() as client:
            for c in self.store.contestants(kind="server", enabled=True, include_missing=False, include_adhoc=False):
                if c["remote"]:
                    continue
                try:
                    if c["api"] == "ollama":
                        response = client.get(c["url"].rstrip("/") + "/api/ps")
                        response.raise_for_status()
                        entry = next((m for m in response.json().get("models") or [] if c["model"] in (m.get("name"), m.get("model"))), None)
                        resident = entry is not None
                        if resident:
                            stamp = str(entry.get("expires_at") or "")
                            if self._expires.get(c["id"]) not in (None, stamp):
                                self._active[c["id"]] = now
                            self._expires[c["id"]] = stamp
                    else:
                        info = probe_openai(client, c["url"])
                        resident = bool(info.get("up"))
                        if info.get("busy"):
                            self._active[c["id"]] = now
                except (httpx.HTTPError, ValueError):
                    resident = False
                if resident:
                    self._seen.setdefault(c["id"], now)
                    up += 1
                else:
                    self._seen.pop(c["id"], None)
                    self._expires.pop(c["id"], None)
                    if c["meta"].get("resident") or c["meta"].get("up"):
                        self.store.update_contestant(c["id"], meta={**c["meta"], "resident": False, "up": False})
        return {"servers_up": up}

    # ------------------------------------------------------------------ who needs measuring
    def candidates(self) -> list[dict[str, Any]]:
        """Models never measured on the quick suite at their current digest, and not tried by the watch within one interval."""
        suite = self.store.find_suite(SMOKE_SUITE)
        if suite is None:
            return []
        measured = {(r["contestant_id"], r["digest"]) for r in self.store.scoring_rows(suite_ids=[suite["id"]])}
        horizon = self.clock() - float(self.settings.get("watch.interval_h")) * 3600
        tried = {cid for run in self.store.runs(source="watch", limit=200) if run["created_ts"] >= horizon for cid in run["contestants"]}
        out = []
        for c in self.store.contestants(enabled=True, include_missing=False, include_adhoc=False):
            if c["remote"] or c["id"] in tried or (c["id"], c["digest"]) in measured or (c["meta"] or {}).get("not_served"):
                continue
            out.append(c)
        return out

    def _eligible(self, c: dict[str, Any]) -> tuple[bool, str]:
        if c["kind"] == "server":
            idle = self.idle_for(c["id"])
            if idle is None:
                return False, "not loaded on a server right now"
            need = float(self.settings.get("watch.idle_min")) * 60
            if idle < need:
                return False, f"server busy recently (idle {idle / 60:.0f} of {need / 60:.0f} min)"
            return True, "loaded and idle"
        try:
            ctx = placement.choose_context(int(self.settings.get("runner.context")), placement.gguf_info(self.store, c, self.runner.meta_reader))
            need_mb = placement.estimate_mb(self.store, c, ctx, self.runner.meta_reader)
            self.gpus.check_possible(need_mb)
        except GaltonError as exc:
            return False, exc.message
        return (True, "an allowed GPU has room") if self.gpus.plan(need_mb) else (False, "no allowed GPU has room right now")

    # ------------------------------------------------------------------ the tick
    def tick(self) -> dict[str, Any]:
        """Probe the servers, then queue at most one quick run if a candidate may be measured now."""
        now = self.clock()
        self.last_tick = now
        decision: dict[str, Any] = {"probe": self.probe()}
        self.last_decision = decision
        if not self.settings.get("watch.enabled") or not self.settings.get("watch.auto_smoke"):
            decision["skipped"] = "auto smoke is off"
            return decision
        if self.settings.get("scheduler.paused"):
            decision["skipped"] = "the scheduler is paused"
            return decision
        if in_quiet_hours(now, float(self.settings.get("watch.quiet_from")), float(self.settings.get("watch.quiet_to"))):
            decision["skipped"] = "quiet hours"
            return decision
        if self.runner.active_run or self.store.runs(states=ACTIVE_RUN_STATES, limit=1):
            decision["skipped"] = "a run is already queued or running"
            return decision
        reasons = {}
        for c in self.candidates():
            ok, why = self._eligible(c)
            reasons[c["name"]] = why
            if not ok:
                continue
            run = self.runner.create(suites=[SMOKE_SUITE], contestants=[c["id"]], settings={"wait_s": 0, "device": "gpu"}, label=text("label_watch", name=c["name"]), source="watch", caller="watch")
            self.submit_run(run["id"])
            decision["queued"] = {"run": run["id"], "contestant": c["name"], "why": why}
            return decision
        decision["waiting"] = reasons
        return decision

    # ------------------------------------------------------------------ regressions
    def detect_regressions(self, run_id: str) -> list[dict[str, Any]]:
        """Compare every model of the run with the results it had before its digest last changed. Raises a notice when it got worse."""
        found = []
        run = self.store.run(run_id)
        for cid in run["contestants"]:
            c = self.store.contestant(cid)
            rows = self.store.scoring_rows(contestant_ids=[cid])
            current = [r for r in rows if r["digest"] == c["digest"] and r["run_id"] == run_id]
            others = {}
            for r in rows:
                if r["digest"] != c["digest"]:
                    others.setdefault(r["digest"], []).append(r)
            if not current or not others:
                continue
            previous_digest = max(others, key=lambda d: max(x["ts"] for x in others[d]))
            cmp = stats.compare(others[previous_digest], current)   # a = before, b = now
            if cmp["n"] < 8 or cmp["verdict"] not in ("better", "worse"):
                continue
            if cmp["verdict"] == "better":    # the earlier version was significantly better: a regression
                info = {"contestant": cid, "name": c["name"], "n": cmp["n"], "p_value": cmp["p_value"], "diff": cmp["diff"], "run": run_id, "previous_digest": previous_digest}
                self.store.add_notice(kind="regression", severity="high", params={"name": c["name"], "n": cmp["n"], "diff": f"{abs(cmp['diff']):.2f}", "p": cmp["p_value"]},
                                      data=info, contestant_id=cid, dedupe=f"regression:{cid}:{c['digest']}")
                self.emit("galton.regression", info)
                found.append(info)
            else:
                self.store.add_notice(kind="improvement", severity="low", params={"name": c["name"], "n": cmp["n"], "diff": f"{abs(cmp['diff']):.2f}"},
                                      data={"contestant": cid, "n": cmp["n"], "p_value": cmp["p_value"], "run": run_id}, contestant_id=cid, dedupe=f"improvement:{cid}:{c['digest']}")
        return found
