"""Regression watch: notice new and changed models, measure them with the quick suite when that disturbs nobody, and say so when one got worse.

* A model that is new, or whose digest changed, or that was never measured on the quick suite, is a candidate.
* A candidate is measured automatically only when (a) it is already served by a shared server (a llama-server or Ollama on this PC, or a server of
  another machine of the person's network) that has been idle for ``watch.idle_min`` minutes, or (b) ``watch.load_local`` is on, it is a GGUF and an
  allowed GPU has room right now. With ``watch.load_local`` off (the default) the watch never starts a llama-server or loads a model into Ollama on
  this PC. The watch never queues for a GPU and never runs in quiet hours (default 01:00-08:00).
* A model the watch failed to measure is not tried again by the watch until its file (digest, path) or its server changes, or somebody runs it by
  hand: the reason stays on the model (``meta.watch_failed``), and the failure is announced once, not on every cycle.
* After a run, each model's results are compared with the ones it had before its last digest change; a significantly worse result raises a notice
  and the family event ``galton.regression``.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Callable, Optional

import httpx

from . import placement, stats
from .errors import GaltonError
from .hoard_link import notify_channels
from .messages import text
from .servers import probe_openai
from .store import ACTIVE_RUN_STATES

log = logging.getLogger("galton.watch")
SMOKE_SUITE = "s_rapida"
#: failures that say nothing about the model itself (somebody was using the server, the server did not answer, a GPU was taken in between): the watch
#: may look again at the next interval, so it does not stop trying; they are still announced only once per model, file and reason
TRANSIENT = frozenset({"server_busy", "no_gpu_free", "no_gpu", "server_unverified", "server_changed", "model_not_served", "server_no_answer", "ollama_no_answer",
                       "ollama_not_loaded", "cancelled_loading", "cpu_no_ram", "load_local_off"})


def failure_of(c: dict[str, Any]) -> Optional[dict[str, Any]]:
    """What the watch recorded when it failed to measure this model, while it still stands for the same file and server; ``None`` otherwise."""
    failed = (c.get("meta") or {}).get("watch_failed")
    if not isinstance(failed, dict):
        return None
    same = failed.get("digest") == c["digest"] and failed.get("path") == c["path"] and failed.get("url") == c["url"] and failed.get("model") == c["model"]
    return failed if same else None


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
                    if c["meta"].get("up") is False:
                        self._server_is_back(c)
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
        """Models never measured on the quick suite at their current digest, not tried by the watch within one interval, and not given up on (a
        failure the watch recorded for the file and server they have now). Models that would have to be loaded on this PC are left out while
        ``watch.load_local`` is off."""
        suite = self.store.find_suite(SMOKE_SUITE)
        if suite is None:
            return []
        measured = {(r["contestant_id"], r["digest"]) for r in self.store.scoring_rows(suite_ids=[suite["id"]])}
        horizon = self.clock() - float(self.settings.get("watch.interval_h")) * 3600
        tried = {cid for run in self.store.runs(source="watch", limit=200) if run["created_ts"] >= horizon for cid in run["contestants"]}
        loads = bool(self.settings.get("watch.load_local"))
        out = []
        for c in self.store.contestants(enabled=True, include_missing=False, include_adhoc=False):
            if c["remote"] or c["id"] in tried or (c["id"], c["digest"]) in measured or (c["meta"] or {}).get("not_served"):
                continue
            if failure_of(c) is not None or (c["kind"] == "gguf" and not loads):
                continue
            out.append(c)
        return out

    def held_back(self) -> list[dict[str, Any]]:
        """Models the watch would measure if it were allowed to load them on this PC (``watch.load_local`` is off): what it leaves alone, and why."""
        if self.settings.get("watch.load_local"):
            return []
        suite = self.store.find_suite(SMOKE_SUITE)
        if suite is None:
            return []
        measured = {(r["contestant_id"], r["digest"]) for r in self.store.scoring_rows(suite_ids=[suite["id"]])}
        return [c for c in self.store.contestants(kind="gguf", enabled=True, include_missing=False, include_adhoc=False)
                if not c["remote"] and (c["id"], c["digest"]) not in measured and failure_of(c) is None]

    # ------------------------------------------------------------------ giving up on a model
    def _server_is_back(self, c: dict[str, Any]) -> None:
        """A server that was not answering answers again: it is up, and whatever kept the watch from its model may be gone, so the watch may look again."""
        fresh = self.store.contestant(c["id"])
        self.store.update_contestant(c["id"], meta={**{k: v for k, v in fresh["meta"].items() if k != "watch_failed"}, "up": True})

    def record_outcome(self, run_id: str) -> list[str]:
        """After a run: a model that failed is marked (why, and for which file and server), so the watch does not start it again by itself; a model
        that was measured loses the mark (and one that somebody runs by hand loses it when the run is created, see ``Runner.create``). A failure
        that says nothing about the model (``TRANSIENT``) marks nothing. Returns the ids of the models marked."""
        run = self.store.find_run(run_id)
        if run is None:
            return []
        marked = []
        for rc in self.store.run_contestants(run_id):
            c = self.store.find_contestant(rc["contestant_id"])
            if c is None:
                continue
            meta = dict(c["meta"] or {})
            if rc["state"] == "done" and "watch_failed" in meta:
                meta.pop("watch_failed")
                self.store.update_contestant(c["id"], meta=meta)
            elif rc["state"] == "failed" and getattr(rc["error"], "key", "") not in TRANSIENT:
                meta["watch_failed"] = {"ts": self.clock(), "run": run_id, "error": str(rc["error"] or ""), "key": getattr(rc["error"], "key", ""), "digest": c["digest"],
                                        "path": c["path"], "url": c["url"], "model": c["model"]}
                self.store.update_contestant(c["id"], meta=meta)
                marked.append(c["id"])
        return marked

    def _eligible(self, c: dict[str, Any]) -> tuple[bool, str]:
        if c["kind"] != "server" and not self.settings.get("watch.load_local"):
            return False, "loading models on this PC is off (watch.load_local)"
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
    def in_quiet_hours(self, ts: float) -> bool:
        """Is ``ts`` inside the quiet window (``watch.quiet_from`` .. ``watch.quiet_to``, hours; it may wrap over midnight)? The shared window
        logic of Hoard Link, which holds everything but urgent notices; the watch is never urgent."""
        start, end = float(self.settings.get("watch.quiet_from")), float(self.settings.get("watch.quiet_to"))
        return notify_channels.in_quiet_hours(datetime.fromtimestamp(ts), round(start * 60), round(end * 60), allow_high=False)

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
        if self.in_quiet_hours(now):
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
            run = self.runner.create(suites=[SMOKE_SUITE], contestants=[c["id"]], settings={"wait_s": 0, "device": "gpu", "load_local": bool(self.settings.get("watch.load_local"))},
                                     label=text("label_watch", name=c["name"]), source="watch", caller="watch")
            self.submit_run(run["id"])
            decision["queued"] = {"run": run["id"], "contestant": c["name"], "why": why}
            return decision
        decision["waiting"] = reasons
        held = self.held_back()
        if held:
            decision["held_back"] = {"setting": "watch.load_local", "models": [c["name"] for c in held]}
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
