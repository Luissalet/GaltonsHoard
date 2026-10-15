"""The agent bench inside the app: campaigns as background threads, their trials in the database, and the views the tools and the screen read.

A campaign does not use the GPU lane: the model is served elsewhere (the campaign names its endpoint) and the trials run in containers. Its thread
keeps ``slots`` trials in flight and copies every finished trial into the database as it lands. A campaign that was running when the app stopped is
started again on the next start: the job folders say what already ran, so only the rest is launched.

``launch=False`` adopts a campaign whose jobs another process runs (the standalone script): the thread only reads its job folders until they stop
changing.
"""

from __future__ import annotations

import logging
import os
import shutil
import threading
import time
from pathlib import Path
from typing import Any, Callable, Optional

from ..errors import GaltonError
from .campaign import Campaign, CampaignConfig, dataset_folder, download
from .harbor import read_job, runner_exe, trial_logs
from .scoring import board
from .spec import BY_ID, CATALOGUE, HarnessSpec, benchmark, parse_harness_line, parse_harnesses, read_dataset, select
from .store import ACTIVE, AgentBenchStore

log = logging.getLogger("galton.agentbench")
MAX_TRIALS = 5000


class AgentBench:
    def __init__(self, store: AgentBenchStore, settings: Any, *, data_dir: Path, emit: Optional[Callable[[str, dict], None]] = None,
                 clock: Callable[[], float] = time.time):
        self.store, self.settings, self.data_dir, self.clock = store, settings, Path(data_dir), clock
        self.emit = emit or (lambda *_: None)
        self.threads: dict[str, tuple[threading.Thread, threading.Event]] = {}
        self.lock = threading.Lock()

    # -------------------------------------------------------------- configuration
    def _dir(self, key: str, default: str) -> Path:
        value = str(self.settings.get(key) or "").strip()
        return Path(value) if value else self.data_dir / "agentbench" / default

    @property
    def jobs_dir(self) -> Path:
        return self._dir("agentbench.jobs_dir", "jobs")

    @property
    def datasets_dir(self) -> Path:
        return self._dir("agentbench.datasets_dir", "datasets")

    def exe(self) -> str:
        return runner_exe(str(self.settings.get("agentbench.runner") or ""))

    def harnesses(self, *, enabled_only: bool = False) -> list[HarnessSpec]:
        specs = parse_harnesses(self.settings.get("agentbench.harnesses") or [])
        return [h for h in specs if h.enabled] if enabled_only else specs

    def catalog(self) -> dict[str, Any]:
        exe = self.exe()
        benches = []
        for b in CATALOGUE:
            folder = dataset_folder(self.datasets_dir, b)
            tasks = read_dataset(folder)
            sub = select(b, tasks) if tasks else {"dev": [], "test": [], "pool": 0}
            benches.append({**b.info(), "downloaded": bool(tasks), "available": len(tasks), "pool": sub["pool"],
                            "dev_tasks": sub["dev"], "test_tasks": sub["test"]})
        bad = [line for line in self.settings.get("agentbench.harnesses") or [] if str(line).strip() and parse_harness_line(line) is None]
        return {"runner": exe, "runner_found": bool(shutil.which(exe) or Path(exe).is_file()),
                "jobs_dir": str(self.jobs_dir), "datasets_dir": str(self.datasets_dir), "benchmarks": benches,
                "harnesses": [h.info() for h in self.harnesses()], "unreadable_lines": bad,
                "defaults": {"model": self.settings.get("agentbench.model"), "base_url": self.settings.get("agentbench.base_url"),
                             "slots": self.settings.get("agentbench.slots"), "per_cell": self.settings.get("agentbench.per_cell"),
                             "retries": self.settings.get("agentbench.retries")}}

    def plan(self, benchmarks: list[str]) -> dict[str, Any]:
        out = {}
        for ref in benchmarks or [b.id for b in CATALOGUE]:
            b = self._bench(ref)
            sub = select(b, read_dataset(dataset_folder(self.datasets_dir, b)))
            out[b.id] = sub
        return out

    def _bench(self, ref: str):
        try:
            return benchmark(ref)
        except KeyError:
            raise GaltonError("invalid", "ab_unknown_benchmark", ref=ref, options=", ".join(BY_ID)) from None

    # -------------------------------------------------------------- campaigns
    def start(self, *, benchmarks: list[str], harnesses: list[str], model: str = "", base_url: str = "", k: int = 1,
              splits: Optional[list[str]] = None, label: str = "", name: str = "", launch: bool = True, source: str = "ui", caller: str = "",
              slots: Optional[int] = None, per_cell: Optional[int] = None) -> dict[str, Any]:
        benches = [self._bench(b) for b in (benchmarks or [b.id for b in CATALOGUE])]
        known = {h.name: h for h in self.harnesses()}
        chosen = harnesses or [h.name for h in self.harnesses(enabled_only=True)]
        missing = [h for h in chosen if h not in known]
        if missing:
            raise GaltonError("invalid", "ab_unknown_harness", ref=", ".join(missing), options=", ".join(known) or "-")
        if not chosen:
            raise GaltonError("invalid", "ab_no_harness")
        model = model or str(self.settings.get("agentbench.model") or "")
        base_url = base_url or str(self.settings.get("agentbench.base_url") or "")
        if not model or not base_url:
            raise GaltonError("invalid", "ab_no_model")
        splits = [s for s in (splits or ["dev", "test"]) if s in ("dev", "test")] or ["dev", "test"]
        k = max(1, min(int(k or 1), 10))
        total = sum(b.n for b in benches) * len(chosen) * k
        if total > MAX_TRIALS:
            raise GaltonError("invalid", "ab_too_many", n=total, max=MAX_TRIALS)
        row = self.store.create(label=label, name=name or "", state="queued", benchmarks=[b.id for b in benches], harnesses=chosen,
                                harness_lines=[known[h].info() | {"line_agent": known[h].agent} for h in chosen], model=model, base_url=base_url, k=k,
                                splits=splits, slots=int(slots or self.settings.get("agentbench.slots")),
                                per_cell=int(per_cell or self.settings.get("agentbench.per_cell")), retries=int(self.settings.get("agentbench.retries")),
                                launch=launch, source=source, caller=caller)
        if not name:
            row = self.store.update(row["id"], name=row["id"].replace("_", "-"))
        self._spawn(row["id"])
        return self.card(self.store.campaign(row["id"]))

    def _campaign_obj(self, row: dict[str, Any], cancel: threading.Event) -> Campaign:
        known = {h.name: h for h in self.harnesses()}
        specs = [known[h] for h in row["harnesses"] if h in known]
        cfg = CampaignConfig(name=row["name"], model=row["model"], base_url=row["base_url"], k=row["k"], slots=row["slots"], per_cell=row["per_cell"],
                             retries=row["retries"], splits=tuple(row["splits"]), exe=self.exe(), jobs_dir=self.jobs_dir, datasets_dir=self.datasets_dir,
                             time_scale=float(self.settings.get("agentbench.time_scale") or 1.0))
        return Campaign(cfg, [benchmark(b) for b in row["benchmarks"]], specs, log=lambda m: log.info("%s: %s", row["name"], m), cancel=cancel,
                        launch=bool(row["launch"]))

    def _spawn(self, cid: str) -> None:
        with self.lock:
            if cid in self.threads and self.threads[cid][0].is_alive():
                return
            cancel = threading.Event()
            t = threading.Thread(target=self._run, args=(cid, cancel), name=f"agentbench-{cid}", daemon=True)
            self.threads[cid] = (t, cancel)
            t.start()

    def _run(self, cid: str, cancel: threading.Event) -> None:
        row = self.store.update(cid, state="running", started_ts=self.clock(), error="")
        self.emit("galton.agentbench.started", {"campaign": cid, "name": row["name"]})
        try:
            camp = self._campaign_obj(row, cancel)
            for b in camp.benches:
                download(camp.cfg.exe, b, camp.cfg.datasets_dir)
            camp.plan()
            last = {"n": -1, "since": time.monotonic()}

            def tick(c: Campaign) -> None:
                self.sync(cid, c)
                if self.store.find(cid) and self.store.campaign(cid)["cancel"]:
                    cancel.set()
                if not c.launch:
                    n = sum(self.store.count(cid).values())
                    if n != last["n"]:
                        last.update(n=n, since=time.monotonic())

            if row["launch"]:
                camp.on_tick = tick
                camp.run(poll_s=float(os.environ.get("GALTON_AB_POLL_S", "15")))
            else:
                # adopted: read the folders until nothing has changed for an hour or the campaign is cancelled
                while not cancel.is_set():
                    tick(camp)
                    if time.monotonic() - last["since"] > 3600:
                        break
                    cancel.wait(30)
            self.sync(cid, camp)
            state = "cancelled" if cancel.is_set() else "done"
            self.store.update(cid, state=state, finished_ts=self.clock(), cancel=False, summary=self.summary(cid))
            self.emit("galton.agentbench.finished", {"campaign": cid, "state": state})
        except Exception as exc:  # noqa: BLE001
            log.exception("agent-bench campaign %s failed", cid)
            self.store.update(cid, state="failed", finished_ts=self.clock(), error=f"{type(exc).__name__}: {exc}"[:500])
            self.emit("galton.agentbench.finished", {"campaign": cid, "state": "failed"})

    def sync(self, cid: str, camp: Optional[Campaign] = None) -> int:
        """Copy every finished trial of the campaign's job folders into the database."""
        if camp is None:
            camp = self._campaign_obj(self.store.campaign(cid), threading.Event())
            camp.plan()
        n = 0
        for cell in camp.cells:
            if not cell.skipped:
                n += self.store.upsert_trials(cid, camp.trials(cell))
        return n

    def resume_all(self) -> list[str]:
        """Start again the campaigns the previous process left running or queued."""
        out = []
        for row in self.store.campaigns(states=ACTIVE, limit=50):
            self._spawn(row["id"])
            out.append(row["id"])
        return out

    def cancel(self, ref: str) -> dict[str, Any]:
        row = self.store.campaign(ref)
        self.store.update(row["id"], cancel=True)
        entry = self.threads.get(row["id"])
        if entry:
            entry[1].set()
        return self.card(self.store.campaign(row["id"]))

    # -------------------------------------------------------------- views
    def card(self, row: dict[str, Any]) -> dict[str, Any]:
        counts = self.store.count(row["id"])
        expected = 0
        for b in row["benchmarks"]:
            try:
                bench = benchmark(b)
            except KeyError:
                continue
            expected += (bench.dev if "dev" in row["splits"] else 0) + ((bench.n - bench.dev) if "test" in row["splits"] else 0)
        expected *= len(row["harnesses"]) * row["k"]
        verified = counts.get("pass", 0) + counts.get("fail", 0)
        return {"id": row["id"], "name": row["name"], "label": row["label"], "state": row["state"], "model": row["model"], "benchmarks": row["benchmarks"],
                "harnesses": row["harnesses"], "k": row["k"], "splits": row["splits"], "launch": row["launch"], "created_ts": row["created_ts"],
                "started_ts": row.get("started_ts"), "finished_ts": row.get("finished_ts"), "error": row["error"],
                "trials": {"expected": expected, "verified": verified, "passed": counts.get("pass", 0), "none": counts.get("none", 0)}}

    def summary(self, ref: str, *, split: str = "", focus: str = "faustus") -> dict[str, Any]:
        row = self.store.campaign(ref)
        trials = self.store.trials(row["id"])
        return board(trials, focus=focus, split=split)

    def result(self, ref: str, *, split: str = "", focus: str = "faustus") -> dict[str, Any]:
        row = self.store.campaign(ref)
        return {"campaign": self.card(row), "board": self.summary(row["id"], split=split, focus=focus)}

    def cell(self, ref: str, bench: str, harness: str) -> dict[str, Any]:
        row = self.store.campaign(ref)
        trials = self.store.trials(row["id"], benchmark=bench, harness=harness)
        if not trials:
            raise GaltonError("not_found", "ab_no_cell", id=row["id"], benchmark=bench, harness=harness)
        from .scoring import cell as cell_of
        return {"campaign": row["id"], "benchmark": bench, "harness": harness, "cell": cell_of(trials),
                "trials": [{k: t[k] for k in ("id", "task", "split", "outcome", "passed", "reward", "reason", "error_type", "tokens_in", "tokens_out",
                                              "agent_s", "total_s")} for t in trials]}

    def trial(self, tid: int, *, limit: int = 20_000) -> dict[str, Any]:
        t = self.store.trial(tid)
        logs = trial_logs(Path(t["trial_dir"]), limit=limit) if t.get("trial_dir") else {}
        return {**t, "logs": logs}

    def campaigns(self, limit: int = 30) -> list[dict[str, Any]]:
        return [self.card(r) for r in self.store.campaigns(limit=limit)]

    def read_folder(self, job: str) -> list[dict[str, Any]]:
        return read_job(self.jobs_dir / job)
