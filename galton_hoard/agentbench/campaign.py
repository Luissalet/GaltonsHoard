"""A campaign: benchmarks x harnesses x k attempts, run as runner jobs with a fixed number of trials in flight.

The job folders are the source of truth: a campaign can be stopped and started again, and it only launches what is missing. A trial whose verifier
never ran (``none``) is launched again, up to ``retries`` times; a verified trial is never run twice.

Scheduling: each cell is one job with ``per_cell`` trials at a time; at most ``slots // per_cell`` jobs run together. Cells are taken benchmark by
benchmark and, inside one benchmark, harness after harness, so a partial campaign already compares every harness on the same benchmark.
"""

from __future__ import annotations

import json
import subprocess
import sys
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

from .harbor import CellJob, job_command, job_finished, read_job
from .spec import Benchmark, HarnessSpec, read_dataset, select


#: a cell whose runner dies this many times before writing anything is skipped
MAX_LAUNCH_FAILURES = 3


@dataclass
class Cell:
    bench: Benchmark
    harness: HarnessSpec
    tasks: list[str]                   # dev + test of the benchmark, in that order
    split_of: dict[str, str]
    k: int
    jobs: list[str] = field(default_factory=list)
    skipped: str = ""

    @property
    def key(self) -> str:
        return f"{self.bench.id}|{self.harness.name}"


@dataclass
class CampaignConfig:
    name: str
    model: str
    base_url: str
    api_key: str = "sk-local"
    k: int = 1
    slots: int = 8
    per_cell: int = 2
    retries: int = 2
    splits: tuple[str, ...] = ("dev", "test")
    exe: str = "harbor"
    jobs_dir: Path = Path("jobs")
    datasets_dir: Path = Path("datasets")
    #: Multiplies every task's agent time limit, the same for all harnesses: the published limits assume a fast hosted
    #: model; a local one serving many trials at once needs more wall time to show what a harness can solve.
    time_scale: float = 1.0


def dataset_folder(datasets_dir: Path, bench: Benchmark) -> Path:
    return datasets_dir / bench.dataset.rsplit("/", 1)[-1]


def download(exe: str, bench: Benchmark, datasets_dir: Path, *, env: Optional[dict[str, str]] = None, timeout: float = 3600) -> Path:
    folder = dataset_folder(datasets_dir, bench)
    if not folder.is_dir() or not any(folder.iterdir()):
        datasets_dir.mkdir(parents=True, exist_ok=True)
        prefix = [sys.executable, exe] if exe.endswith(".py") else [exe]
        subprocess.run([*prefix, "datasets", "download", bench.dataset, "-o", str(datasets_dir)], env=env, timeout=timeout, check=False,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return folder


class Campaign:
    def __init__(self, cfg: CampaignConfig, benches: list[Benchmark], harnesses: list[HarnessSpec], *,
                 log: Callable[[str], None] = print, cancel: Optional[threading.Event] = None,
                 on_trials: Optional[Callable[[Cell, list[dict[str, Any]]], None]] = None,
                 on_tick: Optional[Callable[["Campaign"], None]] = None, launch: bool = True):
        self.cfg, self.benches, self.harnesses, self.log = cfg, benches, harnesses, log
        self.cancel = cancel or threading.Event()
        self.on_trials = on_trials
        self.on_tick = on_tick
        self.launch = launch
        self.cells: list[Cell] = []
        self.procs: dict[str, subprocess.Popen] = {}
        self.launch_failures: dict[str, int] = {}

    # -------------------------------------------------------------- plan
    def plan(self) -> list[Cell]:
        cells: list[Cell] = []
        for bench in self.benches:
            folder = dataset_folder(self.cfg.datasets_dir, bench)
            subset = select(bench, read_dataset(folder))
            split_of = {t: "dev" for t in subset["dev"]}
            split_of.update({t: "test" for t in subset["test"]})
            tasks = [t for s in ("dev", "test") if s in self.cfg.splits for t in subset[s]]
            for h in self.harnesses:
                c = Cell(bench, h, tasks, split_of, self.cfg.k)
                if bench.needs_mcp and not h.mcp:
                    c.skipped = "needs_mcp"
                if not tasks:
                    c.skipped = c.skipped or "no_tasks"
                cells.append(c)
        self.cells = cells
        return cells

    def job_name(self, cell: Cell, attempt: int) -> str:
        return f"{self.cfg.name}-{cell.bench.id}-{cell.harness.name}" + (f"-r{attempt}" if attempt else "")

    # -------------------------------------------------------------- state from disk
    def trials(self, cell: Cell) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for attempt in range(self.cfg.retries + 1):
            job = self.cfg.jobs_dir / self.job_name(cell, attempt)
            for r in read_job(job):
                r.update({"benchmark": cell.bench.id, "harness": cell.harness.name, "split": cell.split_of.get(r["task"], ""), "job": job.name})
                rows.append(r)
        return rows

    def missing(self, cell: Cell) -> list[str]:
        """Tasks with fewer than k verified trials (a task listed once per attempt it still needs)."""
        done: dict[str, int] = defaultdict(int)
        for r in self.trials(cell):
            if r["outcome"] in ("pass", "fail"):
                done[r["task"]] += 1
        return [t for t in cell.tasks if done[t] < cell.k]

    # -------------------------------------------------------------- launching
    def _start(self, cell: Cell, attempt: int, tasks: list[str]) -> subprocess.Popen:
        folder = dataset_folder(self.cfg.datasets_dir, cell.bench)
        job: CellJob = job_command(exe=self.cfg.exe, bench=cell.bench, harness=cell.harness, tasks=tasks, dataset_dir=folder,
                                   jobs_dir=self.cfg.jobs_dir, job_name=self.job_name(cell, attempt), model=self.cfg.model,
                                   base_url=self.cfg.base_url, api_key=self.cfg.api_key, k=cell.k if attempt == 0 else 1,
                                   concurrent=self.cfg.per_cell, time_scale=self.cfg.time_scale)
        self.cfg.jobs_dir.mkdir(parents=True, exist_ok=True)
        log_path = self.cfg.jobs_dir / f"{self.job_name(cell, attempt)}.log"
        out = open(log_path, "ab")
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        proc = subprocess.Popen(job.argv, env=job.env, stdout=out, stderr=subprocess.STDOUT, cwd=str(self.cfg.jobs_dir), creationflags=flags)
        out.close()
        self.log(f"start {self.job_name(cell, attempt)}: {len(tasks)} task(s)")
        return proc

    def next_attempt(self, cell: Cell) -> Optional[tuple[int, list[str]]]:
        for attempt in range(self.cfg.retries + 1):
            job_dir = self.cfg.jobs_dir / self.job_name(cell, attempt)
            if job_dir.exists() and not job_finished(job_dir) and self.job_name(cell, attempt) not in self.procs:
                # a job left unfinished by an earlier run: start the next attempt for what it did not verify
                continue
            if job_dir.exists():
                continue
            todo = self.missing(cell)
            if not todo:
                return None
            if attempt == 0:
                return 0, todo
            return attempt, sorted(set(todo))
        return None

    def run(self, *, poll_s: float = 15.0) -> dict[str, Any]:
        if not self.cells:
            self.plan()
        queue = [c for c in self.cells if not c.skipped]
        max_jobs = max(1, self.cfg.slots // max(1, self.cfg.per_cell))
        running: dict[str, tuple[Cell, subprocess.Popen]] = {}
        while not self.cancel.is_set():
            for name, (cell, proc) in list(running.items()):
                if proc.poll() is not None:
                    running.pop(name)
                    self.procs.pop(name, None)
                    if self.on_trials:
                        self.on_trials(cell, self.trials(cell))
                    self.log(f"done {name} (exit {proc.returncode})")
                    if proc.returncode and not (self.cfg.jobs_dir / name).exists():
                        # The runner died before writing a job folder (the
                        # harness did not import, a bad flag...): relaunching it
                        # would fail the same way forever.
                        fails = self.launch_failures.get(cell.key, 0) + 1
                        self.launch_failures[cell.key] = fails
                        if fails >= MAX_LAUNCH_FAILURES:
                            cell.skipped = f"launch_failed: the runner exited {proc.returncode} {fails} times without starting (see {name}.log)"
                            queue = [c for c in queue if c.key != cell.key]
                            self.log(f"skip {cell.key}: {cell.skipped}")
            if self.on_tick:
                try:
                    self.on_tick(self)
                except Exception as exc:  # noqa: BLE001 - a failed sync must not stop the trials
                    self.log(f"sync failed: {type(exc).__name__}: {exc}")
            if not self.launch:
                break
            busy_cells = {c.key for c, _ in running.values()}
            for cell in queue:
                if len(running) >= max_jobs:
                    break
                if cell.key in busy_cells:
                    continue
                nxt = self.next_attempt(cell)
                if nxt is None:
                    continue
                attempt, tasks = nxt
                proc = self._start(cell, attempt, tasks)
                name = self.job_name(cell, attempt)
                running[name] = (cell, proc)
                self.procs[name] = proc
                busy_cells.add(cell.key)
            if not running and all(self.next_attempt(c) is None for c in queue):
                break
            self.cancel.wait(poll_s)
        if self.cancel.is_set():
            for _, proc in running.values():
                try:
                    proc.terminate()
                except OSError:
                    pass
        return self.summary()

    def all_trials(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for cell in self.cells:
            if not cell.skipped:
                rows.extend(self.trials(cell))
        return rows

    def summary(self) -> dict[str, Any]:
        from .scoring import board
        rows = self.all_trials()
        return {"campaign": self.cfg.name, "model": self.cfg.model, "board": board(rows, focus="faustus"),
                "skipped": [{"cell": c.key, "why": c.skipped} for c in self.cells if c.skipped]}


def save_json(path: Path, data: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    tmp.replace(path)
