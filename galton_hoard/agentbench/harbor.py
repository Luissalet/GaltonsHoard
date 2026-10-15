"""The container trial runner (``harbor``): the command line of one cell's job and the reading of its job folder.

One job = one benchmark x one harness x a list of tasks x k attempts. The runner downloads the task, builds or pulls its container, installs the
harness inside, gives it the task text, then runs the task's hidden tests and writes ``result.json`` per trial. Galton never looks inside the
container: everything it stores comes from those files.

Outcome of a trial, read from its ``result.json``:

* ``pass`` / ``fail``: the verifier ran and gave a reward (1 is a pass; partial rewards are kept as the number);
* ``fail`` with reason ``agent_timeout`` or ``agent_error``: the harness did not finish within the task's time, or stopped with an error of its
  own; the verifier may still have run on what it left, and then its reward decides;
* ``none``: the trial says nothing about the harness: the container could not be built, the harness could not install itself, the model server did
  not answer, the job was cancelled. These are left out of every rate and retried.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

from .spec import Benchmark, HarnessSpec

#: exception types of the runner that are about the infrastructure, not the harness
INFRA_ERRORS = (
    "AgentSetupTimeoutError", "EnvironmentStartTimeoutError", "EnvironmentBuildError", "EnvironmentStartError", "DockerError", "DockerBuildError",
    "ImagePullError", "CancelledError", "KeyboardInterrupt", "VerifierTimeoutError", "VerifierError", "NetworkConnectionError", "ApiConnectionClosedError",
    "ApiInternalServerError", "ApiOverloadedError", "ApiRateLimitError", "UnicodeDecodeError", "FileNotFoundError", "PermissionError", "OSError",
    "RuntimeError", "TimeoutError",
)
AGENT_TIMEOUT = ("AgentTimeoutError",)


def runner_exe(explicit: str = "") -> str:
    """The trial runner's executable: the setting, else the one on PATH."""
    if explicit and Path(explicit).exists():
        return explicit
    found = shutil.which("harbor")
    return found or explicit or "harbor"


@dataclass
class CellJob:
    """What it takes to start one job."""
    argv: list[str]
    env: dict[str, str]
    job_dir: Path


def job_command(*, exe: str, bench: Benchmark, harness: HarnessSpec, tasks: Iterable[str], dataset_dir: Path, jobs_dir: Path, job_name: str,
                model: str, base_url: str, api_key: str = "sk-local", k: int = 1, concurrent: int = 1,
                base_env: Optional[dict[str, str]] = None, time_scale: float = 1.0) -> CellJob:
    tasks = list(tasks)
    prefix = [sys.executable, exe] if exe.endswith(".py") else [exe]      # a runner written in Python (the tests' stand-in) runs with this interpreter
    argv = [*prefix, "run", "-p", str(dataset_dir), "-a", harness.agent, "-m", harness.model_arg(model), "-o", str(jobs_dir), "--job-name", job_name,
            "-y", "-q", "-n", str(max(1, int(concurrent))), "-k", str(max(1, int(k)))]
    for name in tasks:
        argv += ["-i", name]
    if harness.setup and harness.setup != 1.0:
        argv += ["--agent-setup-timeout-multiplier", f"{harness.setup:g}"]
    agent_time = (bench.timeout_multiplier or 1.0) * (time_scale or 1.0)
    if agent_time != 1.0:
        argv += ["--agent-timeout-multiplier", f"{agent_time:g}"]
    for key, value in harness.kwargs:
        argv += ["--ak", f"{key}={harness.render(value, model=model, base_url=base_url, api_key=api_key)}"]
    base = base_url.rstrip("/")
    base_v1 = base if base.endswith("/v1") else base + "/v1"
    env = dict(base_env if base_env is not None else os.environ)
    env.update({"OPENAI_BASE_URL": base_v1, "OPENAI_API_BASE": base_v1, "OPENAI_API_KEY": api_key, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"})
    for key, value in getattr(bench, "env", ()) or ():
        env.setdefault(key, value.replace("{model}", model))
    for key, value in harness.env:
        env[key] = harness.render(value, model=model, base_url=base_url, api_key=api_key)
    if harness.path:
        env["PYTHONPATH"] = harness.path + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    return CellJob(argv=argv, env=env, job_dir=jobs_dir / job_name)


# ---------------------------------------------------------------------------------------------------------------------- reading
def _iso_seconds(span: Any) -> Optional[float]:
    from datetime import datetime
    if not isinstance(span, dict) or not span.get("started_at") or not span.get("finished_at"):
        return None
    try:
        a = datetime.fromisoformat(str(span["started_at"]).replace("Z", "+00:00"))
        b = datetime.fromisoformat(str(span["finished_at"]).replace("Z", "+00:00"))
        return round((b - a).total_seconds(), 1)
    except ValueError:
        return None


def short_task(name: str) -> str:
    """``org/name`` or ``name`` -> ``name`` (the dataset folder)."""
    return str(name or "").rsplit("/", 1)[-1]


def read_trial(result: dict[str, Any]) -> dict[str, Any]:
    """One trial row from the runner's ``result.json``."""
    exc = result.get("exception_info") or {}
    etype = str(exc.get("exception_type") or "")
    emsg = str(exc.get("exception_message") or "")
    rewards = ((result.get("verifier_result") or {}).get("rewards") or {})
    reward: Optional[float] = None
    if rewards:
        value = rewards.get("reward", next(iter(rewards.values())))
        try:
            reward = float(value)
        except (TypeError, ValueError):
            reward = None
    agent = result.get("agent_result") or {}
    info = result.get("agent_info") or {}
    outcome, passed, reason = "none", None, ""
    started = bool((result.get("agent_execution") or {}).get("started_at"))
    if reward is None and etype and not started:
        outcome, passed, reason = "none", None, f"setup:{etype}"
    elif reward is not None:
        passed = reward >= 1.0
        outcome = "pass" if passed else "fail"
        if not passed:
            reason = "timeout" if etype in AGENT_TIMEOUT else ("agent_error" if etype else "tests_failed")
    elif etype in AGENT_TIMEOUT:
        outcome, passed, reason = "fail", False, "timeout"
    elif etype and etype not in INFRA_ERRORS:
        outcome, passed, reason = "fail", False, "agent_error"
    elif etype:
        outcome, passed, reason = "none", None, etype
    else:
        outcome, passed, reason = "none", None, "unfinished"
    return {
        "trial_name": result.get("trial_name") or "",
        "task": short_task(result.get("task_name") or (result.get("task_id") or {}).get("name") or ""),
        "outcome": outcome, "passed": passed, "reward": reward, "reason": reason,
        "error_type": etype, "error": emsg[:2000],
        "tokens_in": agent.get("n_input_tokens"), "tokens_out": agent.get("n_output_tokens"), "tokens_cache": agent.get("n_cache_tokens"),
        "agent_s": _iso_seconds(result.get("agent_execution")), "setup_s": _iso_seconds(result.get("agent_setup")),
        "total_s": _iso_seconds({"started_at": result.get("started_at"), "finished_at": result.get("finished_at")}),
        "agent_version": str(info.get("version") or ""), "started_at": result.get("started_at") or "", "finished_at": result.get("finished_at") or "",
    }


def read_job(job_dir: Path) -> list[dict[str, Any]]:
    """Every finished trial of a job folder (a trial still running has an empty or missing ``result.json`` and is skipped)."""
    rows: list[dict[str, Any]] = []
    if not job_dir.is_dir():
        return rows
    for trial in sorted(p for p in job_dir.iterdir() if p.is_dir()):
        path = trial / "result.json"
        try:
            data = json.loads(path.read_text(encoding="utf-8")) if path.is_file() and path.stat().st_size else None
        except (OSError, ValueError):
            data = None
        if not isinstance(data, dict) or not data.get("finished_at"):
            continue
        row = read_trial(data)
        row["trial_dir"] = str(trial)
        rows.append(row)
    return rows


def job_finished(job_dir: Path) -> bool:
    path = job_dir / "result.json"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return bool(data.get("finished_at"))


def trial_logs(trial_dir: Path, *, limit: int = 20_000) -> dict[str, str]:
    """The text files a trial left (agent log, verifier output), each cut to its last ``limit`` characters, for the cell view."""
    out: dict[str, str] = {}
    for sub in ("agent", "verifier"):
        folder = trial_dir / sub
        if not folder.is_dir():
            continue
        for path in sorted(folder.rglob("*")):
            if path.is_file() and path.suffix in (".txt", ".log", ".md", ".json", ".ndjson", ".err") and path.stat().st_size < 50_000_000:
                try:
                    text = path.read_text(encoding="utf-8", errors="replace")
                except OSError:
                    continue
                out[path.relative_to(trial_dir).as_posix()] = text[-limit:]
    return out
