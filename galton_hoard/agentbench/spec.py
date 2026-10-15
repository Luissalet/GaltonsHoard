"""Harness lines and the benchmark catalogue of the agent bench.

A harness is written in the settings (``agentbench.harnesses``), one line each::

    name = agent [model=TEMPLATE] [path=FOLDER] [setup=N] [mcp=yes|no] [ak.KEY=VALUE ...] [env.KEY=VALUE ...]

``agent`` is an agent the trial runner knows by name or an import path ``package.module:Class``; ``model`` is how that agent wants the model written
(``{model}`` is replaced by the campaign's model, default ``openai/{model}``); ``path`` is a folder put in front of ``PYTHONPATH`` so an agent given by
import path can be found; ``setup`` multiplies the time the runner gives the agent to install itself; ``mcp=no`` says the agent cannot use tools served
over MCP (benchmarks that need them are skipped for it); ``ak.`` lines are keyword arguments of the agent and ``env.`` lines environment variables. In
values, ``{model}``, ``{base_url}`` (ending in /v1), ``{root_url}`` (the same without /v1) and ``{api_key}`` are replaced. A line that starts with ``#`` is
switched off. Nothing about any particular agent is built in.

A benchmark is a published dataset of the runner's registry plus a fixed recipe for the subset Galton uses: filters on the task metadata, a seed, how
many tasks, how many of them form the *dev* split (where a harness may be tuned) and the rest the *test* split (only ever measured). The subset only
depends on the task list and the recipe, so every harness and every later run gets exactly the same tasks.
"""

from __future__ import annotations

import fnmatch
import hashlib
import re
import shlex
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python < 3.11
    tomllib = None  # type: ignore[assignment]

NAME_RE = re.compile(r"^[a-z0-9][a-z0-9_.-]{0,39}$")
PLACEHOLDERS = ("{model}", "{base_url}", "{root_url}", "{api_key}")


@dataclass(frozen=True)
class HarnessSpec:
    name: str
    agent: str
    model: str = "openai/{model}"
    path: str = ""
    setup: float = 1.0
    mcp: bool = True
    kwargs: tuple[tuple[str, str], ...] = ()
    env: tuple[tuple[str, str], ...] = ()
    enabled: bool = True

    def render(self, value: str, *, model: str, base_url: str, api_key: str) -> str:
        base = base_url.rstrip("/")
        base_v1 = base if base.endswith("/v1") else base + "/v1"
        root = base_v1[: -len("/v1")]
        for key, val in (("{model}", model), ("{base_url}", base_v1), ("{root_url}", root), ("{api_key}", api_key)):
            value = value.replace(key, val)
        return value

    def model_arg(self, model: str) -> str:
        return self.model.replace("{model}", model)

    def info(self) -> dict[str, Any]:
        return {"name": self.name, "agent": self.agent, "model": self.model, "path": self.path, "setup": self.setup, "mcp": self.mcp,
                "kwargs": dict(self.kwargs), "env": sorted(k for k, _ in self.env), "enabled": self.enabled}


def _split(text: str) -> list[str]:
    try:
        return shlex.split(text, posix=True)
    except ValueError:
        return text.split()


def parse_harness_line(raw: str) -> Optional[HarnessSpec]:
    text = str(raw or "").strip()
    if not text:
        return None
    enabled = not text.startswith("#")
    text = text.lstrip("# ").strip()
    name, sep, rest = text.partition("=")
    name = name.strip().lower()
    if not sep or not NAME_RE.match(name):
        return None
    tokens = _split(rest.replace("\\", "\\\\")) if "\\" in rest else _split(rest)
    if not tokens:
        return None
    agent, options = tokens[0], tokens[1:]
    fields: dict[str, Any] = {"name": name, "agent": agent, "enabled": enabled}
    kwargs: list[tuple[str, str]] = []
    env: list[tuple[str, str]] = []
    for opt in options:
        key, eq, value = opt.partition("=")
        if not eq:
            return None
        if key.startswith("ak.") and len(key) > 3:
            kwargs.append((key[3:], value))
        elif key.startswith("env.") and len(key) > 4:
            env.append((key[4:], value))
        elif key == "model":
            fields["model"] = value
        elif key == "path":
            fields["path"] = value
        elif key == "setup":
            try:
                fields["setup"] = max(0.1, min(float(value), 50.0))
            except ValueError:
                return None
        elif key == "mcp":
            fields["mcp"] = value.strip().lower() not in ("no", "false", "0", "off")
        else:
            return None
    return HarnessSpec(kwargs=tuple(kwargs), env=tuple(env), **fields)


def parse_harnesses(lines: Iterable[str]) -> list[HarnessSpec]:
    out: list[HarnessSpec] = []
    seen: set[str] = set()
    for line in lines or []:
        spec = parse_harness_line(line)
        if spec and spec.name not in seen:
            seen.add(spec.name)
            out.append(spec)
    return out


# ---------------------------------------------------------------------------------------------------------------------- benchmarks
@dataclass(frozen=True)
class Benchmark:
    id: str
    dataset: str                     # registry reference of the trial runner
    title: str
    kind: str                        # terminal | repo_fix | edit | tools_dialog | algorithmic
    n: int                           # tasks in Galton's subset
    dev: int                         # how many of them form the dev split
    seed: int = 11
    difficulty: tuple[str, ...] = ()      # keep only these difficulty labels (empty: all)
    max_agent_s: float = 0.0              # keep only tasks whose own agent time limit is at most this (0: any)
    max_cpus: int = 0
    max_memory_mb: int = 0
    exclude: tuple[str, ...] = ()         # name globs left out
    include: tuple[str, ...] = ()         # name globs kept (empty: all)
    stratify: str = ""                    # regex with one group over the task name: tasks are drawn evenly from each group
    needs_mcp: bool = False
    timeout_multiplier: float = 1.0
    note: str = ""
    extra: dict[str, Any] = field(default_factory=dict)
    #: environment for the benchmark's own services, ``{model}`` = the campaign's model (an
    #: explicit variable in the caller's environment wins)
    env: tuple[tuple[str, str], ...] = ()

    def info(self) -> dict[str, Any]:
        return {"id": self.id, "dataset": self.dataset, "title": self.title, "kind": self.kind, "n": self.n, "dev": self.dev, "test": self.n - self.dev,
                "seed": self.seed, "needs_mcp": self.needs_mcp, "note": self.note}


CATALOGUE: tuple[Benchmark, ...] = (
    Benchmark("tb2", "terminal-bench/terminal-bench-2", "Terminal-Bench 2.0", "terminal", n=24, dev=8, difficulty=("easy", "medium"),
              max_agent_s=1800, max_cpus=2, max_memory_mb=4096,
              note="Real terminal work in a container: build, debug, configure, data. Hidden tests decide."),
    Benchmark("swev", "swe-bench/swe-bench-verified", "SWE-bench Verified", "repo_fix", n=24, dev=8,
              difficulty=("<15 min fix", "15 min - 1 hour"),
              note="A real issue of a real Python repository; the fix must make the failing tests pass and keep the others passing."),
    Benchmark("polyglot", "aider/aider-polyglot", "Polyglot (Exercism)", "edit", n=30, dev=10, stratify=r"^(?:[^/]*/)?([a-z+]+?)[_-]",
              note="Exercism exercises in six languages; the agent edits the solution file, the unit tests decide."),
    Benchmark("tau3", "sierra-research/tau3-bench", "tau3-bench (retail, airline)", "tools_dialog", n=24, dev=8, include=("*retail*", "*airline*"),
              stratify=r"(retail|airline)", needs_mcp=True,
              # the simulated customer and the judge of the agent's statements run on the
              # campaign's own model, through the same OpenAI-compatible endpoint
              env=(("TAU2_USER_MODEL", "openai/{model}"), ("TAU2_NL_ASSERTIONS_MODEL", "openai/{model}")),
              note="Customer-service dialogues with a simulated user and a policy; the database state and what the agent tells the user decide."),
    Benchmark("lcb", "livecodebench/livecodebench", "LiveCodeBench", "algorithmic", n=24, dev=8,
              note="Competitive programming problems released after the usual training cut-offs; hidden tests decide."),
)
BY_ID = {b.id: b for b in CATALOGUE}


def benchmark(ref: str) -> Benchmark:
    try:
        return BY_ID[str(ref).strip().lower()]
    except KeyError as exc:
        raise KeyError(f"unknown benchmark {ref!r}; known: {', '.join(BY_ID)}") from exc


# ---------------------------------------------------------------------------------------------------------------------- task subsets
@dataclass(frozen=True)
class TaskMeta:
    name: str            # folder name in the downloaded dataset
    difficulty: str = ""
    category: str = ""
    agent_s: float = 0.0
    cpus: int = 0
    memory_mb: int = 0


def read_task(folder: Path) -> TaskMeta:
    meta: dict[str, Any] = {}
    toml = folder / "task.toml"
    if toml.is_file() and tomllib is not None:
        try:
            meta = tomllib.loads(toml.read_text(encoding="utf-8"))
        except (ValueError, OSError):
            meta = {}
    m, env, ag = meta.get("metadata") or {}, meta.get("environment") or {}, meta.get("agent") or {}
    def num(value: Any) -> float:
        try:
            return float(value or 0)
        except (TypeError, ValueError):
            return 0.0
    return TaskMeta(folder.name, str(m.get("difficulty") or ""), str(m.get("category") or ""), num(ag.get("timeout_sec")),
                    int(num(env.get("cpus"))), int(num(env.get("memory_mb"))))


def read_dataset(folder: Path) -> list[TaskMeta]:
    if not folder.is_dir():
        return []
    return [read_task(p) for p in sorted(folder.iterdir()) if p.is_dir() and (p / "instruction.md").exists()]


def _rank(seed: int, name: str) -> str:
    return hashlib.sha256(f"{seed}:{name}".encode()).hexdigest()


def select(bench: Benchmark, tasks: list[TaskMeta]) -> dict[str, list[str]]:
    """The benchmark's subset of ``tasks``: ``{"dev": [...], "test": [...], "pool": n}``. Deterministic in the task names and the recipe."""
    pool = []
    for t in tasks:
        if bench.difficulty and t.difficulty not in bench.difficulty:
            continue
        if bench.max_agent_s and t.agent_s and t.agent_s > bench.max_agent_s:
            continue
        if bench.max_cpus and t.cpus > bench.max_cpus:
            continue
        if bench.max_memory_mb and t.memory_mb > bench.max_memory_mb:
            continue
        if bench.include and not any(fnmatch.fnmatch(t.name, g) for g in bench.include):
            continue
        if any(fnmatch.fnmatch(t.name, g) for g in bench.exclude):
            continue
        pool.append(t.name)
    ordered = sorted(pool, key=lambda name: _rank(bench.seed, name))
    rx = re.compile(bench.stratify) if bench.stratify else None
    groups: dict[str, list[str]] = {}
    for name in ordered:
        m = rx.search(name) if rx else None
        groups.setdefault(m.group(1) if m else "", []).append(name)
    keys = sorted(groups)
    chosen: dict[str, list[str]] = {k: [] for k in keys}
    total = 0
    while total < bench.n and any(groups[k] for k in keys):
        for k in keys:
            if groups[k] and total < bench.n:
                chosen[k].append(groups[k].pop(0))
                total += 1
    # the dev split takes the same share of every group, so both splits keep the mix of the subset
    dev_n = min(bench.dev, total)
    quota = {k: (len(chosen[k]) * dev_n) // max(1, total) for k in keys}
    rest = dev_n - sum(quota.values())
    for k in sorted(keys, key=lambda k: -((len(chosen[k]) * dev_n) % max(1, total))):
        if rest <= 0:
            break
        if quota[k] < len(chosen[k]):
            quota[k] += 1
            rest -= 1
    dev = [name for k in keys for name in chosen[k][: quota[k]]]
    test = [name for k in keys for name in chosen[k][quota[k]:]]
    return {"dev": sorted(dev, key=lambda n: _rank(bench.seed, n)), "test": sorted(test, key=lambda n: _rank(bench.seed, n)), "pool": len(pool)}
