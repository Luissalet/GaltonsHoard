"""Run the bench (tasks x variants x pressures x k samples) and summarise it per variant and per variant x pressure.

``generate(system, user, sample) -> str`` is any callable; ``ChatGenerator`` is the real one over a chat-completions compatible
endpoint (``backends.OpenAIBackend``), with temperature above zero so the k samples differ.
"""

from __future__ import annotations

import math
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Iterable, Optional

from ..backends import ChatRequest, OpenAIBackend
from .scoring import Sample, pass_at_k, pass_hat_k, score_answer
from .tasks import TASKS
from .variants import PRESSURES, VARIANTS, system_prompt, user_prompt

Generate = Callable[[str, str, int], str]

#: The acceptance line of the first measurement: a variant earns the default when it cuts the lines changed outside the
#: request by this fraction against A without lowering pass@k.
TARGET_EXCESS_CUT = 0.30


class ChatGenerator:
    def __init__(self, base_url: str, model: str, *, temperature: float = 0.7, max_tokens: int = 2048, timeout_s: float = 300.0,
                 effort: Optional[str] = None, seed: Optional[int] = None, transport=None):
        self.backend = OpenAIBackend(base_url, model, transport=transport)
        self.temperature, self.max_tokens, self.timeout_s, self.effort, self.seed = temperature, max_tokens, timeout_s, effort, seed

    def __call__(self, system: str, user: str, sample: int) -> str:
        seed = None if self.seed is None else self.seed + sample
        done = self.backend.chat(ChatRequest(messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                                             temperature=self.temperature, max_tokens=self.max_tokens, timeout_s=self.timeout_s,
                                             effort=self.effort, seed=seed))
        if done.error:
            raise RuntimeError(done.error)
        return done.text


def select_tasks(only: Iterable[str] = (), kinds: Iterable[str] = ()) -> list[dict]:
    only, kinds = set(only), set(kinds)
    return [t for t in TASKS if (not only or t["id"] in only) and (not kinds or t["kind"] in kinds)]


def run_bench(tasks: list[dict], generate: Generate, *, variants: Iterable[str] = ("A", "B", "C"),
              pressures: Iterable[str] = ("neutral",), k: int = 5, rules_override: Optional[dict[str, str]] = None,
              timeout_s: float = 60.0, on_sample: Optional[Callable[[Sample], None]] = None, keep_answers: bool = True,
              concurrency: int = 1) -> list[Sample]:
    """Every (task, variant, pressure, sample) once; with ``concurrency`` > 1 the calls run in parallel threads and the
    returned list keeps the same order as a sequential run."""
    variants, pressures = list(variants), list(pressures)
    for v in variants:
        if v not in VARIANTS and v not in (rules_override or {}):
            raise ValueError(f"unknown variant {v!r}")
    for p in pressures:
        if p not in PRESSURES:
            raise ValueError(f"unknown pressure {p!r}")
    jobs = []
    for task in tasks:
        for variant in variants:
            system = system_prompt(variant if variant in VARIANTS else "A", (rules_override or {}).get(variant, ""))
            for pressure in pressures:
                user = user_prompt(task["request"], task["files"], pressure)
                jobs.extend((task, variant, pressure, system, user, i) for i in range(k))

    report_lock = threading.Lock()

    def one(job) -> Sample:
        task, variant, pressure, system, user, i = job
        try:
            answer, error = generate(system, user, i), ""
        except Exception as exc:  # a failed call is a scored, explained sample, never a crash of the run
            answer, error = "", f"{type(exc).__name__}: {exc}"[:300]
        s = score_answer(task, answer, variant=variant, pressure=pressure, sample=i, error=error, timeout_s=timeout_s,
                         keep_answer=keep_answers)
        if on_sample:
            with report_lock:
                on_sample(s)
        return s

    if concurrency <= 1:
        return [one(j) for j in jobs]
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        return list(pool.map(one, jobs))


def _block(rows: list[Sample], k: int) -> dict[str, Any]:
    by_task: dict[str, list[Sample]] = {}
    for r in rows:
        by_task.setdefault(f"{r.task_id}|{r.pressure}", []).append(r)
    at_k, hat_k = [], []
    for group in by_task.values():
        n, c = len(group), sum(s.passed for s in group)
        kk = min(k, n)
        at_k.append(pass_at_k(n, c, kk))
        hat_k.append(pass_hat_k(n, c, kk))
    n = len(rows)
    tests = [r for r in rows if r.wrote_test]
    mean = lambda xs: round(sum(xs) / len(xs), 4) if xs else None  # noqa: E731
    return {
        "samples": n,
        "errors": sum(1 for r in rows if r.error),
        "pass@1": mean([float(r.passed) for r in rows]),
        f"pass@{k}": mean([x for x in at_k if not math.isnan(x)]),
        f"pass^{k}": mean([x for x in hat_k if not math.isnan(x)]),
        "mean_lines_changed": mean([r.lines_changed for r in rows]),
        "mean_excess_lines": mean([r.excess_lines for r in rows]),
        "outside_file_rate": mean([float(bool(r.outside_files)) for r in rows]),
        "kept_missing_rate": mean([float(bool(r.kept_missing)) for r in rows]),
        "test_written_rate": mean([float(r.wrote_test) for r in rows]),
        "test_first_rate": mean([float(r.test_first) for r in tests]),
    }


def summarise(samples: list[Sample], k: int) -> dict[str, Any]:
    variants = sorted({s.variant for s in samples})
    pressures = sorted({s.pressure for s in samples})
    out: dict[str, Any] = {"k": k, "by_variant": {}, "by_variant_pressure": {}, "by_kind": {}, "verdict": {}}
    for v in variants:
        out["by_variant"][v] = _block([s for s in samples if s.variant == v], k)
        for p in pressures:
            rows = [s for s in samples if s.variant == v and s.pressure == p]
            if rows:
                out["by_variant_pressure"][f"{v}/{p}"] = _block(rows, k)
        for kind in sorted({s.kind for s in samples}):
            rows = [s for s in samples if s.variant == v and s.kind == kind]
            if rows:
                out["by_kind"][f"{v}/{kind}"] = _block(rows, k)
    out["verdict"] = _verdict(out["by_variant"], k)
    # per pressure as well: under "contra" the user asks for the tidy-up, so pooling it hides what the rules do unprompted
    out["verdict_by_pressure"] = {}
    for p in pressures:
        rows = {v: out["by_variant_pressure"][f"{v}/{p}"] for v in variants if f"{v}/{p}" in out["by_variant_pressure"]}
        verdict = _verdict(rows, k)
        if verdict:
            out["verdict_by_pressure"][p] = verdict
    return out


def _verdict(rows: dict[str, dict], k: int) -> dict[str, Any]:
    base = rows.get("A")
    if not base:
        return {}
    out = {}
    for v, row in rows.items():
        if v == "A":
            continue
        b_excess, v_excess = base["mean_excess_lines"] or 0.0, row["mean_excess_lines"] or 0.0
        cut = (b_excess - v_excess) / b_excess if b_excess else (0.0 if v_excess == 0 else -1.0)
        key = f"pass@{k}"
        holds = (row[key] or 0.0) >= (base[key] or 0.0)
        out[v] = {"excess_cut_vs_A": round(cut, 4), "pass_at_k_held": holds, "meets_target": bool(holds and cut >= TARGET_EXCESS_CUT)}
    return out


def samples_from_report(report: dict[str, Any]) -> list[Sample]:
    """The samples of a saved JSON report, to summarise it again without calling any model."""
    fields = set(Sample.__dataclass_fields__)
    return [Sample(**{key: val for key, val in row.items() if key in fields}) for row in report.get("samples", [])]
