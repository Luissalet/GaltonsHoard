"""Cells, ranking and paired comparisons of the agent bench.

A cell is one benchmark x one harness. Its pass rate counts only verified trials (outcome ``pass`` or ``fail``); ``none`` trials are reported
apart and never enter a denominator. With several attempts per task the rate is over trials, and ``tasks_solved`` counts tasks with at least one pass.

The ranking of a benchmark sorts harnesses by pass rate. Two harnesses are compared fairly only on the tasks both have a verified result for: the
paired comparison counts the tasks one solves and the other does not, both ways, and gives the exact two-sided sign-test p-value of that split. A
lead with a p-value above 0.05 is a lead on this sample, not evidence; the text says so.
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Iterable, Optional

Z95 = 1.959963984540054


def wilson(k: int, n: int, z: float = Z95) -> tuple[Optional[float], Optional[float]]:
    if n <= 0:
        return None, None
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return round(max(0.0, centre - half), 4), round(min(1.0, centre + half), 4)


def sign_test(wins: int, losses: int) -> float:
    """Exact two-sided binomial p-value for ``wins`` vs ``losses`` under p = 1/2 (ties left out)."""
    n = wins + losses
    if n == 0:
        return 1.0
    k = min(wins, losses)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return round(min(1.0, 2 * tail), 4)


def _mean(values: Iterable[Optional[float]]) -> Optional[float]:
    vals = [float(v) for v in values if isinstance(v, (int, float))]
    return round(sum(vals) / len(vals), 1) if vals else None


def cell(trials: list[dict[str, Any]]) -> dict[str, Any]:
    verified = [t for t in trials if t.get("outcome") in ("pass", "fail")]
    none = [t for t in trials if t.get("outcome") not in ("pass", "fail")]
    k = sum(1 for t in verified if t.get("passed"))
    n = len(verified)
    low, high = wilson(k, n)
    per_task: dict[str, list[bool]] = defaultdict(list)
    for t in verified:
        per_task[t["task"]].append(bool(t.get("passed")))
    reasons: dict[str, int] = defaultdict(int)
    for t in verified:
        if not t.get("passed"):
            reasons[t.get("reason") or "tests_failed"] += 1
    none_reasons: dict[str, int] = defaultdict(int)
    for t in none:
        none_reasons[t.get("reason") or "unknown"] += 1
    return {
        "n": n, "passed": k, "rate": round(k / n, 4) if n else None, "ci": [low, high],
        "tasks": len(per_task), "tasks_solved": sum(1 for v in per_task.values() if any(v)),
        "tasks_all": sum(1 for v in per_task.values() if v and all(v)),
        "none_n": len(none), "none_reasons": dict(none_reasons), "fail_reasons": dict(reasons),
        "tokens_in_mean": _mean(t.get("tokens_in") for t in verified), "tokens_out_mean": _mean(t.get("tokens_out") for t in verified),
        "agent_s_mean": _mean(t.get("agent_s") for t in verified),
    }


def task_results(trials: list[dict[str, Any]]) -> dict[str, float]:
    """Per task, the share of verified attempts that passed."""
    acc: dict[str, list[bool]] = defaultdict(list)
    for t in trials:
        if t.get("outcome") in ("pass", "fail"):
            acc[t["task"]].append(bool(t.get("passed")))
    return {task: sum(v) / len(v) for task, v in acc.items() if v}


def paired(a: list[dict[str, Any]], b: list[dict[str, Any]]) -> dict[str, Any]:
    """Harness A against B on the tasks both have a verified result for (a task counts as solved when most of its attempts passed)."""
    ra, rb = task_results(a), task_results(b)
    shared = sorted(set(ra) & set(rb))
    wins = sum(1 for t in shared if ra[t] > 0.5 >= rb[t])
    losses = sum(1 for t in shared if rb[t] > 0.5 >= ra[t])
    both = sum(1 for t in shared if ra[t] > 0.5 and rb[t] > 0.5)
    neither = len(shared) - wins - losses - both
    return {"shared": len(shared), "wins": wins, "losses": losses, "both": both, "neither": neither, "p": sign_test(wins, losses),
            "a_rate": round(sum(ra[t] for t in shared) / len(shared), 4) if shared else None,
            "b_rate": round(sum(rb[t] for t in shared) / len(shared), 4) if shared else None}


def board(trials: list[dict[str, Any]], *, focus: str = "", split: str = "") -> dict[str, Any]:
    """Every cell of a set of trials, the ranking per benchmark and, with ``focus``, its paired comparison against every other harness.

    ``split`` keeps only the trials of that split (``dev`` or ``test``)."""
    if split:
        trials = [t for t in trials if t.get("split") == split]
    by_cell: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for t in trials:
        by_cell[(t["benchmark"], t["harness"])].append(t)
    benches = sorted({b for b, _ in by_cell})
    harnesses = sorted({h for _, h in by_cell})
    cells = {f"{b}|{h}": cell(v) for (b, h), v in by_cell.items()}
    ranking: dict[str, list[dict[str, Any]]] = {}
    duels: dict[str, list[dict[str, Any]]] = {}
    for b in benches:
        rows = [{"harness": h, **cells[f"{b}|{h}"]} for h in harnesses if f"{b}|{h}" in cells]
        rows.sort(key=lambda r: (-(r["rate"] if r["rate"] is not None else -1), -r["n"], r["harness"]))
        for i, r in enumerate(rows, 1):
            r["rank"] = i if r["rate"] is not None else None
        ranking[b] = rows
        if focus and (b, focus) in by_cell:
            out = []
            for h in harnesses:
                if h != focus and (b, h) in by_cell:
                    out.append({"vs": h, **paired(by_cell[(b, focus)], by_cell[(b, h)])})
            duels[b] = out
    verdict = {}
    if focus:
        for b in benches:
            rows = [r for r in ranking[b] if r["rate"] is not None]
            mine = next((r for r in rows if r["harness"] == focus), None)
            if not mine:
                verdict[b] = {"state": "not_measured"}
                continue
            best_other = max((r["rate"] for r in rows if r["harness"] != focus), default=None)
            lead = None if best_other is None else round(mine["rate"] - best_other, 4)
            if best_other is None:
                state = "alone"
            elif mine["rate"] > best_other:
                state = "first"
            elif mine["rate"] == best_other:
                state = "tied_first"
            else:
                state = "behind"
            verdict[b] = {"state": state, "rank": mine["rank"], "rate": mine["rate"], "best_other": best_other, "lead": lead,
                          "weak_evidence": any(d["p"] > 0.05 for d in duels.get(b, []) if d["wins"] > d["losses"])}
    return {"benchmarks": benches, "harnesses": harnesses, "cells": cells, "ranking": ranking, "duels": duels, "verdict": verdict,
            "split": split or "all", "trials": len(trials)}
