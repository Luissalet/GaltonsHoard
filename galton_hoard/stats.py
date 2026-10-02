"""Statistics: intervals for pass rates and mean scores, paired comparison of two models, speed summaries, arena ratings.

Everything is plain Python and seeded, so the same results always give the same numbers. Results of the same case are first averaged (a case
is the unit, repeats only reduce noise), so asking a case ten times does not make a model look ten times better measured.
"""

from __future__ import annotations

import math
import random
from collections import defaultdict
from functools import lru_cache
from typing import Any, Iterable, Optional, Sequence

from .messages import text
from .util import percentile

Z95 = 1.959964
BOOTSTRAP_RESAMPLES = 2000
MIN_CASES_WARNING = 20
TIE_EPS = 1e-9


# ------------------------------------------------------------------------------------------------- intervals
def wilson(successes: float, n: float, z: float = Z95) -> tuple[float, float]:
    """Wilson score interval for a proportion. ``(0, 1)`` when there is nothing to measure."""
    if n <= 0:
        return 0.0, 1.0
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def weighted_mean(values: Sequence[float], weights: Sequence[float]) -> float:
    total = sum(weights)
    return sum(v * w for v, w in zip(values, weights)) / total if total else 0.0


def bootstrap_ci(values: Sequence[float], weights: Optional[Sequence[float]] = None, *, resamples: int = BOOTSTRAP_RESAMPLES, seed: int = 20251,
                 alpha: float = 0.05) -> tuple[float, float, float]:
    """``(mean, low, high)``: weighted mean with a percentile bootstrap interval. A single value has no spread: its interval is the whole 0..1 range of doubt."""
    n = len(values)
    if n == 0:
        return 0.0, 0.0, 1.0
    w = tuple(weights) if weights is not None else (1.0,) * n
    return _bootstrap(tuple(values), w, resamples, seed, alpha)


@lru_cache(maxsize=1024)
def _bootstrap(values: tuple[float, ...], weights: tuple[float, ...], resamples: int, seed: int, alpha: float) -> tuple[float, float, float]:
    n = len(values)
    mean = weighted_mean(values, weights)
    if n == 1:
        return mean, 0.0, 1.0
    rng = random.Random(seed)
    indices = range(n)
    means = []
    for _ in range(resamples):
        pick = rng.choices(indices, k=n)
        total = sum(weights[i] for i in pick)
        means.append(sum(values[i] * weights[i] for i in pick) / total if total else 0.0)
    means.sort()
    lo = means[int((alpha / 2) * resamples)]
    hi = means[min(resamples - 1, int((1 - alpha / 2) * resamples))]
    return mean, lo, hi


# ------------------------------------------------------------------------------------------------- per-case aggregation
def per_case(rows: Iterable[dict[str, Any]]) -> dict[str, dict[str, float]]:
    """Average the repeats of each case: ``case_id -> {score, passed, weight}`` (weight is the case weight times the mean confidence of its results)."""
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        groups[r["case_id"]].append(r)
    out = {}
    for case_id, items in groups.items():
        out[case_id] = {"score": sum(i["score"] for i in items) / len(items), "passed": sum(1 for i in items if i["passed"]) / len(items),
                        "weight": (sum(i["weight"] for i in items) / len(items)) * (sum(i.get("confidence", 1.0) for i in items) / len(items)), "n": len(items)}
    return out


def summarize(rows: Iterable[dict[str, Any]], *, seed: int = 20251) -> dict[str, Any]:
    """Mean score with its bootstrap interval, pass rate with its Wilson interval, over the cases in ``rows``."""
    cases = per_case(rows)
    n = len(cases)
    if n == 0:
        return {"n": 0, "n_results": 0, "score": None, "ci": None, "pass_rate": None, "pass_ci": None}
    ids = sorted(cases)
    scores = [cases[i]["score"] for i in ids]
    weights = [cases[i]["weight"] for i in ids]
    mean, lo, hi = bootstrap_ci(scores, weights, seed=seed)
    passed = sum(cases[i]["passed"] for i in ids)
    plo, phi = wilson(passed, n)
    return {"n": n, "n_results": sum(c["n"] for c in cases.values()), "score": round(mean, 4), "ci": [round(lo, 4), round(hi, 4)],
            "pass_rate": round(passed / n, 4), "pass_ci": [round(plo, 4), round(phi, 4)]}


#: a contestant with more than this share of cut-off results in a category is flagged: its score there understates the model
TRUNCATED_WARN_SHARE = 0.05


def truncation(rows: Iterable[dict[str, Any]], key: str = "category") -> dict[str, dict[str, Any]]:
    """Per ``key`` (the category by default): how many results there are, how many ended with no answer because the token budget ran out, their
    share, and whether the share is above ``TRUNCATED_WARN_SHARE``."""
    total: dict[str, int] = defaultdict(int)
    cut: dict[str, int] = defaultdict(int)
    for r in rows:
        total[r[key]] += 1
        cut[r[key]] += 1 if r.get("truncated") else 0
    return {k: {"results": total[k], "truncated": cut[k], "share": round(cut[k] / total[k], 4), "warn": cut[k] / total[k] > TRUNCATED_WARN_SHARE} for k in sorted(total)}


# ------------------------------------------------------------------------------------------------- paired comparison
def binomial_two_sided(k: int, n: int) -> float:
    """Exact two-sided p-value of ``k`` successes out of ``n`` at p = 0.5 (the McNemar exact test on the discordant pairs)."""
    if n == 0:
        return 1.0
    k = min(k, n - k)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n)
    return min(1.0, 2 * tail)


def compare(rows_a: Iterable[dict[str, Any]], rows_b: Iterable[dict[str, Any]], *, seed: int = 20251) -> dict[str, Any]:
    """A against B on the cases both answered: wins/losses/ties, McNemar's exact test on pass/fail, paired bootstrap on the score difference."""
    a, b = per_case(rows_a), per_case(rows_b)
    common = sorted(set(a) & set(b))
    n = len(common)
    out: dict[str, Any] = {"n": n, "only_a": len(set(a) - set(b)), "only_b": len(set(b) - set(a)), "warnings": []}
    if n == 0:
        out.update(wins=0, losses=0, ties=0, verdict="no_data", p_value=None, mcnemar={"b": 0, "c": 0, "p": None}, diff=None, diff_ci=None, per_case=[])
        out["warnings"].append(text("cmp_no_shared"))
        return out
    diffs = [a[c]["score"] - b[c]["score"] for c in common]
    weights = [(a[c]["weight"] + b[c]["weight"]) / 2 for c in common]
    wins = sum(1 for d in diffs if d > TIE_EPS)
    losses = sum(1 for d in diffs if d < -TIE_EPS)
    pa = [a[c]["passed"] >= 0.5 for c in common]
    pb = [b[c]["passed"] >= 0.5 for c in common]
    only_a_pass = sum(1 for x, y in zip(pa, pb) if x and not y)
    only_b_pass = sum(1 for x, y in zip(pa, pb) if y and not x)
    p = binomial_two_sided(only_a_pass, only_a_pass + only_b_pass)
    mean, lo, hi = bootstrap_ci(diffs, weights, seed=seed) if n > 1 else (diffs[0], -1.0, 1.0)
    significant = p < 0.05 and (lo > 0 or hi < 0)
    if n < 8:
        verdict = "no_clear_difference"
    elif significant and mean > 0:
        verdict = "better"
    elif significant and mean < 0:
        verdict = "worse"
    else:
        verdict = "no_clear_difference"
    if n < MIN_CASES_WARNING:
        out["warnings"].append(text("cmp_few_cases", n=n, min=MIN_CASES_WARNING))
    if (p < 0.05) != (lo > 0 or hi < 0):
        out["warnings"].append(text("cmp_disagree"))
    out.update(wins=wins, losses=losses, ties=n - wins - losses, verdict=verdict, p_value=round(p, 4), mcnemar={"b": only_a_pass, "c": only_b_pass, "p": round(p, 4)},
               diff=round(mean, 4), diff_ci=[round(lo, 4), round(hi, 4)],
               per_case=[{"case_id": c, "a": round(a[c]["score"], 3), "b": round(b[c]["score"], 3), "diff": round(d, 3)} for c, d in zip(common, diffs)])
    return out


# ------------------------------------------------------------------------------------------------- speed
def speed_summary(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    rows = list(rows)
    decode = [r["decode_tps"] for r in rows if r.get("decode_tps")]
    prompt = [r["prompt_tps"] for r in rows if r.get("prompt_tps")]
    ttft = [r["ttft_ms"] for r in rows if r.get("ttft_ms")]
    r1 = lambda x: round(x, 1) if x is not None else None  # noqa: E731
    return {"n": len(decode), "decode_tps_median": r1(percentile(decode, 50)), "decode_tps_p90": r1(percentile(decode, 90)), "prompt_tps_median": r1(percentile(prompt, 50)),
            "ttft_ms_median": r1(percentile(ttft, 50))}


# ------------------------------------------------------------------------------------------------- arena
Vote = tuple[str, str, str]  # (a, b, "a" | "b" | "tie" | "both_bad")


def bradley_terry(votes: Sequence[Vote], *, iterations: int = 200) -> dict[str, float]:
    """Bradley-Terry strengths by the MM algorithm; ties count half a win each, ``both_bad`` carries no information. Returned on an Elo-like scale (mean 1000)."""
    wins: dict[tuple[str, str], float] = defaultdict(float)
    players: set[str] = set()
    for a, b, outcome in votes:
        if outcome == "both_bad":
            continue
        players.update((a, b))
        if outcome == "a":
            wins[(a, b)] += 1
        elif outcome == "b":
            wins[(b, a)] += 1
        else:
            wins[(a, b)] += 0.5
            wins[(b, a)] += 0.5
    if not players:
        return {}
    strength = {p: 1.0 for p in players}
    total_wins = {p: sum(w for (x, _), w in wins.items() if x == p) for p in players}
    pairs: dict[str, list[tuple[str, float]]] = defaultdict(list)
    games: dict[tuple[str, str], float] = defaultdict(float)
    for (x, y), w in wins.items():
        games[tuple(sorted((x, y)))] += w  # type: ignore[arg-type]
    for (x, y), g in games.items():
        pairs[x].append((y, g))
        pairs[y].append((x, g))
    for _ in range(iterations):
        new = {}
        for p in players:
            denom = sum(g / (strength[p] + strength[q]) for q, g in pairs[p])
            new[p] = (total_wins[p] + 0.01) / (denom + 0.02 / (strength[p] + 1.0))
        scale = math.exp(sum(math.log(v) for v in new.values()) / len(new))
        new = {p: v / scale for p, v in new.items()}
        if max(abs(new[p] - strength[p]) for p in players) < 1e-9:
            strength = new
            break
        strength = new
    return {p: 1000 + 400 * math.log10(v) for p, v in strength.items()}


def arena_ratings(votes: Sequence[Vote], *, min_votes: int = 5, resamples: int = 200, seed: int = 20251) -> list[dict[str, Any]]:
    """Ratings with bootstrap intervals over the votes. A model with fewer than ``min_votes`` decisive comparisons is listed without a rating."""
    counts: dict[str, int] = defaultdict(int)
    for a, b, outcome in votes:
        if outcome != "both_bad":
            counts[a] += 1
            counts[b] += 1
    base = bradley_terry(votes)
    rng = random.Random(seed)
    samples: dict[str, list[float]] = defaultdict(list)
    if len(votes) >= 2:
        for _ in range(resamples):
            for p, r in bradley_terry([votes[rng.randrange(len(votes))] for _ in votes], iterations=60).items():
                samples[p].append(r)
    out = []
    for p, rating in base.items():
        shown = counts[p] >= min_votes
        s = sorted(samples[p])
        ci = [round(s[int(0.025 * len(s))], 1), round(s[min(len(s) - 1, int(0.975 * len(s)))], 1)] if len(s) >= 20 else None
        out.append({"contestant": p, "votes": counts[p], "rating": round(rating, 1) if shown else None, "ci": ci if shown else None, "shown": shown})
    return sorted(out, key=lambda r: (r["rating"] is None, -(r["rating"] or 0)))
