"""Turn trials into the model x harness matrix: one cell per pair, margins, and a short automatic conclusion.

A trial whose verifier never ran (``outcome == "none"``: timeout, network, server error) is not a zero. It is left out of every rate and every
denominator and counted apart in the cell (``none_n`` and ``none_reasons``); a cell with nothing left shows no number at all. The conclusion compares
two spreads of pass@1 over the cells that have a number: how much a model moves when only the harness changes, and how much the models differ
under one harness.
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Any, Iterable, Optional

from ..rulebench.scoring import pass_at_k, pass_hat_k
from ..stats import wilson

WEAK_BELOW = 20             # fewer valid trials than this per cell: the conclusion says the evidence is weak
SENSITIVE_POINTS = 15.0     # a model whose best and worst harness differ by this many points depends on the harness
SIMILAR_POINTS = 2.0        # the two spreads differ by less than this: no clear difference
LOW_PASS = 0.5              # a model whose best harness stays under this fails whatever the harness


def _mean(values: Iterable[Optional[float]], digits: int = 3) -> Optional[float]:
    items = [float(v) for v in values if v is not None]
    return round(sum(items) / len(items), digits) if items else None


def _round(x: Optional[float], digits: int = 4) -> Optional[float]:
    return None if x is None or (isinstance(x, float) and math.isnan(x)) else round(x, digits)


def cell_stats(trials: list[dict[str, Any]], k: int = 1) -> dict[str, Any]:
    """Every metric of one (model, harness) cell. Rates use the trials that ran (not ``none``) as denominator."""
    valid = [t for t in trials if t["outcome"] != "none"]
    nones = [t for t in trials if t["outcome"] == "none"]
    n = len(valid)
    passed = sum(1 for t in valid if t["outcome"] == "pass")
    counts = Counter(t["outcome"] for t in trials)
    out: dict[str, Any] = {"trials": len(trials), "valid": n, "none_n": len(nones), "none_reasons": dict(Counter((t.get("detail") or t.get("reason") or "?")[:80] for t in nones)),
                           "passed": passed, "counts": {o: counts.get(o, 0) for o in ("pass", "fail_answer", "fail_format", "none")}}
    out["pass_at_1"] = _round(passed / n) if n else None
    out["ci"] = [_round(x) for x in wilson(passed, n)] if n else None
    out["format_fail_rate"] = _round(counts.get("fail_format", 0) / n) if n else None
    out["answer_fail_rate"] = _round(counts.get("fail_answer", 0) / n) if n else None
    own = [t for t in valid if t.get("format_failures") is not None]          # command-line agents do not expose their calls
    out["format_error_rate"] = _round(sum(1 for t in own if t["format_failures"] > 0) / len(own)) if own else None
    out["extra_calls_mean"] = _mean((t["extra_calls"] for t in own), 3)
    out["extra_rate"] = _round(sum(1 for t in own if (t["extra_calls"] or 0) > 0) / len(own)) if own else None
    out["turns_mean"] = _mean((t.get("turns") for t in valid), 2)
    out["calls_mean"] = _mean((t.get("calls") for t in valid), 2)
    out["tokens_out_mean"] = _mean((t.get("tokens_out") for t in valid), 1)
    out["tokens_in_mean"] = _mean((t.get("tokens_in") for t in valid), 1)
    out["latency_ms_mean"] = _mean((t.get("latency_ms") for t in valid), 1)
    if k > 1 and n:
        by_task: dict[str, list[bool]] = {}
        for t in valid:
            by_task.setdefault(t["task_id"], []).append(t["outcome"] == "pass")
        at_k, hat_k = [], []
        for results in by_task.values():
            size, good = len(results), sum(results)
            kk = min(k, size)
            at_k.append(pass_at_k(size, good, kk))
            hat_k.append(pass_hat_k(size, good, kk))
        out["pass_at_k"] = _mean((x for x in at_k if not math.isnan(x)), 4)
        out["pass_hat_k"] = _mean((x for x in hat_k if not math.isnan(x)), 4)
    return out


def build_matrix(trials: list[dict[str, Any]], model_ids: list[str], harnesses: list[str], k: int = 1) -> dict[str, Any]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for t in trials:
        grouped.setdefault((t["contestant_id"], t["harness"]), []).append(t)
    cells = {m: {h: cell_stats(grouped.get((m, h), []), k) for h in harnesses} for m in model_ids}
    by_model = {m: _mean((c["pass_at_1"] for c in cells[m].values()), 4) for m in model_ids}
    by_harness = {h: _mean((cells[m][h]["pass_at_1"] for m in model_ids), 4) for h in harnesses}
    return {"cells": cells, "by_model": by_model, "by_harness": by_harness}


def _pts(x: float) -> str:
    return f"{x:.1f}"


def conclude(matrix: dict[str, Any], names: dict[str, str], harnesses: list[str]) -> dict[str, Any]:
    """The short automatic reading of a matrix. ``names`` maps a model id to the name shown."""
    cells = matrix["cells"]
    ids = list(cells)
    value = lambda m, h: cells[m][h]["pass_at_1"]  # noqa: E731
    per_model: dict[str, dict[str, Any]] = {}
    model_spreads, harness_spreads = [], []
    for m in ids:
        have = [(value(m, h), h) for h in harnesses if value(m, h) is not None]
        if not have:
            per_model[m] = {"name": names.get(m, m), "best": None, "verdict": "no_data"}
            continue
        order = {h: i for i, h in enumerate(harnesses)}
        ranked = sorted(have, key=lambda vh: (-vh[0], cells[m][vh[1]]["format_fail_rate"] or 0.0, cells[m][vh[1]]["tokens_out_mean"] or 0.0, order[vh[1]]))
        best, worst = ranked[0], min(have, key=lambda vh: (vh[0], -order[vh[1]]))
        spread = (best[0] - worst[0]) * 100 if len(have) > 1 else None
        fails = [(cells[m][h]["counts"]["fail_format"], cells[m][h]["counts"]["fail_answer"]) for h in harnesses]
        total_fail = sum(f + a for f, a in fails)
        second = ranked[1] if len(ranked) > 1 else None
        clear = bool(second and cells[m][best[1]]["ci"][0] > cells[m][second[1]]["ci"][1])
        if best[0] < LOW_PASS:
            verdict = "weak_everywhere"
        elif spread is not None and spread >= SENSITIVE_POINTS:
            verdict = "harness_sensitive"
        else:
            verdict = "stable"
        per_model[m] = {"name": names.get(m, m), "best": best[1], "best_pass": best[0], "worst": worst[1] if len(have) > 1 else None, "worst_pass": worst[0] if len(have) > 1 else None,
                        "spread_points": _round(spread, 1), "second": second[1] if second else None, "margin_points": _round((best[0] - second[0]) * 100, 1) if second else None,
                        "best_is_clear": clear, "format_share_of_failures": _round(sum(f for f, _ in fails) / total_fail) if total_fail else None, "verdict": verdict}
        if spread is not None:
            model_spreads.append(spread)
    for h in harnesses:
        have = [value(m, h) for m in ids if value(m, h) is not None]
        if len(have) > 1:
            harness_spreads.append((max(have) - min(have)) * 100)
    x = round(sum(model_spreads) / len(model_spreads), 1) if model_spreads else None       # same model, other harness
    y = round(sum(harness_spreads) / len(harness_spreads), 1) if harness_spreads else None  # same harness, other model
    if x is None or y is None:
        verdict = "not_comparable"
    elif x - y > SIMILAR_POINTS:
        verdict = "harness_larger"
    elif y - x > SIMILAR_POINTS:
        verdict = "model_larger"
    else:
        verdict = "similar"
    valid_counts = [c["valid"] for row in cells.values() for c in row.values() if c["trials"]]
    min_valid = min(valid_counts) if valid_counts else 0
    weak = min_valid < WEAK_BELOW
    none_total = sum(c["none_n"] for row in cells.values() for c in row.values())
    best_by_model = {names.get(m, m): d["best"] for m, d in per_model.items() if d.get("best")}
    out = {"verdict": verdict, "harness_spread_points": x, "model_spread_points": y, "harness_spread_max": _round(max(model_spreads), 1) if model_spreads else None,
           "model_spread_max": _round(max(harness_spreads), 1) if harness_spreads else None, "best_harness_by_model": best_by_model, "per_model": per_model,
           "min_valid_per_cell": min_valid, "weak": weak, "none_total": none_total}
    out["summary"] = {"en": _text("en", out, names), "es": _text("es", out, names)}
    return out


def _text(lang: str, c: dict[str, Any], names: dict[str, str]) -> str:
    es = lang == "es"
    x, y = c["harness_spread_points"], c["model_spread_points"]
    parts: list[str] = []
    if c["verdict"] == "not_comparable":
        parts.append("Con un solo modelo o un solo arnés no se pueden comparar las dos diferencias." if es else "With a single model or a single harness the two spreads cannot be compared.")
        if x is not None:
            parts.append(f"Cambiar de arnés mueve el pass@1 de un mismo modelo {_pts(x)} puntos de media." if es else f"Changing the harness moves a model's pass@1 by {_pts(x)} points on average.")
        if y is not None:
            parts.append(f"Cambiar de modelo con el mismo arnés lo mueve {_pts(y)} puntos de media." if es else f"Changing the model under one harness moves it by {_pts(y)} points on average.")
    else:
        word_es = {"harness_larger": "mayor", "model_larger": "menor", "similar": "parecida"}[c["verdict"]]
        word_en = {"harness_larger": "larger", "model_larger": "smaller", "similar": "similar"}[c["verdict"]]
        if es:
            parts.append(f"La diferencia entre arneses ({_pts(x)} puntos de pass@1 para un mismo modelo) es {word_es} que la diferencia entre modelos ({_pts(y)} puntos con un mismo arnés).")
        else:
            parts.append(f"The difference between harnesses ({_pts(x)} pass@1 points for the same model) is {word_en} than the difference between models ({_pts(y)} points under the same harness).")
    for m, d in c["per_model"].items():
        if d.get("best") is None:
            continue
        name = d["name"]
        if d["worst"] is None:
            parts.append(f"{name}: solo un arnés medido ({d['best']}, {round(d['best_pass'] * 100)} %)." if es else f"{name}: only one harness measured ({d['best']}, {round(d['best_pass'] * 100)} %).")
            continue
        base = f"{name}: {d['best']} {round(d['best_pass'] * 100)} %, {d['worst']} {round(d['worst_pass'] * 100)} %"
        tail = {"harness_sensitive": ("depende del arnés", "depends on the harness"), "weak_everywhere": ("falla con cualquier arnés", "fails whatever the harness"),
                "stable": ("estable entre arneses", "stable across harnesses")}[d["verdict"]]
        parts.append(f"{base}: {tail[0] if es else tail[1]}.")
    if c["none_total"]:
        parts.append(f"{c['none_total']} pruebas sin resultado (infraestructura) quedan fuera de los porcentajes." if es else f"{c['none_total']} trials without a result (infrastructure) are left out of the percentages.")
    if c["weak"]:
        parts.append(f"Evidencia débil: solo {c['min_valid_per_cell']} pruebas válidas en la celda con menos." if es else f"Weak evidence: only {c['min_valid_per_cell']} valid trials in the smallest cell.")
    return " ".join(parts)
