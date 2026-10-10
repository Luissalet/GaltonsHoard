"""Gate correlation: do two checks of the same case fail together?

A *gate* is one check inside a case: the leaf checkers of its ``checker`` spec (``all`` / ``any`` combinators are opened; ``regex``,
``citations``, ``judge``... are the gates). For every case a model answered, each gate says ``pass``, ``fail`` or ``abstain`` (the check was
not available: the judge was not live, a service was missing). This module turns the stored results into that outcome matrix and measures,
for every pair of gates that were evaluated on the same answers, how much their failures overlap:

* ``phi``: correlation of the two failure indicators (-1..1);
* ``lift``: joint failure rate divided by the product of the two failure rates (1 = independent);
* ``kappa``: Cohen's kappa of the two verdicts (agreement beyond chance);
* ``p_a_given_b`` / ``p_b_given_a``: P(A fails | B fails) and the reverse;

each with a seeded percentile bootstrap interval over the answers, and with a minimum-sample guard: below ``min_samples`` shared answers the
numbers are shown but no conclusion is drawn. Everything is plain Python and deterministic.

The unit is one answer of one model to one case (repeats and runs of the same model on the same case are collapsed: a gate fails when at
least half of the repeats it judged failed). Only cases with two or more gates produce pairs; a case with a single check contributes to that
gate's failure rate only.
"""

from __future__ import annotations

import math
import random
from collections import defaultdict
from typing import Any, Iterable, Optional

from . import stats

COMBINATORS = ("all", "any")
#: Gates whose verdict needs a model or another service: the costlier ones, dropped first when two are equally redundant.
COSTLY = frozenset({"judge", "family", "ifmt"})
PASS, FAIL, ABSTAIN = "pass", "fail", "abstain"
MIN_SAMPLES = 20
STRONG_PHI = 0.7
#: The lower end of the interval of a redundant pair must stay above this: "tight CI".
TIGHT_PHI_LOW = 0.5
RESAMPLES = 1000
SEED = 20260
DISJOINT_LISTED = 8      # pairs named in the single "never share an answer" finding; the rest are counted
MAX_GATES = 40
KEYS = ("phi", "lift", "kappa", "p_a_given_b", "p_b_given_a")


# ------------------------------------------------------------------------------------------------- extraction
def _outcome(verdict: dict[str, Any]) -> str:
    if verdict.get("unavailable") or verdict.get("judge_pending"):
        return ABSTAIN
    return PASS if verdict.get("passed") else FAIL


def leaves(spec: Any, verdict: dict[str, Any]) -> list[tuple[str, str]]:
    """``[(gate, outcome)]`` of one stored verdict against its checker spec. Combinators are opened when the stored detail lines up with the spec."""
    kind = spec.get("type") if isinstance(spec, dict) else None
    if not kind or kind == "none":
        return []
    detail = verdict.get("detail") if isinstance(verdict.get("detail"), dict) else {}
    subs, parts = spec.get("checks"), detail.get("checks")
    if kind in COMBINATORS and isinstance(subs, list) and isinstance(parts, list) and len(subs) == len(parts):
        out: list[tuple[str, str]] = []
        for sub_spec, part in zip(subs, parts):
            out.extend(leaves(sub_spec, part if isinstance(part, dict) else {}))
        return out
    return [(kind, _outcome(verdict))]


def _merge(outcomes: Iterable[str]) -> str:
    """Several verdicts of the same gate on one answer (two ``regex`` checks, a nested combinator): fails if any fails."""
    seen = set(outcomes)
    return FAIL if FAIL in seen else PASS if PASS in seen else ABSTAIN


def build_matrix(rows: Iterable[dict[str, Any]], specs: dict[str, dict[str, Any]], *, by_digest: bool = False) -> dict[str, Any]:
    """The outcome matrix. ``rows`` are result rows with ``detail``, ``passed``, ``unavailable``, ``judge_pending``, ``contestant_id``, ``case_id``, ``digest``;
    ``specs`` maps a case id to its checker spec. Returns ``{gates, units: {unit: {gate: outcome}}, answers}``."""
    per_unit: dict[tuple, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    answers = 0
    for row in rows:
        spec = specs.get(row["case_id"])
        if not spec:
            continue
        verdict = {"passed": row.get("passed"), "unavailable": row.get("unavailable"), "judge_pending": row.get("judge_pending"), "detail": row.get("detail")}
        found = leaves(spec, verdict)
        if not found:
            continue
        answers += 1
        unit = (row["contestant_id"], row.get("digest", "") if by_digest else "", row["case_id"])
        by_gate: dict[str, list[str]] = defaultdict(list)
        for gate, outcome in found:
            by_gate[gate].append(outcome)
        for gate, outcomes in by_gate.items():
            per_unit[unit][gate].append(_merge(outcomes))
    units: dict[tuple, dict[str, str]] = {}
    for unit, gates in per_unit.items():
        collapsed = {}
        for gate, repeats in gates.items():
            fails, passes = repeats.count(FAIL), repeats.count(PASS)
            collapsed[gate] = ABSTAIN if not fails and not passes else FAIL if fails >= passes else PASS
        units[unit] = collapsed
    gate_names = sorted({g for gs in units.values() for g in gs})
    return {"gates": gate_names, "units": units, "answers": answers}


# ------------------------------------------------------------------------------------------------- the 2x2 statistics
def _phi(n11: int, n10: int, n01: int, n00: int) -> Optional[float]:
    r1, r0, c1, c0 = n11 + n10, n01 + n00, n11 + n01, n10 + n00
    if min(r1, r0, c1, c0) == 0:
        return None
    return (n11 * n00 - n10 * n01) / math.sqrt(r1 * r0 * c1 * c0)


def _kappa(n11: int, n10: int, n01: int, n00: int) -> Optional[float]:
    n = n11 + n10 + n01 + n00
    if n == 0:
        return None
    pa, pb = (n11 + n10) / n, (n11 + n01) / n
    expected = pa * pb + (1 - pa) * (1 - pb)
    if expected >= 1.0:
        return None
    return ((n11 + n00) / n - expected) / (1 - expected)


def _lift(n11: int, n10: int, n01: int, n00: int) -> Optional[float]:
    n = n11 + n10 + n01 + n00
    pa, pb = ((n11 + n10) / n, (n11 + n01) / n) if n else (0.0, 0.0)
    return (n11 / n) / (pa * pb) if n and pa and pb else None


def _cond(num: int, den: int) -> Optional[float]:
    return num / den if den else None


def _measures(c: tuple[int, int, int, int]) -> dict[str, Optional[float]]:
    n11, n10, n01, n00 = c
    return {"phi": _phi(*c), "lift": _lift(*c), "kappa": _kappa(*c), "p_a_given_b": _cond(n11, n11 + n01), "p_b_given_a": _cond(n11, n11 + n10)}


def _binomial(rng: random.Random, n: int, p: float) -> int:
    """A binomial draw: exact for small ``n`` or a small mean, normal approximation otherwise (the bootstrap does not need more)."""
    if n <= 0 or p <= 0:
        return 0
    if p >= 1:
        return n
    if n <= 64:
        return sum(1 for _ in range(n) if rng.random() < p)
    q = min(p, 1 - p)
    if n * q < 20:                                  # inversion on the rare side
        u, k, term = rng.random(), 0, (1 - q) ** n
        cdf = term
        while u > cdf and k < n:
            term *= (n - k) / (k + 1) * q / (1 - q)
            k += 1
            cdf += term
        return k if q == p else n - k
    return max(0, min(n, round(rng.gauss(n * p, math.sqrt(n * p * (1 - p))))))


def _resample(rng: random.Random, c: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    """One bootstrap replicate of the four cells (a multinomial of the same size, drawn cell by cell)."""
    left, remaining, out = sum(c), sum(c), []
    for i, count in enumerate(c):
        if i == 3:
            out.append(left)
            break
        k = _binomial(rng, left, count / remaining) if remaining else 0
        out.append(k)
        left -= k
        remaining -= count
    return tuple(out)  # type: ignore[return-value]


def bootstrap_measures(c: tuple[int, int, int, int], *, resamples: int = RESAMPLES, seed: int = SEED) -> dict[str, Optional[tuple[float, float]]]:
    """95 % percentile intervals of every measure. ``None`` when more than half of the replicates leave the measure undefined (a gate that never failed)."""
    if sum(c) < 2:
        return {k: None for k in KEYS}
    rng = random.Random(seed)
    draws: dict[str, list[float]] = {k: [] for k in KEYS}
    for _ in range(resamples):
        for key, value in _measures(_resample(rng, c)).items():
            if value is not None:
                draws[key].append(value)
    out: dict[str, Optional[tuple[float, float]]] = {}
    for key, values in draws.items():
        if len(values) * 2 < resamples:
            out[key] = None
            continue
        values.sort()
        out[key] = (values[int(0.025 * len(values))], values[min(len(values) - 1, int(0.975 * len(values)))])
    return out


def _r(value: Optional[float], digits: int = 3) -> Optional[float]:
    return None if value is None else round(value, digits)


def _ci(interval: Optional[tuple[float, float]]) -> Optional[list[float]]:
    return None if interval is None else [round(interval[0], 3), round(interval[1], 3)]


def classify(stat: dict[str, Any], *, min_samples: int, strong_phi: float) -> str:
    """The reading of one pair: insufficient / never_fail / never_disagree / redundant / probably_redundant / complementary / independent / mixed."""
    c = stat["counts"]
    if stat["n"] < min_samples:
        return "insufficient"
    if c["fail_both"] + c["fail_a_only"] + c["fail_b_only"] == 0:
        return "never_fail"                           # neither gate ever fails here: no information about failures
    if c["fail_a_only"] + c["fail_b_only"] == 0:
        return "never_disagree"
    phi, ci = stat["phi"], stat["phi_ci"]
    if phi is None:
        return "mixed"
    if phi >= strong_phi:
        return "redundant" if ci and ci[0] >= TIGHT_PHI_LOW else "probably_redundant"
    if phi <= -0.5 and ci and ci[1] <= -0.3:
        return "complementary"
    if abs(phi) < 0.3 and ci and ci[0] <= 0 <= ci[1]:
        return "independent"
    return "mixed"


def pair_stat(a: str, b: str, units: dict[Any, dict[str, str]], *, min_samples: int, strong_phi: float, resamples: int, seed: int) -> dict[str, Any]:
    """All the numbers of one pair of gates over the answers on which both were evaluated (abstentions are left out of that pair only)."""
    n11 = n10 = n01 = n00 = 0
    for gates in units.values():
        x, y = gates.get(a, ABSTAIN), gates.get(b, ABSTAIN)
        if ABSTAIN in (x, y):
            continue
        if x == FAIL and y == FAIL:
            n11 += 1
        elif x == FAIL:
            n10 += 1
        elif y == FAIL:
            n01 += 1
        else:
            n00 += 1
    c = (n11, n10, n01, n00)
    n = sum(c)
    m = _measures(c) if n else {k: None for k in KEYS}
    ci = bootstrap_measures(c, resamples=resamples, seed=seed)
    stat: dict[str, Any] = {"a": a, "b": b, "n": n, "counts": {"fail_both": n11, "fail_a_only": n10, "fail_b_only": n01, "pass_both": n00},
                            "fail_rate_a": _r((n11 + n10) / n if n else None), "fail_rate_b": _r((n11 + n01) / n if n else None),
                            "joint_fail_rate": _r(n11 / n if n else None),
                            "expected_joint": _r(((n11 + n10) / n) * ((n11 + n01) / n) if n else None, 4),
                            "agreement": _r((n11 + n00) / n if n else None)}
    for key in KEYS:
        stat[key] = _r(m[key])
        stat[f"{key}_ci"] = _ci(ci[key])
    stat["reading"] = classify(stat, min_samples=min_samples, strong_phi=strong_phi)
    return stat


# ------------------------------------------------------------------------------------------------- findings
def _rank_drop(stat: dict[str, Any], single: dict[str, dict[str, Any]]) -> tuple[str, str, str]:
    """Which gate of a redundant pair to drop: ``(drop, keep, why)``. The one that catches fewer failures the other misses; ties: the costlier, then the one that abstains more."""
    a, b = stat["a"], stat["b"]
    unique = {a: stat["counts"]["fail_a_only"], b: stat["counts"]["fail_b_only"]}
    other = {a: b, b: a}
    if unique[a] != unique[b]:
        drop = a if unique[a] < unique[b] else b
        return drop, other[drop], "unique"
    if (a in COSTLY) != (b in COSTLY):
        drop = a if a in COSTLY else b
        return drop, other[drop], "cost"
    abstain = {g: single[g]["abstain_rate"] or 0.0 for g in (a, b)}
    if abstain[a] != abstain[b]:
        drop = a if abstain[a] > abstain[b] else b
        return drop, other[drop], "abstain"
    drop = max(a, b)
    return drop, other[drop], "tie"


_WHY_ES = {"unique": "recoge menos fallos que la otra no vea", "cost": "es la más cara de ejecutar y no ve nada que la otra no vea",
           "abstain": "se queda más veces sin veredicto", "tie": "no hay diferencia; se propone la última por orden alfabético"}
_WHY_EN = {"unique": "it catches fewer failures the other misses", "cost": "it is the costlier one and sees nothing the other does not",
           "abstain": "it abstains more often", "tie": "no difference; the last one alphabetically is proposed"}


def _pct(v: Optional[float]) -> str:
    return "?" if v is None else f"{round(v * 100)} %"


def _span(ci: Optional[list[float]]) -> tuple[str, str]:
    return (f"{ci[0]:+.2f}", f"{ci[1]:+.2f}") if ci else ("?", "?")


def build_findings(single: dict[str, dict[str, Any]], pairs: list[dict[str, Any]], *, min_samples: int) -> list[dict[str, Any]]:
    """Plain-language findings, Spanish first (``es``) with an English twin (``en``). Each carries ``kind``, ``severity`` and the gates involved."""
    out: list[dict[str, Any]] = []

    def add(kind: str, severity: str, gates: list[str], es: str, en: str, **extra: Any) -> None:
        out.append({"kind": kind, "severity": severity, "gates": gates, "es": es, "en": en, **extra})

    for g, s in single.items():
        if s["n"] < min_samples:
            add("few_samples", "info", [g], f"La comprobación «{g}» solo se evaluó en {s['n']} respuestas (mínimo {min_samples}): no se concluye nada sobre ella.",
                f"Gate '{g}' was evaluated on only {s['n']} answers (minimum {min_samples}): no conclusion about it.")
        elif s["fails"] == 0:
            add("never_fails", "warn", [g], f"«{g}» no falló nunca en {s['n']} respuestas: con estos datos no distingue a unos modelos de otros.",
                f"'{g}' never failed in {s['n']} answers: with this data it does not tell models apart.")
        elif s["fails"] == s["n"]:
            add("always_fails", "warn", [g], f"«{g}» falló en las {s['n']} respuestas: revisa si la comprobación o los casos están rotos.",
                f"'{g}' failed in all {s['n']} answers: check whether the check or the cases are broken.")
        if s["abstain_rate"] and s["abstain_rate"] >= 0.25 and s["n"] + s["abstained"] >= min_samples:
            add("often_abstains", "info", [g], f"«{g}» se quedó sin veredicto en {_pct(s['abstain_rate'])} de las respuestas (juez o servicio no disponible); sus cifras usan el resto.",
                f"'{g}' gave no verdict on {_pct(s['abstain_rate'])} of answers (judge or service unavailable); its numbers use the rest.")
    # Pairs that never share an answer live in different cases: there is nothing
    # to compare, and one line each buried the real conclusions (67 of 97 on real data).
    disjoint = [p for p in pairs if p["reading"] == "insufficient" and not p["n"]]
    if disjoint:
        names = [f"{p['a']}·{p['b']}" for p in disjoint]
        shown = ", ".join(names[:DISJOINT_LISTED])
        more_es = f" y {len(names) - DISJOINT_LISTED} más" if len(names) > DISJOINT_LISTED else ""
        more_en = f" and {len(names) - DISJOINT_LISTED} more" if len(names) > DISJOINT_LISTED else ""
        add("pairs_disjoint", "info", sorted({g for p in disjoint for g in (p["a"], p["b"])}),
            f"{len(names)} pares de comprobaciones nunca se evalúan sobre la misma respuesta (están en casos distintos), así que no se comparan: {shown}{more_es}.",
            f"{len(names)} gate pairs are never evaluated on the same answer (they belong to different cases), so they are not compared: {shown}{more_en}.",
            pairs=[[p["a"], p["b"]] for p in disjoint])
    for p in pairs:
        a, b, r = p["a"], p["b"], p["reading"]
        if r == "insufficient" and not p["n"]:
            continue
        if r == "insufficient":
            add("pair_insufficient", "info", [a, b], f"«{a}» y «{b}» solo coinciden en {p['n']} respuestas (mínimo {min_samples}): no hay base para medir si fallan juntas.",
                f"'{a}' and '{b}' share only {p['n']} answers (minimum {min_samples}): not enough to say whether they fail together.")
        elif r == "never_fail":
            continue
        elif r == "never_disagree":
            drop, keep, _ = _rank_drop(p, single)
            add("never_disagree", "warn", [a, b], f"«{a}» y «{b}» nunca discreparon en {p['n']} respuestas (fallaron a la vez {p['counts']['fail_both']} veces). Una de las dos sobra: se propone quitar «{drop}» y conservar «{keep}».",
                f"'{a}' and '{b}' never disagreed in {p['n']} answers (they failed together {p['counts']['fail_both']} times). One is redundant: drop '{drop}', keep '{keep}'.",
                recommend_drop=drop, keep=keep)
        elif r in ("redundant", "probably_redundant"):
            drop, keep, why = _rank_drop(p, single)
            sure = r == "redundant"
            lo, hi = _span(p["phi_ci"])
            add(r, "warn" if sure else "info", [a, b],
                f"«{a}» y «{b}» fallan casi a la vez (phi {p['phi']:+.2f}, intervalo {lo} a {hi}; si «{a}» falla, «{b}» falla en {_pct(p['p_b_given_a'])} de los casos, y a la inversa en {_pct(p['p_a_given_b'])}). "
                f"{'Aportan casi la misma señal' if sure else 'Parecen aportar la misma señal, pero la muestra deja dudas'}: se propone quitar «{drop}» ({_WHY_ES[why]}).",
                f"'{a}' and '{b}' fail almost together (phi {p['phi']:+.2f}, interval {lo} to {hi}; when '{a}' fails '{b}' fails {_pct(p['p_b_given_a'])} of the time, the reverse {_pct(p['p_a_given_b'])}). "
                f"{'They carry nearly the same signal' if sure else 'They seem to carry the same signal, but the sample leaves doubt'}: drop '{drop}' ({_WHY_EN[why]}).",
                recommend_drop=drop if sure else None, keep=keep, candidate=drop)
        elif r == "complementary":
            add("complementary", "info", [a, b], f"«{a}» y «{b}» tienden a fallar en respuestas distintas (phi {p['phi']:+.2f}): se complementan, conviene mantener las dos.",
                f"'{a}' and '{b}' tend to fail on different answers (phi {p['phi']:+.2f}): complementary, keep both.")
        elif r == "mixed" and p["phi"] is not None:
            add("partial_overlap", "info", [a, b], f"«{a}» y «{b}» se solapan en parte (phi {p['phi']:+.2f}): ninguna sobra, pero no son del todo independientes.",
                f"'{a}' and '{b}' overlap in part (phi {p['phi']:+.2f}): neither is redundant, but they are not fully independent.")
        elif r == "independent":
            add("independent", "ok", [a, b], f"«{a}» y «{b}» fallan de forma independiente (phi {p['phi']:+.2f}): cada una aporta señal propia.",
                f"'{a}' and '{b}' fail independently (phi {p['phi']:+.2f}): each adds its own signal.")
    for g in single:
        mine = [p for p in pairs if g in (p["a"], p["b"]) and p["reading"] not in ("insufficient", "never_fail")]
        if len(mine) >= 2 and all(p["reading"] == "never_disagree" for p in mine):
            add("never_disagrees_with_any", "warn", [g], f"«{g}» no discrepó de ninguna otra comprobación: no aporta información independiente.",
                f"'{g}' never disagreed with any other gate: no independent information.")
    return out


def analyze(matrix: dict[str, Any], *, min_samples: int = MIN_SAMPLES, strong_phi: float = STRONG_PHI, resamples: int = RESAMPLES, seed: int = SEED) -> dict[str, Any]:
    """The report: per-gate summary, every pair with its measures and reading, the findings and the recommendation."""
    units, names = matrix["units"], matrix["gates"][:MAX_GATES]
    single: dict[str, dict[str, Any]] = {}
    for g in names:
        outcomes = [gs[g] for gs in units.values() if g in gs]
        fails, passes, abstained = outcomes.count(FAIL), outcomes.count(PASS), outcomes.count(ABSTAIN)
        n = fails + passes
        lo, hi = stats.wilson(fails, n) if n else (0.0, 1.0)
        single[g] = {"gate": g, "n": n, "fails": fails, "passes": passes, "abstained": abstained, "fail_rate": _r(fails / n if n else None), "fail_rate_ci": [round(lo, 3), round(hi, 3)],
                     "abstain_rate": _r(abstained / (n + abstained) if n + abstained else 0.0), "costly": g in COSTLY}
    pairs = [pair_stat(a, names[j], units, min_samples=min_samples, strong_phi=strong_phi, resamples=resamples, seed=seed + 31 * i + j)
             for i, a in enumerate(names) for j in range(i + 1, len(names))]
    findings = build_findings(single, pairs, min_samples=min_samples)
    drops: list[dict[str, Any]] = []
    seen: set[str] = set()
    for f in findings:
        d = f.get("recommend_drop")
        if d and d not in seen:
            seen.add(d)
            drops.append({"gate": d, "keep": f["keep"], "kind": f["kind"], "phi": next((p["phi"] for p in pairs if {p["a"], p["b"]} == {d, f["keep"]}), None)})
    evaluable = [p for p in pairs if p["reading"] != "insufficient"]
    if not names:
        es, en = "No hay respuestas con comprobaciones evaluables en este ámbito.", "There are no answers with evaluable gates in this scope."
    elif len(names) < 2:
        es, en = f"Solo hay una comprobación ({names[0]}) en este ámbito: no hay pares que comparar.", f"Only one gate ({names[0]}) in this scope: no pairs to compare."
    elif not evaluable:
        es = f"Ningún par de comprobaciones comparte al menos {min_samples} respuestas: faltan datos para concluir."
        en = f"No pair of gates shares at least {min_samples} answers: not enough data to conclude."
    elif drops:
        es = "Posible redundancia: " + "; ".join(f"«{d['gate']}» sobra frente a «{d['keep']}»" for d in drops) + "."
        en = "Possible redundancy: " + "; ".join(f"'{d['gate']}' is redundant next to '{d['keep']}'" for d in drops) + "."
    else:
        es, en = "Ninguna comprobación resulta redundante con los datos actuales.", "No gate is redundant with the current data."
    return {"gates": [single[g] for g in names], "pairs": pairs, "findings": findings, "recommendations": drops, "units": len(units), "answers": matrix["answers"],
            "thresholds": {"min_samples": min_samples, "strong_phi": strong_phi, "tight_phi_low": TIGHT_PHI_LOW, "resamples": resamples},
            "summary": {"es": es, "en": en}, "truncated_gates": max(0, len(matrix["gates"]) - MAX_GATES)}
