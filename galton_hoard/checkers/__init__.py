"""Checkers: pure functions ``(spec, output, context) -> {score, passed, detail, unavailable?}``.

``run_checker`` dispatches on ``spec["type"]`` and never raises: a checker that crashes returns a failed verdict that says so.
``all`` and ``any`` combine other checkers (a RAG case checks the citations and the fact in one go).
"""

from __future__ import annotations

from typing import Any, Callable

from .basic import check_choice, check_contains, check_exact, check_none, check_regex
from .code import check_python_tests
from .constraints import check_constraints
from .ifmt import check_ifmt
from .judge import check_family, check_judge
from .jsoncheck import check_json, check_tool_call
from .mathcheck import check_math
from .numbers import check_number
from .rag import check_citations, check_needle
from .types import CheckContext, ModelOutput, bad_spec, verdict

Checker = Callable[[dict[str, Any], ModelOutput, CheckContext], dict[str, Any]]

CHECKERS: dict[str, Checker] = {
    "exact": check_exact,
    "contains": check_contains,
    "regex": check_regex,
    "choice": check_choice,
    "number": check_number,
    "math_equiv": check_math,
    "json": check_json,
    "tool_call": check_tool_call,
    "python_tests": check_python_tests,
    "constraints": check_constraints,
    "needle": check_needle,
    "citations": check_citations,
    "judge": check_judge,
    "family": check_family,
    "ifmt": check_ifmt,
    "none": check_none,
}
COMBINATORS = ("all", "any")
#: Checkers whose verdict depends on a model or another service, so the reference answer of a case cannot be verified offline.
NON_DETERMINISTIC = frozenset({"judge", "family", "ifmt", "none"})


def checker_types() -> list[str]:
    return [*CHECKERS, *COMBINATORS]


def _combine(kind: str, spec: dict[str, Any], output: ModelOutput, ctx: CheckContext) -> dict[str, Any]:
    checks = spec.get("checks")
    if not isinstance(checks, list) or not checks:
        return bad_spec(f"{kind} needs a non-empty `checks` list")
    weights = spec.get("weights") or [1.0] * len(checks)
    if len(weights) != len(checks):
        return bad_spec(f"{kind}: `weights` must have one entry per check")
    parts = [(run_checker(c, output, ctx), float(w)) for c, w in zip(checks, weights)]
    usable = [(r, w) for r, w in parts if not r.get("unavailable")]
    detail = {"checks": [r for r, _ in parts]}
    if not usable:
        return {**verdict(0.0, False, detail, unavailable=True)}
    if kind == "all":
        total = sum(w for _, w in usable) or 1.0
        score = sum(r["score"] * w for r, w in usable) / total
        return verdict(score, all(r["passed"] for r, _ in usable), detail)
    best = max(usable, key=lambda rw: rw[0]["score"])[0]
    return verdict(best["score"], any(r["passed"] for r, _ in usable), detail)


def run_checker(spec: dict[str, Any], output: ModelOutput, ctx: CheckContext | None = None) -> dict[str, Any]:
    """Check ``output`` against the checker ``spec``. Never raises."""
    ctx = ctx or CheckContext()
    kind = (spec or {}).get("type")
    try:
        if kind in COMBINATORS:
            return _combine(kind, spec, output, ctx)
        func = CHECKERS.get(kind or "")
        if func is None:
            return bad_spec(f"unknown checker type {kind!r}")
        return func(spec, output, ctx)
    except Exception as exc:  # noqa: BLE001 — a bug in a checker must not take a whole run down
        return verdict(0.0, False, {"error": f"checker {kind} crashed: {type(exc).__name__}: {exc}"})


def is_deterministic(spec: dict[str, Any]) -> bool:
    """Can the reference answer of a case with this checker be verified without a model or another app?"""
    kind = (spec or {}).get("type")
    if kind in COMBINATORS:
        return all(is_deterministic(c) for c in spec.get("checks", []))
    return kind not in NON_DETERMINISTIC


__all__ = ["CHECKERS", "COMBINATORS", "CheckContext", "ModelOutput", "checker_types", "is_deterministic", "run_checker"]
