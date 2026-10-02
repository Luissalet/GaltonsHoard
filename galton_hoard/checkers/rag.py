"""Retrieval checkers: ``needle`` (a planted value must appear in the answer) and ``citations`` (the answer cites the right passages)."""

from __future__ import annotations

import re
from typing import Any

from ..util import fold
from .basic import visible
from .types import CheckContext, ModelOutput, bad_spec, verdict


def _squash_digits(text: str) -> str:
    """``48.213`` / ``48 213`` / ``48,213`` -> ``48213`` so thousand separators do not hide a correct number."""
    return re.sub(r"(?<=\d)[.,\s  ](?=\d{3}(?!\d))", "", text)


def check_needle(spec: dict[str, Any], output: ModelOutput, ctx: CheckContext) -> dict[str, Any]:
    needles = spec.get("needles") if "needles" in spec else spec.get("needle")
    needles = [needles] if isinstance(needles, (str, int, float)) else list(needles or [])
    if not needles:
        return bad_spec("needle needs `needle` or `needles`")
    text = visible(output)
    hay = fold(text).casefold()
    if spec.get("ignore_digit_separators", True):
        hay = _squash_digits(hay)
    missing = []
    for n in needles:
        want = fold(str(n)).casefold()
        if spec.get("ignore_digit_separators", True):
            want = _squash_digits(want)
        if want not in hay:
            missing.append(str(n))
    forbidden = [str(f) for f in (spec.get("forbid") or []) if fold(str(f)).casefold() in hay]
    score = (len(needles) - len(missing)) / len(needles)
    if forbidden:
        score *= 0.5
    return verdict(score, not missing and not forbidden, {"missing": missing, "forbidden_present": forbidden})


_BRACKET = re.compile(r"\[\s*([^\[\]]{1,60}?)\s*\]")
_ID = re.compile(r"^[A-Za-z]{0,3}\d{1,3}$")


def cited_ids(text: str, allowed: set[str] | None = None, pattern: str = "") -> list[str]:
    """Passage ids cited as ``[p3]``, ``[p3, p4]`` or ``[3]``, in order of appearance without repeats."""
    found: list[str] = []
    if pattern:
        tokens = [m.group(1) for m in re.finditer(pattern, text)]
    else:
        tokens = []
        for m in _BRACKET.finditer(text):
            tokens.extend(t for t in re.split(r"[,;\s]+", m.group(1)) if t)
    for token in tokens:
        token = token.casefold()
        if not _ID.match(token) or (allowed is not None and token not in allowed):
            continue
        if token not in found:
            found.append(token)
    return found


def check_citations(spec: dict[str, Any], output: ModelOutput, ctx: CheckContext) -> dict[str, Any]:
    correct = [str(c).casefold() for c in (spec.get("correct") or [])]
    if not correct:
        return bad_spec("citations needs `correct`")
    allowed = {str(i).casefold() for i in spec["ids"]} if spec.get("ids") else None
    extra_ok = {str(i).casefold() for i in (spec.get("allowed_extra") or [])}
    cited = cited_ids(visible(output), allowed, spec.get("pattern", ""))
    hits = [c for c in cited if c in correct]
    wrong = [c for c in cited if c not in correct and c not in extra_ok]
    recall = len(hits) / len(correct)
    counted = [c for c in cited if c not in extra_ok]
    precision = len(hits) / len(counted) if counted else 0.0
    score = 0.0 if recall == 0 or precision == 0 else 2 * recall * precision / (recall + precision)
    return verdict(score, recall == 1.0 and not wrong, {"cited": cited, "correct": correct, "wrong": wrong})
