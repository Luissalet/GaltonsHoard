"""Reading numbers out of free text (Spanish ``1.234,5`` and English ``1,234.5``) and the ``number`` checker."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Optional

from .numberwords import first_number_words
from .textutil import split_reasoning
from .types import CheckContext, ModelOutput, bad_spec, verdict

_NUMBER = re.compile(
    r"(?<![\w.,])[-−–+]?(?:\d{1,3}(?:[.,  ' ]\d{3})+(?!\d)|\d+)(?:[.,]\d+)?(?:[eE][-+]?\d+)?(?:\s?%)?"
)
_FRACTION = re.compile(r"(?<![\w.,/])[-−–+]?\d+\s*/\s*\d+(?![\w/])")
_MARKER = re.compile(r"(respuesta(?:\s+final)?|resultado(?:\s+final)?|soluci[oó]n|final answer|answer|result|\\boxed)", re.I)


@dataclass
class Token:
    raw: str
    values: list[float]   # more than one when the notation is ambiguous (``1,234`` is 1.234 in Spanish and 1234 in English)
    start: int
    end: int


def interpret(raw: str, locale: str = "auto") -> list[float]:
    """The values a numeric token can mean, most likely first. ``locale``: ``es``, ``en`` or ``auto`` (keep both when ambiguous)."""
    token = raw.strip().replace("−", "-").replace("–", "-").replace("%", "").strip()
    sign = -1.0 if token.startswith("-") else 1.0
    token = token.lstrip("+-").strip()
    exp = ""
    m = re.search(r"[eE][-+]?\d+$", token)
    if m:
        exp, token = m.group(0), token[: m.start()]
    token = re.sub(r"[  ' ]", "", token)
    if not token:
        return []
    seps = [c for c in token if c in ".,"]

    def num(text: str) -> float:
        return sign * float(text + exp)

    if not seps:
        return [num(token)]
    kinds = set(seps)
    if len(kinds) == 2:  # both appear: the last one is the decimal separator
        dec = token[max(token.rfind("."), token.rfind(","))]
        thousands = "." if dec == "," else ","
        cleaned = token.replace(thousands, "").replace(dec, ".")
        return [num(cleaned)] if cleaned.count(".") <= 1 else []
    sep = seps[0]
    if len(seps) > 1:  # the same separator several times: thousands
        groups = token.split(sep)
        if all(len(g) == 3 for g in groups[1:]) and 1 <= len(groups[0]) <= 3:
            return [num("".join(groups))]
        return []
    head, tail = token.split(sep)
    if not head:
        head = "0"
    if len(tail) == 3 and head not in ("0", "00") and len(head) <= 3:
        as_thousands, as_decimal = num(head + tail), num(f"{head}.{tail}")
        if locale == "es":
            return [as_thousands] if sep == "." else [as_decimal]
        if locale == "en":
            return [as_thousands] if sep == "," else [as_decimal]
        return [as_thousands, as_decimal] if sep == "." else [as_decimal, as_thousands]
    return [num(f"{head}.{tail}")]


def find_numbers(text: str, locale: str = "auto", *, fractions: bool = False) -> list[Token]:
    tokens: list[Token] = []
    taken: list[tuple[int, int]] = []
    if fractions:
        for m in _FRACTION.finditer(text):
            a, _, b = m.group(0).replace("−", "-").partition("/")
            try:
                den = float(b)
                if den:
                    tokens.append(Token(m.group(0), [float(a.replace(" ", "")) / den], m.start(), m.end()))
                    taken.append((m.start(), m.end()))
            except ValueError:
                continue
    for m in _NUMBER.finditer(text):
        if any(s <= m.start() < e for s, e in taken):
            continue
        values = interpret(m.group(0), locale)
        if values:
            tokens.append(Token(m.group(0).strip(), values, m.start(), m.end()))
    tokens.sort(key=lambda t: t.start)
    return tokens


# What may sit between two numbers of one written-out calculation (``2^10 = 1024``, ``15 × 4 = 60``, ``2x + 3 = 11``): blanks,
# brackets, operators and a multiplication "x" right after a number. Letters and line breaks end the calculation.
_BLANKS = r"[ \t()\[\]{}]*"
_RESULT_OP = r"(?:={1,2}>?|[≈≃⇒→])"
_ANY_OP = rf"(?:{_RESULT_OP}|\*\*|[+\-*/^×÷·])"
_GAP = re.compile(rf"^[ \t]*(?:[xX](?![^\W\d_]))?{_BLANKS}(?:{_ANY_OP}{_BLANKS})*$")
_RESULT_GAP = re.compile(_RESULT_OP)


def _calculation(text: str, tokens: list[Token], first: int) -> list[tuple[Token, str]]:
    """The numbers of the calculation that starts at ``tokens[first]``, each with the text that separates it from the one before
    (empty for the first), up to the first thing that is not part of it."""
    chain = [(tokens[first], "")]
    for token in tokens[first + 1:]:
        gap = text[chain[-1][0].end:token.start]
        if not gap.strip(" \t()[]{}") or not _GAP.match(gap):
            break
        chain.append((token, gap))
    return chain


def pick_marked(text: str, tokens: list[Token]) -> Optional[Token]:
    """The number meant by the last «Respuesta:»-style marker: the first number after it (within 120 characters) and, when that
    number starts a written-out calculation on the same line (``Respuesta: 2^10 = 1024``), the value after its final ``=``
    instead of its first number. ``None`` when there is no marker."""
    markers = list(_MARKER.finditer(text))
    if not markers:
        return None
    start = markers[-1].end()
    for index, token in enumerate(tokens):
        if token.start >= start and token.start - start <= 120:
            result = token
            for item, gap in _calculation(text, tokens, index)[1:]:
                if _RESULT_GAP.search(gap):
                    result = item
            return result
    return None


def marker_line(text: str) -> Optional[str]:
    """The answer line after the last «Respuesta:»-style marker: the rest of the marker's line, or the next non-empty line when that rest is
    blank. ``None`` when there is no marker."""
    markers = list(_MARKER.finditer(text))
    if not markers:
        return None
    rest = text[markers[-1].end():].lstrip(" \t:：*_`-–—")
    lines = rest.splitlines()
    first = lines[0].strip() if lines else ""
    if first:
        return first
    for line in lines[1:]:
        if line.strip():
            return line.strip()
    return ""


def pick_worded(text: str) -> Optional[tuple[float, str]]:
    """The number written with words on the answer line («Respuesta: El octavo día» is 8), only when that line has no digit at all: with a
    digit on the line the digits are the answer. ``None`` when there is no marker, the line has a digit or no number word."""
    line = marker_line(text)
    if not line or re.search(r"\d", line):
        return None
    found = first_number_words(line)
    return (float(found[0]), found[1]) if found else None


def close(value: float, expected: float, abs_tol: float, rel_tol: float) -> bool:
    return abs(value - expected) <= max(abs_tol, rel_tol * abs(expected)) + 1e-12


def check_number(spec: dict[str, Any], output: ModelOutput, ctx: CheckContext) -> dict[str, Any]:
    if "expected" not in spec:
        return bad_spec("number needs `expected`")
    try:
        expected = float(spec["expected"])
    except (TypeError, ValueError):
        return bad_spec("number: `expected` must be numeric")
    tol = spec.get("tolerance") or {}
    if not isinstance(tol, dict):
        tol = {"abs": tol}
    abs_tol, rel_tol = float(tol.get("abs", spec.get("abs", 0.0))), float(tol.get("rel", spec.get("rel", 0.0)))
    which = spec.get("which", "marked")
    locale = spec.get("locale", "auto")
    text, _ = split_reasoning(output.text)
    tokens = find_numbers(text, locale, fractions=bool(spec.get("fractions", False)))
    if which not in ("first", "last", "any"):
        worded = pick_worded(text)
        if worded is not None:
            value, words = worded
            ok = close(value, expected, abs_tol, rel_tol)
            return verdict(1.0 if ok else 0.0, ok, {"expected": expected, "got": value, "raw": words, "from_words": True})
    if not tokens:
        return verdict(0.0, False, {"reason": "no number in the answer", "expected": expected})
    if which == "first":
        chosen = [tokens[0]]
    elif which == "last":
        chosen = [tokens[-1]]
    elif which == "any":
        chosen = tokens
    else:
        marked = pick_marked(text, tokens)
        chosen = [marked or tokens[-1]]
    for token in chosen:
        for value in token.values:
            if close(value, expected, abs_tol, rel_tol):
                return verdict(1.0, True, {"expected": expected, "got": value, "raw": token.raw})
    token = chosen[0]
    return verdict(0.0, False, {"expected": expected, "got": token.values[0], "raw": token.raw})
