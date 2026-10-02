"""Plain-text checkers: ``exact``, ``contains``, ``regex``, ``choice`` and ``none``."""

from __future__ import annotations

import re
from typing import Any, Optional

from ..util import fold
from .textutil import SCOPES, answer_part, first_line, last_line, normalise, split_reasoning
from .types import CheckContext, ModelOutput, bad_spec, unavailable, verdict


def _as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [str(v) for v in value]
    return [str(value)]


def visible(output: ModelOutput) -> str:
    """The answer without any <think> block."""
    return split_reasoning(output.text)[0]


def scope_problem(spec: dict[str, Any]) -> Optional[dict[str, Any]]:
    """A bad-spec verdict when ``scope`` is neither absent, ``all`` nor ``answer``."""
    scope = spec.get("scope") or ""
    return None if scope in SCOPES else bad_spec(f"`scope` must be 'answer' or 'all', not {scope!r}")


def scoped(output: ModelOutput, spec: dict[str, Any]) -> str:
    """The text a checker looks at: the whole visible answer, or with ``scope: "answer"`` only its final answer (see ``answer_part``)."""
    text = visible(output)
    return answer_part(text) if spec.get("scope") == "answer" else text


def _marker_tail(text: str) -> str:
    """What follows the last «Respuesta:»-style marker, else the last non-empty line."""
    markers = list(re.finditer(r"(?:respuesta(?:\s+final)?|resultado|soluci[oó]n|final answer|answer)\s*[:：]\s*", text, re.I))
    if markers:
        tail = text[markers[-1].end():].strip()
        return tail.splitlines()[0].strip() if tail else ""
    return last_line(text)


def check_exact(spec: dict[str, Any], output: ModelOutput, ctx: CheckContext) -> dict[str, Any]:
    expected = _as_list(spec.get("expected"))
    if not expected:
        return bad_spec("exact needs `expected`")
    ignore_accents = bool(spec.get("ignore_accents", False))
    markup = bool(spec.get("strip_markup", True))
    text = visible(output)
    where = spec.get("extract", "whole")
    if where == "last_line":
        text = last_line(text)
    elif where == "first_line":
        text = first_line(text)
    elif where == "marker":
        text = _marker_tail(text)
    got = normalise(text, ignore_accents=ignore_accents, strip_markup=markup)
    wanted = [normalise(e, ignore_accents=ignore_accents, strip_markup=markup) for e in expected]
    ok = got in wanted
    return verdict(1.0 if ok else 0.0, ok, {"expected": expected, "got": got[:300]})


def _find(term: str, text: str, *, case_sensitive: bool, ignore_accents: bool, whole_words: bool) -> bool:
    hay, needle = text, term
    if ignore_accents:
        hay, needle = fold(hay), fold(needle)
    if not case_sensitive:
        hay, needle = hay.casefold(), needle.casefold()
    if whole_words:
        return re.search(r"(?<!\w)" + re.escape(needle) + r"(?!\w)", hay, re.UNICODE) is not None
    return needle in hay


def check_contains(spec: dict[str, Any], output: ModelOutput, ctx: CheckContext) -> dict[str, Any]:
    all_of, any_of, none_of = _as_list(spec.get("all")), _as_list(spec.get("any")), _as_list(spec.get("none"))
    if not (all_of or any_of or none_of):
        return bad_spec("contains needs `all`, `any` or `none`")
    if (bad := scope_problem(spec)) is not None:
        return bad
    opts = {"case_sensitive": bool(spec.get("case_sensitive", False)), "ignore_accents": bool(spec.get("ignore_accents", False)),
            "whole_words": bool(spec.get("words", False))}
    text = scoped(output, spec)
    parts: list[float] = []
    detail: dict[str, Any] = {}
    if all_of:
        missing = [t for t in all_of if not _find(t, text, **opts)]
        parts.append((len(all_of) - len(missing)) / len(all_of))
        detail["missing"] = missing
    if any_of:
        hit = [t for t in any_of if _find(t, text, **opts)]
        parts.append(1.0 if hit else 0.0)
        detail["any_hit"] = hit
        if not hit:
            detail["any_wanted"] = any_of
    if none_of:
        present = [t for t in none_of if _find(t, text, **opts)]
        parts.append((len(none_of) - len(present)) / len(none_of))
        detail["forbidden_present"] = present
    score = sum(parts) / len(parts)
    return verdict(score, all(p == 1.0 for p in parts), detail)


def check_regex(spec: dict[str, Any], output: ModelOutput, ctx: CheckContext) -> dict[str, Any]:
    patterns = _as_list(spec.get("patterns") if "patterns" in spec else spec.get("pattern"))
    forbid = _as_list(spec.get("forbid"))
    if not patterns and not forbid:
        return bad_spec("regex needs `pattern`, `patterns` or `forbid`")
    if (bad := scope_problem(spec)) is not None:
        return bad
    flags = 0
    for ch in str(spec.get("flags", "")).lower():
        flags |= {"i": re.I, "s": re.S, "m": re.M}.get(ch, 0)
    text = scoped(output, spec)
    if spec.get("extract") == "last_line":
        text = last_line(text)
    mode = spec.get("mode", "search")
    parts: list[float] = []
    detail: dict[str, Any] = {}
    try:
        if patterns:
            miss = []
            for p in patterns:
                rx = re.compile(p, flags)
                hit = rx.fullmatch(text.strip()) if mode == "fullmatch" else rx.search(text)
                if not hit:
                    miss.append(p)
            parts.append((len(patterns) - len(miss)) / len(patterns))
            detail["unmatched"] = miss
        if forbid:
            present = [p for p in forbid if re.search(p, text, flags)]
            parts.append((len(forbid) - len(present)) / len(forbid))
            detail["forbidden_present"] = present
    except re.error as exc:
        return bad_spec(f"regex: {exc}")
    return verdict(sum(parts) / len(parts), all(p == 1.0 for p in parts), detail)


# ---------------------------------------------------------------------------------------------------- multiple choice
_LETTERS = "ABCDEFGH"


def extract_choice(text: str, letters: str = _LETTERS, options: Optional[dict[str, str]] = None) -> Optional[str]:
    """The option letter an answer picks. Looks for «respuesta: B», «la opción correcta es la C», «(D)», a lone letter, a leading «A)»;
    when the answer repeats the text of exactly one option, that letter."""
    t = re.sub(r"[*_`]+", "", split_reasoning(text)[0]).strip()
    if not t:
        return None
    cls = f"[{letters}{letters.lower()}]"
    upper = f"[{letters}]"
    tail = (r"\s*(?:final\s*)?(?:es|is|:|-|=)?\s*(?:la\s+|el\s+|the\s+)?(?:opci[oó]n\s+|letra\s+|option\s+|letter\s+)?[\(\[\"']?(" + cls
            + r")[\)\]\"']?(?![A-Za-zÁ-ú])")
    # «Respuesta: B. La opción A es incorrecta porque…»: a marker that names THE answer wins over one that only mentions an option
    # while explaining it; only without one of those does the last «opción X» count.
    strong = list(re.finditer(r"(?:respuesta(?:\s+correcta)?|(?:opci[oó]n|alternativa|option|choice)\s+correcta|correcta|correct\s+(?:answer|option|choice)|answer)"
                              + tail, t, re.I))
    if strong:
        return strong[-1].group(1).upper()
    weak = list(re.finditer(r"(?:opci[oó]n|alternativa|choice|option)" + tail, t, re.I))
    if weak:
        return weak[-1].group(1).upper()
    alone = re.fullmatch(r"[\(\[]?(" + cls + r")[\)\].:]?", t)
    if alone:
        return alone.group(1).upper()
    lead = re.match(r"^[\(\[]?(" + upper + r")[\)\].:]\s", t)
    if lead:
        return lead.group(1).upper()
    lines = [ln.strip() for ln in t.splitlines() if ln.strip()]
    for ln in reversed(lines):
        m = re.fullmatch(r"[\(\[]?(" + upper + r")[\)\].:]?", ln)
        if m:
            return m.group(1).upper()
    if options:
        norm = {k: normalise(v) for k, v in options.items()}
        found = [k for k, v in norm.items() if v and v in normalise(t)]
        if len(found) == 1:
            return found[0].upper()
    singles = re.findall(r"(?<![A-Za-zÁ-ú])[\(\[](" + upper + r")[\)\]](?![A-Za-zÁ-ú])", t)
    if len(set(singles)) == 1:
        return singles[0].upper()
    return None


def check_choice(spec: dict[str, Any], output: ModelOutput, ctx: CheckContext) -> dict[str, Any]:
    answer = str(spec.get("answer", "")).strip().upper()
    if not answer:
        return bad_spec("choice needs `answer`")
    options = spec.get("options") if isinstance(spec.get("options"), dict) else None
    picked = extract_choice(output.text, options=options)
    ok = picked == answer
    return verdict(1.0 if ok else 0.0, ok, {"expected": answer, "got": picked})


def check_none(spec: dict[str, Any], output: ModelOutput, ctx: CheckContext) -> dict[str, Any]:
    """A case with no checker: the answer is recorded (for the arena, or to read it) and does not count towards any score."""
    return unavailable("unscored case", unscored=True)
