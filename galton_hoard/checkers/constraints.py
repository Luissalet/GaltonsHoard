"""The ``constraints`` checker: instruction following that can be measured (counts, words that must or must not appear, language, case,
how the text starts and ends, JSON only...). Each constraint is satisfied or not; the score is the fraction satisfied."""

from __future__ import annotations

import re
from typing import Any, Callable

from .basic import _find, scope_problem, scoped
from .textutil import (bullet_lines, detect_language, english_leaks, extract_json, paragraphs, split_sentences, word_count, words)
from .types import CheckContext, ModelOutput, bad_spec, verdict

#: Business anglicisms that have a common Spanish equivalent (used by ``forbid_anglicisms``).
ANGLICISMS = (
    "feedback", "deadline", "meeting", "briefing", "brainstorming", "coaching", "coach", "target", "stakeholder", "stakeholders", "workshop",
    "networking", "freelance", "call", "performance", "know-how", "staff", "timing", "kickoff", "kick-off", "pitch", "email", "e-mail", "mail",
    "link", "insight", "insights", "challenge", "teamwork", "deal", "output", "report", "manager", "business", "wishlist", "padlet", "sprint",
)

Check = Callable[[str, Any], tuple[bool, Any]]


def _range(count: int, lo: Any = None, hi: Any = None, exact: Any = None) -> tuple[bool, Any]:
    ok = (exact is None or count == exact) and (lo is None or count >= lo) and (hi is None or count <= hi)
    return ok, count


def _letters(text: str) -> str:
    return "".join(c for c in text if c.isalpha())


def _strip_md(text: str) -> str:
    return re.sub(r"[*_`#>]+", "", text).strip()


def evaluate(text: str, spec: dict[str, Any]) -> list[dict[str, Any]]:
    """One entry ``{name, ok, got, want}`` per constraint present in ``spec``."""
    entries: list[dict[str, Any]] = []

    def add(name: str, ok: bool, got: Any, want: Any) -> None:
        entries.append({"name": name, "ok": bool(ok), "got": got, "want": want})

    for noun, counter in (("words", word_count), ("sentences", lambda t: len(split_sentences(t))), ("bullets", lambda t: len(bullet_lines(t))),
                          ("paragraphs", lambda t: len(paragraphs(t))), ("chars", lambda t: len(t.strip())),
                          ("lines", lambda t: len([ln for ln in t.splitlines() if ln.strip()]))):
        lo, hi, exact = spec.get(f"min_{noun}"), spec.get(f"max_{noun}"), spec.get(f"exact_{noun}")
        if lo is None and hi is None and exact is None:
            continue
        count = counter(text)
        ok, got = _range(count, lo, hi, exact)
        want = exact if exact is not None else (f"{lo if lo is not None else 0}-{hi if hi is not None else '∞'}")
        add(noun, ok, got, want)
    for key, wanted in (("include", True), ("exclude", False)):
        terms = spec.get(key)
        if terms:
            terms = [terms] if isinstance(terms, str) else list(terms)
            for term in terms:
                present = _find(term, text, case_sensitive=False, ignore_accents=bool(spec.get("ignore_accents", True)), whole_words=True)
                add(f"{key}:{term}", present == wanted, present, wanted)
    if spec.get("include_any"):
        options = list(spec["include_any"])
        hits = [t for t in options if _find(t, text, case_sensitive=False, ignore_accents=True, whole_words=True)]
        add("include_any", bool(hits), hits, options)
    if spec.get("language"):
        want = spec["language"]
        got = detect_language(text)
        add("language", got == want, got, want)
    if spec.get("no_english"):
        leaks = english_leaks(text)
        add("no_english", not leaks, leaks, [])
    if spec.get("forbid_anglicisms"):
        found = [a for a in ANGLICISMS if _find(a, text, case_sensitive=False, ignore_accents=True, whole_words=True)]
        add("forbid_anglicisms", not found, found, [])
    if spec.get("case"):
        letters = _letters(text)
        ok = letters.isupper() if spec["case"] == "upper" else not any(c.isupper() for c in letters)
        add(f"case:{spec['case']}", ok, "mixed" if not ok else spec["case"], spec["case"])
    if spec.get("starts_with"):
        start = _strip_md(text).casefold()
        wants = [spec["starts_with"]] if isinstance(spec["starts_with"], str) else list(spec["starts_with"])
        add("starts_with", any(start.startswith(w.casefold()) for w in wants), start[:40], wants)
    if spec.get("ends_with"):
        end = re.sub(r"[\s*_`]+$", "", text).casefold()
        wants = [spec["ends_with"]] if isinstance(spec["ends_with"], str) else list(spec["ends_with"])
        add("ends_with", any(end.endswith(w.casefold()) for w in wants), end[-40:], wants)
    if spec.get("json_only"):
        add("json_only", _is_only_json(text), None, True)
    if spec.get("no_bullets"):
        n = len(bullet_lines(text))
        add("no_bullets", n == 0, n, 0)
    if spec.get("no_commas"):
        add("no_commas", "," not in text, text.count(","), 0)
    if spec.get("no_digits"):
        add("no_digits", not re.search(r"\d", text), None, True)
    if spec.get("no_markdown"):
        add("no_markdown", not re.search(r"[*_`#]{1,}\S|^\s*#", text, re.M), None, True)
    if spec.get("all_bullets_start_with"):
        prefix = spec["all_bullets_start_with"]
        bad = [b for b in bullet_lines(text) if not re.sub(r"^\s*(?:[-*•–—▪◦]|\d{1,3}[.)])\s+", "", b).casefold().startswith(prefix.casefold())]
        add("all_bullets_start_with", not bad and bool(bullet_lines(text)), bad[:3], prefix)
    if spec.get("max_bullet_words"):
        longest = max((word_count(b) for b in bullet_lines(text)), default=0)
        add("max_bullet_words", longest <= spec["max_bullet_words"] and bool(bullet_lines(text)), longest, spec["max_bullet_words"])
    if spec.get("unique_first_letters"):
        letters = [w[0].casefold() for w in words(text)]
        add("unique_first_letters", len(letters) == len(set(letters)) and bool(letters), len(letters), True)
    for pattern in _listify(spec.get("require_regex")):
        add(f"regex:{pattern}", re.search(pattern, text, re.I | re.S) is not None, None, pattern)
    for pattern in _listify(spec.get("forbid_regex")):
        add(f"no_regex:{pattern}", re.search(pattern, text, re.I | re.S) is None, None, pattern)
    return entries


def _listify(value: Any) -> list[str]:
    if not value:
        return []
    return [value] if isinstance(value, str) else list(value)


def _is_only_json(text: str) -> bool:
    stripped = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())
    found = extract_json(stripped)
    return found is not None and stripped.strip()[:1] in "{[" and stripped.strip()[-1:] in "}]"


CONSTRAINT_KEYS = {
    f"{p}_{n}" for p in ("min", "max", "exact") for n in ("words", "sentences", "bullets", "paragraphs", "chars", "lines")
} | {"include", "include_any", "exclude", "language", "no_english", "forbid_anglicisms", "case", "starts_with", "ends_with", "json_only", "no_bullets",
     "no_commas", "no_digits", "no_markdown", "all_bullets_start_with", "max_bullet_words", "unique_first_letters", "require_regex", "forbid_regex"}
#: options that change how the constraints are read, not constraints themselves
OPTION_KEYS = ("type", "ignore_accents", "weight", "scope")


def check_constraints(spec: dict[str, Any], output: ModelOutput, ctx: CheckContext) -> dict[str, Any]:
    unknown = [k for k in spec if k not in CONSTRAINT_KEYS and k not in OPTION_KEYS]
    if unknown:
        return bad_spec(f"constraints: unknown keys {unknown}")
    if (bad := scope_problem(spec)) is not None:
        return bad
    entries = evaluate(scoped(output, spec), spec)
    if not entries:
        return bad_spec("constraints needs at least one constraint")
    ok = sum(1 for e in entries if e["ok"])
    return verdict(ok / len(entries), ok == len(entries), {"constraints": entries})
