"""Importing cases from JSONL or CSV text, and the one-line checker shorthand used by imports and by assistants.

A row needs a prompt. Everything else is optional: ``title``, ``system``, ``expected`` (what a good answer contains), ``checker`` (a full object, or a
shorthand such as ``contains:foo|bar``), ``weight``, ``tags`` and ``notes``. Column and key names may be Spanish or English. A row with an expected
answer and no checker is checked with ``number`` when the answer is a number and ``contains`` otherwise; a row with neither is kept unscored (its
answers are still useful in the arena). Imported text is untrusted data: nothing in it is executed.
"""

from __future__ import annotations

import csv
import io
import json
import re
from typing import Any, Optional

from .errors import GaltonError
from .messages import text as message
from .suites import normalise_case, validate_case
from .util import fold

ALIASES: dict[str, tuple[str, ...]] = {
    "title": ("title", "titulo", "name", "nombre"),
    "prompt": ("prompt", "question", "pregunta", "input", "text", "texto", "enunciado"),
    "system": ("system", "sistema"),
    "expected": ("expected", "answer", "respuesta", "reference", "referencia", "esperado", "solucion"),
    "checker": ("checker", "check", "comprobacion", "verificacion"),
    "weight": ("weight", "peso"),
    "tags": ("tags", "etiquetas"),
    "notes": ("notes", "notas"),
}
MAX_ROWS = 2000
NUMBER = re.compile(r"^[-+]?\d+(?:[.,]\d+)?$")


def parse_number(text: str) -> Optional[float]:
    text = text.strip()
    if not NUMBER.match(text):
        return None
    return float(text.replace(",", "."))


def shorthand(text: str) -> dict[str, Any]:
    """``type:value`` to a checker spec. ``contains:a|b`` needs all of the words, ``any:a|b`` one of them, ``none`` means unscored."""
    text = text.strip()
    kind, _, value = text.partition(":")
    kind, value = kind.strip().lower(), value.strip()
    words = [w.strip() for w in value.split("|") if w.strip()]
    if kind in ("none", "ninguno", ""):
        return {"type": "none"}
    if not value:
        raise GaltonError("invalid", "shorthand_needs_value", text=text)
    if kind in ("exact", "exacto"):
        return {"type": "exact", "expected": words if len(words) > 1 else value, "ignore_accents": True}
    if kind in ("contains", "contiene"):
        return {"type": "contains", "all": words, "ignore_accents": True}
    if kind in ("any", "alguno"):
        return {"type": "contains", "any": words, "ignore_accents": True}
    if kind in ("number", "numero", "número"):
        number = parse_number(value)
        if number is None:
            raise GaltonError("invalid", "not_a_number", value=value)
        return {"type": "number", "expected": number}
    if kind in ("choice", "opcion", "opción"):
        return {"type": "choice", "answer": value.upper()}
    if kind == "regex":
        try:
            re.compile(value)
        except re.error as exc:
            raise GaltonError("invalid", "regex_invalid", detail=str(exc)) from exc
        return {"type": "regex", "pattern": value}
    if kind in ("math", "math_equiv"):
        return {"type": "math_equiv", "expected": value}
    if kind in ("judge", "juez"):
        return {"type": "judge", "rubric": value}
    raise GaltonError("invalid", "shorthand_unknown", kind=kind)


def derive_checker(expected: str) -> dict[str, Any]:
    """The checker for a row that only has an expected answer."""
    expected = expected.strip()
    if not expected:
        return {"type": "none"}
    number = parse_number(expected)
    if number is not None:
        return {"type": "number", "expected": number}
    return {"type": "contains", "all": [expected], "ignore_accents": True}


def _pick(row: dict[str, Any], field: str) -> Any:
    folded = {fold(str(k)).strip(): v for k, v in row.items()}
    for alias in ALIASES[field]:
        value = folded.get(fold(alias))
        if value not in (None, ""):
            return value
    return None


def _tags(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]
    return [t.strip() for t in re.split(r"[;,]", str(value or "")) if t.strip()]


def row_to_case(row: dict[str, Any], *, default_checker: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    prompt = _pick(row, "prompt")
    if prompt is None:
        raise GaltonError("invalid", "row_no_prompt")
    expected = _pick(row, "expected")
    expected = "" if expected is None else (expected if isinstance(expected, str) else json.dumps(expected, ensure_ascii=False))
    raw_checker = _pick(row, "checker")
    if isinstance(raw_checker, dict):
        checker = raw_checker
    elif isinstance(raw_checker, str) and raw_checker.strip().startswith("{"):
        try:
            checker = json.loads(raw_checker)
        except ValueError as exc:
            raise GaltonError("invalid", "checker_json", detail=str(exc)) from exc
    elif isinstance(raw_checker, str):
        checker = shorthand(raw_checker)
    else:
        checker = default_checker or derive_checker(expected)
    prompt_obj: Any = prompt if isinstance(prompt, dict) else str(prompt)
    system = _pick(row, "system")
    if system and isinstance(prompt_obj, str):
        prompt_obj = {"text": prompt_obj, "system": str(system)}
    try:
        weight = float(_pick(row, "weight") or 1.0)
    except (TypeError, ValueError) as exc:
        raise GaltonError("invalid", "weight_number") from exc
    case = normalise_case({"title": _pick(row, "title") or "", "prompt": prompt_obj, "checker": checker, "weight": weight, "tags": _tags(_pick(row, "tags")),
                           "reference": expected, "notes": str(_pick(row, "notes") or ""), "tools": row.get("tools") or []})
    problems = validate_case(case)
    if problems:
        raise GaltonError("invalid", "case_invalid", problems="; ".join(problems), problem_items=list(problems))
    return case


def read_rows(text: str, fmt: str = "auto") -> tuple[list[tuple[int, dict[str, Any]]], list[dict[str, Any]]]:
    """Split the text into ``(line, row)`` pairs; lines that cannot be read go to the error list."""
    text = text.lstrip("﻿")
    stripped = text.strip()
    if not stripped:
        raise GaltonError("invalid", "import_empty")
    if fmt == "auto":
        fmt = "jsonl" if stripped[0] in "{[" else "csv"
    rows: list[tuple[int, dict[str, Any]]] = []
    errors: list[dict[str, Any]] = []
    if fmt == "jsonl":
        if stripped[0] == "[":
            try:
                items = json.loads(stripped)
            except ValueError as exc:
                raise GaltonError("invalid", "json_array_invalid", detail=str(exc)) from exc
            for i, item in enumerate(items, 1):
                (rows.append((i, item)) if isinstance(item, dict) else errors.append({"row": i, "error": message("row_not_object")}))
        else:
            for i, line in enumerate(stripped.splitlines(), 1):
                if not line.strip():
                    continue
                try:
                    item = json.loads(line)
                except ValueError as exc:
                    errors.append({"row": i, "error": message("row_bad_json", detail=str(exc))})
                    continue
                (rows.append((i, item)) if isinstance(item, dict) else errors.append({"row": i, "error": message("row_not_object")}))
    elif fmt == "csv":
        try:
            dialect = csv.Sniffer().sniff(stripped.splitlines()[0], delimiters=",;\t")
        except csv.Error:
            dialect = csv.excel
        reader = csv.DictReader(io.StringIO(stripped), dialect=dialect)
        for i, item in enumerate(reader, 2):
            rows.append((i, {k: v for k, v in item.items() if k}))
    else:
        raise GaltonError("invalid", "import_format")
    if len(rows) + len(errors) > MAX_ROWS:
        raise GaltonError("too_large", "import_rows", max=MAX_ROWS)
    return rows, errors


def parse_cases(text: str, fmt: str = "auto", *, default_checker: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    """``{"cases": [...], "errors": [{"row", "error"}]}``: every readable row becomes a normalised case, the rest is reported."""
    rows, errors = read_rows(text, fmt)
    cases: list[dict[str, Any]] = []
    for line, row in rows:
        try:
            cases.append(row_to_case(row, default_checker=default_checker))
        except GaltonError as exc:
            errors.append({"row": line, "error": exc.coded()})
    return {"cases": cases, "errors": sorted(errors, key=lambda e: e["row"])}
