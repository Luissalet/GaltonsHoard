"""The ``json`` checker (extract the first JSON value of an answer, validate it against a JSON-schema subset, compare expected fields)
and the ``tool_call`` checker (did the model call the right function with the right arguments)."""

from __future__ import annotations

import json
import re
from datetime import date
from typing import Any, Iterator, Optional

from ..util import fold, squash
from .numbers import close, interpret
from .textutil import balanced_slice, extract_json, parse_loose, split_reasoning
from .types import CheckContext, ModelOutput, bad_spec, verdict

# ---------------------------------------------------------------------------------------------------- JSON schema subset
_TYPE_CHECK = {
    "string": lambda v: isinstance(v, str),
    "number": lambda v: isinstance(v, (int, float)) and not isinstance(v, bool),
    "integer": lambda v: (isinstance(v, int) and not isinstance(v, bool)) or (isinstance(v, float) and v.is_integer()),
    "boolean": lambda v: isinstance(v, bool),
    "null": lambda v: v is None,
    "object": lambda v: isinstance(v, dict),
    "array": lambda v: isinstance(v, list),
}


def validate_schema(value: Any, schema: dict[str, Any], path: str = "$") -> list[str]:
    """Errors of ``value`` against the subset of JSON Schema this app supports: type, required, properties, additionalProperties,
    enum, const, items, minItems, maxItems, minimum, maximum, minLength, maxLength, pattern. Empty list: valid."""
    errors: list[str] = []
    types = schema.get("type")
    if types is not None:
        wanted = types if isinstance(types, list) else [types]
        if not any(_TYPE_CHECK.get(t, lambda v: False)(value) for t in wanted):
            return [f"{path}: expected {'/'.join(wanted)}, got {type(value).__name__}"]
    if "enum" in schema and value not in schema["enum"]:
        errors.append(f"{path}: {value!r} is not one of {schema['enum']}")
    if "const" in schema and value != schema["const"]:
        errors.append(f"{path}: expected {schema['const']!r}")
    if isinstance(value, dict):
        for name in schema.get("required", []):
            if name not in value:
                errors.append(f"{path}.{name}: missing")
        props = schema.get("properties", {})
        for name, sub in props.items():
            if name in value:
                errors.extend(validate_schema(value[name], sub, f"{path}.{name}"))
        if schema.get("additionalProperties") is False:
            for name in value:
                if name not in props:
                    errors.append(f"{path}.{name}: not allowed")
    if isinstance(value, list):
        if "minItems" in schema and len(value) < schema["minItems"]:
            errors.append(f"{path}: at least {schema['minItems']} items")
        if "maxItems" in schema and len(value) > schema["maxItems"]:
            errors.append(f"{path}: at most {schema['maxItems']} items")
        if isinstance(schema.get("items"), dict):
            for index, item in enumerate(value):
                errors.extend(validate_schema(item, schema["items"], f"{path}[{index}]"))
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if "minimum" in schema and value < schema["minimum"]:
            errors.append(f"{path}: below {schema['minimum']}")
        if "maximum" in schema and value > schema["maximum"]:
            errors.append(f"{path}: above {schema['maximum']}")
    if isinstance(value, str):
        if "minLength" in schema and len(value) < schema["minLength"]:
            errors.append(f"{path}: shorter than {schema['minLength']}")
        if "maxLength" in schema and len(value) > schema["maxLength"]:
            errors.append(f"{path}: longer than {schema['maxLength']}")
        if "pattern" in schema and not re.search(schema["pattern"], value):
            errors.append(f"{path}: does not match {schema['pattern']}")
    return errors


# ---------------------------------------------------------------------------------------------------- field comparison
def get_path(data: Any, path: str) -> tuple[bool, Any]:
    """Follow ``a.b.0.c`` / ``a[0].b`` through dicts and lists. ``(found, value)``."""
    current = data
    for part in [p for p in re.split(r"[.\[\]]+", path) if p != ""]:
        if isinstance(current, dict) and part in current:
            current = current[part]
        elif isinstance(current, list) and part.lstrip("-").isdigit() and -len(current) <= int(part) < len(current):
            current = current[int(part)]
        else:
            return False, None
    return True, current


def leaf_paths(expected: Any, prefix: str = "") -> list[tuple[str, Any]]:
    """Every leaf of an expected structure as (path, value); dicts are expanded, lists and scalars are leaves."""
    if isinstance(expected, dict) and expected:
        out: list[tuple[str, Any]] = []
        for key, value in expected.items():
            out.extend(leaf_paths(value, f"{prefix}.{key}" if prefix else str(key)))
        return out
    return [(prefix, expected)]


def _to_number(value: Any) -> Optional[float]:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        stripped = re.sub(r"[€$£\s]|EUR|USD|eur|usd", "", value)
        found = interpret(stripped) if re.fullmatch(r"[-+−]?[\d.,']+", stripped) else []
        return found[0] if found else None
    return None


def _as_date(value: Any) -> Optional[date]:
    if not isinstance(value, str):
        return None
    text = value.strip()
    m = re.fullmatch(r"(\d{4})-(\d{2})-(\d{2})", text)
    if m:
        y, mo, d = (int(x) for x in m.groups())
    else:
        m = re.fullmatch(r"(\d{1,2})[/.\-](\d{1,2})[/.\-](\d{4})", text)
        if not m:
            return None
        d, mo, y = (int(x) for x in m.groups())
    try:
        return date(y, mo, d)
    except ValueError:
        return None


#: Generic words that label a place or a thing and may stand before its name («sala Magallanes», «calle Mayor»): the ``norm`` rule drops one.
GENERIC_LEADS = ("sala", "calle", "c/", "c.", "avda.", "avda", "av.", "avenida", "plaza", "pza.", "paseo", "ctra.", "carretera", "provincia", "ciudad",
                 "localidad", "oficina", "edificio", "planta", "piso", "aula", "despacho", "tienda", "room", "street", "st.", "avenue", "ave.", "road",
                 "rd.", "city", "province")
_ARTICLE = r"(?:(?:de\s+(?:la|las|los|el)|del|de|la|el)\s+)?"


def strip_generic(value: str, words: Any = GENERIC_LEADS) -> str:
    """``value`` folded (no accents, lower case, no final full stop) without a leading generic word and the article after it:
    ``Sala Magallanes`` -> ``magallanes``, ``c/ del Olmo`` -> ``olmo``. A value that is only the generic word is left as it is."""
    text = fold(squash(value)).casefold().rstrip(".")
    alts = "|".join(re.escape(fold(w)) for w in sorted(words, key=len, reverse=True))
    match = re.match(r"^(?:(?:la|el|las|los|the)\s+)?(?:" + alts + r")(?:\s+|(?<=[/.])\s*)" + _ARTICLE, text)
    return text[match.end():].strip() or text if match else text


def values_equal(got: Any, want: Any, rule: Any = None, *, strict_types: bool = False) -> bool:
    """Compare one extracted value with the expected one under a rule (a mode name or ``{mode, abs, rel}``).

    String modes: ``strict`` (same text, accents and case included), ``contains`` (the expected text is somewhere in the answer), ``norm`` (equal
    once a leading generic word such as ``sala`` or ``calle`` is dropped from both; ``{mode: "norm", words: [...]}`` replaces the list of words),
    ``date``, ``regex``; by default equal without accents, case or a final full stop. List mode: ``unordered``."""
    spec = {"mode": rule} if isinstance(rule, str) else dict(rule or {})
    mode = spec.get("mode", "")
    if mode == "any":
        return got is not None
    if isinstance(want, bool):
        if isinstance(got, str) and not strict_types:
            got = {"true": True, "false": False, "sí": True, "si": True, "no": False}.get(got.strip().casefold(), got)
        return isinstance(got, bool) and got == want
    if want is None:
        return got is None
    if isinstance(want, (int, float)):
        number = got if (isinstance(got, (int, float)) and not isinstance(got, bool)) else (None if strict_types else _to_number(got))
        if number is None:
            return False
        return close(float(number), float(want), float(spec.get("abs", 1e-6)), float(spec.get("rel", 0.0)))
    if isinstance(want, str):
        if mode == "date":
            a, b = _as_date(got), _as_date(want)
            return a is not None and a == b
        if mode == "regex":
            return isinstance(got, str) and re.search(want, got) is not None
        if not isinstance(got, str):
            if isinstance(got, (int, float)) and not isinstance(got, bool) and not strict_types:
                got = str(int(got)) if float(got).is_integer() else str(got)
            else:
                return False
        if mode == "strict":
            return squash(got) == squash(want)
        if mode == "contains":
            return fold(squash(want)).casefold() in fold(squash(got)).casefold()
        if mode == "norm":
            words = spec.get("words") if isinstance(spec.get("words"), list) and spec.get("words") else GENERIC_LEADS
            return strip_generic(got, words) == strip_generic(want, words)
        return fold(squash(got)).casefold().rstrip(".") == fold(squash(want)).casefold().rstrip(".")
    if isinstance(want, list):
        if not isinstance(got, list) or len(got) != len(want):
            return False
        if mode == "unordered":
            remaining = list(got)
            for item in want:
                for candidate in remaining:
                    if values_equal(candidate, item, None, strict_types=strict_types):
                        remaining.remove(candidate)
                        break
                else:
                    return False
            return True
        return all(values_equal(g, w, None, strict_types=strict_types) for g, w in zip(got, want))
    if isinstance(want, dict):
        return isinstance(got, dict) and all(k in got and values_equal(got[k], v, None, strict_types=strict_types) for k, v in want.items())
    return got == want


def compare_fields(data: Any, expected: Any, rules: Optional[dict[str, Any]] = None, *, strict_types: bool = False) -> tuple[float, list[dict[str, Any]]]:
    """Fraction of expected leaves that match, and one entry per leaf."""
    leaves = leaf_paths(expected)
    entries = []
    for path, want in leaves:
        found, got = get_path(data, path) if path else (True, data)
        ok = found and values_equal(got, want, (rules or {}).get(path), strict_types=strict_types)
        entries.append({"path": path or "$", "ok": bool(ok), "want": want, "got": got if found else "<missing>"})
    return (sum(1 for e in entries if e["ok"]) / len(entries) if entries else 1.0), entries


def check_json(spec: dict[str, Any], output: ModelOutput, ctx: CheckContext) -> dict[str, Any]:
    schema, expected = spec.get("schema"), spec.get("expected")
    if schema is None and expected is None:
        return bad_spec("json needs `schema` and/or `expected`")
    text = split_reasoning(output.text)[0]
    found = extract_json(text, expect=spec.get("expect"))
    if found is None:
        return verdict(0.0, False, {"reason": "no JSON in the answer"})
    value, repaired = found
    detail: dict[str, Any] = {"repaired": repaired}
    if spec.get("strict_json") and not _only_json(text):
        detail["extra_text"] = True
    errors = validate_schema(value, schema) if isinstance(schema, dict) else []
    detail["schema_errors"] = errors[:8]
    frac, entries = compare_fields(value, expected, spec.get("rules"), strict_types=bool(spec.get("strict_types"))) if expected is not None else (1.0, [])
    if entries:
        detail["fields"] = entries
    schema_ok = not errors and not detail.get("extra_text")
    score = frac * (1.0 if schema_ok else 0.5)
    return verdict(score, schema_ok and frac == 1.0, detail)


def _only_json(text: str) -> bool:
    stripped = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())
    try:
        json.loads(stripped)
        return True
    except ValueError:
        return False


# ---------------------------------------------------------------------------------------------------- tool calls
def iter_json_values(text: str) -> Iterator[Any]:
    """Every top-level JSON object or array in the text, in order (tool calls may come one after another)."""
    decoder = json.JSONDecoder()
    index = 0
    while index < len(text):
        match = re.search(r"[{\[]", text[index:])
        if not match:
            return
        start = index + match.start()
        try:
            value, end = decoder.raw_decode(text[start:])
            yield value
            index = start + end
            continue
        except ValueError:
            piece = balanced_slice(text, start)
            if piece:
                try:
                    yield parse_loose(piece)[0]
                    index = start + len(piece)
                    continue
                except ValueError:
                    pass
        index = start + 1


def normalise_call(item: Any) -> Optional[dict[str, Any]]:
    """``{name, arguments}`` from the many shapes models use (the ``function`` wrapper, ``parameters``, JSON-string arguments)."""
    if not isinstance(item, dict):
        return None
    if isinstance(item.get("function"), dict):
        item = {**item["function"], **{k: v for k, v in item.items() if k != "function"}}
    name = item.get("name") or item.get("tool") or item.get("function_name") or (item.get("function") if isinstance(item.get("function"), str) else None)
    if not isinstance(name, str) or not name:
        return None
    args = next((item[k] for k in ("arguments", "parameters", "args", "input") if k in item), {})
    if isinstance(args, str):
        try:
            args = parse_loose(args)[0] if args.strip() else {}
        except ValueError:
            args = {"_unparsed": args}
    if not isinstance(args, dict):
        args = {"_value": args}
    return {"name": name, "arguments": args}


def calls_from_output(output: ModelOutput) -> list[dict[str, Any]]:
    calls = [c for c in (normalise_call(tc) for tc in output.tool_calls) if c]
    if calls:
        return calls
    text = split_reasoning(output.text)[0]
    found: list[dict[str, Any]] = []
    for value in iter_json_values(text):
        items = value if isinstance(value, list) else [value]
        for item in items:
            if isinstance(item, dict) and isinstance(item.get("tool_calls"), list):
                found.extend(c for c in (normalise_call(x) for x in item["tool_calls"]) if c)
                continue
            call = normalise_call(item)
            if call:
                found.append(call)
    return found


def _score_call(call: dict[str, Any], want: dict[str, Any], rules: dict[str, Any], strict_types: bool, exact_args: bool) -> tuple[float, bool, list[dict[str, Any]]]:
    if fold(call["name"]).casefold() != fold(str(want.get("name", ""))).casefold():
        return 0.0, False, []
    wanted_args = want.get("arguments") or {}
    frac, entries = compare_fields(call["arguments"], wanted_args, rules, strict_types=strict_types) if wanted_args else (1.0, [])
    extra = sorted(set(call["arguments"]) - set(wanted_args)) if exact_args else []
    full = frac == 1.0 and not extra
    if extra:
        entries.append({"path": "*", "ok": False, "want": "no other arguments", "got": extra})
    return 0.4 + 0.6 * frac * (0.5 if extra else 1.0), full, entries


def check_tool_call(spec: dict[str, Any], output: ModelOutput, ctx: CheckContext) -> dict[str, Any]:
    calls = calls_from_output(output)
    if spec.get("none"):
        ok = not calls
        return verdict(1.0 if ok else 0.0, ok, {"expected": "no tool call", "got": [c["name"] for c in calls]})
    expected = spec.get("expected")
    if isinstance(expected, dict):
        expected = [expected]
    if not expected or not all(isinstance(e, dict) and e.get("name") for e in expected):
        return bad_spec("tool_call needs `expected` ({name, arguments}) or `none`")
    rules, strict_types, exact_args = spec.get("rules") or {}, bool(spec.get("strict_types")), bool(spec.get("exact_args"))
    if not calls:
        return verdict(0.0, False, {"reason": "no tool call", "expected": [e["name"] for e in expected]})
    remaining = list(calls)
    scores: list[float] = []
    entries_all: list[dict[str, Any]] = []
    all_full = True
    for want in expected:
        best: Optional[tuple[float, bool, list, dict]] = None
        for call in remaining:
            s, full, entries = _score_call(call, want, rules, strict_types, exact_args)
            if best is None or s > best[0]:
                best = (s, full, entries, call)
        if best is None or best[0] == 0.0:
            scores.append(0.0)
            all_full = False
            entries_all.append({"call": want["name"], "ok": False, "got": [c["name"] for c in calls]})
            continue
        remaining.remove(best[3])
        scores.append(best[0])
        all_full = all_full and best[1]
        entries_all.append({"call": want["name"], "ok": best[1], "fields": best[2]})
    extra_calls = [c["name"] for c in remaining]
    score = sum(scores) / len(scores)
    if extra_calls and not spec.get("allow_extra_calls", False):
        score *= 0.8
        all_full = False
    return verdict(score, all_full, {"calls": entries_all, "extra_calls": extra_calls, "got": [c["name"] for c in calls]})
