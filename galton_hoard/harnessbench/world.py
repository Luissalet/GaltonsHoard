"""The tools of the harness bench and the small in-memory world they act on.

Every tool is a pure function of the world: a calculator, a unit converter, a document store (search by words, read by id), a virtual file
system (list, read, write) and a record lookup that fails on its first call. Nothing touches the real disk or the network, so a trial is
deterministic and a model cannot do harm. ``World.run`` validates the arguments against the tool's schema first: that rejection is what the
bench counts as a format failure (the call never reached the tool), as opposed to a call that ran and was simply wrong.
"""

from __future__ import annotations

import ast
import copy
import math
import operator
import re
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    parameters: dict[str, Any]

    def schema(self) -> dict[str, Any]:
        """The tool in chat-completions function-calling form."""
        return {"type": "function", "function": {"name": self.name, "description": self.description, "parameters": self.parameters}}

    def required(self) -> list[str]:
        return list(self.parameters.get("required", []))


def _spec(name: str, description: str, props: dict[str, tuple[str, str]], required: Optional[list[str]] = None) -> ToolSpec:
    properties = {key: {"type": kind, "description": desc} for key, (kind, desc) in props.items()}
    return ToolSpec(name, description, {"type": "object", "properties": properties, "required": list(props) if required is None else required})


TOOL_SPECS: dict[str, ToolSpec] = {s.name: s for s in (
    _spec("calculator", "Evaluate an arithmetic expression. Supports + - * / // % ** parentheses and sqrt(), abs(), round().",
          {"expression": ("string", "The expression, for example (3 + 4) * 2.")}),
    _spec("convert", "Convert a value between units of length (km, mi, m, ft, in, cm), mass (kg, lb, g, oz), temperature (c, f, k), time (s, min, h, d) or volume (l, ml, gal).",
          {"value": ("number", "The number to convert."), "from_unit": ("string", "Unit of the value."), "to_unit": ("string", "Unit wanted.")}),
    _spec("search_docs", "Search the documentation store by words. Returns the ids and titles of the best matching documents, not their text.",
          {"query": ("string", "Words to look for.")}),
    _spec("read_doc", "Read the full text of one document of the documentation store by its id.",
          {"id": ("string", "Document id as returned by search_docs.")}),
    _spec("list_dir", "List the files and folders directly inside a folder of the virtual file system.",
          {"path": ("string", "Absolute folder path, for example /data.")}),
    _spec("read_file", "Read a text file of the virtual file system.",
          {"path": ("string", "Absolute file path.")}),
    _spec("write_file", "Write a text file in the virtual file system (replaces it when it exists).",
          {"path": ("string", "Absolute file path."), "content": ("string", "The whole new content.")}),
    _spec("lookup_record", "Look up a record of the records service by its id and return its fields.",
          {"record_id": ("string", "Record id, for example REC-100.")}),
)}


# ------------------------------------------------------------------------------------------------- the documentation store
DOCS: dict[str, dict[str, str]] = {d["id"]: d for d in (
    {"id": "doc-001", "title": "Orchid service", "text": "The Orchid service listens on port 8443 and stores its data in /var/orchid. Restarts require the token ORC-7731."},
    {"id": "doc-002", "title": "Basalt cache", "text": "Basalt keeps cache entries for 90 seconds by default. The maximum cache size is 512 MB. Eviction is least-recently-used."},
    {"id": "doc-003", "title": "Cinder queue", "text": "Cinder processes up to 40 jobs per minute. Failed jobs are retried 3 times with a 20 second delay between attempts."},
    {"id": "doc-004", "title": "Delta backups", "text": "Delta backups run every day at 02:30 and are kept for 14 days. The backup bucket is named delta-archive-eu."},
    {"id": "doc-005", "title": "Ember gateway", "text": "The Ember gateway allows 120 requests per minute per client. Its admin console is served on port 9100."},
    {"id": "doc-006", "title": "Fjord database", "text": "Fjord has 6 replicas and each replica holds 250 GB. Writes always go to the primary, named fjord-p1."},
    {"id": "doc-007", "title": "Granite deploys", "text": "Granite deploys happen on Tuesdays and Thursdays. A deploy freeze starts on 12 December."},
    {"id": "doc-008", "title": "Harbor logs", "text": "Harbor keeps its logs for 30 days. Each log file rotates when it reaches 100 MB."},
)}

RECORDS: dict[str, str] = {
    "REC-204": "status=shipped; weight_kg=12; carrier=Northline",
    "REC-310": "status=pending; weight_kg=4; carrier=Eastway",
}

_WORD = re.compile(r"[a-z0-9]{3,}")


def _words(text: str) -> set[str]:
    return set(_WORD.findall(text.lower()))


# ------------------------------------------------------------------------------------------------- calculator
_BIN = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv,
        ast.Mod: operator.mod, ast.Pow: operator.pow}
_FUNCS = {"sqrt": math.sqrt, "abs": abs, "round": round}


class CalcError(ValueError):
    pass


def _eval(node: ast.AST) -> float:
    if isinstance(node, ast.Expression):
        return _eval(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
        return node.value
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        value = _eval(node.operand)
        return value if isinstance(node.op, ast.UAdd) else -value
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN:
        left, right = _eval(node.left), _eval(node.right)
        if isinstance(node.op, ast.Pow) and (abs(right) > 1000 or abs(left) > 1e9):
            raise CalcError("the exponent is too large")
        try:
            return _BIN[type(node.op)](left, right)
        except ZeroDivisionError as exc:
            raise CalcError("division by zero") from exc
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in _FUNCS and not node.keywords:
        try:
            return _FUNCS[node.func.id](*[_eval(a) for a in node.args])
        except (TypeError, ValueError) as exc:
            raise CalcError(f"bad call to {node.func.id}") from exc
    raise CalcError("only numbers, + - * / // % **, parentheses and sqrt(), abs(), round() are allowed")


def number_text(value: float) -> str:
    if isinstance(value, float) and value.is_integer() and abs(value) < 1e15:
        return str(int(value))
    return format(value, ".10g") if isinstance(value, float) else str(value)


def calculate(expression: str) -> str:
    expr = str(expression).strip()
    if "^" in expr:
        raise CalcError("^ is not a power here; write powers with **")
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as exc:
        raise CalcError("not a valid expression") from exc
    return number_text(_eval(tree))


# ------------------------------------------------------------------------------------------------- unit conversion
_UNITS: dict[str, tuple[str, float]] = {}      # unit -> (dimension, factor to the base unit)
_ALIASES: dict[str, str] = {}


def _units(dimension: str, table: dict[str, float], aliases: dict[str, str]) -> None:
    for unit, factor in table.items():
        _UNITS[unit] = (dimension, factor)
    _ALIASES.update(aliases)


_units("length", {"m": 1.0, "km": 1000.0, "cm": 0.01, "mi": 1609.344, "ft": 0.3048, "in": 0.0254},
       {"meter": "m", "meters": "m", "metre": "m", "metres": "m", "kilometer": "km", "kilometers": "km", "kilometre": "km", "kilometres": "km", "mile": "mi",
        "miles": "mi", "foot": "ft", "feet": "ft", "inch": "in", "inches": "in", "centimeter": "cm", "centimeters": "cm"})
_units("mass", {"kg": 1.0, "g": 0.001, "lb": 0.45359237, "oz": 0.028349523125},
       {"kilogram": "kg", "kilograms": "kg", "gram": "g", "grams": "g", "pound": "lb", "pounds": "lb", "lbs": "lb", "ounce": "oz", "ounces": "oz"})
_units("time", {"s": 1.0, "min": 60.0, "h": 3600.0, "d": 86400.0},
       {"second": "s", "seconds": "s", "minute": "min", "minutes": "min", "hour": "h", "hours": "h", "day": "d", "days": "d"})
_units("volume", {"l": 1.0, "ml": 0.001, "gal": 3.785411784}, {"liter": "l", "liters": "l", "litre": "l", "litres": "l", "gallon": "gal", "gallons": "gal"})
_TEMPS = {"c": "c", "celsius": "c", "°c": "c", "f": "f", "fahrenheit": "f", "°f": "f", "k": "k", "kelvin": "k"}


def known_units() -> str:
    return ", ".join([*sorted(_UNITS), "c", "f", "k"])


def convert_value(value: float, from_unit: str, to_unit: str) -> str:
    a, b = str(from_unit).strip().lower(), str(to_unit).strip().lower()
    a, b = _ALIASES.get(a, a), _ALIASES.get(b, b)
    if a in _TEMPS or b in _TEMPS:
        ta, tb = _TEMPS.get(a), _TEMPS.get(b)
        if ta is None or tb is None:
            bad = a if ta is None else b
            raise CalcError(f"unknown unit '{bad}'. Known units: {known_units()}")
        kelvin = {"c": value + 273.15, "f": (value - 32) * 5 / 9 + 273.15, "k": value}[ta]
        result = {"c": kelvin - 273.15, "f": (kelvin - 273.15) * 9 / 5 + 32, "k": kelvin}[tb]
        return f"{number_text(float(value))} {ta} = {format(result, '.6g')} {tb}"
    for unit in (a, b):
        if unit not in _UNITS:
            raise CalcError(f"unknown unit '{unit}'. Known units: {known_units()}")
    (da, fa), (db, fb) = _UNITS[a], _UNITS[b]
    if da != db:
        raise CalcError(f"cannot convert {da} ({a}) to {db} ({b})")
    return f"{number_text(float(value))} {a} = {format(value * fa / fb, '.6g')} {b}"


# ------------------------------------------------------------------------------------------------- the world
@dataclass
class ToolResult:
    ok: bool
    text: str


def _norm_path(path: str) -> str:
    parts = [p for p in str(path).strip().split("/") if p not in ("", ".")]
    return "/" + "/".join(parts)


@dataclass
class World:
    """The state a trial acts on: virtual files, and how many first calls of the records service still fail."""

    files: dict[str, str] = field(default_factory=dict)
    records_fail_first: int = 0
    record_calls: int = 0
    writes: list[str] = field(default_factory=list)

    @classmethod
    def build(cls, spec: Optional[dict[str, Any]]) -> "World":
        spec = copy.deepcopy(spec or {})
        return cls(files={_norm_path(p): c for p, c in (spec.get("files") or {}).items()}, records_fail_first=int(spec.get("records_fail_first", 0)))

    # -- arguments -----------------------------------------------------------------------------------------------
    @staticmethod
    def check_args(spec: ToolSpec, args: Any) -> tuple[Optional[dict[str, Any]], str]:
        """The arguments as the tool will get them, or the reason the call is refused before it runs."""
        if not isinstance(args, dict):
            return None, "the arguments must be a JSON object"
        if "_unparsed" in args:
            return None, "the arguments are not valid JSON"
        clean: dict[str, Any] = {}
        for key in spec.required():
            if key not in args or args[key] is None:
                return None, f"missing required argument '{key}'"
        for key, meta in spec.parameters["properties"].items():
            if key not in args:
                continue
            value, kind = args[key], meta["type"]
            if kind == "string":
                if isinstance(value, (dict, list)) or value is None:
                    return None, f"argument '{key}' must be a string"
                value = str(value) if not isinstance(value, str) else value
            elif kind == "number":
                if isinstance(value, bool):
                    return None, f"argument '{key}' must be a number"
                if isinstance(value, str):
                    try:
                        value = float(value.strip())
                    except ValueError:
                        return None, f"argument '{key}' must be a number"
                if not isinstance(value, (int, float)):
                    return None, f"argument '{key}' must be a number"
            clean[key] = value
        return clean, ""

    # -- the tools ------------------------------------------------------------------------------------------------
    def run(self, name: str, args: dict[str, Any]) -> ToolResult:
        """Run an already validated call."""
        try:
            return ToolResult(True, getattr(self, f"_t_{name}")(**args))
        except CalcError as exc:
            return ToolResult(False, f"Error: {exc}")
        except _ToolFail as exc:
            return ToolResult(False, f"Error: {exc}")

    @staticmethod
    def _t_calculator(expression: str) -> str:
        return calculate(expression)

    @staticmethod
    def _t_convert(value: float, from_unit: str, to_unit: str) -> str:
        return convert_value(float(value), from_unit, to_unit)

    @staticmethod
    def _t_search_docs(query: str) -> str:
        wanted = _words(query)
        scored = []
        for doc in DOCS.values():
            hits = len(wanted & _words(doc["title"])) * 3 + len(wanted & _words(doc["text"]))
            if hits:
                scored.append((-hits, doc["id"], doc["title"]))
        if not scored:
            return "No documents match."
        return "\n".join(f"{doc_id}: {title}" for _, doc_id, title in sorted(scored)[:3])

    @staticmethod
    def _t_read_doc(id: str) -> str:  # noqa: A002 - the tool's own argument name
        doc = DOCS.get(str(id).strip())
        if doc is None:
            digits = re.sub(r"\D", "", str(id))
            guess = f"doc-{int(digits):03d}" if digits else ""
            hint = f" Did you mean '{guess}'?" if guess in DOCS else " Use search_docs to find ids."
            raise _ToolFail(f"no document '{id}'.{hint}")
        return f"{doc['title']}\n{doc['text']}"

    def _t_list_dir(self, path: str) -> str:
        base = _norm_path(path)
        prefix = base.rstrip("/") + "/"
        names = sorted({p[len(prefix):].split("/")[0] + ("/" if "/" in p[len(prefix):] else "") for p in self.files if p.startswith(prefix)})
        if not names:
            raise _ToolFail(f"no such folder: {base}")
        return "\n".join(names)

    def _t_read_file(self, path: str) -> str:
        key = _norm_path(path)
        if key not in self.files:
            folder = key.rsplit("/", 1)[0] or "/"
            near = sorted(p.rsplit("/", 1)[1] for p in self.files if (p.rsplit("/", 1)[0] or "/") == folder)
            raise _ToolFail(f"file not found: {key}." + (f" Files in {folder}: {', '.join(near)}" if near else ""))
        return self.files[key]

    def _t_write_file(self, path: str, content: str) -> str:
        key = _norm_path(path)
        self.files[key] = str(content)
        self.writes.append(key)
        return f"Wrote {len(str(content))} characters to {key}"

    def _t_lookup_record(self, record_id: str) -> str:
        self.record_calls += 1
        if self.record_calls <= self.records_fail_first:
            raise _ToolFail("the records service is temporarily unavailable (503). Try the same call again.")
        rid = str(record_id).strip().upper()
        if rid not in RECORDS:
            raise _ToolFail(f"no record '{record_id}'")
        return RECORDS[rid]


class _ToolFail(Exception):
    pass
