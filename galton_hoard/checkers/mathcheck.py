"""The ``math_equiv`` checker: is the final expression of the answer equal to the expected one?

The answer is untrusted text. It is cleaned (LaTeX, Unicode, decimal comma), restricted to a whitelist of characters and names, and
only then parsed: with sympy when it is installed (a restricted namespace, never ``from sympy import *``), otherwise with a small
evaluator built on ``ast`` that supports the same grammar numerically. Equality is decided by evaluating both sides at several
points (and symbolically when sympy is there and the numeric test could not decide).
"""

from __future__ import annotations

import ast
import cmath
import math
import re
from typing import Any, Optional

from .textutil import split_reasoning
from .types import CheckContext, ModelOutput, bad_spec, verdict

try:  # sympy is an optional dependency ([math] extra); without it the numeric evaluator below does the work
    import sympy
    from sympy.parsing.sympy_parser import convert_xor, implicit_multiplication_application, parse_expr, standard_transformations
    HAVE_SYMPY = True
except Exception:  # noqa: BLE001
    sympy = None  # type: ignore[assignment]
    HAVE_SYMPY = False

MAX_EXPR_CHARS = 300
FUNCTIONS = {"sqrt", "sin", "cos", "tan", "asin", "acos", "atan", "sinh", "cosh", "tanh", "exp", "ln", "log", "abs", "sen", "tg", "arcsen",
             "arccos", "arctan", "cbrt", "log2", "log10", "floor", "ceil", "sign"}
CONSTANTS = {"pi", "e", "oo"}
SAMPLE_POINTS = [1.37, 2.11, 3.07, 1.83, 2.59, 0.61, 4.4]
_ALLOWED_CHARS = re.compile(r"^[0-9A-Za-z+\-*/^().,\s=<>!]*$")
_NAME = re.compile(r"[A-Za-z]+")


# ---------------------------------------------------------------------------------------------------- cleaning
_SUPERSCRIPT = {"²": "**2", "³": "**3", "⁴": "**4", "¹": "**1", "⁰": "**0"}


def _replace_braced(text: str, command: str, template: str, args: int) -> str:
    """Replace ``\\command{a}{b}`` (``args`` braced groups) by ``template``; innermost first, repeated until stable."""
    pattern = re.compile(r"\\" + command + r"\s*" + r"\{([^{}]*)\}" * args)
    for _ in range(12):
        new = pattern.sub(lambda m: template.format(*m.groups()), text)
        if new == text:
            break
        text = new
    return text


def latex_to_text(text: str) -> str:
    t = text
    t = re.sub(r"\\(?:left|right|big|Big|bigg|Bigg)\s*", "", t)
    t = re.sub(r"\\[,;:! ]|\\quad|\\qquad", " ", t)
    t = re.sub(r"\\text(?:bf|it|rm)?\{([^{}]*)\}", r" \1 ", t)
    t = re.sub(r"\\mathrm\{([^{}]*)\}", r"\1", t)
    for _ in range(12):
        before = t
        t = _replace_braced(t, "(?:d|t)?frac", "(({0})/({1}))", 2)
        t = re.sub(r"\\sqrt\s*\[([^\]]*)\]\s*\{([^{}]*)\}", r"((\2)**(1/(\1)))", t)
        t = _replace_braced(t, "sqrt", "sqrt({0})", 1)
        t = re.sub(r"\\sqrt\s*(\d+(?:\.\d+)?|[A-Za-z])(?![\w{])", r"sqrt(\1)", t)
        t = re.sub(r"(?<![\w\\])e\s*\^\s*\{([^{}]*)\}", r"exp(\1)", t)                 # e^{...} -> exp(...)
        t = re.sub(r"\^\{([^{}]*)\}", r"**(\1)", t)
        t = re.sub(r"_\{([^{}]*)\}", "", t)
        if t == before:
            break
    t = t.replace("\\cdot", "*").replace("\\times", "*").replace("\\div", "/").replace("\\pi", "pi").replace("\\infty", "oo")
    t = re.sub(r"\\displaystyle\b", "", t)
    # a function name takes a blank before it: ``2\\ln(x)`` must not become the unknown word ``2ln``
    t = re.sub(r"\\(ln|log|sin|cos|tan|exp|arcsin|arccos|arctan|sinh|cosh|tanh)\b", r" \1", t)
    t = t.replace("\\", "")
    return t


def clean_expression(raw: str) -> str:
    """Turn what a model wrote into a plain ``a*b**2/(c)`` string (not yet validated)."""
    t = (raw or "").strip()
    t = t.replace("$", "")
    t = t.replace("\\(", "").replace("\\)", "").replace("\\[", "").replace("\\]", "")
    t = latex_to_text(t)
    for sup, repl in _SUPERSCRIPT.items():
        t = t.replace(sup, repl)
    t = t.replace("√", "sqrt").replace("π", "pi").replace("×", "*").replace("·", "*").replace("⋅", "*").replace("−", "-").replace("–", "-")
    t = t.replace("÷", "/").replace("∞", "oo").replace("^", "**").replace("{", "(").replace("}", ")").replace("[", "(").replace("]", ")")
    t = re.sub(r"(?<=\d),(?=\d)", ".", t)   # decimal comma
    t = re.sub(r"(?<=\d)\s(?=\d{3}\b)", "", t)  # 1 000 -> 1000
    t = re.sub(r"(?<=[\d)])(?=(?:ln|log|sen|sin|cos|tan|tg|exp|sqrt)\s*\()", " ", t)    # ``2ln(x)`` -> ``2 ln(x)``
    t = re.sub(r"\bsen\b", "sin", t)
    t = re.sub(r"\barcsen\b", "asin", t)
    t = re.sub(r"\barc(sin|cos|tan)\b", r"a\1", t)
    t = re.sub(r"\btg\b", "tan", t)
    t = re.sub(r"\bln\b", "log", t)
    t = t.rstrip(" .;")
    return re.sub(r"\s+", " ", t).strip()


def is_safe(expr: str, constants: tuple[str, ...] = ()) -> bool:
    if not expr or len(expr) > MAX_EXPR_CHARS or not _ALLOWED_CHARS.match(expr):
        return False
    if re.search(r"\.(?!\d)|(?<!\d)\.", expr):
        return False
    for name in _NAME.findall(expr):
        if name in FUNCTIONS or name in CONSTANTS or name in constants:
            continue
        if len(name) <= 2:  # variables are one or two letters: ``xy`` is x times y
            continue
        return False
    return True


# ---------------------------------------------------------------------------------------------------- answer extraction
_BOXED = re.compile(r"\\boxed\s*\{")
_MARKER = re.compile(r"(?:respuesta(?:\s+final)?|resultado(?:\s+final)?|soluci[oó]n|final answer|answer|result)\s*[:：]?\s*", re.I)


def _plain(line: str) -> str:
    """A line of the answer without the Markdown emphasis around it."""
    return re.sub(r"^[\s*_`]+|[\s*_`]+$", "", line)


def extract_candidates(text: str) -> list[str]:
    """Where the final expression may be, most specific first."""
    text, _ = split_reasoning(text)
    out: list[str] = []
    boxed = [m for m in _BOXED.finditer(text)]
    if boxed:
        start = boxed[-1].end()
        depth, i = 1, start
        while i < len(text) and depth:
            depth += {"{": 1, "}": -1}.get(text[i], 0)
            i += 1
        out.append(text[start:i - 1])
    markers = list(_MARKER.finditer(text))
    if markers:
        tail = text[markers[-1].end():].strip().splitlines()
        if tail and _plain(tail[0]):
            out.append(_plain(tail[0]))
        elif len(tail) > 1:
            out.append(_plain(tail[1]))
    lines = [ln.strip() for ln in text.strip().splitlines() if ln.strip()]
    if lines:
        last = _MARKER.sub("", lines[-1], count=1) if _MARKER.match(_plain(lines[-1])) else lines[-1]
        out.append(_plain(last))
    return [c for c in out if c]


def strip_lhs(expr: str) -> str:
    """``f(x) = 2x`` / ``y = 3`` / ``x = 4`` -> the right-hand side (only when the left one is a bare symbol or function)."""
    if expr.count("=") == 1 and not re.search(r"[<>!]=|=[<>]", expr):
        left, right = expr.split("=")
        if re.fullmatch(r"\s*[A-Za-z]\w{0,2}\s*(?:'|\([A-Za-z]\w?\))?\s*", left) and right.strip():
            return right.strip()
    if expr.count("=") > 1:
        return expr.split("=")[-1].strip()
    return expr


def _tidy_candidate(raw: str, *, answer: bool = False) -> str:
    """One candidate as a bare expression: no prose after it, no left-hand side, no unit. ``answer=True`` (the model's line, not the expected
    value) keeps what follows the last ``=`` of the line, so ``f'(x) = 3x^2 + 1 = ...`` and ``y = 2x`` both give the final expression."""
    t = clean_expression(raw)
    prose = re.search(r"[A-Za-zÁ-ú]{3,}\s+[A-Za-zÁ-ú]{3,}", t)
    if prose and re.search(r"\d", t[: prose.start()]):
        t = t[: prose.start()]
    if answer and "=" in t and not re.search(r"[<>!]=|=[<>]|[<>]", t):
        t = t.rsplit("=", 1)[-1].strip() or t
    else:
        t = strip_lhs(t)
    t = re.sub(r"(?<=\d)\s*(?:cm|km|kg|mm|m|g|s|h|%|€|euros?)\s*$", "", t)
    return t.strip().rstrip(".")


# ---------------------------------------------------------------------------------------------------- numeric evaluator (no sympy)
class _Evaluator(ast.NodeVisitor):
    def __init__(self, env: dict[str, complex]):
        self.env = env

    def visit_Expression(self, node: ast.Expression) -> complex:
        return self.visit(node.body)

    def visit_Constant(self, node: ast.Constant) -> complex:
        if isinstance(node.value, (int, float)):
            return complex(node.value)
        raise ValueError("constant")

    def visit_Name(self, node: ast.Name) -> complex:
        if node.id in self.env:
            return self.env[node.id]
        raise ValueError(f"name {node.id}")

    def visit_UnaryOp(self, node: ast.UnaryOp) -> complex:
        value = self.visit(node.operand)
        if isinstance(node.op, ast.USub):
            return -value
        if isinstance(node.op, ast.UAdd):
            return value
        raise ValueError("unary")

    def visit_BinOp(self, node: ast.BinOp) -> complex:
        a, b = self.visit(node.left), self.visit(node.right)
        if isinstance(node.op, ast.Add):
            return a + b
        if isinstance(node.op, ast.Sub):
            return a - b
        if isinstance(node.op, ast.Mult):
            return a * b
        if isinstance(node.op, ast.Div):
            return a / b
        if isinstance(node.op, ast.Pow):
            if abs(b) > 1000 or abs(a) > 1e9:
                raise ValueError("too large")
            return a ** b
        raise ValueError("operator")

    def visit_Call(self, node: ast.Call) -> complex:
        if not isinstance(node.func, ast.Name) or node.keywords:
            raise ValueError("call")
        fn = _FUNCS.get(node.func.id)
        if fn is None:
            raise ValueError(f"function {node.func.id}")
        return fn(*[self.visit(a) for a in node.args])

    def generic_visit(self, node: ast.AST) -> complex:
        raise ValueError(type(node).__name__)


_FUNCS = {
    "sqrt": cmath.sqrt, "sin": cmath.sin, "cos": cmath.cos, "tan": cmath.tan, "asin": cmath.asin, "acos": cmath.acos, "atan": cmath.atan,
    "sinh": cmath.sinh, "cosh": cmath.cosh, "tanh": cmath.tanh, "exp": cmath.exp, "log": cmath.log, "abs": lambda z: complex(abs(z)),
    "cbrt": lambda z: z ** (1 / 3), "log2": lambda z: cmath.log(z, 2), "log10": cmath.log10, "floor": lambda z: complex(math.floor(z.real)),
    "ceil": lambda z: complex(math.ceil(z.real)), "sign": lambda z: complex((z.real > 0) - (z.real < 0)),
}


def _implicit(expr: str, constants: tuple[str, ...] = ()) -> str:
    """Insert the multiplication that humans omit: ``2x`` -> ``2*x``, ``xy`` -> ``x*y``, ``x(x+1)`` -> ``x*(x+1)``, ``(a)(b)`` -> ``(a)*(b)``."""
    known = FUNCTIONS | CONSTANTS | set(constants)

    def split(match: re.Match) -> str:
        name = match.group(0)
        return name if name in known or len(name) == 1 else "*".join(name)

    t = re.sub(r"[A-Za-z]+", split, expr)
    t = re.sub(r"(\d)\s*([A-Za-z(])", r"\1*\2", t)
    t = re.sub(r"\)\s*([A-Za-z0-9(])", r")*\1", t)
    t = re.sub(r"\b(?!(?:" + "|".join(sorted(known, key=len, reverse=True)) + r")\b)([A-Za-z])\s*\(", r"\1*(", t)
    t = re.sub(r"\b([A-Za-z])\s+(?=[A-Za-z0-9(])", r"\1*", t)
    return t


def free_names(expr: str, constants: tuple[str, ...]) -> list[str]:
    names: list[str] = []
    for name in _NAME.findall(_implicit(expr, constants)):
        if name in FUNCTIONS or name in CONSTANTS or name in constants or name in names:
            continue
        names.append(name)
    return names


def numeric_values(expr: str, names: list[str], constants: tuple[str, ...]) -> list[complex]:
    tree = ast.parse(_implicit(expr, constants), mode="eval")
    out: list[complex] = []
    for index in range(len(SAMPLE_POINTS)):
        env: dict[str, complex] = {"pi": complex(math.pi), "e": complex(math.e)}
        for c in constants:
            env[c] = 0j
        for offset, name in enumerate(names):
            env[name] = complex(SAMPLE_POINTS[(index + offset * 2) % len(SAMPLE_POINTS)])
        try:
            out.append(_Evaluator(env).visit(tree))
        except (ValueError, ZeroDivisionError, OverflowError, TypeError):
            out.append(complex("nan"))
    return out


def _numeric_equal(a: str, b: str, constants: tuple[str, ...]) -> Optional[bool]:
    names = sorted(set(free_names(a, constants)) | set(free_names(b, constants)))
    try:
        va, vb = numeric_values(a, names, constants), numeric_values(b, names, constants)
    except (SyntaxError, ValueError, RecursionError, MemoryError):
        return None
    compared = 0
    for x, y in zip(va, vb):
        if any(cmath.isnan(z) or cmath.isinf(z) for z in (x, y)):
            continue
        compared += 1
        if abs(x - y) > 1e-7 * max(1.0, abs(x), abs(y)):
            return False
    return True if compared >= 2 else None


# ---------------------------------------------------------------------------------------------------- sympy
def _sympy_namespace() -> dict[str, Any]:
    s = sympy
    return {"Integer": s.Integer, "Float": s.Float, "Rational": s.Rational, "Symbol": s.Symbol, "Function": s.Function, "sqrt": s.sqrt,
            "sin": s.sin, "cos": s.cos, "tan": s.tan, "asin": s.asin, "acos": s.acos, "atan": s.atan, "sinh": s.sinh, "cosh": s.cosh,
            "tanh": s.tanh, "exp": s.exp, "log": s.log, "Abs": s.Abs, "abs": s.Abs, "cbrt": s.cbrt, "floor": s.floor, "ceiling": s.ceiling,
            "ceil": s.ceiling, "sign": s.sign, "pi": s.pi, "E": s.E, "oo": s.oo, "log2": lambda z: s.log(z, 2), "log10": lambda z: s.log(z, 10)}


def _sympy_parse(expr: str, constants: tuple[str, ...]):
    local = {"e": sympy.E}
    for c in constants:
        local[c] = sympy.Integer(0)
    return parse_expr(expr, local_dict=local, global_dict=_sympy_namespace(), evaluate=True,
                      transformations=standard_transformations + (implicit_multiplication_application, convert_xor))


def _sympy_equal(a: str, b: str, constants: tuple[str, ...]) -> Optional[bool]:
    try:
        ea, eb = _sympy_parse(a, constants), _sympy_parse(b, constants)
    except Exception:  # noqa: BLE001 — anything the parser dislikes is "cannot decide here"
        return None
    try:
        diff = sympy.simplify(ea - eb)
        if diff == 0:
            return True
        if getattr(diff, "is_number", False):
            return bool(abs(complex(sympy.N(diff))) < 1e-9)
        free = sorted(diff.free_symbols, key=str)
        compared = 0
        for index in range(len(SAMPLE_POINTS)):
            subs = {sym: SAMPLE_POINTS[(index + off * 2) % len(SAMPLE_POINTS)] for off, sym in enumerate(free)}
            value = complex(sympy.N(diff.subs(subs)))
            if cmath.isnan(value) or cmath.isinf(value):
                continue
            compared += 1
            if abs(value) > 1e-7 * max(1.0, abs(complex(sympy.N(ea.subs(subs))))):
                return False
        return True if compared >= 2 else None
    except Exception:  # noqa: BLE001
        return None


def equivalent(answer: str, expected: str, constants: tuple[str, ...] = ()) -> tuple[bool, str]:
    """``(equal, method)``; ``method`` is ``sympy``, ``numeric`` or ``unparsed``."""
    # an expected value that is itself an equation (``x**2 + y**2 = 25``) is compared whole; otherwise the answer is read after its last "="
    a, b = _tidy_candidate(answer, answer="=" not in str(expected)), _tidy_candidate(expected)
    if not is_safe(a, constants) or not is_safe(b, constants):
        return False, "unparsed"
    if a.replace(" ", "") == b.replace(" ", ""):
        return True, "text"
    if HAVE_SYMPY:
        result = _sympy_equal(a, b, constants)
        if result is not None:
            return result, "sympy"
    result = _numeric_equal(a, b, constants)
    if result is not None:
        return result, "numeric"
    return False, "unparsed"


def _split_items(text: str) -> list[str]:
    t = clean_expression(text)
    t = re.sub(r"\b(?:y|o|and|or)\b", ",", t)
    t = t.replace(";", ",")
    items = []
    for part in re.split(r",(?![^()]*\))", t):
        part = strip_lhs(part.strip())
        if part:
            items.append(part)
    return items


def check_math(spec: dict[str, Any], output: ModelOutput, ctx: CheckContext) -> dict[str, Any]:
    if "expected" not in spec:
        return bad_spec("math_equiv needs `expected`")
    constants = tuple(spec.get("constants") or ())
    mode = spec.get("mode", "expr")
    candidates = extract_candidates(output.text)
    if not candidates:
        return verdict(0.0, False, {"reason": "empty answer"})
    if mode == "set":
        expected = [str(x) for x in (spec["expected"] if isinstance(spec["expected"], list) else [spec["expected"]])]
        best: Optional[dict[str, Any]] = None
        for cand in candidates:
            items = _split_items(cand)
            unmatched = list(items)
            hits = 0
            for want in expected:
                for item in list(unmatched):
                    if equivalent(item, want, constants)[0]:
                        unmatched.remove(item)
                        hits += 1
                        break
            score = hits / len(expected) * (1.0 if not unmatched else 0.5)
            current = {"score": score, "passed": hits == len(expected) and not unmatched, "items": items, "extra": unmatched}
            if best is None or current["score"] > best["score"]:
                best = current
        assert best is not None
        return verdict(best["score"], best["passed"], {"expected": expected, "items": best["items"], "extra": best["extra"]})
    expected_text = str(spec["expected"])
    last: dict[str, Any] = {}
    for cand in candidates:
        ok, method = equivalent(cand, expected_text, constants)
        last = {"expected": expected_text, "got": _tidy_candidate(cand, answer="=" not in expected_text), "method": method, "sympy": HAVE_SYMPY}
        if ok:
            return verdict(1.0, True, last)
    return verdict(0.0, False, last)
