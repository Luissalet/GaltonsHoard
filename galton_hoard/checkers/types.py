"""Shared types of the checkers: what a model produced, what a checker may need, and the shape of a verdict."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional


@dataclass
class ModelOutput:
    """What a model answered: the visible text, its reasoning (if the server separated it) and the tool calls it made."""

    text: str = ""
    reasoning: str = ""
    tool_calls: list[dict[str, Any]] = field(default_factory=list)  # [{"name": str, "arguments": dict}]


@dataclass
class CheckContext:
    """Services a checker may use. Everything is optional: a checker that needs one that is missing reports ``unavailable``."""

    case: dict[str, Any] = field(default_factory=dict)
    allow_code: bool = True
    code_timeout_s: float = 10.0
    judge: Optional[Callable[[dict[str, Any]], dict[str, Any]]] = None
    family_call: Optional[Callable[..., dict[str, Any]]] = None
    contestant_id: str = ""


def verdict(score: float, passed: bool, detail: Optional[dict[str, Any]] = None, *, unavailable: bool = False) -> dict[str, Any]:
    """The common result of every checker: ``{score, passed, detail, unavailable?}`` with the score clamped to 0..1."""
    out: dict[str, Any] = {"score": round(max(0.0, min(1.0, float(score))), 4), "passed": bool(passed), "detail": detail or {}}
    if unavailable:
        out["unavailable"] = True
    return out


def unavailable(reason: str, **detail: Any) -> dict[str, Any]:
    return verdict(0.0, False, {"reason": reason, **detail}, unavailable=True)


def bad_spec(message: str) -> dict[str, Any]:
    """A malformed checker spec is a bug in the case, not a wrong answer: it is reported, never scored as a pass."""
    return verdict(0.0, False, {"error": f"invalid checker: {message}"})
