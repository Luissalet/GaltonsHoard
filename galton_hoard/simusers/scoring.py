"""Scoring of one conversation: goal reached, turns used, clarifying question asked."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from .cases import Case

#: Weights of the composite score. The three parts are always reported on their own as well.
W_GOAL, W_CLARIFY, W_TURNS = 0.6, 0.25, 0.15


@dataclass
class CaseResult:
    case_id: str
    persona: str
    ambiguous: bool
    goal_reached: bool
    turns: int                       # user messages sent, the opening one included
    asked_clarifying: bool           # the first assistant reply asked a question
    clarify_score: float
    turn_score: float
    score: float
    final_answer: str = ""
    ended: str = ""                  # simulator_done | max_turns | target_error | simulator_error
    error: str = ""
    transcript: list[dict[str, str]] = field(default_factory=list)
    check: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def asked_question(text: str) -> bool:
    """A reply asks something when it contains a question mark outside code and quotes of the user's own words."""
    return "?" in (text or "") or "¿" in (text or "")


def score_case(case: Case, transcript: list[dict[str, str]], *, ended: str, error: str = "") -> CaseResult:
    replies = [m["content"] for m in transcript if m["role"] == "assistant"]
    turns = sum(1 for m in transcript if m["role"] == "user")
    final = replies[-1] if replies else ""
    check = case.check(final) if replies else {"score": 0.0, "passed": False, "detail": {"reason": "no assistant reply"}}
    reached = bool(check.get("passed")) and not error
    asked = asked_question(replies[0]) if replies else False
    # An ambiguous request should be clarified; a clear one should be answered without a detour (asking costs half).
    clarify = (1.0 if asked else 0.0) if case.ambiguous else (0.5 if asked else 1.0)
    span = max(1, case.max_turns - 1)
    turn_score = max(0.0, 1.0 - (turns - 1) / span) if reached else 0.0
    score = W_GOAL * (1.0 if reached else 0.0) + W_CLARIFY * clarify + W_TURNS * turn_score
    if error:                                    # a conversation that broke measured nothing about the assistant
        clarify, score = 0.0, 0.0
    return CaseResult(case.id, case.persona, case.ambiguous, reached, turns, asked, clarify, round(turn_score, 4), round(score, 4),
                      final_answer=final, ended=ended, error=error, transcript=transcript, check=check)


def summarise(results: list[CaseResult]) -> dict[str, Any]:
    """Aggregate numbers, overall and per persona."""
    def block(rows: list[CaseResult]) -> dict[str, Any]:
        n = len(rows)
        amb = [r for r in rows if r.ambiguous]
        return {
            "cases": n,
            "goal_rate": round(sum(r.goal_reached for r in rows) / n, 4) if n else 0.0,
            "mean_turns": round(sum(r.turns for r in rows) / n, 2) if n else 0.0,
            "clarify_rate_ambiguous": round(sum(r.asked_clarifying for r in amb) / len(amb), 4) if amb else None,
            "needless_questions": sum(1 for r in rows if not r.ambiguous and r.asked_clarifying),
            "mean_score": round(sum(r.score for r in rows) / n, 4) if n else 0.0,
        }
    out = {"overall": block(results), "by_persona": {}}
    for persona in sorted({r.persona for r in results}):
        out["by_persona"][persona] = block([r for r in results if r.persona == persona])
    return out
