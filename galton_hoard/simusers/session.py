"""Running a conversation: the persona opens, the target answers, the simulator replies, until it says it is done or the turns run out.

``target(history) -> str`` receives the conversation so far as ``[{"role": "user"|"assistant", "content": ...}]`` and returns the
assistant's next message. ``simulator(system, history) -> str`` receives the persona prompt and the same history seen from the
other side (the persona's own messages are ``assistant``), and returns the next user message. Both may raise ``RuntimeError``: the
case then ends with that error and is scored as not reached.
"""

from __future__ import annotations

from typing import Callable, Iterable, Optional

from .cases import END_MARKER, Case, persona_prompt
from .scoring import CaseResult, score_case

Target = Callable[[list[dict[str, str]]], str]
Simulator = Callable[[str, list[dict[str, str]]], str]


def _flipped(history: list[dict[str, str]]) -> list[dict[str, str]]:
    return [{"role": "assistant" if m["role"] == "user" else "user", "content": m["content"]} for m in history]


def run_case(case: Case, target: Target, simulator: Simulator) -> CaseResult:
    history: list[dict[str, str]] = [{"role": "user", "content": case.opening}]
    system = persona_prompt(case)
    ended = "max_turns"
    error = ""
    while True:
        try:
            reply = target(list(history))
        except RuntimeError as exc:
            ended, error = "target_error", str(exc)
            break
        history.append({"role": "assistant", "content": reply})
        if sum(1 for m in history if m["role"] == "user") >= case.max_turns:
            break
        try:
            nxt = (simulator(system, _flipped(history)) or "").strip()
        except RuntimeError as exc:
            ended, error = "simulator_error", str(exc)
            break
        if not nxt or nxt.startswith(END_MARKER):
            ended = "simulator_done"
            break
        history.append({"role": "user", "content": nxt})
    return score_case(case, history, ended=ended, error=error)


def run_family(cases: Iterable[Case], target_factory: Callable[[], Target], simulator: Simulator,
               on_result: Optional[Callable[[CaseResult], None]] = None) -> list[CaseResult]:
    """Every case gets a fresh target (a fresh chat), so one conversation never leaks into the next."""
    out = []
    for case in cases:
        res = run_case(case, target_factory(), simulator)
        out.append(res)
        if on_result:
            on_result(res)
    return out
