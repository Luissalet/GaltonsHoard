"""One trial: a model, a harness and a task, run through a minimal agent loop with tools.

``chat(messages, tools)`` is any callable that returns a ``backends.Completion`` (the real one wraps a session, the tests use fake models). The
loop asks, lets the harness read the reply, refuses what is malformed (a format failure the model is told about), runs what is valid against the
virtual world, gives the results back and repeats until a final answer, the turn limit, too many format failures, or an infrastructure error.

The outcome of a trial is one of:

* ``pass``: the verifier accepted the final answer and the state of the files.
* ``fail_answer``: the harness could read everything the model wrote but the result is wrong (or there was no final answer in time).
* ``fail_format``: the trial failed and the cause was format: the last thing the model produced was a call the harness could not read or refused,
  or it kept producing them until the limit, or the server refused the tools field.
* ``none``: the verifier did not get to run because the server or the network failed (timeout, connection, server error). It is not a zero:
  it is left out of every rate and reported apart.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Optional

from ..backends import Completion
from .harnesses import KEEP_OBSERVATIONS, Call, Harness, Rejected, trim_history
from .tasks import is_solved, new_world
from .world import TOOL_SPECS, World

Chat = Callable[[list[dict[str, Any]], list[dict[str, Any]]], Completion]

OUTCOMES = ("pass", "fail_answer", "fail_format", "none")
EMPTY_REPLY = "your reply was empty: write either the tool call in the required format or the final answer"
TEXT_CAP = 2000
RESULT_CAP = 600


@dataclass
class Trial:
    task_id: str
    harness: str
    sample: int = 0
    outcome: str = "none"
    passed: Optional[bool] = None            # None: the verifier did not run
    reason: str = ""                         # short machine key: wrong_answer, turn_limit, format_limit, truncated, tools_unsupported, server_error...
    detail: str = ""                         # what the verifier or the infrastructure said
    turns: Optional[int] = 0                 # None for a harness that does not expose them (command-line agents)
    calls: Optional[int] = 0
    format_failures: Optional[int] = 0
    extra_calls: Optional[int] = 0
    extra_repeat: Optional[int] = 0
    extra_after_solved: Optional[int] = 0
    tokens_in: Optional[int] = 0
    tokens_out: Optional[int] = 0
    latency_ms: float = 0.0
    final: str = ""
    trace: list[dict[str, Any]] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4) if text else 0


def tools_refused(error: str) -> bool:
    """The server answered that it does not take the ``tools`` field."""
    low = str(error).lower()
    return "tool" in low and any(f"http {code}" in low for code in (400, 422, 501))


def _short(text: str, cap: int) -> str:
    return text if len(text) <= cap else text[:cap] + f"… [{len(text) - cap} more characters]"


def run_trial(task: dict[str, Any], harness: Harness, chat: Chat, *, api: str = "openai", max_turns: int = 8, max_format_errors: int = 3, sample: int = 0) -> Trial:
    specs = [TOOL_SPECS[name] for name in task["tools"]]
    payload = harness.tools_payload(specs)
    world = new_world(task)
    entries: list[dict[str, Any]] = [{"kind": "system", "msg": {"role": "system", "content": harness.system_prompt(specs)}},
                                     {"kind": "user", "msg": {"role": "user", "content": task["prompt"]}}]
    t = Trial(task_id=task["id"], harness=harness.id, sample=sample)
    observations: list[str] = []
    done_ok: set[tuple[str, str]] = set()
    solved = is_solved(task, world, observations)
    keep = KEEP_OBSERVATIONS if harness.compact == "trim" else 0
    final: Optional[str] = None
    infra = ""
    stop = ""                    # why the loop ended without a final answer: format_limit, truncated, tools_unsupported
    last_was_format = False

    for turn_no in range(1, max_turns + 1):
        sent = trim_history(entries, keep)
        comp = chat(sent, payload)
        t.turns = turn_no
        t.latency_ms += comp.latency_ms or 0.0
        t.tokens_in += comp.prompt_tokens if comp.prompt_tokens is not None else sum(estimate_tokens(str(m.get("content") or "")) for m in sent)
        t.tokens_out += comp.completion_tokens if comp.completion_tokens is not None else estimate_tokens(comp.text)
        record: dict[str, Any] = {"turn": turn_no, "text": _short(comp.text, TEXT_CAP), "finish": comp.finish_reason, "calls": [], "rejected": [], "format_error": ""}
        t.trace.append(record)
        if comp.error:
            record["error"] = str(comp.error)[:300]
            if harness.native and tools_refused(comp.error):
                t.format_failures += 1
                stop, last_was_format = "tools_unsupported", True
                t.detail = str(comp.error)[:300]
            else:
                infra = str(comp.error)
            break
        if comp.truncated:
            stop = "truncated"
            break
        parsed = harness.parse(comp)
        entries.append({"kind": "assistant", "msg": harness.assistant_message(comp, parsed, api)})
        if parsed.format_error:
            t.format_failures += 1
            last_was_format = True
            record["format_error"] = parsed.format_error
            entries.append({"kind": "user", "msg": harness.format_error_message(parsed.format_error)})
            if t.format_failures >= max_format_errors:
                stop = "format_limit"
                break
            continue
        if not parsed.calls and not parsed.rejected:
            if not (parsed.final or "").strip() and not (comp.text or "").strip():
                # Nothing visible came back (some servers end the turn on a special token, or strip it): the model
                # tried to act but the harness got no call and no answer. That is a format problem, not a wrong answer.
                t.format_failures += 1
                last_was_format = True
                record["format_error"] = "empty reply"
                thought = str(getattr(comp, "reasoning", "") or "")
                if thought:
                    record["reasoning"] = _short(thought, TEXT_CAP)
                entries.append({"kind": "user", "msg": harness.format_error_message(EMPTY_REPLY)})
                if t.format_failures >= max_format_errors:
                    stop = "format_limit"
                    break
                continue
            final = parsed.final or ""
            last_was_format = False
            break

        # -- the calls: refuse what is malformed, run the rest in order ------------------------------------------------
        results: list[tuple[Any, str]] = []
        rejected = list(parsed.rejected)
        for call in parsed.calls:
            spec = TOOL_SPECS.get(call.name)
            if call.name not in task["tools"] or spec is None:
                rejected.append(Rejected(f"unknown tool '{call.name}'. Available tools: {', '.join(task['tools'])}", call.name, call.id))
                continue
            clean, why = World.check_args(spec, call.args)
            if clean is None:
                rejected.append(Rejected(why, call.name, call.id))
                continue
            key = (call.name, json.dumps(clean, sort_keys=True, ensure_ascii=False, default=str))
            extra = "after_solved" if solved else ("repeat" if key in done_ok else "")
            outcome = world.run(call.name, clean)
            t.calls += 1
            if outcome.ok:
                done_ok.add(key)
                observations.append(outcome.text)
            if extra:
                t.extra_calls += 1
                if extra == "repeat":
                    t.extra_repeat += 1
                else:
                    t.extra_after_solved += 1
            solved = is_solved(task, world, observations)
            record["calls"].append({"name": call.name, "args": clean, "ok": outcome.ok, "result": _short(outcome.text, RESULT_CAP), "extra": extra})
            results.append((Call(call.name, clean, call.id), outcome.text))
        for item in rejected:
            t.format_failures += 1
            record["rejected"].append({"name": item.name, "reason": item.reason})
            results.append((item, f"Error: call refused, {item.reason}."))
        last_was_format = bool(rejected) and not record["calls"]
        for full, short in zip(harness.observation(results, api), harness.observation(results, api, omit=True)):
            entries.append({"kind": "obs", "msg": full, "short": short})
        if t.format_failures >= max_format_errors:
            stop = "format_limit"
            break

    t.final = _short(final or "", TEXT_CAP)
    if infra:
        t.outcome, t.passed, t.reason, t.detail = "none", None, "server_error", infra[:300]
        return t
    ok, why = task["verify"](final or "", world)
    if ok and stop != "tools_unsupported":
        t.outcome, t.passed, t.reason = "pass", True, ""
        return t
    t.passed = False
    t.detail = (t.detail or why)[:300]
    if stop == "tools_unsupported" or stop == "format_limit" or (final is None and last_was_format):
        t.outcome, t.reason = "fail_format", stop or "format_at_turn_limit"
    elif stop == "truncated":
        t.outcome, t.reason = "fail_answer", "truncated"
    elif final is None:
        t.outcome, t.reason = "fail_answer", "turn_limit"
    else:
        t.outcome, t.reason = "fail_answer", "wrong_answer"
    return t
