"""The judge: a contestant chosen in settings (``judge.contestant``) that grades answers against a rubric.

* The judge never grades itself silently: when it is the model being judged the result is marked ``self_judged`` with half the confidence.
* Grades are cached by a hash of what the judge saw, so re-checking a stored answer costs nothing.
* When the judge cannot be reached (a GGUF that is not loaded, a server that is down) the result waits as ``judge_pending`` and is graded
  later; it never counts as a failure and never counts at all until it is graded.
"""

from __future__ import annotations

import logging
from typing import Any, Callable

from .backends import Backend, ChatRequest
from .checkers.judge import build_judge_messages, parse_judge_reply
from .util import stable_hash

log = logging.getLogger("galton.judge")


def has_judge(spec: dict[str, Any]) -> bool:
    """Does this checker (or one inside a combinator) need the judge?"""
    if not isinstance(spec, dict):
        return False
    if spec.get("type") == "judge":
        return True
    return any(has_judge(c) for c in spec.get("checks", []) or [])


def make_ask(store: Any, backend: Backend, judge: dict[str, Any], tested_id: str, *, cancel: Callable[[], bool] = lambda: False) -> Callable[[dict[str, Any]], dict[str, Any]]:
    """The ``ctx.judge`` callable for one judge backend and the contestant being judged."""
    self_judged = judge["id"] == tested_id

    def ask(request: dict[str, Any]) -> dict[str, Any]:
        key = stable_hash(judge["id"], judge.get("digest", ""), request.get("prompt", ""), request.get("rubric", ""), request.get("answer", ""), request.get("reference", ""))
        cached = store.judge_cached(key)
        if cached and "score" in cached:
            return {**cached, "cached": True, "self_judged": self_judged}
        completion = backend.chat(ChatRequest(messages=build_judge_messages(request), temperature=0.0, max_tokens=500, effort="off", timeout_s=120.0), cancel)
        if completion.error:
            return {"error": completion.error}
        reply = parse_judge_reply(completion.text)
        if reply.get("error"):
            return reply
        reply["judge"] = judge["name"]
        store.judge_store(key, {k: v for k, v in reply.items() if k != "self_judged"})
        return {**reply, "self_judged": self_judged}

    return ask


def pending_reply(_: dict[str, Any]) -> dict[str, Any]:
    """Used while the judge is not live: marks the grade as owed instead of failing it."""
    return {"error": "judge not live", "pending": True}
