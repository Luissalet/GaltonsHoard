"""One error type for every expected failure, so the API, the agent tools and the UI report it the same way."""

from __future__ import annotations

from typing import Any

from .hoard_link.agentkit import AppError
from .messages import ERRORS, KEY, CodedText, fields_of, fill, scalar


class GaltonError(AppError):
    """An expected, explainable failure: a stable ``code``, a human ``message`` and an actionable ``hint``.

    An :class:`~hoard_link.agentkit.AppError`: the shared error handlers and the agent router answer it with its own status and body (this app's
    handler adds the message parts the bundled interface translates)."""

    STATUS = {
        "not_found": 404,
        "confirm_required": 400,
        "invalid": 400,
        "forbidden": 403,
        "builtin_readonly": 403,
        "no_gpu": 409,
        "gpu_not_allowed": 403,
        "busy": 409,
        "unavailable": 503,
        "remote_not_allowed": 403,
        "unsupported": 415,
        "too_large": 413,
        "conflict": 409,
    }

    def __init__(self, code: str, message: str, hint: str = "", *, status: int | None = None, **details: Any):
        """``GaltonError("invalid", "no_suite", ref=x)`` takes the message and hint of the catalogue entry ``no_suite`` (``messages.ERRORS``) and
        fills them with the keyword arguments; the UI translates them from the ``key`` and ``params``. ``GaltonError(code, "free text", hint)`` is
        for a message that cannot be known in advance (a text the system gave us); it has no key."""
        self.key = ""
        if KEY.match(message) and message in ERRORS:
            self.key = message
            template, hint_template = ERRORS[message]
            used = fields_of(template) | fields_of(hint_template)
            self.params = {k: scalar(v) for k, v in details.items() if k in used}
            details = {k: v for k, v in details.items() if k not in used}        # what is left travels with the error as it is (lists, ids, a log path)
            message, hint = fill(template, self.params), fill(hint_template, self.params)
        elif KEY.match(message) and " " not in message and "_" in message:
            raise KeyError(f"{message!r} is not in messages.ERRORS")
        else:
            self.params = {}
        super().__init__(code, message, hint=hint, status=status, details=details)

    def to_dict(self) -> dict[str, Any]:
        body: dict[str, Any] = {**self.details, "error": self.message, "code": self.code}
        if self.hint:
            body["hint"] = self.hint
        if self.key:
            body["key"] = self.key
            body["params"] = self.params
        return body

    def coded(self) -> CodedText:
        """The message as a ``CodedText``: a ``str`` that keeps its key and parameters, for plan lines and warnings the UI translates."""
        return CodedText(self.message, self.key, self.params)

    def item(self) -> dict[str, Any]:
        """The wire form kept in stored rows and plan lines: ``{code, key, params, text, hint}``."""
        return {"code": self.code, "key": self.key, "params": self.params, "text": self.message, "hint": self.hint}
