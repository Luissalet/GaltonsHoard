"""Generators for the cases that are built when they run (long-context documents, vision images)."""

from __future__ import annotations

from typing import Any

from . import haystack, vision


def materialize(prompt: dict[str, Any]) -> dict[str, Any]:
    """Resolve a generated prompt into ``{"system", "text", "images": [png bytes]}``. A plain prompt is returned as it is."""
    spec = prompt.get("generate")
    if not spec:
        return {"system": prompt.get("system"), "text": prompt.get("text", ""), "images": []}
    kind = spec.get("kind")
    if kind == "haystack":
        return {"system": prompt.get("system"), "text": haystack.build(spec)["text"], "images": []}
    if kind == "vision":
        return {"system": prompt.get("system"), "text": prompt.get("text") or vision.plan(spec)["question"], "images": [vision.render(spec)]}
    raise ValueError(f"unknown generator {kind!r}")


def suites() -> list[dict[str, Any]]:
    """The generated builtin suites."""
    return [haystack.suite_def(), vision.suite_def()]


__all__ = ["haystack", "materialize", "suites", "vision"]
