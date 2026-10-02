"""Builtin suites: the JSON files in this folder plus the generated ones (long context, vision), validated, hashed and synced into the database.

A builtin suite is read-only for the user; they duplicate it to edit. ``sync`` is idempotent: it only touches a suite whose content hash changed.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

from ..checkers import COMBINATORS, CHECKERS
from ..errors import GaltonError
from ..generators import materialize as materialize_generated, suites as generated_suites
from ..messages import text
from ..util import stable_hash

CATEGORIES = ("writing_es", "reasoning", "math", "code", "extraction", "tool_use", "instruction", "long_context", "rag", "vision", "translation", "summary", "custom")
FOLDER = Path(__file__).parent


def suite_id(slug: str) -> str:
    return slug if slug.startswith("s_") else f"s_{slug}"


def case_id(suite: str, title: str, salt: str = "") -> str:
    return "k_" + stable_hash(suite, title, salt)[:14]


# ------------------------------------------------------------------------------------------------- validation
def validate_checker(spec: Any, path: str = "checker") -> list[str]:  # the items are ``CodedText``: strings that keep their key for the UI
    """Problems with a checker spec (empty when it is usable). Combinators are checked recursively."""
    if not isinstance(spec, dict):
        return [text("prob_checker_object", path=path)]
    kind = spec.get("type")
    if kind in COMBINATORS:
        checks = spec.get("checks")
        if not isinstance(checks, list) or not checks:
            return [text("prob_checker_checks", path=path, kind=kind)]
        out: list[str] = []
        for i, sub in enumerate(checks):
            out += validate_checker(sub, f"{path}.checks[{i}]")
        return out
    if kind not in CHECKERS:
        return [text("prob_checker_unknown", path=path, kind=kind, options=[*CHECKERS, *COMBINATORS])]
    return []


def validate_case(case: dict[str, Any]) -> list[str]:
    """Problems with a case dict (title, prompt, checker, weight, tools, images)."""
    problems: list[str] = []
    prompt = case.get("prompt")
    if isinstance(prompt, str):
        has_prompt = bool(prompt.strip())
    elif isinstance(prompt, dict):
        has_prompt = bool((prompt.get("text") or "").strip() or prompt.get("messages") or prompt.get("generate"))
        messages = prompt.get("messages")
        if messages is not None and (not isinstance(messages, list) or not all(isinstance(m, dict) and m.get("role") in ("system", "user", "assistant") for m in messages)):
            problems.append(text("prob_messages"))
    else:
        has_prompt = False
    if not has_prompt:
        problems.append(text("prob_prompt_empty"))
    problems += validate_checker(case.get("checker"))
    try:
        if float(case.get("weight", 1.0)) <= 0:
            problems.append(text("prob_weight_positive"))
    except (TypeError, ValueError):
        problems.append(text("prob_weight_number"))
    tools = case.get("tools") or []
    if not isinstance(tools, list) or not all(isinstance(t, dict) and isinstance(t.get("function"), dict) and t["function"].get("name") for t in tools):
        problems.append(text("prob_tools"))
    return problems


def normalise_prompt(prompt: Any) -> dict[str, Any]:
    """The stored shape of a prompt: ``{"text", "system"?, "messages"?, "generate"?}``."""
    if isinstance(prompt, str):
        return {"text": prompt}
    return {k: v for k, v in dict(prompt or {}).items() if v not in (None, "")}


def normalise_case(raw: dict[str, Any]) -> dict[str, Any]:
    """Accept the loose shape an assistant or an import file gives (``prompt`` as text) and return the stored shape."""
    case = {
        "title": str(raw.get("title") or "").strip(),
        "prompt": normalise_prompt(raw.get("prompt")),
        "checker": raw.get("checker") or {},
        "weight": float(raw.get("weight", 1.0) or 1.0),
        "tags": [str(t) for t in (raw.get("tags") or [])],
        "tools": list(raw.get("tools") or []),
        "images": list(raw.get("images") or []),
        "reference": str(raw.get("reference") or ""),
        "notes": str(raw.get("notes") or ""),
    }
    for key in ("max_tokens", "min_context"):
        if raw.get(key):
            case[key] = int(raw[key])
    if not case["title"]:
        text = case["prompt"].get("text") or ""
        case["title"] = (text.strip().splitlines() or [""])[0][:70] or "Caso sin título"
    return case


# ------------------------------------------------------------------------------------------------- loading
def _read_json_suites() -> list[dict[str, Any]]:
    out = []
    for path in sorted(FOLDER.glob("*.json")):
        out.append(json.loads(path.read_text(encoding="utf-8")))
    return out


def _finish_case(suite: str, position: int, raw: dict[str, Any], salt: str = "") -> dict[str, Any]:
    case = normalise_case(raw)
    case.update({"id": case_id(suite, case["title"], salt), "suite_id": suite, "position": position, "source": "builtin"})
    return case


def load_definitions() -> list[dict[str, Any]]:
    """Every builtin suite fully resolved: ``{id, name, ..., cases: [...]}`` with stable ids. ``rapida`` copies its cases from the others."""
    raw_suites = [*_read_json_suites(), *generated_suites()]
    by_slug = {s["id"]: s for s in raw_suites}
    defs: list[dict[str, Any]] = []
    for raw in sorted(raw_suites, key=lambda s: s.get("position", 99)):
        sid = suite_id(raw["id"])
        if "picks" in raw:
            cases = []
            for pick in raw["picks"]:
                source = by_slug.get(pick["suite"])
                found = next((c for c in (source or {}).get("cases", []) if c["title"] == pick["title"]), None)
                if found is None:
                    raise GaltonError("invalid", "rapida_pick_missing", suite=pick["suite"], title=pick["title"])
                cases.append(_finish_case(sid, len(cases), found, salt=pick["suite"]))
        else:
            cases = [_finish_case(sid, i, c) for i, c in enumerate(raw["cases"])]
        suite = {k: raw[k] for k in ("name", "description", "category", "version", "max_tokens", "position") if k in raw}
        suite.update({"id": sid, "builtin": True, "cases": cases})
        suite["content_hash"] = stable_hash({k: v for k, v in suite.items() if k != "cases"}, cases)
        defs.append(suite)
    return defs


def sync(store: Any) -> dict[str, int]:
    """Create or update the builtin suites in the database. Idempotent: an unchanged suite is not touched."""
    counts = {"created": 0, "updated": 0, "unchanged": 0, "removed": 0}
    defs = load_definitions()
    for d in defs:
        existing = store.find_suite(d["id"])
        fields = {k: d[k] for k in ("name", "description", "category", "version", "max_tokens", "position", "content_hash") if k in d}
        if existing is None:
            store.create_suite(id=d["id"], builtin=True, **fields)
            counts["created"] += 1
        elif existing.get("content_hash") == d["content_hash"] and existing.get("builtin"):
            counts["unchanged"] += 1
            continue
        else:
            store.update_suite(d["id"], builtin=True, **fields)
            counts["updated"] += 1
        wanted = {c["id"]: c for c in d["cases"]}
        for old in store.cases(d["id"]):
            if old["id"] not in wanted:
                store.delete_case(old["id"])
                counts["removed"] += 1
        for case in d["cases"]:
            body = {k: v for k, v in case.items() if k != "id"}
            if store.find_case(case["id"]) is None:
                store.create_case(id=case["id"], **body)
            else:
                store.update_case(case["id"], **body)
    known = {d["id"] for d in defs}
    for suite in store.suites():
        if suite["builtin"] and suite["id"] not in known:
            store.delete_suite(suite["id"])
            counts["removed"] += 1
    return counts


# ------------------------------------------------------------------------------------------------- running a case
def materialize_case(case: dict[str, Any], images_dir: Optional[Path] = None) -> dict[str, Any]:
    """What is sent to a model for ``case``: ``{"system", "messages", "text", "images": [bytes]}``.

    ``messages`` holds the chat turns without the images (a backend attaches ``images`` to the last user turn in its own format).
    """
    prompt = case.get("prompt") or {}
    built = materialize_generated(prompt)
    images: list[bytes] = list(built["images"])
    for name in case.get("images") or []:
        path = (images_dir / name) if images_dir else None
        if path is not None and path.is_file():
            images.append(path.read_bytes())
    messages = [dict(m) for m in prompt.get("messages") or []]
    system = built.get("system") or prompt.get("system")
    if not messages:
        messages = [{"role": "user", "content": built["text"]}]
    if system and messages[0].get("role") != "system":
        messages = [{"role": "system", "content": system}, *messages]
    return {"system": system, "messages": messages, "text": built["text"], "images": images}


def needs_vision(case: dict[str, Any]) -> bool:
    prompt = case.get("prompt") or {}
    return bool(case.get("images")) or (prompt.get("generate") or {}).get("kind") == "vision"


def rough_tokens(case: dict[str, Any]) -> int:
    """Context a case needs, without building the prompt when the case says it."""
    if case.get("min_context"):
        return int(case["min_context"])
    prompt = case.get("prompt") or {}
    text = (prompt.get("text") or "") + (prompt.get("system") or "") + json.dumps(prompt.get("messages") or [], ensure_ascii=False)
    return int(len(text) / 3.5) + 200 + int(case.get("max_tokens") or 0)
