"""Small pure helpers: identifiers, text folding, time formatting and a few numeric utilities."""

from __future__ import annotations

import hashlib
import json
import math
import re
from datetime import datetime
from typing import Any, Iterable, Optional

from .hoard_link.ids import new_ulid
from .hoard_link.text import fold, slugify as _slugify  # noqa: F401 - ``fold`` is re-exported: the checkers import it from here


def new_id(prefix: str, now: Optional[float] = None) -> str:
    """``<prefix>_<ulid>`` in lowercase: a shared ULID, so ids sort by creation (strictly, inside this process) and never collide. ``now`` pins the
    time (imports, tests). Ids made by older versions (``<prefix>_<12 characters>``) keep working: they are only ever compared as text."""
    return f"{prefix}_{new_ulid(now).lower()}"


_BLOB_STEM = re.compile(r"^sha256[-:][0-9a-f]{16,}$", re.I)
_PATHLIKE = re.compile(r"^(?:[A-Za-z]:[\\/]|[\\/]|~|\.{1,2}[\\/])|\\|/.*\.gguf$", re.I)


def usable_alias(name: Any) -> bool:
    """Can a server or an app report this as the name of a model? Not a file path (those have their own fields) and not the digest name of an
    Ollama blob (``sha256-<hex>``); an Ollama tag such as ``hf.co/org/model:q4`` is a name."""
    value = str(name or "").strip()
    if not value or _PATHLIKE.search(value):
        return False
    return not _BLOB_STEM.match(re.sub(r"\.gguf$", "", value, flags=re.I))


def clean_aliases(*groups: Iterable[Any]) -> list[str]:
    """The usable names of every group, in order, each once (accent- and case-insensitive)."""
    seen: dict[str, str] = {}
    for group in groups:
        for alias in group or []:
            alias = str(alias or "").strip()
            if usable_alias(alias) and fold(alias) not in seen:
                seen[fold(alias)] = alias
    return list(seen.values())


def squash(text: str) -> str:
    """Collapse every run of whitespace (non-breaking spaces included) into one space and trim."""
    return re.sub(r"\s+", " ", (text or "").replace(" ", " ")).strip()


def clamp_text(text: str, limit: int) -> str:
    text = squash(text)
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def slugify(text: str) -> str:
    """ASCII slug (the shared one, uncut); ``x`` when nothing is left."""
    return _slugify(text, max_len=0, fallback="x")


def stable_hash(*parts: Any, length: int = 32) -> str:
    """SHA-256 of the JSON form of ``parts`` (stable key order), shortened."""
    blob = json.dumps(parts, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:length]


def iso_local(ts: float) -> str:
    """ISO 8601 with the local UTC offset, seconds precision: ``2026-10-02T10:00:00+02:00``."""
    return datetime.fromtimestamp(ts).astimezone().isoformat(timespec="seconds")


def median(values: Iterable[float]) -> Optional[float]:
    data = sorted(v for v in values if v is not None and not math.isnan(v))
    if not data:
        return None
    mid = len(data) // 2
    return data[mid] if len(data) % 2 else (data[mid - 1] + data[mid]) / 2


def percentile(values: Iterable[float], q: float) -> Optional[float]:
    """Linear-interpolated percentile (q in 0..100)."""
    data = sorted(v for v in values if v is not None and not math.isnan(v))
    if not data:
        return None
    if len(data) == 1:
        return data[0]
    pos = (len(data) - 1) * q / 100.0
    lo = int(math.floor(pos))
    hi = min(lo + 1, len(data) - 1)
    return data[lo] + (data[hi] - data[lo]) * (pos - lo)


def round_or_none(value: Optional[float], digits: int = 3) -> Optional[float]:
    return None if value is None else round(float(value), digits)


def human_bytes(n: Optional[float]) -> str:
    if not n:
        return ""
    value = float(n)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024 or unit == "TB":
            return f"{value:.0f} {unit}" if unit == "B" else f"{value:.1f} {unit}"
        value /= 1024
    return ""
