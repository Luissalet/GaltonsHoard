"""Small pure helpers: identifiers, text folding, time formatting and a few numeric utilities."""

from __future__ import annotations

import hashlib
import json
import math
import re
import secrets
import time
import unicodedata
from datetime import datetime
from typing import Any, Iterable, Optional

_ID_ALPHABET = "0123456789abcdefghjkmnpqrstvwxyz"


def new_id(prefix: str, now: Optional[float] = None) -> str:
    """``<prefix>_<time><random>``: sortable by creation time, unique enough for a local database."""
    millis = int((now if now is not None else time.time()) * 1000)
    stamp = ""
    for _ in range(7):
        millis, rest = divmod(millis, 32)
        stamp = _ID_ALPHABET[rest] + stamp
    tail = "".join(secrets.choice(_ID_ALPHABET) for _ in range(5))
    return f"{prefix}_{stamp}{tail}"


def fold_char(ch: str) -> str:
    base = unicodedata.normalize("NFD", ch)[:1] or ch
    low = base.lower()
    return low if len(low) == 1 else base


def fold(text: str) -> str:
    """Lowercase and strip accents without changing the length."""
    return "".join(fold_char(c) for c in text)


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
    return re.sub(r"[^a-z0-9]+", "-", fold(text)).strip("-") or "x"


def stable_hash(*parts: Any, length: int = 32) -> str:
    """SHA-256 of the JSON form of ``parts`` (stable key order), shortened."""
    blob = json.dumps(parts, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:length]


def iso_local(ts: float) -> str:
    """ISO 8601 with the local UTC offset, seconds precision: ``2026-10-02T10:00:00+02:00``."""
    return datetime.fromtimestamp(ts).astimezone().isoformat(timespec="seconds")


def local_hour(ts: float) -> float:
    d = datetime.fromtimestamp(ts)
    return d.hour + d.minute / 60.0


def in_quiet_hours(ts: float, start: float, end: float) -> bool:
    """True when the local hour of ``ts`` is in [start, end); wraps over midnight when start > end."""
    if start == end:
        return False
    hour = local_hour(ts)
    return start <= hour < end if start < end else (hour >= start or hour < end)


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
