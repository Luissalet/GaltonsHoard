"""Typed settings stored in the database as JSON text. One table of specs drives validation, the defaults and what the UI shows."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

from .errors import GaltonError

ROUTES_POLICY_DEFAULT: dict[str, Any] = {
    "days": 30,                # only results from the last N days count
    "include_remote": False,   # remote endpoints are never routed to unless asked
    "min_tok_s": 8.0,          # slower candidates are excluded (when their speed is known)
    "max_vram_gb": 30.0,       # larger candidates are excluded (when their memory is known)
    "weight_quality": 1.0,     # rank = quality * lower bound of the 95 % interval + speed * relative speed
    "weight_speed": 0.0,
    "min_cases": 10,           # a category with fewer checked cases is never published
    "top_k": 3,                # how many candidates per category go in the table
}
ROUTES_POLICY_BOUNDS: dict[str, tuple[float, float]] = {
    "days": (1, 3650), "min_tok_s": (0, 10_000), "max_vram_gb": (1, 1000), "weight_quality": (0, 100), "weight_speed": (0, 100),
    "min_cases": (1, 10_000), "top_k": (1, 10),
}

DEFAULT_GGUF_FOLDER = "D:\\LocalAI\\models"
DEFAULT_LLAMA_SERVER = "D:\\LocalAI\\llama.cpp\\llama-server.exe"


@dataclass(frozen=True)
class Spec:
    key: str
    kind: str                       # bool | int | float | str | enum | int_list | str_list | hour | dict | secret
    default: Any
    group: str
    low: Optional[float] = None
    high: Optional[float] = None
    choices: tuple[str, ...] = ()
    default_fn: Optional[Callable[[], Any]] = field(default=None, compare=False)

    def default_value(self) -> Any:
        return self.default_fn() if self.default_fn else self.default


def _default_gguf_folders() -> list[str]:
    try:
        return [DEFAULT_GGUF_FOLDER] if Path(DEFAULT_GGUF_FOLDER).is_dir() else []
    except OSError:
        return []


SPECS: list[Spec] = [
    Spec("ui.language", "enum", "es", "general", choices=("es", "en")),
    Spec("scheduler.paused", "bool", False, "general"),
    Spec("gpus.allowed", "int_list", [2, 3], "gpus", low=0, high=63),
    Spec("gpus.reserved", "int_list", [0, 1], "gpus", low=0, high=63),
    Spec("llama.server_path", "str", "", "llama", default_fn=lambda: DEFAULT_LLAMA_SERVER),
    Spec("llama.extra_args", "str", "", "llama"),
    Spec("llama.load_timeout_s", "int", 600, "llama", low=30, high=3600),
    Spec("gguf.folders", "str_list", [], "discovery", default_fn=_default_gguf_folders),
    Spec("ollama.url", "str", "http://127.0.0.1:11434", "discovery"),
    Spec("ollama.models_dir", "str", "", "discovery"),
    Spec("faustus.url", "str", "http://127.0.0.1:7000", "discovery"),
    Spec("faustus.token", "secret", "", "discovery"),
    Spec("judge.contestant", "str", "", "judge"),
    Spec("runner.idle_grace_s", "int", 20, "runner", low=0, high=3600),
    Spec("runner.wait_idle_max_s", "int", 3600, "runner", low=0, high=604_800),
    Spec("runner.allow_ollama_load", "bool", False, "runner"),
    Spec("runner.timeout_s", "int", 120, "runner", low=5, high=3600),
    Spec("runner.gpu_wait_s", "int", 900, "runner", low=0, high=7200),
    Spec("runner.cpu_fallback", "bool", True, "runner"),
    Spec("runner.cpu_max_gb", "float", 4.0, "runner", low=0.1, high=256),
    Spec("runner.context", "int", 8192, "runner", low=512, high=1_048_576),
    Spec("runner.reasoning_tokens", "int", 8192, "runner", low=0, high=131_072),
    Spec("runner.check_threads", "int", 4, "runner", low=1, high=16),
    Spec("checks.allow_code_execution", "bool", True, "checks"),
    Spec("checks.code_timeout_s", "int", 10, "checks", low=1, high=60),
    Spec("routes.policy", "dict", ROUTES_POLICY_DEFAULT, "routes"),
    Spec("watch.enabled", "bool", True, "watch"),
    Spec("watch.interval_h", "float", 6.0, "watch", low=0.05, high=168),
    Spec("watch.auto_smoke", "bool", True, "watch"),
    Spec("watch.quiet_from", "hour", 1, "watch", low=0, high=24),
    Spec("watch.quiet_to", "hour", 8, "watch", low=0, high=24),
    Spec("watch.idle_min", "int", 10, "watch", low=0, high=1440),
    Spec("arena.min_votes", "int", 5, "arena", low=1, high=1000),
]
SPEC_BY_KEY = {s.key: s for s in SPECS}
MASK = "…"


def _coerce(spec: Spec, value: Any) -> Any:
    kind, key = spec.kind, spec.key
    if kind == "bool":
        if isinstance(value, str):
            if value.strip().lower() in ("1", "true", "yes", "on", "sí", "si"):
                return True
            if value.strip().lower() in ("0", "false", "no", "off", ""):
                return False
            raise GaltonError("invalid", "setting_bool", setting=key)
        return bool(value)
    if kind in ("int", "float", "hour"):
        try:
            number = float(value)
        except (TypeError, ValueError) as exc:
            raise GaltonError("invalid", "setting_number", setting=key) from exc
        if spec.low is not None and not spec.low <= number <= (spec.high if spec.high is not None else number):
            raise GaltonError("invalid", "setting_range", setting=key, low=f"{spec.low:g}", high=f"{spec.high:g}")
        return number if kind == "float" else (int(number) if kind == "int" else round(number, 2))
    if kind == "enum":
        text = str(value).strip()
        if text not in spec.choices:
            raise GaltonError("invalid", "setting_choice", setting=key, options=list(spec.choices))
        return text
    if kind in ("str", "secret"):
        return str(value if value is not None else "").strip()
    if kind == "int_list":
        items = value.replace(";", ",").split(",") if isinstance(value, str) else list(value or [])
        out: list[int] = []
        for item in items:
            if str(item).strip() == "":
                continue
            try:
                number = int(str(item).strip())
            except ValueError as exc:
                raise GaltonError("invalid", "setting_intlist", setting=key) from exc
            if spec.low is not None and not spec.low <= number <= (spec.high or number):
                raise GaltonError("invalid", "setting_intlist_range", setting=key, number=number)
            if number not in out:
                out.append(number)
        return sorted(out)
    if kind == "str_list":
        items = value.replace(";", "\n").splitlines() if isinstance(value, str) else list(value or [])
        return [str(i).strip() for i in items if str(i).strip()]
    if kind == "dict":
        if not isinstance(value, dict):
            raise GaltonError("invalid", "setting_object", setting=key)
        merged = dict(spec.default)
        for name, item in value.items():
            if name not in spec.default:
                raise GaltonError("invalid", "setting_unknown_field", name=name, setting=key, options=list(spec.default))
            if isinstance(spec.default[name], bool):
                merged[name] = _coerce(Spec(f"{key}.{name}", "bool", False, spec.group), item)
            else:
                try:
                    number = float(item)
                except (TypeError, ValueError) as exc:
                    raise GaltonError("invalid", "setting_field_number", setting=key, name=name) from exc
                low, high = ROUTES_POLICY_BOUNDS.get(name, (0, 1e9))
                if not low <= number <= high:
                    raise GaltonError("invalid", "setting_field_range", setting=key, name=name, low=f"{low:g}", high=f"{high:g}")
                merged[name] = int(number) if isinstance(spec.default[name], int) else number
        return merged
    raise GaltonError("invalid", "setting_type", setting=key)


def _plausible(spec: Spec, value: Any) -> bool:
    """Is a stored value of the kind the setting expects? A text that is not JSON comes back from the database as that text."""
    kind = spec.kind
    if kind == "bool":
        return isinstance(value, bool)
    if kind in ("int", "float", "hour"):
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if kind in ("enum", "str", "secret"):
        return isinstance(value, str)
    if kind in ("int_list", "str_list"):
        return isinstance(value, list)
    if kind == "dict":
        return isinstance(value, dict)
    return True


class Settings:
    """Read and write the typed settings. ``get`` always returns a value (the default when nothing is stored)."""

    def __init__(self, db: Any):
        self.db = db

    def get(self, key: str) -> Any:
        spec = SPEC_BY_KEY.get(key)
        if spec is None:
            raise GaltonError("invalid", "setting_unknown", setting=key, options=list(SPEC_BY_KEY))
        value = self.db.get_setting(key, None)       # the shared database decodes the JSON the setting was stored as
        if value is None:
            value = spec.default_value()
            return dict(value) if isinstance(value, dict) else (list(value) if isinstance(value, list) else value)
        if not _plausible(spec, value):               # text that was never JSON (a hand edit) or of another kind: the default
            return spec.default_value()
        if spec.kind == "dict":
            return {**spec.default, **(value if isinstance(value, dict) else {})}
        return value

    def all(self, *, mask_secrets: bool = True) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for spec in SPECS:
            value = self.get(spec.key)
            if spec.kind == "secret" and mask_secrets:
                out[spec.key] = {"configured": bool(value), "value": (MASK + value[-4:]) if len(value) >= 8 else ("****" if value else "")}
            else:
                out[spec.key] = value
        return out

    def describe(self) -> list[dict[str, Any]]:
        return [{"key": s.key, "kind": s.kind, "group": s.group, "default": s.default_value(), "choices": list(s.choices),
                 "min": s.low, "max": s.high} for s in SPECS]

    def validate(self, values: dict[str, Any], *, confirm_reserved: bool = False) -> dict[str, Any]:
        clean: dict[str, Any] = {}
        for key, value in values.items():
            spec = SPEC_BY_KEY.get(key)
            if spec is None:
                raise GaltonError("invalid", "setting_unknown", setting=key, options=list(SPEC_BY_KEY))
            if spec.kind == "secret" and isinstance(value, dict):
                value = ""
            clean[key] = _coerce(spec, value)
        if "gpus.allowed" in clean:
            reserved = set(clean.get("gpus.reserved", self.get("gpus.reserved")))
            wanted = set(clean["gpus.allowed"]) & reserved
            if wanted and not confirm_reserved:
                raise GaltonError("confirm_required", "gpu_reserved", gpus=sorted(wanted), reserved=sorted(wanted))
        return clean

    def set_many(self, values: dict[str, Any], *, confirm_reserved: bool = False) -> dict[str, Any]:
        clean = self.validate(values, confirm_reserved=confirm_reserved)
        for key, value in clean.items():
            self.db.set_setting(key, value)
        return self.all()

    def reset(self, key: str) -> None:
        self.db.delete_setting(key)
