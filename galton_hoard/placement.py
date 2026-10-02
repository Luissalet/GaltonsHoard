"""Where and how a contestant would run: memory estimate, context, and whether it fits on one allowed GPU."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from . import gguf_meta
from .errors import GaltonError
from .messages import text

FITS_16GB_MB = 16 * 1024 - 600


def gguf_info(store: Any, contestant: dict[str, Any], reader: Callable[[Any], dict[str, Any]] = gguf_meta.read_meta) -> dict[str, Any]:
    """The GGUF header summary of a ``gguf`` contestant, cached in its metadata. ``{}`` when the file cannot be read (demo models, a moved file)."""
    cached = (contestant.get("meta") or {}).get("gguf")
    if cached and cached.get("block_count"):
        return cached
    path = contestant.get("path") or ""
    if not path or not Path(path).is_file():
        return cached or {}
    try:
        info = reader(path)
    except GaltonError:
        return cached or {}
    keep = {k: info.get(k) for k in ("architecture", "block_count", "kv_layers", "head_count_kv", "head_dim", "context_length", "params_b", "quant", "has_chat_template", "expert_count")}
    store.update_contestant(contestant["id"], meta={**contestant.get("meta", {}), "gguf": keep}, **({"context": info["context_length"]} if info.get("context_length") and not contestant.get("context") else {}))
    return keep


def choose_context(requested: int, info: dict[str, Any]) -> int:
    trained = info.get("context_length")
    return int(min(requested, trained)) if trained else int(requested)


def estimate_mb(store: Any, contestant: dict[str, Any], context: int, reader: Callable[[Any], dict[str, Any]] = gguf_meta.read_meta) -> int:
    """MiB a ``gguf`` contestant needs at ``context`` tokens (file × 1.08 + KV cache + projector + 600 MB)."""
    info = gguf_info(store, contestant, reader)
    size = int(contestant.get("size_bytes") or 0)
    if not size and contestant.get("path") and Path(contestant["path"]).is_file():
        size = Path(contestant["path"]).stat().st_size
    if not size:
        raise GaltonError("invalid", "size_unknown", name=contestant["name"])
    mmproj = 0
    if contestant.get("mmproj") and Path(contestant["mmproj"]).is_file():
        mmproj = Path(contestant["mmproj"]).stat().st_size
    return gguf_meta.estimate_vram_mb(size, {**info, "params_b": info.get("params_b") or contestant.get("params_b")}, context, mmproj)


def file_gb(contestant: dict[str, Any]) -> float:
    """Size of the model file in GiB (0 when unknown)."""
    size = int(contestant.get("size_bytes") or 0)
    if not size and contestant.get("path"):
        try:
            size = Path(contestant["path"]).stat().st_size
        except OSError:
            size = 0
    return size / 1024 ** 3


def cpu_eligible(settings: Any, contestant: dict[str, Any], device: str = "auto") -> bool:
    """May this ``gguf`` contestant run on the CPU? ``device`` ``cpu``: always (the run asked for it); ``gpu``: never; ``auto``: when the setting
    ``runner.cpu_fallback`` is on and the file is at most ``runner.cpu_max_gb``."""
    if contestant.get("kind") != "gguf":
        return False
    if device == "cpu":
        return True
    if device == "gpu" or settings is None:
        return False
    size = file_gb(contestant)
    return bool(settings.get("runner.cpu_fallback")) and 0 < size <= float(settings.get("runner.cpu_max_gb"))


def describe(store: Any, contestant: dict[str, Any], requested_context: int, gpus: Any, reader: Callable[[Any], dict[str, Any]] = gguf_meta.read_meta, *,
             settings: Any = None, device: str = "auto") -> dict[str, Any]:
    """What the UI shows next to a contestant when choosing it: where it runs and how much memory it takes. ``device`` is the run setting
    (``auto``, ``gpu`` or ``cpu``): with ``auto`` a small file that no allowed GPU can take right now is shown on the CPU."""
    if contestant["kind"] == "server":
        resident = contestant.get("meta", {}).get("resident")
        where = text("where_server_resident", url=contestant["url"]) if resident else text("where_server", url=contestant["url"])
        return {"runs_on": "server", "where": where, "vram_mb": None, "fits_one_16gb": None, "context": contestant.get("context"), "warnings": []}
    info = gguf_info(store, contestant, reader)
    context = choose_context(requested_context, info)
    out: dict[str, Any] = {"runs_on": "llama.cpp", "context": context, "warnings": []}
    try:
        need = estimate_mb(store, contestant, context, reader)
    except GaltonError as exc:
        return {**out, "vram_mb": None, "where": exc.coded(), "fits_one_16gb": None, "warnings": [exc.coded()]}
    cpu = {"vram_mb": None, "ram_mb": need, "fits_one_16gb": None, "gpus": [], "device": "cpu"}
    if device == "cpu":
        return {**out, **cpu, "where": text("where_cpu_forced")}
    out.update(vram_mb=need, fits_one_16gb=need <= FITS_16GB_MB, device="gpu")
    cpu_ok = cpu_eligible(settings, contestant, device)
    try:
        gpus.check_possible(need)
        plan = gpus.plan(need)
        if plan is None and cpu_ok:
            return {**out, **cpu, "where": text("where_cpu")}
        out["where"] = text("where_gpus", gpus=" + ".join(str(g) for g in plan)) if plan else text("where_wait")
        out["gpus"] = list(plan) if plan else []
    except GaltonError as exc:
        if cpu_ok and exc.code == "no_gpu":
            return {**out, **cpu, "where": text("where_cpu")}
        out["where"] = exc.coded()
        out["warnings"].append(exc.coded())
    return out
