"""Ollama's on-disk model store: find the folder, list the manifests and resolve a model name to its GGUF blob (and vision projector)."""

from __future__ import annotations

import json
import ntpath
import os
import sys
from pathlib import Path
from typing import Any, Optional, Sequence

DEFAULT_HOST = "registry.ollama.ai"
DEFAULT_NAMESPACE = "library"
MODEL_LAYER = "application/vnd.ollama.image.model"
PROJECTOR_LAYER = "application/vnd.ollama.image.projector"
CANDIDATES_WINDOWS = (r"D:\LocalAI\ollama-models",)


IS_WINDOWS = sys.platform.startswith("win")
#: where Windows keeps environment variables set for a user and for the machine; a program started before they were set (or by a
#: shortcut, a service or a scheduler that did not inherit them) does not have them in ``os.environ``
REGISTRY_LOCATIONS = (("HKEY_CURRENT_USER", "Environment"), ("HKEY_LOCAL_MACHINE", r"SYSTEM\CurrentControlSet\Control\Session Manager\Environment"))


def registry_variable(name: str) -> Optional[str]:
    """An environment variable as Windows has it stored (the user's first, then the machine's), expanded; ``None`` when it is not set there
    or this is not Windows."""
    if not IS_WINDOWS:
        return None
    try:
        import winreg
    except ImportError:
        return None
    for hive, subkey in REGISTRY_LOCATIONS:
        try:
            with winreg.OpenKey(getattr(winreg, hive), subkey) as key:
                value, kind = winreg.QueryValueEx(key, name)
        except OSError:
            continue
        text = str(value or "").strip()
        if not text:
            continue
        return ntpath.expandvars(text) if kind == getattr(winreg, "REG_EXPAND_SZ", None) else text
    return None


def environment_variable(name: str) -> str:
    """The process environment first; on Windows, when the variable is missing there, the registry."""
    value = os.environ.get(name, "").strip()
    if value:
        return value
    return (registry_variable(name) or "").strip() if IS_WINDOWS else ""


def candidate_dirs(configured: str = "") -> list[Path]:
    """Where the models may be, in order: the setting, ``OLLAMA_MODELS`` (the environment, or the registry on Windows), ``~/.ollama/models`` and the Windows default."""
    out: list[Path] = []
    for raw in (configured, environment_variable("OLLAMA_MODELS"), str(Path.home() / ".ollama" / "models"), *CANDIDATES_WINDOWS):
        if raw and Path(raw) not in out:
            out.append(Path(raw))
    return out


def _has_manifest_files(manifests: Path) -> bool:
    try:
        return any(files for _root, _dirs, files in os.walk(manifests))
    except OSError:
        return False


def store_dirs(configured: str = "") -> list[Path]:
    """Every candidate that has a ``manifests`` folder, those that really hold manifests first (each group in candidate order). A store that
    exists but is empty (Ollama creates ``~/.ollama/models/manifests`` even when ``OLLAMA_MODELS`` points elsewhere) must not hide the real one."""
    filled: list[Path] = []
    empty: list[Path] = []
    for path in candidate_dirs(configured):
        try:
            if not (path / "manifests").is_dir():
                continue
            (filled if _has_manifest_files(path / "manifests") else empty).append(path)
        except OSError:
            continue
    return [*filled, *empty]


def models_dir(configured: str = "") -> Optional[Path]:
    """The store to list from: the first candidate that holds manifests, else the first that only has the folder, else ``None``."""
    found = store_dirs(configured)
    return found[0] if found else None


def parse_ref(ref: str) -> tuple[str, str, str, str]:
    """``qwen3:8b`` -> (registry.ollama.ai, library, qwen3, 8b); ``user/model`` -> (.., user, model, latest); ``host/ns/name:tag`` is kept."""
    text = (ref or "").strip()
    tag = "latest"
    last = text.rsplit("/", 1)[-1]
    if ":" in last:
        text, tag = text.rsplit(":", 1)
    parts = [p for p in text.split("/") if p]
    if len(parts) >= 3:
        return parts[-3], parts[-2], parts[-1], tag
    if len(parts) == 2:
        return DEFAULT_HOST, parts[0], parts[1], tag
    return DEFAULT_HOST, DEFAULT_NAMESPACE, parts[0] if parts else "", tag


def display_name(host: str, namespace: str, name: str, tag: str) -> str:
    prefix = ""
    if host != DEFAULT_HOST:
        prefix = f"{host}/{namespace}/"
    elif namespace != DEFAULT_NAMESPACE:
        prefix = f"{namespace}/"
    return f"{prefix}{name}:{tag}"


def _blob(root: Path, digest: str) -> Path:
    return root / "blobs" / digest.replace(":", "-")


def read_manifest(root: Path, host: str, namespace: str, name: str, tag: str) -> Optional[dict[str, Any]]:
    path = root / "manifests" / host / namespace / name / tag
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def resolve(ref: str, root: Optional[Path | Sequence[Path]]) -> Optional[dict[str, Any]]:
    """``{ref, model_path, mmproj_path, digest, size_bytes, missing}`` for an Ollama tag, or ``None`` when there is no manifest.

    ``root`` is a store or a list of stores (``store_dirs``): they are tried in order until one has the manifest.

    ``digest`` is the blob's sha256 as Ollama stores it, so it changes exactly when the model does. ``missing`` lists blobs the manifest names but the disk lacks.
    """
    if root is None:
        return None
    if isinstance(root, (list, tuple)):
        for candidate in root:
            found = resolve(ref, candidate)
            if found is not None:
                return found
        return None
    host, ns, name, tag = parse_ref(ref)
    manifest = read_manifest(root, host, ns, name, tag)
    if manifest is None:
        return None
    model: Optional[dict[str, Any]] = None
    projector: Optional[dict[str, Any]] = None
    for layer in manifest.get("layers") or []:
        if not isinstance(layer, dict):
            continue
        if layer.get("mediaType") == MODEL_LAYER and model is None:
            model = layer
        elif layer.get("mediaType") == PROJECTOR_LAYER and projector is None:
            projector = layer
    if model is None or not model.get("digest"):
        return None
    model_path = _blob(root, model["digest"])
    mmproj_path = _blob(root, projector["digest"]) if projector and projector.get("digest") else None
    missing = [str(p) for p in (model_path, mmproj_path) if p is not None and not p.is_file()]
    return {"ref": display_name(host, ns, name, tag), "model_path": str(model_path), "mmproj_path": str(mmproj_path) if mmproj_path else None,
            "digest": str(model["digest"]), "size_bytes": int(model.get("size") or 0), "mmproj_size": int((projector or {}).get("size") or 0), "missing": missing}


def list_models(root: Optional[Path]) -> list[str]:
    """Every installed model name (``name:tag``), found by walking the manifests folder."""
    if root is None:
        return []
    base = root / "manifests"
    out: list[str] = []
    try:
        for host in sorted(p for p in base.iterdir() if p.is_dir()):
            for ns in sorted(p for p in host.iterdir() if p.is_dir()):
                for name in sorted(p for p in ns.iterdir() if p.is_dir()):
                    for tag in sorted(p for p in name.iterdir() if p.is_file()):
                        out.append(display_name(host.name, ns.name, name.name, tag.name))
    except OSError:
        return out
    return out
