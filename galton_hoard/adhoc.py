"""Turn what a caller gives into contestants: an id or name, or a spec ``{kind: gguf|ollama|server, ...}`` for a model nobody registered yet.

This is how another app asks "evaluate the file I just produced": ``{"kind": "gguf", "path": "D:\\\\models\\\\tuned.gguf"}``.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlsplit

from . import gguf_meta, ollama_models
from .discovery import name_variants
from .errors import GaltonError
from .util import clean_aliases, slugify

LOOPBACK = {"127.0.0.1", "localhost", "::1", "0.0.0.0"}


def is_remote(url: str) -> bool:
    host = (urlsplit(url if "//" in url else "//" + url).hostname or "").lower()
    return host not in LOOPBACK


def _gguf_spec(store: Any, spec: dict[str, Any], meta_reader: Callable[[Any], dict[str, Any]], digest_fn: Callable[[Any], str]) -> dict[str, Any]:
    raw = str(spec.get("path") or "").strip()
    if not raw:
        raise GaltonError("invalid", "gguf_needs_path")
    path = Path(raw)
    if not path.is_absolute():
        raise GaltonError("invalid", "path_not_absolute", path=raw)
    if not path.is_file():
        raise GaltonError("not_found", "file_missing", path=raw)
    meta = meta_reader(path)
    if meta.get("is_projector"):
        raise GaltonError("invalid", "is_projector", name=path.name)
    mmproj = str(spec.get("mmproj") or "")
    if mmproj and not Path(mmproj).is_file():
        raise GaltonError("not_found", "no_projector", path=mmproj)
    digest = digest_fn(path)
    key = f"gguf:{path}"
    stem = str(spec.get("name") or "") or re.sub(r"\.gguf$", "", path.name, flags=re.I)
    fields = {"kind": "gguf", "name": stem, "path": str(path), "mmproj": mmproj, "provider": "llamacpp", "aliases": clean_aliases(name_variants(path.name), [stem]), "digest": digest,
              "family": meta.get("architecture") or "", "params_b": meta.get("params_b"), "quant": meta.get("quant") or "", "context": meta.get("context_length"),
              "vision": bool(mmproj), "size_bytes": meta["size_bytes"], "source": "spec",
              "meta": {"gguf": {k: meta.get(k) for k in ("architecture", "block_count", "kv_layers", "head_count_kv", "head_dim", "context_length", "params_b", "quant", "has_chat_template", "expert_count")}}}
    existing = store.contestant_by_key(key)
    if existing is not None:
        update = {k: v for k, v in fields.items() if k not in ("name", "aliases", "meta")}
        return store.update_contestant(existing["id"], missing=False, **update)
    return store.create_contestant(key=key, adhoc=True, enabled=True, **fields)


def resolve_one(store: Any, settings: Any, item: Any, *, meta_reader: Callable[[Any], dict[str, Any]] = gguf_meta.read_meta,
                digest_fn: Callable[[Any], str] = gguf_meta.file_digest) -> dict[str, Any]:
    if isinstance(item, str):
        found = store.find_contestant(item)
        if found is None:
            raise GaltonError("not_found", "no_model_spec", ref=item)
        return found
    if not isinstance(item, dict):
        raise GaltonError("invalid", "contestant_spec")
    kind = item.get("kind")
    if kind == "gguf":
        return _gguf_spec(store, item, meta_reader, digest_fn)
    if kind == "ollama":
        model = str(item.get("model") or "").strip()
        if not model:
            raise GaltonError("invalid", "ollama_needs_model")
        via_server = item.get("via") == "server"
        pool = store.contestants(include_adhoc=True)
        wanted = {name.casefold() for name in name_variants(model)}
        matches = [c for c in pool if c["ollama_ref"].casefold() in wanted or any(a.casefold() in wanted for a in c["aliases"])]
        matches = [c for c in matches if (c["kind"] == "server") == via_server] or matches
        if matches:
            return matches[0]
        resolved = ollama_models.resolve(model, ollama_models.store_dirs(str(settings.get("ollama.models_dir") or "")))
        if resolved and not resolved["missing"] and not via_server:
            return _gguf_spec(store, {"path": resolved["model_path"], "mmproj": resolved["mmproj_path"] or "", "name": f"{resolved['ref']} (llama.cpp)"}, meta_reader, digest_fn)
        raise GaltonError("not_found", "ollama_not_installed", model=model)
    if kind == "server":
        url, model = str(item.get("url") or "").rstrip("/"), str(item.get("model") or "")
        if not url or not model:
            raise GaltonError("invalid", "server_needs_url_model")
        api = item.get("api") or ("ollama" if ":11434" in url else "openai")
        host = urlsplit(url).netloc
        key = f"server:adhoc:{host}:{slugify(model)}"
        existing = store.contestant_by_key(key)
        if existing is not None:
            return existing
        remote = is_remote(url)
        return store.create_contestant(key=key, kind="server", name=model, url=url, api=api, model=model, provider="remote" if remote else ("ollama" if api == "ollama" else "openai_compat"),
                                       aliases=[model], remote=remote, remote_ok=False, enabled=True, source="spec", adhoc=True)
    raise GaltonError("invalid", "unknown_kind", kind=kind)


def resolve_many(store: Any, settings: Any, items: list[Any], **kwargs: Any) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for item in items:
        contestant = resolve_one(store, settings, item, **kwargs)
        if contestant["id"] not in {c["id"] for c in out}:
            out.append(contestant)
    return out
