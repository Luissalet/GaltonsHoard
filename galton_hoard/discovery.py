"""Find the models on this computer and keep the contestants in step with them.

Sources: Ollama (``/api/tags``, ``/api/ps``, ``/api/show`` and the manifests on disk), llama-server on ports 8080-8090, the Faustus registry,
configured GGUF folders. Each thing found becomes or updates a contestant. Nothing is deleted: a model that disappeared is marked missing,
because its measurements are still worth having.
"""

from __future__ import annotations

import logging
import re
import time
from pathlib import Path
from typing import Any, Callable, Optional
from urllib.parse import urlsplit

import httpx

from . import gguf_meta, identity, ollama_models
from .errors import GaltonError
from .messages import text as coded
from .hoard_link import _faustus
from .servers import probe_openai
from .util import clean_aliases, fold, slugify

log = logging.getLogger("galton.discovery")
LLAMA_PORTS = range(8080, 8091)
SHARD = re.compile(r"-(\d{5})-of-(\d{5})\.gguf$", re.I)
#: answers to a one-token chat request that mean "this server does not chat" (a 503 is a model still loading, a 429 a busy server: no verdict)
NOT_CHAT_STATUSES = frozenset({400, 404, 405, 415, 422, 500, 501})
NOT_CHAT_RECHECK_S = 6 * 3600


def _merge_aliases(*groups: list[str]) -> list[str]:
    """The names of every group once each, only those a server or app can report (no paths, no blob digests)."""
    return clean_aliases(*groups)


def name_variants(name: str) -> list[str]:
    """Every spelling another app may see for a model: ``qwen3:8b``, ``qwen3:8b`` without ``:latest``, the file stem. A path gives its file stem
    only, and a blob digest nothing: aliases are names, the path stays in its own field."""
    name = str(name or "").strip()
    out = [name]
    if name.endswith(":latest"):
        out.append(name[: -len(":latest")])
    base = re.split(r"[\\/]", name)[-1]
    out.append(re.sub(r"\.gguf$", "", base, flags=re.I))
    return clean_aliases(out)


BLOB_NAME = re.compile(r"^sha256-([0-9a-f]{64})$", re.I)


def blob_digest(path: str) -> str:
    """``sha256:<hex>`` when ``path`` is a blob of Ollama's store (``...\\blobs\\sha256-<hex>``, either kind of separator), else empty."""
    match = BLOB_NAME.match(re.split(r"[\\/]", str(path or "").strip())[-1])
    return f"sha256:{match.group(1).lower()}" if match else ""


def host_port(url: str) -> str:
    parts = urlsplit(url if "//" in url else "//" + url)
    return f"{(parts.hostname or '').lower()}:{parts.port or 80}"


class Discovery:
    def __init__(self, store: Any, settings: Any, *, client_factory: Optional[Callable[[], httpx.Client]] = None, clock: Callable[[], float] = time.time,
                 meta_reader: Callable[[Any], dict[str, Any]] = gguf_meta.read_meta, digest_fn: Callable[[Any], str] = gguf_meta.file_digest, offline: bool = False,
                 token_fallback: Callable[[], str] = lambda: ""):
        self.store, self.settings, self.clock = store, settings, clock
        self._client_factory = client_factory or (lambda: httpx.Client(trust_env=False, timeout=4.0))
        self._meta, self._digest = meta_reader, digest_fn
        self.offline = offline
        self.token_fallback = token_fallback

    # ------------------------------------------------------------------ upsert
    def _upsert(self, key: str, fields: dict[str, Any], summary: dict[str, Any], *, new_enabled: bool = True) -> dict[str, Any]:
        existing = self.store.contestant_by_key(key)
        now = self.clock()
        if "aliases" in fields:
            fields = {**fields, "aliases": _merge_aliases(fields["aliases"])}
        if existing is None:
            created = self.store.create_contestant(key=key, enabled=new_enabled, last_seen_ts=now, missing=False, **fields)
            summary["new"].append(created["id"])
            if not (created["meta"].get("chat") or {}).get("ok") is False:          # a server that cannot chat is marked as such, not announced
                self.store.add_notice(kind="new_model", severity="low", params={"name": created["name"]},
                                      data={"contestant": created["id"], "digest": created["digest"]}, contestant_id=created["id"], dedupe=f"new:{key}", once=True)
            return created
        update = dict(fields)
        update["aliases"] = _merge_aliases(existing["aliases"], fields.get("aliases", []))
        for keep in ("name",):  # a name the user chose stays
            if existing.get("meta", {}).get("renamed"):
                update.pop(keep, None)
        old_digest, new_digest = existing["digest"], fields.get("digest", "")
        chat_now = (fields.get("meta") or {}).get("chat") or {}
        if chat_now.get("ok") is False and (existing["meta"].get("chat") or {}).get("ok") is not False:
            update["enabled"] = False            # found out just now that it cannot chat: it leaves the measuring lists (the user may switch it back on)
        if old_digest and new_digest and old_digest != new_digest and chat_now.get("ok") is not False:
            update["meta"] = {**fields.get("meta", {}), "previous_digest": old_digest, "digest_changed_ts": now}
            summary["changed"].append(existing["id"])
            self.store.add_notice(kind="changed_model", severity="medium", params={"name": existing["name"]},
                                  data={"contestant": existing["id"], "old": old_digest, "new": new_digest}, contestant_id=existing["id"], dedupe=f"changed:{key}:{new_digest}")
        else:
            update["meta"] = {**existing["meta"], **fields.get("meta", {})}
        updated = self.store.update_contestant(existing["id"], last_seen_ts=now, missing=False, **update)
        summary["updated"].append(updated["id"])
        return updated

    # ------------------------------------------------------------------ Ollama
    def _ollama(self, client: httpx.Client, summary: dict[str, Any]) -> set[str]:
        base = str(self.settings.get("ollama.url")).rstrip("/")
        seen: set[str] = set()
        try:
            tags = client.get(base + "/api/tags")
            tags.raise_for_status()
            models = tags.json().get("models") or []
        except (httpx.HTTPError, ValueError) as exc:
            summary["sources"]["ollama"] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
            return seen
        resident: dict[str, dict[str, Any]] = {}
        try:
            ps = client.get(base + "/api/ps")
            if ps.status_code == 200:
                resident = {m.get("name") or m.get("model"): m for m in ps.json().get("models") or []}
        except (httpx.HTTPError, ValueError):
            pass
        stores = ollama_models.store_dirs(str(self.settings.get("ollama.models_dir") or ""))
        summary["sources"]["ollama"] = {"ok": True, "models": len(models), "resident": len(resident), "manifests": str(stores[0]) if stores else None,
                                        "stores": [str(p) for p in stores]}
        for item in models:
            name = item.get("name") or item.get("model")
            if not name:
                continue
            digest = item.get("digest", "")
            info = self._ollama_show(client, base, name, digest)
            details = item.get("details") or info.get("details") or {}
            caps = info.get("capabilities") or []
            arch = (info.get("model_info") or {}).get("general.architecture", "")
            context = (info.get("model_info") or {}).get(f"{arch}.context_length") if arch else None
            params = gguf_meta.params_from_text(str(details.get("parameter_size") or "")) or gguf_meta.params_from_text(name)
            common = {"family": details.get("family") or arch or name.split(":")[0], "params_b": params, "quant": details.get("quantization_level") or gguf_meta.quant_from_name(name),
                      "context": int(context) if context else None, "vision": "vision" in caps, "size_bytes": item.get("size")}
            resolved = ollama_models.resolve(name, stores)
            res = resident.get(name)
            server_key = f"server:ollama:{name}"
            seen.add(server_key)
            twin = bool(resolved and not resolved["missing"])
            self._upsert(server_key, {
                "kind": "server", "name": name, "url": base, "api": "ollama", "model": name, "provider": "ollama", "aliases": name_variants(name), "digest": digest,
                "ollama_ref": name, "source": "ollama", **common,
                "meta": {"resident": bool(res), "size_vram": (res or {}).get("size_vram"), "size": (res or {}).get("size"), "capabilities": caps, "gguf_twin": twin,
                         "model_digest": resolved["digest"] if resolved else ""}},
                summary, new_enabled=not twin)
            if twin:
                gkey = f"gguf:ollama:{name}"
                seen.add(gkey)
                size = (resolved["size_bytes"] or 0) + (resolved["mmproj_size"] or 0)
                self._upsert(gkey, {
                    "kind": "gguf", "name": f"{name} (llama.cpp)", "path": resolved["model_path"], "mmproj": resolved["mmproj_path"] or "", "ollama_ref": name,
                    "provider": "llamacpp", "aliases": name_variants(name), "digest": resolved["digest"], "source": "ollama", **{**common, "size_bytes": size or common["size_bytes"]},
                    "vision": common["vision"] or bool(resolved["mmproj_path"]), "meta": {"blob": True, "ollama_digest": digest, "model_digest": resolved["digest"]}},
                    summary)
        return seen

    def _ollama_show(self, client: httpx.Client, base: str, name: str, digest: str) -> dict[str, Any]:
        cached = self.store.ollama_info(digest) if digest else None
        if cached is not None:
            return cached
        try:
            response = client.post(base + "/api/show", json={"model": name}, timeout=10.0)
            if response.status_code != 200:
                return {}
            info = response.json()
        except (httpx.HTTPError, ValueError):
            return {}
        trimmed = {"details": info.get("details") or {}, "capabilities": info.get("capabilities") or [],
                   "model_info": {k: v for k, v in (info.get("model_info") or {}).items() if k == "general.architecture" or k.endswith(".context_length")}}
        if digest:
            self.store.store_ollama_info(digest, trimmed)
        return trimmed

    # ------------------------------------------------------------------ llama-server
    def _llama(self, client: httpx.Client, summary: dict[str, Any]) -> set[str]:
        seen: set[str] = set()
        found = 0
        for port in LLAMA_PORTS:
            base = f"http://127.0.0.1:{port}"
            if base.endswith(":11434"):
                continue
            probe = probe_openai(client, base)
            if not probe.get("up"):
                continue
            found += 1
            for model_id in probe["models"][:1] or []:
                alias = probe.get("alias") or model_id
                path = probe.get("model_path") or ""
                shown = next(iter(name_variants(str(alias)) + name_variants(model_id) + name_variants(path)), f"llama-server:{port}")   # a name, never a path
                stub = self._known_stub(base)
                meta, digest = {}, ""
                if path and Path(path).is_file():
                    try:
                        meta = self._meta(path)
                        digest = self._digest(path)
                    except (GaltonError, OSError):
                        meta = {}
                key = stub["key"] if stub is not None else f"server:llamacpp:{slugify(str(alias))}"
                seen.add(key)
                chat = self._chat_state(client, probe, base, model_id, self.store.contestant_by_key(key))
                self._upsert(key, {
                    "kind": "server", "name": shown, "url": base, "api": "openai", "model": model_id, "provider": "llamacpp",
                    "aliases": _merge_aliases(name_variants(str(alias)), name_variants(model_id), name_variants(path) if path else []), "digest": digest, "path": path,
                    "family": meta.get("architecture") or shown.split("-")[0], "params_b": meta.get("params_b") or gguf_meta.params_from_text(str(alias)),
                    "quant": meta.get("quant") or gguf_meta.quant_from_name(str(alias) + " " + path), "context": probe.get("context") or meta.get("context_length"),
                    "vision": bool(probe.get("vision")), "size_bytes": meta.get("size_bytes"), "source": "llamacpp",
                    "meta": {"up": True, "busy": probe.get("busy"), "port": port, "model_digest": blob_digest(path), **({"chat": chat} if chat else {})}}, summary,
                             new_enabled=chat.get("ok") is not False)
        summary["sources"]["llamacpp"] = {"ok": True, "servers": found}
        # a shared server that is not answering now is down, not gone
        for c in self.store.contestants(kind="server"):
            if c["source"] == "llamacpp" and c["key"] not in seen and c["meta"].get("up", True):
                self.store.update_contestant(c["id"], meta={**c["meta"], "up": False})
        return seen

    # ------------------------------------------------------------------ servers that now serve another model
    def _check_served(self, client: httpx.Client, summary: dict[str, Any]) -> None:
        """A shared server can be restarted with another model on the same address. The entry that stood for the old model is not deleted (its
        history stays) but marked as not served: it cannot be measured until the server serves it again, when the mark goes. The model that is
        served now already has its own entry (the llama-server scan above made it)."""
        groups: dict[str, list[dict[str, Any]]] = {}
        for c in self.store.contestants(kind="server", include_missing=False, include_adhoc=True):
            if c["api"] != "openai" or not c["url"] or (c["remote"] and not c["remote_ok"]):
                continue                                  # a remote endpoint is never called unless it was enabled
            groups.setdefault(identity.endpoint(c["url"]), []).append(c)
        for group in groups.values():
            served = identity.look_openai(client, group[0]["url"])
            if served is None:
                continue                                  # not answering: down, not changed
            for c in group:
                if identity.needs_pin(c):
                    continue                              # a model name nobody has checked against the server yet
                verdict, label = identity.judge(c, served)
                if verdict == "different":
                    if identity.mark_not_served(self.store, c, label, c["url"], self.clock()):
                        summary["not_served"].append(c["id"])
                elif verdict == "same" and identity.clear_not_served(self.store, c):
                    summary["served_again"].append(c["id"])

    # ------------------------------------------------------------------ servers that answer /v1/models but cannot chat
    def _known_stub(self, base: str) -> Optional[dict[str, Any]]:
        """The entry of a server at ``base`` that was already found not to chat: its listing may change its model id on every look, the entry stays one."""
        for c in self.store.contestants(kind="server", include_missing=True, include_adhoc=True):
            if c["source"] == "llamacpp" and c["url"] == base and (c["meta"].get("chat") or {}).get("ok") is False:
                return c
        return None

    def _chat_state(self, client: httpx.Client, probe: dict[str, Any], base: str, model_id: str, existing: Optional[dict[str, Any]]) -> dict[str, Any]:
        """Can this server chat? ``{"ok": bool, "reason"?, "status"?, "checked_ts", "model"}``, or ``{}`` while unknown.

        A server with no chat template (per /props) cannot chat. Otherwise one single-token request settles it: a 404, 500 and the like mean it
        is not a chat model. The verdict is kept on the entry and asked again only for another model id, or (for a "no") after
        ``NOT_CHAT_RECHECK_S``; a connection error or a 503 (still loading) is no verdict."""
        now = self.clock()
        prior = (existing or {}).get("meta", {}).get("chat") or {}
        if probe.get("has_template") is False:
            return {"ok": False, "reason": "no_template", "checked_ts": now, "model": model_id}
        fresh = prior.get("model") == model_id and (prior.get("ok") is True or now - float(prior.get("checked_ts") or 0) < NOT_CHAT_RECHECK_S)
        if prior and fresh and prior.get("reason") != "no_template":
            return prior
        if self.offline:
            return prior
        try:
            r = client.post(base + "/v1/chat/completions", json={"model": model_id, "messages": [{"role": "user", "content": "Hi"}], "max_tokens": 1, "stream": False}, timeout=20.0)
        except (httpx.HTTPError, ValueError):
            return prior
        if r.status_code in NOT_CHAT_STATUSES:
            return {"ok": False, "reason": "failed", "status": r.status_code, "checked_ts": now, "model": model_id}
        if r.status_code == 200:
            return {"ok": True, "checked_ts": now, "model": model_id}
        return prior

    # ------------------------------------------------------------------ the same weights served two ways
    def _link_same_weights(self) -> int:
        """A llama-server whose model file is a blob of Ollama's store runs the very weights of the Ollama tag(s) whose manifest names that blob.
        Those entries are one model seen twice: each gets the other's names as aliases (so the routing table answers to every name) and the
        others' ids in ``meta.same_weights``. Aliases are only ever added; the sibling lists are recomputed on every refresh. Returns how many
        contestants are in a group."""
        groups: dict[str, list[dict[str, Any]]] = {}
        every = self.store.contestants(include_missing=True, include_adhoc=True)
        for c in every:
            digest = str(c["meta"].get("model_digest") or "")
            if digest and c["source"] in ("ollama", "llamacpp"):
                groups.setdefault(digest, []).append(c)
        linked: set[str] = set()
        for members in groups.values():
            if {m["source"] for m in members} != {"ollama", "llamacpp"}:
                continue                         # two Ollama entries of one tag are not news; the link is Ollama against a server
            for c in members:
                others = [o for o in members if o["id"] != c["id"]]
                self.store.update_contestant(c["id"], aliases=_merge_aliases(c["aliases"], *[self._names_of(o) for o in others]),
                                             meta={**c["meta"], "same_weights": sorted(o["id"] for o in others)})
                linked.add(c["id"])
        for c in every:                          # a contestant that no longer shares its weights with anyone loses the marker
            if c["meta"].get("same_weights") and c["id"] not in linked:
                self.store.update_contestant(c["id"], meta={k: v for k, v in c["meta"].items() if k != "same_weights"})
        return len(linked)

    @staticmethod
    def _names_of(c: dict[str, Any]) -> list[str]:
        """The names a contestant answers to, for another one to adopt: its Ollama tag, its server alias and model, not its file blob or '(llama.cpp)' label."""
        out: list[str] = []
        for raw in (c["ollama_ref"], c["model"], c["name"]):
            raw = str(raw or "").strip()
            if not raw or raw.endswith(" (llama.cpp)") or BLOB_NAME.match(re.split(r"[\\/]", raw)[-1]):
                continue
            out.extend(name_variants(raw))
        return _merge_aliases(out)

    # ------------------------------------------------------------------ Faustus
    def _faustus(self, client: httpx.Client, summary: dict[str, Any]) -> None:
        base = str(self.settings.get("faustus.url") or "").rstrip("/")
        if not base:
            return
        token = str(self.settings.get("faustus.token") or self.token_fallback() or "")
        try:
            response = client.get(base + "/api/models", headers=_faustus.auth_headers(token) or {})
            response.raise_for_status()
            items = _faustus.model_items(response.json())
        except (httpx.HTTPError, ValueError) as exc:
            summary["sources"]["faustus"] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
            return
        known = {host_port(c["url"]): c for c in self.store.contestants(kind="server") if c["url"]}
        added = 0
        for item in items:
            if not _faustus.is_local_item(item) or item.get("model_type") not in ("llm", "vision", "chat", None):
                continue
            url = str(item.get("url"))
            name = str(item.get("name") or item.get("model") or "")
            if not name:
                continue
            existing = known.get(host_port(url))
            if existing is not None:
                self.store.update_contestant(existing["id"], aliases=_merge_aliases(existing["aliases"], [name]))
                continue
            api = _faustus.api_for_backend(item.get("backend"))
            self._upsert(f"server:faustus:{slugify(name)}", {"kind": "server", "name": name, "url": url.rstrip("/"), "api": api, "model": str(item.get("model") or name),
                                                             "provider": "ollama" if api == "ollama" else "openai_compat", "aliases": [name], "source": "faustus", "meta": {"up": None}}, summary)
            added += 1
        summary["sources"]["faustus"] = {"ok": True, "items": len(items), "added": added}

    # ------------------------------------------------------------------ GGUF folders
    def _folders(self, summary: dict[str, Any]) -> set[str]:
        seen: set[str] = set()
        blobs = {c["path"] for c in self.store.contestants(kind="gguf") if c["meta"].get("blob")}
        count = 0
        for folder in self.settings.get("gguf.folders"):
            root = Path(folder)
            try:
                files = sorted(root.rglob("*.gguf")) if root.is_dir() else []
            except OSError:
                files = []
            by_dir: dict[Path, list[Path]] = {}
            for f in files:
                by_dir.setdefault(f.parent, []).append(f)
            for f in files:
                if "mmproj" in f.name.lower() or str(f) in blobs:
                    continue
                shard = SHARD.search(f.name)
                if shard and shard.group(1) != "00001":
                    continue
                key = f"gguf:{f}"
                seen.add(key)
                count += 1
                existing = self.store.contestant_by_key(key)
                try:
                    stat = f.stat()
                    size = stat.st_size
                    if shard:
                        size = sum(p.stat().st_size for p in by_dir[f.parent] if p.name.startswith(f.name[: shard.start()]))
                    stamp = f"{stat.st_size}:{int(stat.st_mtime)}"
                    if existing and existing["meta"].get("stamp") == stamp and existing["digest"]:
                        meta = existing["meta"].get("gguf", {})
                        digest = existing["digest"]
                    else:
                        meta, digest = self._meta(f), self._digest(f)
                except (GaltonError, OSError) as exc:
                    summary["errors"].append(coded("refresh_file_error", name=f.name, detail=exc.coded() if isinstance(exc, GaltonError) else str(exc)))
                    continue
                if meta.get("is_projector"):
                    continue
                proj = self._projector_for(f, by_dir.get(f.parent, []))
                stem = re.sub(r"\.gguf$", "", SHARD.sub("", f.name) if shard else f.name, flags=re.I)
                self._upsert(key, {
                    "kind": "gguf", "name": stem, "path": str(f), "mmproj": str(proj) if proj else "", "provider": "llamacpp", "aliases": [stem, f.name], "digest": digest,
                    "family": meta.get("architecture") or stem.split("-")[0], "params_b": meta.get("params_b") or gguf_meta.params_from_text(stem),
                    "quant": meta.get("quant") or gguf_meta.quant_from_name(stem), "context": meta.get("context_length"), "vision": bool(proj), "size_bytes": size, "source": "folder",
                    "meta": {"stamp": stamp, "gguf": {k: meta.get(k) for k in ("architecture", "block_count", "kv_layers", "head_count_kv", "head_dim", "context_length", "params_b", "quant", "has_chat_template", "expert_count")}}},
                    summary)
        summary["sources"]["folders"] = {"ok": True, "files": count, "folders": len(self.settings.get("gguf.folders"))}
        return seen

    @staticmethod
    def _projector_for(model: Path, siblings: list[Path]) -> Optional[Path]:
        projectors = [p for p in siblings if "mmproj" in p.name.lower()]
        if not projectors:
            return None
        base = fold(re.sub(r"[-_.]?(q\d.*|f16|bf16|f32|iq\d.*)$", "", model.stem, flags=re.I))
        for p in projectors:
            if fold(p.stem).startswith(base[:12]) or base[:12] in fold(p.stem):
                return p
        return projectors[0] if len(projectors) == 1 else None

    # ------------------------------------------------------------------ entry point
    def refresh(self) -> dict[str, Any]:
        summary: dict[str, Any] = {"new": [], "updated": [], "changed": [], "missing": [], "not_served": [], "served_again": [], "errors": [], "sources": {}, "offline": self.offline}
        if self.offline:
            return summary
        seen: set[str] = set()
        with self._client_factory() as client:
            ollama_seen = self._ollama(client, summary)
            seen |= ollama_seen
            seen |= self._llama(client, summary)
            self._faustus(client, summary)
            self._check_served(client, summary)
        summary["same_weights"] = self._link_same_weights()
        folder_seen = self._folders(summary)
        seen |= folder_seen
        ollama_ok = summary["sources"].get("ollama", {}).get("ok")
        for c in self.store.contestants():
            if c["missing"] or c["adhoc"] or c["source"] in ("manual", "demo", "spec"):
                continue
            gone = False
            if c["kind"] == "gguf":
                if c["meta"].get("blob") and ollama_ok:
                    gone = c["key"] not in ollama_seen
                elif not c["meta"].get("blob"):
                    gone = bool(c["path"]) and not Path(c["path"]).is_file()
            elif c["source"] == "ollama" and ollama_ok:
                gone = c["key"] not in ollama_seen
            if gone:
                self.store.update_contestant(c["id"], missing=True)
                summary["missing"].append(c["id"])
        return summary
