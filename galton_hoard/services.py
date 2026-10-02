"""Wiring: database, settings, GPU manager, launcher, discovery, runner, arena, routes, watch and scheduler behind one object that the API
routers and the agent tools share. Also the views (cards, dashboard, status, overview) built from the store."""

from __future__ import annotations

import hashlib
import json
import logging
import time
from pathlib import Path
from typing import Any, Callable, Optional

import httpx

from . import SERVICE, __version__, adhoc, gguf_meta, placement, suites as suite_lib
from .arena import Arena
from .backends import make_backend
from .board import Board, memory_gb
from .config import Config
from .db import Database
from .discovery import Discovery
from .errors import GaltonError
from .fakes import FakeGpus, FakeWorld, demo_seed
from .gpus import GpuManager
from .hoard_link import family
from .hoard_link.docs.sniff import sniff
from .hoard_link.tokens import read_or_create_token, write_url
from .messages import CodedText, hint_of, text
from .routes import TASKS, Routes
from .runner import Runner, normalise_settings
from .scheduler import Scheduler
from .servers import Launcher, probe_openai
from .settings import Settings
from .store import ACTIVE_RUN_STATES, Store
from .util import clean_aliases, slugify
from .watch import SMOKE_SUITE, Watch

log = logging.getLogger("galton")

GENERATED_SUITES = ("s_contexto-largo", "s_vision")
CARD_FIELDS = ("id", "key", "name", "kind", "url", "api", "model", "provider", "aliases", "family", "params_b", "quant", "context", "vision", "digest", "remote",
               "remote_ok", "enabled", "path", "mmproj", "ollama_ref", "source", "size_bytes", "missing", "adhoc", "last_seen_ts")
OUTPUT_PREVIEW = 1500


def _family_call(app: str, tool: str, arguments: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    return family.call(app, tool, arguments, timeout=60.0)


class Services:
    def __init__(self, config: Config, *, clock_fn: Callable[[], float] = time.time, world: Optional[FakeWorld] = None, fake_gpus: Optional[FakeGpus] = None,
                 gpus: Optional[GpuManager] = None, launcher: Any = None, backend_factory: Optional[Callable[..., Any]] = None,
                 client_factory: Optional[Callable[[], httpx.Client]] = None, family_call: Optional[Callable[..., dict[str, Any]]] = None,
                 meta_reader: Callable[[Any], dict[str, Any]] = gguf_meta.read_meta, digest_fn: Callable[[Any], str] = gguf_meta.file_digest,
                 sleep: Callable[[float], None] = time.sleep):
        self.config = config
        self.clock = clock_fn
        self.started_at = time.time()
        self.meta_reader, self.digest_fn = meta_reader, digest_fn
        for d in (config.data_dir, config.cache_dir, config.logs_dir, config.images_dir):
            d.mkdir(parents=True, exist_ok=True)
        self.token = read_or_create_token(config.token_path)      # persistent: created once, reused on every later start
        write_url(config.url_path, f"http://127.0.0.1:{config.port}")
        self.db = Database(config.db_path)
        self.store = Store(self.db, clock_fn)
        self.settings = Settings(self.db)
        self.fake = bool(config.fake or world is not None)
        self.fake_world: Optional[FakeWorld] = world
        if config.fake and world is None:
            self.fake_world = FakeWorld(delay_s=0.03)
        if self.fake:
            fake_gpus = fake_gpus or FakeGpus()
            if config.routes_file is None:      # the demo never writes into the real routing table
                config.routes_file = config.data_dir / "routes-demo.json"
        self.fake_gpus = fake_gpus
        self.gpus = gpus or (GpuManager(self.settings, probe=fake_gpus.probe, hub=fake_gpus.hub, lease_factory=fake_gpus.lease_factory, names=fake_gpus.names,
                                        sleep=lambda _s: None) if fake_gpus else GpuManager(self.settings))
        self.launcher = launcher or self.fake_world or Launcher(self.settings, config.logs_dir, config.servers_file)
        self.backend_factory = backend_factory or (self.fake_world.backend_factory if self.fake_world else make_backend)
        self.client_factory = client_factory
        self.family_call = family_call if family_call is not None else (None if (config.offline or self.fake) else _family_call)
        self.sync_info = suite_lib.sync(self.store)
        self.aliases_cleaned = self.store.clean_aliases()
        self.discovery = Discovery(self.store, self.settings, client_factory=client_factory, clock=clock_fn, meta_reader=meta_reader, digest_fn=digest_fn,
                                   offline=config.offline or self.fake, token_fallback=lambda: config.secret("FAUSTUS_TOKEN"))
        self.runner = Runner(self.store, self.settings, self.gpus, self.launcher, self.backend_factory, images_dir=config.images_dir, clock=clock_fn, sleep=sleep,
                             family_call=self.family_call, emit=self.emit, client_factory=client_factory, meta_reader=meta_reader, digest_fn=digest_fn,
                             base_url=lambda: f"http://127.0.0.1:{config.port}")
        self.arena = Arena(self.store, self.settings)
        self.routes = Routes(self.store, self.settings, config.routes_path, clock=clock_fn, emit=self.emit)
        self.board = Board(self.store, self.settings, self.routes, clock_fn)
        self.watch = Watch(self.store, self.settings, self.runner, self.gpus, clock=clock_fn, submit_run=self.submit_run, emit=self.emit, client_factory=client_factory,
                           offline=config.offline or self.fake)
        self.scheduler = Scheduler({"run": self._job_run, "judge": self._job_judge, "refresh": self._job_refresh, "watch": self._job_watch,
                                    "housekeeping": self._job_housekeeping}, self.store, clock=clock_fn, enabled=config.scheduler and not self.fake,
                                   paused=lambda: bool(self.settings.get("scheduler.paused")), refresh_every_h=lambda: float(self.settings.get("watch.interval_h")),
                                   watch_enabled=lambda: bool(self.settings.get("watch.enabled")) and not config.offline)
        self.last_refresh: dict[str, Any] = {}
        self._routes_cache: tuple[Any, dict[str, Any]] = (None, {})
        if config.fake:
            demo_seed(self)

    # ------------------------------------------------------------------ lifecycle
    def start(self) -> None:
        try:
            reaped = self.launcher.reap_orphans()
            if reaped:
                log.info("stopped %d llama-server process(es) left over from a previous run", len(reaped))
        except Exception:  # noqa: BLE001
            log.exception("could not look for leftover llama-server processes")
        self._abandon_stale_runs()
        self.scheduler.start()
        self._requeue_queued_runs()

    def _requeue_queued_runs(self) -> None:
        """Runs still ``queued`` lived only in the old process's in-memory queue: put them back on the GPU lane, oldest first.
        Without this they would wait forever, and the regression watch (which skips while any run is queued) would never run again."""
        if not self.scheduler._alive():
            return
        for run in sorted(self.store.runs(states=("queued",), limit=200), key=lambda r: r["created_ts"]):
            self.scheduler.submit("run", run["id"], "restart")

    def stop(self) -> None:
        self.scheduler.stop()
        self.db.close()

    def emit(self, type_: str, data: dict[str, Any]) -> None:
        try:
            family.emit(type_, data)
        except Exception:  # noqa: BLE001 — events are hints; the database is the truth
            pass

    def _abandon_stale_runs(self) -> None:
        """A run that was running when the process died can never finish: say so instead of showing it as live forever."""
        for run in self.store.runs(states=("running", "waiting_gpu", "waiting_server"), limit=100):
            self.store.update_run(run["id"], state="failed", finished_ts=self.clock(), error=text("run_stopped"))
            self.runner.jobs.finished(self.store.run(run["id"]), "failed", str(text("run_stopped")))
            for rc in self.store.run_contestants(run["id"]):
                if rc["state"] not in ("done", "failed", "cancelled"):
                    self.store.upsert_run_contestant(run["id"], rc["contestant_id"], state="failed", error=text("rc_interrupted"), wait_since=None)

    # ------------------------------------------------------------------ settings
    def setting(self, key: str) -> Any:
        return self.settings.get(key)

    def settings_view(self) -> dict[str, Any]:
        return {"values": self.settings.all(), "spec": self.settings.describe(), "gpus": self.gpus.state()}

    # ------------------------------------------------------------------ jobs
    def submit_run(self, run_id: str, wait_s: float = 0.0) -> None:
        """Queue a run on the GPU lane. Without running lanes (tests) it runs inline."""
        if not self.scheduler._alive():
            self.scheduler.run_now("run", run_id)
            return
        job = self.scheduler.submit("run", run_id, "manual")
        if job is not None and wait_s > 0:
            job.done.wait(wait_s)

    def _job_run(self, run_id: str) -> dict[str, Any]:
        t0 = time.monotonic()
        run = self.runner.execute(run_id)
        regressions: list[dict[str, Any]] = []
        if run["state"] == "done":
            try:
                regressions = self.watch.detect_regressions(run_id)
            except Exception:  # noqa: BLE001
                log.exception("regression check failed for %s", run_id)
        elif run["state"] == "failed":
            self.store.add_notice(kind="run_failed", severity="medium", params={"label": run["label"], "error": run["error"][:400]}, data={"run": run_id}, dedupe=f"run_failed:{run_id}")
        self.store.add_activity("run", run_id, run["state"] == "done", int((time.monotonic() - t0) * 1000), run["state"])
        self.routes_changed()
        return {"run": run_id, "state": run["state"], "regressions": len(regressions)}

    def _job_judge(self, _ref: str) -> dict[str, Any]:
        out = self.runner.judge_pending()
        self.store.add_activity("judge", "", True, 0, str(out)[:200])
        return out

    def _job_refresh(self, _ref: str) -> dict[str, Any]:
        t0 = time.monotonic()
        try:
            summary = self.discovery.refresh()
        except Exception as exc:  # noqa: BLE001
            self.store.add_activity("refresh", "", False, int((time.monotonic() - t0) * 1000), f"{type(exc).__name__}: {exc}")
            raise
        self.last_refresh = {"ts": self.clock(), **summary}
        self.store.add_activity("refresh", "", True, int((time.monotonic() - t0) * 1000),
                                f"{len(summary['new'])} new, {len(summary['changed'])} changed, {len(summary['missing'])} missing, {len(summary['not_served'])} not served")
        if summary["new"] or summary["changed"] or summary["missing"] or summary["not_served"] or summary["served_again"]:
            self.emit("galton.models.updated", {"new": len(summary["new"]), "changed": len(summary["changed"]), "missing": len(summary["missing"]),
                                                "not_served": len(summary["not_served"])})
        return summary

    def _job_watch(self, _ref: str) -> dict[str, Any]:
        return self.watch.tick()

    def _job_housekeeping(self, _ref: str) -> dict[str, Any]:
        out: dict[str, Any] = {}
        referenced = {name for row in self.db.query("SELECT images FROM cases WHERE images != '[]'") for name in json.loads(row["images"] or "[]")}
        removed = 0
        if self.config.images_dir.is_dir():
            for f in self.config.images_dir.iterdir():
                if f.is_file() and f.name not in referenced and f.stat().st_mtime < self.clock() - 86400:
                    f.unlink(missing_ok=True)
                    removed += 1
        out["images_removed"] = removed
        cutoff = self.clock() - 90 * 86400
        out["judge_cache_removed"] = self.db.execute("DELETE FROM judge_cache WHERE ts < ?", (cutoff,)).rowcount
        out["ollama_info_removed"] = self.db.execute("DELETE FROM ollama_info WHERE ts < ?", (self.clock() - 180 * 86400,)).rowcount
        out["arena_pairs_removed"] = self.db.execute("DELETE FROM arena_pairs WHERE voted = 0 AND created_ts < ?", (self.clock() - 7 * 86400,)).rowcount
        try:
            out["orphan_servers_stopped"] = len(self.launcher.reap_orphans())
        except Exception:  # noqa: BLE001
            out["orphan_servers_stopped"] = 0
        return out

    # ------------------------------------------------------------------ cards
    def contestant_card(self, c: dict[str, Any], last: Optional[dict[str, Any]] = None) -> dict[str, Any]:
        card = {k: c.get(k) for k in CARD_FIELDS}
        meta = c.get("meta") or {}
        info = (last or {}).get(c["id"]) or {}
        digests = {d["digest"] for d in info.get("digests", [])}
        card.update(up=meta.get("up"), resident=meta.get("resident"), busy=meta.get("busy"), demo=bool(meta.get("demo")), blob=bool(meta.get("blob")),
                    same_weights=list(meta.get("same_weights") or []), not_chat=(meta.get("chat") or {}).get("ok") is False, not_chat_reason=self.not_chat_reason(meta),
                    not_served=bool(meta.get("not_served")), not_served_reason=self.not_served_reason(c),
                    measured_n=info.get("n", 0), last_measured_ts=info.get("last_ts"),
                    stale=bool(digests) and bool(c["digest"]) and c["digest"] not in digests, never_measured=not info)
        mem, method = memory_gb(self.store, c)
        card.update(memory_gb=mem, memory_method=method, fits_one_16gb=(mem <= placement.FITS_16GB_MB / 1024) if mem is not None else None)
        return card

    @staticmethod
    def not_chat_reason(meta: dict[str, Any]) -> Optional[CodedText]:
        """Why a server was marked as not a chat model (``None`` for every other entry)."""
        chat = meta.get("chat") or {}
        if chat.get("ok") is not False:
            return None
        return text("not_chat_no_template") if chat.get("reason") == "no_template" else text("not_chat_failed", status=chat.get("status") or "")

    @staticmethod
    def not_served_reason(c: dict[str, Any]) -> Optional[CodedText]:
        """Why a server entry cannot be measured now: its address serves another model (``None`` for every other entry)."""
        flag = (c.get("meta") or {}).get("not_served")
        if not flag:
            return None
        return GaltonError("unavailable", "model_not_served", name=c["name"], url=flag.get("url") or c["url"], model=flag.get("model") or "?").coded()

    def same_weights(self, c: dict[str, Any]) -> list[dict[str, Any]]:
        """The other entries that run the very same weights as ``c`` (an Ollama tag and the llama-server that loads its blob), as short cards."""
        out = []
        for sid in (c.get("meta") or {}).get("same_weights") or []:
            other = self.store.find_contestant(sid)
            if other is not None:
                out.append({"id": other["id"], "name": other["name"], "kind": other["kind"], "source": other["source"], "ollama_ref": other["ollama_ref"],
                            "url": other["url"], "enabled": other["enabled"], "missing": other["missing"], "relation": "same_weights"})
        return out

    def contestant_cards(self, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        last = self.store.last_measured()
        return [self.contestant_card(c, last) for c in rows]

    def suite_card(self, s: dict[str, Any]) -> dict[str, Any]:
        return {"id": s["id"], "name": s["name"], "description": s["description"], "category": s["category"], "builtin": s["builtin"], "version": s["version"],
                "max_tokens": s["max_tokens"], "notes": s.get("notes", ""), "cases": s.get("cases", s.get("n_cases", 0)), "generated": s["id"] in GENERATED_SUITES}

    def checker_label(self, checker: dict[str, Any]) -> str:
        kind = (checker or {}).get("type", "?")
        if kind in ("all", "any"):
            return kind + "(" + ", ".join(self.checker_label(c) for c in checker.get("checks", [])) + ")"
        return kind

    def case_card(self, k: dict[str, Any], *, full: bool = False) -> dict[str, Any]:
        prompt = k["prompt"]
        text = prompt.get("text") or " ".join(str(m.get("content", "")) for m in prompt.get("messages", []) if m.get("role") == "user")
        generated = prompt.get("generate")
        card = {"id": k["id"], "suite_id": k["suite_id"], "position": k["position"], "title": k["title"], "checker": k["checker"], "checker_label": self.checker_label(k["checker"]),
                "weight": k["weight"], "tags": k["tags"], "source": k["source"], "notes": k["notes"], "max_tokens": k.get("max_tokens"), "min_context": k.get("min_context") or 0,
                "has_tools": bool(k["tools"]), "has_images": bool(k["images"]) or bool(generated and generated.get("kind") == "vision"),
                "generated": generated.get("kind") if generated else None,
                "prompt_preview": (f"[{generated.get('kind')}] {text}" if generated and not text else text)[:240]}
        if full:
            card.update(prompt=prompt, tools=k["tools"], images=k["images"], reference=k["reference"], results=self.db_count("results", "case_id", k["id"]))
        return card

    def db_count(self, table: str, column: str, value: Any) -> int:
        row = self.db.one(f"SELECT COUNT(*) FROM {table} WHERE {column} = ?", (value,))
        return int(row[0]) if row else 0

    def case_titles(self, ids: list[str]) -> dict[str, str]:
        out: dict[str, str] = {}
        ids = sorted(set(ids))
        for i in range(0, len(ids), 400):
            chunk = ids[i:i + 400]
            for row in self.db.query(f"SELECT id, title FROM cases WHERE id IN ({', '.join('?' for _ in chunk)})", chunk):
                out[row["id"]] = row["title"]
        return out

    def result_card(self, r: dict[str, Any], titles: dict[str, str], names: dict[str, str], *, output: bool = True, limit: int = OUTPUT_PREVIEW) -> dict[str, Any]:
        card = {"id": r["id"], "run": r["run_id"], "contestant": r["contestant_id"], "contestant_name": names.get(r["contestant_id"], r["contestant_id"]), "suite": r["suite_id"],
                "case": r["case_id"], "title": titles.get(r["case_id"], r["case_id"]), "category": r["category"], "repeat": r["repeat"], "score": r["score"], "passed": r["passed"],
                "skipped": r["skipped"], "unavailable": r["unavailable"], "truncated": r["truncated"], "cpu": r["cpu"], "judge_pending": r["judge_pending"], "self_judged": r["self_judged"], "error": r["error"],
                "latency_ms": r["latency_ms"], "ttft_ms": r["ttft_ms"], "decode_tps": r["decode_tps"], "prompt_tokens": r["prompt_tokens"],
                "completion_tokens": r["completion_tokens"], "detail": r["detail"]}
        if output:
            text = r["output"] or ""
            card.update(output=text[:limit], output_truncated=len(text) > limit, reasoning=(r["reasoning"] or "")[:limit], tool_calls=r["tool_calls"])
        return card

    def _waiting(self, rc: dict[str, Any]) -> dict[str, Any]:
        """How long a model has waited for a shared server somebody else was using: ``waiting_s`` is the current wait (``None`` when it is not
        waiting), ``waited_s`` all the waiting of this run so far, the current wait included."""
        since = rc.get("wait_since")
        current = max(0.0, self.clock() - since) if since is not None and rc.get("state") == "waiting_server" else None
        return {"waiting_s": round(current, 1) if current is not None else None, "waited_s": round(float(rc.get("waited_s") or 0.0) + (current or 0.0), 1)}

    def run_card(self, run: dict[str, Any]) -> dict[str, Any]:
        rcs = {rc["contestant_id"]: rc for rc in self.store.run_contestants(run["id"])}
        contestants = []
        for cid in run["contestants"]:
            c = self.store.find_contestant(cid)
            rc = rcs.get(cid) or {}
            counts = self.store.count_results(run["id"], cid)
            contestants.append({"id": cid, "name": c["name"] if c else cid, "kind": c["kind"] if c else "", "state": rc.get("state", "queued"), "done": rc.get("done", 0),
                                "total": rc.get("total", 0), "error": rc.get("error", ""), "hint": hint_of(rc.get("error")), "runs_on": rc.get("runs_on", ""), "load_ms": rc.get("load_ms"), "vram_mb": rc.get("vram_mb"),
                                "vram_method": rc.get("vram_method", ""), "gpus": rc.get("gpus", []), "device": rc.get("device", ""), "context": rc.get("context"), "warnings": rc.get("warnings", []),
                                "spill": rc.get("spill", ""), "passed": counts.get("passed", 0), "skipped": counts.get("skipped", 0), "errors": counts.get("errors", 0),
                                "truncated": counts.get("truncated", 0), **self._waiting(rc)})
        total = sum(c["total"] for c in contestants)
        done = sum(c["done"] for c in contestants)
        suites = []
        for sid in run["suites"]:
            s = self.store.find_suite(sid)
            suites.append({"id": sid, "name": s["name"] if s else sid})
        position = None
        if run["state"] == "queued":
            queued = sorted(self.store.runs(states=("queued",), limit=200), key=lambda r: r["created_ts"])
            ids = [r["id"] for r in queued]
            position = (ids.index(run["id"]) + 1 + (1 if self.runner.active_run else 0)) if run["id"] in ids else None
        earlier = self.store.find_run(run["continues"]) if run.get("continues") else None
        return {"id": run["id"], "label": run["label"], "state": run["state"], "source": run["source"], "continues": run.get("continues", ""), "continues_label": earlier["label"] if earlier else "", "caller": run["caller"], "created_ts": run["created_ts"],
                "started_ts": run["started_ts"], "finished_ts": run["finished_ts"], "error": run["error"], "settings": run["settings"], "suites": suites, "contestants": contestants,
                "progress": {"done": done, "total": total, "pct": round(100 * done / total, 1) if total else 0.0}, "summary": run["summary"],
                "active": run["id"] == self.runner.active_run, "queue_position": position, "pending_judge": len(self.store.pending_judge(run["id"])) if run["state"] in ("done", "cancelled") else 0,
                "discarded": bool(run["discarded"]), "discard_reason": run["discard_reason"], "discarded_ts": run["discarded_ts"]}

    def run_events(self, run_id: str, after: int = 0, limit: int = 300) -> dict[str, Any]:
        """Polling feed for the run page: the run card plus the results written after result ``after``."""
        run = self.store.run(run_id)
        rows = self.store.results(run_id=run_id, after_id=after, limit=limit, with_output=False)
        titles = self.case_titles([r["case_id"] for r in rows])
        names = {c["id"]: c["name"] for c in self.store.contestants()}
        return {"run": self.run_card(run), "results": [self.result_card(r, titles, names, output=False) for r in rows], "last_id": rows[-1]["id"] if rows else after,
                "finished": run["state"] in ("done", "failed", "cancelled")}

    # ------------------------------------------------------------------ cases
    def attach_image(self, path: str) -> str:
        """Copy an image into the content-addressed image folder and return the name a case stores. Only PNG, JPEG, WebP and GIF, at most 10 MB."""
        source = Path(path)
        if not source.is_absolute():
            raise GaltonError("invalid", "image_not_absolute", path=path)
        if not source.is_file():
            raise GaltonError("not_found", "image_missing", path=path)
        if source.stat().st_size > 10 * 1024 * 1024:
            raise GaltonError("too_large", "image_too_large")
        data = source.read_bytes()
        found = sniff(source.name, data)          # by content: a RIFF audio file is not a WebP picture
        ext = found.ext if found.kind == "image" and found.ext in ("png", "jpg", "webp", "gif") else ""
        if not ext:
            raise GaltonError("unsupported", "image_type", name=source.name)
        name = f"{hashlib.sha256(data).hexdigest()[:24]}.{ext}"
        target = self.config.images_dir / name
        if not target.exists():
            target.write_bytes(data)
        return name

    def editable_suite(self, ref: str) -> dict[str, Any]:
        suite = self.store.find_suite(ref)
        if suite is None:
            raise GaltonError("not_found", "no_suite", ref=ref)
        if suite["builtin"]:
            raise GaltonError("builtin_readonly", "suite_builtin", name=suite["name"])
        return suite

    # ------------------------------------------------------------------ models
    def add_model(self, *, url: str = "", model: str = "", api: str = "", path: str = "", mmproj: str = "", name: str = "", remote_ok: bool = False) -> dict[str, Any]:
        """Register a server (chat-completions compatible, like llama-server, or Ollama) or a GGUF file by hand."""
        if bool(url) == bool(path):
            raise GaltonError("invalid", "add_model_args")
        if path:
            found = adhoc.resolve_one(self.store, self.settings, {"kind": "gguf", "path": path, "mmproj": mmproj, "name": name}, meta_reader=self.meta_reader, digest_fn=self.digest_fn)
            return self.store.update_contestant(found["id"], adhoc=False, source="manual", enabled=True, **({"name": name} if name else {}))
        url = url.rstrip("/")
        if not url.startswith(("http://", "https://")):
            raise GaltonError("invalid", "url_scheme")
        remote = adhoc.is_remote(url)
        if self.config.offline or self.config.fake:
            raise GaltonError("unavailable", "offline_no_probe")
        factory = self.client_factory or (lambda: httpx.Client(trust_env=False, timeout=5.0))
        models: list[str] = []
        detected = api
        with factory() as client:
            if detected in ("", "ollama"):
                try:
                    r = client.get(url + "/api/tags")
                    if r.status_code == 200 and isinstance(r.json().get("models"), list):
                        detected = "ollama"
                        models = [m.get("name") or m.get("model") for m in r.json()["models"] if isinstance(m, dict)]
                except (httpx.HTTPError, ValueError):
                    pass
            if detected in ("", "openai"):
                probe = probe_openai(client, url)
                if probe.get("up"):
                    detected = "openai"
                    models = probe["models"]
        if not detected:
            raise GaltonError("unavailable", "server_no_answer_kind", url=url)
        if not model:
            if len(models) == 1:
                model = models[0]
            else:
                raise GaltonError("invalid", "server_pick_model", options=models[:20] or ["(nothing listed)"], models=models[:50])
        elif models and model not in models:
            raise GaltonError("not_found", "server_lacks_model", model=model, options=models[:20], models=models[:50])
        key = f"server:manual:{url.split('//', 1)[-1]}:{slugify(model)}"
        existing = self.store.contestant_by_key(key)
        fields = dict(kind="server", name=name or model, url=url, api=detected, model=model, provider="remote" if remote else ("ollama" if detected == "ollama" else "openai_compat"),
                      aliases=[model], remote=remote, remote_ok=bool(remote_ok) and remote, enabled=True, source="manual", adhoc=False)
        return self.store.update_contestant(existing["id"], missing=False, **fields) if existing else self.store.create_contestant(key=key, **fields)

    def update_model(self, ref: str, **changes: Any) -> dict[str, Any]:
        c = self.store.find_contestant(ref)
        if c is None:
            raise GaltonError("not_found", "no_model", ref=ref)
        fields: dict[str, Any] = {}
        if changes.get("enabled") is not None:
            fields["enabled"] = bool(changes["enabled"])
        if changes.get("name"):
            fields["name"] = changes["name"].strip()
            fields["meta"] = {**c["meta"], "renamed": True}
        if changes.get("aliases") is not None:
            fields["aliases"] = clean_aliases(changes["aliases"])
        if changes.get("add_aliases"):
            fields["aliases"] = clean_aliases(c["aliases"], changes["add_aliases"])
        if changes.get("remote_ok") is not None:
            if not c["remote"] and changes["remote_ok"]:
                raise GaltonError("invalid", "remote_ok_local", name=c["name"])
            fields["remote_ok"] = bool(changes["remote_ok"])
        if changes.get("vision") is not None:
            fields["vision"] = bool(changes["vision"])
        if changes.get("mmproj") is not None:
            if changes["mmproj"] and not Path(changes["mmproj"]).is_file():
                raise GaltonError("not_found", "no_projector", path=changes["mmproj"])
            fields["mmproj"] = changes["mmproj"]
            if changes["mmproj"] and changes.get("vision") is None:
                fields["vision"] = True
        if changes.get("adhoc") is False:
            fields["adhoc"] = False
        return self.store.update_contestant(c["id"], **fields)

    def remove_model(self, ref: str) -> dict[str, Any]:
        c = self.store.find_contestant(ref)
        if c is None:
            raise GaltonError("not_found", "no_model", ref=ref)
        if self.runner.active_run and c["id"] in self.store.run(self.runner.active_run)["contestants"]:
            raise GaltonError("busy", "model_busy_run", name=c["name"])
        results = self.db_count("results", "contestant_id", c["id"])
        self.store.delete_contestant(c["id"])
        self.db.execute("DELETE FROM results WHERE contestant_id = ?", (c["id"],))
        self.db.execute("DELETE FROM run_contestants WHERE contestant_id = ?", (c["id"],))
        return {"removed": c["name"], "id": c["id"], "results_removed": results,
                "note": text("model_removed_note") if c["source"] not in ("manual", "spec") else ""}

    # ------------------------------------------------------------------ runs
    def preflight(self, contestants: list[dict[str, Any]], rs: dict[str, Any]) -> list[dict[str, Any]]:
        """Problems that would make a model fail before it answers anything: no llama-server, a file too big for the allowed GPUs, no GPU at all."""
        problems = []
        for c in contestants:
            if c["kind"] == "server":
                try:
                    self.runner.ensure_served(c)
                except GaltonError as exc:
                    problems.append({"contestant": c["id"], "name": c["name"], "problem": exc.coded(), "hint": exc.hint, "code": exc.code, "key": exc.key, "params": exc.params})
                continue
            if c["kind"] != "gguf":
                continue
            try:
                if hasattr(self.launcher, "binary"):
                    self.launcher.binary()
                info = placement.gguf_info(self.store, c, self.meta_reader)
                context = placement.choose_context(int(rs["context"] or self.settings.get("runner.context")), info)
                need = placement.estimate_mb(self.store, c, context, self.meta_reader)
                if rs["device"] != "cpu":
                    try:
                        self.gpus.check_possible(need)
                    except GaltonError as exc:
                        if not (exc.code == "no_gpu" and placement.cpu_eligible(self.settings, c, rs["device"])):
                            raise
            except GaltonError as exc:
                problems.append({"contestant": c["id"], "name": c["name"], "problem": exc.coded(), "hint": exc.hint, "code": exc.code, "key": exc.key, "params": exc.params})
        return problems

    def plan_run(self, suites: list[str], contestants: list[Any], settings: Optional[dict[str, Any]] = None) -> dict[str, Any]:
        """What starting this run would do, without creating it: cases per model, memory, where each would run, problems."""
        rs = normalise_settings(settings)
        suite_rows = self.runner.resolve_suites(suites)
        rows = self.runner.resolve_contestants(contestants)
        out = []
        for c in rows:
            planned, notes = self.runner.plan_cases(c, suite_rows, rs)
            where = placement.describe(self.store, c, int(rs["context"] or self.settings.get("runner.context")), self.gpus, self.meta_reader, settings=self.settings,
                                       device=rs["device"])
            out.append({"id": c["id"], "name": c["name"], "kind": c["kind"], "cases": len(planned), "notes": notes, "enabled": c["enabled"], "missing": c["missing"], **where})
        return {"suites": [{"id": s["id"], "name": s["name"], "cases": self.store.count_cases(s["id"])} for s in suite_rows], "contestants": out,
                "problems": self.preflight(rows, rs), "settings": rs, "total_cases": sum(c["cases"] for c in out)}

    def start_run(self, *, suites: list[str], contestants: list[Any], settings: Optional[dict[str, Any]] = None, label: str = "", source: str = "ui", caller: str = "",
                  wait_s: float = 0.0) -> dict[str, Any]:
        rs = normalise_settings(settings)
        rows = self.runner.resolve_contestants(contestants)
        problems = self.preflight(rows, rs)
        self._refuse_hopeless(rows, problems)
        run = self.runner.create(suites=suites, contestants=[c["id"] for c in rows], settings=rs, label=label, source=source, caller=caller)
        self.submit_run(run["id"], wait_s)
        card = self.run_card(self.store.run(run["id"]))
        out = {"run": card, "warnings": [text("warn_will_fail", name=p["name"], problem=p["problem"]) for p in problems]}
        if wait_s > 0 and card["state"] in ACTIVE_RUN_STATES:
            out["still_running"] = True      # the wait ran out first: the run goes on, ask run_status again
        return out

    @staticmethod
    def _refuse_hopeless(rows: list[dict[str, Any]], problems: list[dict[str, Any]]) -> None:
        """A run in which every model is a file that cannot be run at all is refused with the first reason, instead of being created to fail."""
        if problems and len(problems) == len([c for c in rows if c["kind"] == "gguf"]) == len(rows):
            first = problems[0]
            if first["key"]:
                raise GaltonError(first["code"], first["key"], **{**first["params"], "problems": problems})
            raise GaltonError(first["code"], first["problem"], first["hint"], problems=problems)

    def resume_run(self, ref: str, *, source: str = "ui", caller: str = "", wait_s: float = 0.0) -> dict[str, Any]:
        """Queue a run that finishes an interrupted one (failed or cancelled): same suites, models and settings, asking only the cases it did not measure."""
        earlier = self.store.run(ref)
        self.runner.ensure_resumable(earlier)
        rs = normalise_settings(earlier["settings"])
        rows = [c for c in (self.store.find_contestant(cid) for cid in earlier["contestants"]) if c is not None]
        problems = self.preflight([c for c in rows if c["enabled"] and not c["missing"]], rs)
        self._refuse_hopeless(rows, problems)
        run = self.runner.resume(earlier["id"], source=source, caller=caller)
        self.submit_run(run["id"], wait_s)
        card = self.run_card(self.store.run(run["id"]))
        out = {"run": card, "continues": earlier["id"], "warnings": [text("warn_will_fail", name=p["name"], problem=p["problem"]) for p in problems]}
        if wait_s > 0 and card["state"] in ACTIVE_RUN_STATES:
            out["still_running"] = True
        return out

    def measure_new(self, *, suite: str = SMOKE_SUITE, include_stale: bool = True, source: str = "ui", caller: str = "") -> dict[str, Any]:
        """Run the quick suite on every enabled model that was never measured on it, or whose file changed since."""
        target = self.store.find_suite(suite)
        if target is None:
            raise GaltonError("not_found", "no_suite", ref=suite)
        measured = {(r["contestant_id"], r["digest"]) for r in self.store.scoring_rows(suite_ids=[target["id"]])}
        measured_ids = {cid for cid, _ in measured}
        todo = []
        for c in self.store.contestants(enabled=True, include_missing=False, include_adhoc=False):
            if (c["remote"] and not c["remote_ok"]) or (c["meta"] or {}).get("not_served"):
                continue
            if c["id"] not in measured_ids or (include_stale and (c["id"], c["digest"]) not in measured and c["digest"]):
                todo.append(c)
        if not todo:
            return {"started": False, "note": text("measure_new_none", name=target["name"])}
        started = self.start_run(suites=[target["id"]], contestants=[c["id"] for c in todo], label=text("label_measure_new", n=len(todo)), source=source, caller=caller)
        return {"started": True, "models": [c["name"] for c in todo], **started}

    # ------------------------------------------------------------------ discarding a run
    def discard_run(self, ref: str, reason: str) -> dict[str, Any]:
        """Take the results of a finished run out of every statistic, ranking, route, comparison and regression check. The run stays in the history
        with its reason. ``restore_run`` undoes it."""
        run = self.store.run(ref)
        if run["state"] in ACTIVE_RUN_STATES:
            raise GaltonError("conflict", "run_not_finished", id=run["id"], state=run["state"])
        self.store.update_run(run["id"], discarded=True, discard_reason=reason.strip(), discarded_ts=self.clock())
        self.store.delete_run_notices(run["id"], ("regression", "improvement"))      # what the run said about a model is not true any more
        self.routes_changed()
        return {"run": self.run_card(self.store.run(run["id"])), "discarded": True, "results": self.db_count("results", "run_id", run["id"]),
                "routes_has_changes": self.routes_view()["diff"]["has_changes"], "note": text("run_discard_note")}

    def restore_run(self, ref: str) -> dict[str, Any]:
        run = self.store.run(ref)
        was = bool(run["discarded"])
        if was:
            self.store.update_run(run["id"], discarded=False, discard_reason="", discarded_ts=None)
            self.routes_changed()
            if run["state"] == "done":
                try:
                    self.watch.detect_regressions(run["id"])
                except Exception:  # noqa: BLE001
                    log.exception("regression check failed for %s", run["id"])
        return {"run": self.run_card(self.store.run(run["id"])), "discarded": False, "restored": was, "results": self.db_count("results", "run_id", run["id"]),
                "routes_has_changes": self.routes_view()["diff"]["has_changes"], "note": text("run_restore_note") if was else text("run_not_discarded_note")}

    def galton_run(self, models: list[str], *, suites: Optional[list[str]] = None, label: str = "", source: str = "assistant", caller: str = "",
                   wait_s: float = 0.0) -> dict[str, Any]:
        """Queue a run of the named models (the quick suite unless ``suites`` says otherwise). This is what another app asks after it
        published a model: a name Galton has not seen yet triggers one rediscovery (Ollama, servers, GGUF folders) before giving up.
        Names that still cannot be found are listed in ``not_found``; the run goes ahead with the ones that resolved."""
        found: list[dict[str, Any]] = []
        missing: list[str] = []
        refreshed = False
        for name in [str(m).strip() for m in models if str(m).strip()]:
            row = self.store.find_contestant(name)
            if row is None and not refreshed:
                refreshed = True
                try:
                    self.discovery.refresh()
                except Exception:  # noqa: BLE001 — discovery trouble is reported as "not found" below
                    log.exception("rediscovery before galton_run failed")
                row = self.store.find_contestant(name)
            if row is None:
                try:
                    row = self.runner.resolve_contestants([{"kind": "ollama", "model": name}])[0]
                except GaltonError:
                    row = None
            if row is None:
                missing.append(name)
            elif row["id"] not in [c["id"] for c in found]:
                found.append(row)
        if not found:
            raise GaltonError("not_found", "no_model_spec", ref=", ".join(missing) or "(none)")
        started = self.start_run(suites=suites or [SMOKE_SUITE], contestants=[c["id"] for c in found], label=label or text("label_galton_run", n=len(found)),
                                 source=source, caller=caller, wait_s=wait_s)
        return {**started, "models": [c["name"] for c in found], "not_found": missing}

    # ------------------------------------------------------------------ routes
    def routes_changed(self) -> None:
        self._routes_cache = (None, {})

    def routes_view(self) -> dict[str, Any]:
        """``Routes.get`` with a one-minute cache keyed on the data it depends on, so the dashboard can poll it."""
        row = self.db.one("SELECT COALESCE(MAX(id), 0) AS m, COUNT(*) AS n FROM results")
        key = (row["m"], row["n"], (self.db.one("SELECT COALESCE(MAX(updated_ts), 0) FROM contestants") or [0])[0], str(self.settings.get("routes.policy")), int(self.clock() // 60),
               self.routes.path().stat().st_mtime if self.routes.path().exists() else 0)
        if self._routes_cache[0] == key:
            return self._routes_cache[1]
        value = self.routes.get()
        self._routes_cache = (key, value)
        return value

    # ------------------------------------------------------------------ dashboard, status, overview
    def counts(self) -> dict[str, int]:
        return self.store.counts()

    def routes_summary(self) -> list[dict[str, Any]]:
        view = self.routes_view()
        proposed, published = view["proposed"]["tasks"], ((view["published"] or {}).get("tasks") or {})
        out = []
        for category in TASKS:
            detail = view["detail"].get(category, {})
            top = (proposed.get(category) or {}).get("prefer") or []
            ranked = detail.get("ranked") or []
            out.append({"category": category, "winner": top[0] if top else None, "alternatives": top[1:], "explain": (proposed.get(category) or {}).get("explain", ""),
                        "measured_ts": ranked[0]["last_ts"] if ranked else None,
                        "candidates": detail.get("candidates", 0), "excluded": detail.get("excluded", []), "published": bool(published.get(category)),
                        "published_name": ((published.get(category) or {}).get("prefer") or [{}])[0].get("names", [None])[0] if published.get(category) else None})
        return out

    def attention(self) -> dict[str, Any]:
        rows = self.contestant_cards(self.store.contestants(enabled=True, include_missing=False, include_adhoc=False))
        slim = lambda c: {"id": c["id"], "name": c["name"], "kind": c["kind"]}  # noqa: E731
        return {"never_measured": [slim(c) for c in rows if c["never_measured"]], "stale": [slim(c) for c in rows if c["stale"]],
                "missing": [{"id": c["id"], "name": c["name"]} for c in self.store.contestants() if c["missing"] and not c["adhoc"]]}

    def dashboard(self) -> dict[str, Any]:
        runs = self.store.runs(limit=12)
        active = next((r for r in runs if r["state"] in ("running", "waiting_gpu", "waiting_server")), None)
        queued = [r for r in runs if r["state"] == "queued"]
        return {"now": self.clock(), "counts": self.counts(), "demo": self.config.fake, "offline": self.config.offline, "routes": self.routes_summary(),
                "routes_diff": self.routes_view()["diff"], "routes_path": str(self.routes.path()), "running": self.run_card(active) if active else None,
                "queued": [self.run_card(r) for r in queued[:5]], "recent_runs": [self.run_card(r) for r in runs if r not in queued and r is not active][:6],
                "notices": self.store.notices(limit=30), "attention": self.attention(), "gpus": self.gpus.state(), "scheduler": self.scheduler.status(),
                "watch": {"last_tick": self.watch.last_tick, "decision": self.watch.last_decision, "enabled": bool(self.settings.get("watch.enabled")),
                          "auto_smoke": bool(self.settings.get("watch.auto_smoke"))}, "last_refresh_ts": self.last_refresh.get("ts")}

    def mark_visit(self) -> dict[str, Any]:
        return {"marked_seen": self.store.mark_notices_seen()}

    def status(self) -> dict[str, Any]:
        binary: dict[str, Any]
        try:
            binary = {"found": True, "path": self.launcher.binary()} if hasattr(self.launcher, "binary") else {"found": True, "path": "(demo launcher)"}
        except GaltonError as exc:
            binary = {"found": False, "detail": exc.message}
        return {"service": SERVICE, "version": __version__, "data_dir": str(self.config.data_dir), "uptime_s": int(time.time() - self.started_at), "counts": self.counts(),
                "scheduler": self.scheduler.status(), "watch": {"last_tick": self.watch.last_tick, "decision": self.watch.last_decision}, "gpus": self.gpus.state(),
                "llama_server": binary, "last_refresh": {k: self.last_refresh.get(k) for k in ("ts", "sources", "errors") if k in self.last_refresh},
                "routes_path": str(self.routes.path()), "suites_synced": self.sync_info, "demo": self.config.fake, "offline": self.config.offline,
                "judge": str(self.settings.get("judge.contestant") or ""), "active_run": self.runner.active_run, "recent_activity": self.store.activity(limit=8)}

    def overview(self) -> dict[str, Any]:
        d = self.dashboard()
        attention = d["attention"]
        gpus = [{"index": g["index"], "name": g["name"], "free_mb": g["free_mb"], "total_mb": g["total_mb"], "allowed": g["allowed"], "reserved": g["reserved"]} for g in d["gpus"]["gpus"]]
        table = {r["category"]: {"model": (r["winner"]["names"] or ["?"])[0], "score": r["winner"]["score"], "ci": r["winner"]["ci"], "n": r["winner"]["n"],
                                 "tok_s": r["winner"]["tok_s"]} for r in d["routes"] if r["winner"]}
        steps = []
        if attention["never_measured"]:
            steps.append(f"{len(attention['never_measured'])} model(s) were never measured: measure_new runs the quick suite on them.")
        if attention["stale"]:
            steps.append(f"{len(attention['stale'])} model(s) changed since they were measured: measure_new re-measures them.")
        if d["routes_diff"]["has_changes"]:
            steps.append("The measured table differs from the published one: routes_get shows the difference, routes_publish applies it.")
        if not d["counts"]["models"]:
            steps.append("No models yet: models_refresh looks for Ollama models, llama-server and GGUF files.")
        return {"counts": d["counts"], "demo": d["demo"], "routes": table, "routes_published": bool(self.routes.published()), "routes_has_changes": d["routes_diff"]["has_changes"],
                "running": ({"id": d["running"]["id"], "label": d["running"]["label"], "state": d["running"]["state"], "progress": d["running"]["progress"]} if d["running"] else None),
                "queued": len(d["queued"]), "never_measured": attention["never_measured"], "stale": attention["stale"], "missing": attention["missing"],
                "notices": [{k: n[k] for k in ("id", "ts", "kind", "severity", "title", "body")} for n in d["notices"] if not n["seen"]][:10], "gpus": gpus,
                "allowed_gpus": d["gpus"]["allowed"], "scheduler": {"running": d["scheduler"]["running"], "paused": d["scheduler"]["paused"]}, "next_steps": steps}
