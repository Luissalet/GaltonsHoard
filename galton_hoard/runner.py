"""Runs: ask each contestant every case of the chosen suites, check each answer as soon as it arrives, record everything.

One run at a time (the scheduler's GPU lane calls ``execute``). A contestant is a ``session``: a server that is already up (probed, and waited
for when it is busy) or a llama-server that Galton starts itself on an allowed GPU under a lease and stops afterwards. Checks run in a thread
pool so the next question is asked while the last answer is still being checked. Results are written one by one, so the UI shows progress
and a cancelled run keeps what it measured.
"""

from __future__ import annotations

import logging
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterator, Optional

import httpx

from . import adhoc, cpu as cpu_lib, gguf_meta, identity, placement, suites as suite_lib
from .backends import Backend, Cancelled, ChatRequest, Completion
from .checkers import CheckContext, ModelOutput, run_checker
from .errors import GaltonError
from .messages import text
from .gpus import GpuManager
from .hoard_link import reasoning
from .identity import IdentityGuard
from .idle import LlamaGuard, OllamaGuard, wait_until_idle
from .judging import has_judge, make_ask, pending_reply
from .servers import LaunchSpec, probe_openai, slots_busy

log = logging.getLogger("galton.runner")

RUN_DEFAULTS: dict[str, Any] = {"temperature": 0.0, "top_p": None, "max_tokens": None, "effort": None, "repeats": 1, "seed": None, "context": None, "timeout_s": None, "wait_s": None, "device": "auto"}
DEVICES = ("auto", "gpu", "cpu")
TERMINAL = ("done", "failed", "cancelled")
#: extra seconds of timeout per token of thinking allowance (a 27B model on one GPU thinks at 10-20 tokens per second; this leaves room for the slow end)
SECONDS_PER_REASONING_TOKEN = 0.1


def normalise_settings(raw: Optional[dict[str, Any]]) -> dict[str, Any]:
    """Validate the per-run settings and fill the defaults."""
    raw = dict(raw or {})
    unknown = [k for k in raw if k not in RUN_DEFAULTS]
    if unknown:
        raise GaltonError("invalid", "run_unknown_settings", names=unknown, options=list(RUN_DEFAULTS))
    out = {**RUN_DEFAULTS, **{k: v for k, v in raw.items() if v is not None}}

    def number(key: str, low: float, high: float, integer: bool = False) -> None:
        if out[key] is None:
            return
        try:
            value = float(out[key])
        except (TypeError, ValueError) as exc:
            raise GaltonError("invalid", "setting_number", setting=key) from exc
        if not low <= value <= high:
            raise GaltonError("invalid", "setting_range", setting=key, low=f"{low:g}", high=f"{high:g}")
        out[key] = int(value) if integer else value

    number("temperature", 0, 2)
    number("top_p", 0.01, 1)
    number("max_tokens", 16, 65_536, True)
    number("repeats", 1, 10, True)
    number("seed", 0, 2 ** 31 - 1, True)
    number("context", 512, 1_048_576, True)
    number("timeout_s", 5, 3600)
    number("wait_s", 0, 7200)
    out["device"] = str(out["device"]).strip().lower() or "auto"
    if out["device"] not in DEVICES:
        raise GaltonError("invalid", "setting_choice", setting="device", options=list(DEVICES))
    if out["effort"] is not None:
        level = reasoning.normalize(out["effort"])
        if level is None and str(out["effort"]).lower() != "auto":
            raise GaltonError("invalid", "setting_choice", setting="effort", options=list(reasoning.LEVELS))
        out["effort"] = level
    return out


@dataclass
class Session:
    """A contestant ready to be asked: how to reach it and what was measured while getting there."""

    backend: Backend
    runs_on: str
    url: str = ""
    context: Optional[int] = None
    load_ms: Optional[float] = None
    vram_mb: Optional[int] = None
    vram_method: str = ""
    gpus: list[int] = field(default_factory=list)
    device: str = ""                    # cpu: a llama-server of our own on the processor (its speed is not a GPU speed); gpu: our own on GPUs; "": a shared server
    warnings: list[str] = field(default_factory=list)
    spill: Callable[[], str] = lambda: ""
    reasons: Optional[bool] = None      # does the model think before it answers? None while unknown
    reasons_how: str = ""               # how that is known: props (chat template), show (Ollama capabilities), stored, answer (reasoning content arrived)
    guard: Optional[Any] = None         # a shared server: says whether somebody else is using it (see ``idle.py``); None for a server of our own
    identity: Optional[IdentityGuard] = None   # a shared server: keeps checking that it still serves the model the contestant stands for (see ``identity.py``)


_endpoint = identity.endpoint     # ``host:port`` of an address, every spelling of this computer as one host


def _find_flag(detail: Any, flag: str) -> bool:
    if isinstance(detail, dict):
        return bool(detail.get(flag)) or any(_find_flag(v, flag) for v in detail.values())
    if isinstance(detail, list):
        return any(_find_flag(v, flag) for v in detail)
    return False


class Runner:
    def __init__(self, store: Any, settings: Any, gpus: GpuManager, launcher: Any, backend_factory: Callable[..., Backend], *, images_dir: Path,
                 clock: Callable[[], float] = time.time, sleep: Callable[[float], None] = time.sleep, family_call: Optional[Callable[..., dict[str, Any]]] = None,
                 emit: Optional[Callable[[str, dict[str, Any]], None]] = None, client_factory: Optional[Callable[[], httpx.Client]] = None,
                 meta_reader: Callable[[Any], dict[str, Any]] = gguf_meta.read_meta, digest_fn: Callable[[Any], str] = gguf_meta.file_digest,
                 cpu_threads: Optional[Callable[[], int]] = None, ram_free_mb: Optional[Callable[[], Optional[int]]] = None):
        self.store, self.settings, self.gpus, self.launcher, self.backend_factory = store, settings, gpus, launcher, backend_factory
        self.images_dir, self.clock, self.sleep, self.family_call = images_dir, clock, sleep, family_call
        self.emit = emit or (lambda *_: None)
        self.client_factory = client_factory or (lambda: httpx.Client(trust_env=False, timeout=5.0))
        self.meta_reader, self.digest_fn = meta_reader, digest_fn
        self.cpu_threads = cpu_threads or (lambda: cpu_lib.cpu_threads())       # looked up when asked, so tests can replace the module's functions
        self.ram_free_mb = ram_free_mb or (lambda: cpu_lib.ram_free_mb())
        self._cancel: dict[str, threading.Event] = {}
        self._lock = threading.Lock()
        self.active_run: Optional[str] = None
        self.active_info: dict[str, Any] = {}

    # ------------------------------------------------------------------ creating a run
    def resolve_contestants(self, items: list[Any]) -> list[dict[str, Any]]:
        return adhoc.resolve_many(self.store, self.settings, items, meta_reader=self.meta_reader, digest_fn=self.digest_fn)

    def resolve_suites(self, refs: list[str]) -> list[dict[str, Any]]:
        out = []
        for ref in refs:
            suite = self.store.find_suite(ref)
            if suite is None:
                raise GaltonError("not_found", "no_suite", ref=ref)
            if self.store.count_cases(suite["id"]) == 0:
                raise GaltonError("invalid", "suite_empty", name=suite["name"])
            if suite["id"] not in [s["id"] for s in out]:
                out.append(suite)
        if not out:
            raise GaltonError("invalid", "no_suite_chosen")
        return out

    def plan_cases(self, contestant: dict[str, Any], suites: list[dict[str, Any]], settings: dict[str, Any]) -> tuple[list[tuple[dict, dict, int]], list[str]]:
        """``[(suite, case, repeat)]`` for this contestant, and notes about cases left out (vision cases for a model without eyes)."""
        planned, notes = [], []
        dropped = 0
        for suite in suites:
            for case in self.store.cases(suite["id"]):
                if suite_lib.needs_vision(case) and not contestant["vision"]:
                    dropped += 1
                    continue
                for repeat in range(int(settings["repeats"])):
                    planned.append((suite, case, repeat))
        if dropped:
            notes.append(text("note_vision_dropped", n=dropped, name=contestant["name"]))
        return planned, notes

    def create(self, *, suites: list[str], contestants: list[Any], settings: Optional[dict[str, Any]] = None, label: str = "", source: str = "ui", caller: str = "") -> dict[str, Any]:
        rs = normalise_settings(settings)
        suite_rows = self.resolve_suites(suites)
        rows = self.resolve_contestants(contestants)
        if not rows:
            raise GaltonError("invalid", "no_model_chosen")
        for c in rows:
            if not c["enabled"]:
                raise GaltonError("invalid", "model_disabled", name=c["name"])
            if c["missing"]:
                raise GaltonError("not_found", "model_gone", name=c["name"])
            self.ensure_served(c)
        run = self.store.create_run(label=label or ", ".join(s["name"] for s in suite_rows)[:80], state="queued", suites=[s["id"] for s in suite_rows],
                                    contestants=[c["id"] for c in rows], settings=rs, source=source, caller=caller)
        for c in rows:
            planned, notes = self.plan_cases(c, suite_rows, rs)
            self.store.upsert_run_contestant(run["id"], c["id"], state="queued", total=len(planned), done=0, digest=c["digest"], warnings=notes)
        return self.store.run(run["id"])

    def ensure_served(self, c: dict[str, Any]) -> None:
        """A server contestant marked as not served (its address serves another model) cannot be measured. The mark is checked against the server
        first: when it serves the model again the mark goes and the contestant is fine."""
        flag = (c.get("meta") or {}).get("not_served")
        if c["kind"] != "server" or not flag:
            return
        served = identity.look(self.client_factory, c["api"] or "openai", c["url"])
        label = flag.get("model") or "?"
        if served is not None:
            verdict, label = identity.judge(c, served)
            if verdict != "different":
                identity.clear_not_served(self.store, c)
                return
        raise GaltonError("unavailable", "model_not_served", name=c["name"], url=c["url"], model=label)

    # ------------------------------------------------------------------ cancel
    def _event(self, run_id: str) -> threading.Event:
        with self._lock:
            return self._cancel.setdefault(run_id, threading.Event())

    def cancel(self, run_id: str) -> dict[str, Any]:
        run = self.store.run(run_id)
        if run["state"] in TERMINAL:
            return {"run": run_id, "state": run["state"], "cancelled": False, "note": text("run_finished")}
        self._event(run_id).set()
        self.store.update_run(run_id, cancel=True)
        if run["state"] == "queued":
            self.store.update_run(run_id, state="cancelled", finished_ts=self.clock())
            for rc in self.store.run_contestants(run_id):
                self.store.upsert_run_contestant(run_id, rc["contestant_id"], state="cancelled")
            return {"run": run_id, "state": "cancelled", "cancelled": True}
        return {"run": run_id, "state": run["state"], "cancelled": True, "note": self._stopping_note(run)}

    def _stopping_note(self, run: dict[str, Any]) -> Any:
        """What cancelling does, said for the kinds of contestant the run still has to go through: only a llama-server that Galton started is
        stopped; a shared server (somebody else's llama-server, Ollama, any endpoint) is never stopped, killed or told to unload anything."""
        states = {rc["contestant_id"]: rc["state"] for rc in self.store.run_contestants(run["id"])}
        own, shared = False, []
        for cid in run["contestants"]:
            if states.get(cid) in TERMINAL:
                continue
            c = self.store.find_contestant(cid)
            if c is None:
                continue
            if c["kind"] == "gguf":
                own = True
            elif c["url"]:
                shared.append(c["url"])
        urls = ", ".join(dict.fromkeys(shared))
        if own and shared:
            return text("run_stopping_both", urls=urls)
        if shared:
            return text("run_stopping_shared", urls=urls)
        return text("run_stopping_own") if own else text("run_stopping_none")

    def is_cancelled(self, run_id: str) -> bool:
        return self._event(run_id).is_set()

    # ------------------------------------------------------------------ sessions
    def _wait_s(self, rs: dict[str, Any], key: str) -> float:
        """How long to wait for a GPU: the run's own ``wait_s`` (the watch passes 0: it never queues) or the setting."""
        return float(rs["wait_s"]) if rs.get("wait_s") is not None else float(self.settings.get(key))

    def _idle_limit(self, rs: dict[str, Any]) -> Optional[float]:
        """The longest a run waits for a shared server somebody else is using: the run's own ``wait_s`` (0: do not wait at all) or the setting
        ``runner.wait_idle_max_s`` (0: for ever, ``None`` here)."""
        if rs.get("wait_s") is not None:
            return float(rs["wait_s"])
        limit = float(self.settings.get("runner.wait_idle_max_s"))
        return limit if limit > 0 else None

    def _wait_for_server(self, busy: Callable[[], Optional[bool]], rs: dict[str, Any], cancel: Callable[[], bool], note: Callable[[str], None], url: str,
                         busy_now: bool = False) -> float:
        """Wait while somebody else uses the shared server at ``url``, telling the run (``waiting_server``) while it does."""
        return wait_until_idle(busy, grace_s=float(self.settings.get("runner.idle_grace_s")), max_s=self._idle_limit(rs), cancel=cancel, clock=self.clock,
                               sleep=self.sleep, on_wait=lambda waited: note("waiting_server"), url=url, busy_now=busy_now)

    def yield_to_others(self, session: Session, rs: dict[str, Any], cancel: Callable[[], bool], note: Callable[[str], None]) -> float:
        """Before a question: when the shared server turned out to be busy with somebody else's request, wait for it the same way as at the start,
        so a measurement is never interleaved with a chat. Returns the seconds waited."""
        guard = session.guard
        if guard is None or not guard.others_busy():
            return 0.0
        if guard.settle_s:           # a slot of our own last answer may take a moment to free: look once more before taking it for somebody else's
            self.sleep(guard.settle_s)
            if not guard.others_busy():
                return 0.0
        try:
            return self._wait_for_server(guard.others_busy, rs, cancel, note, session.url, busy_now=True)
        finally:
            note("running")

    @contextmanager
    def session(self, contestant: dict[str, Any], rs: dict[str, Any], cancel: Callable[[], bool], note: Callable[[str], None] = lambda s: None) -> Iterator[Session]:
        if contestant["remote"] and not contestant["remote_ok"]:
            raise GaltonError("remote_not_allowed", "remote_not_allowed", name=contestant["name"])
        if contestant["kind"] == "gguf":
            with self._gguf_session(contestant, rs, cancel, note) as s:
                yield s
        else:
            with self._server_session(contestant, rs, cancel, note) as s:
                yield s

    @contextmanager
    def _gguf_session(self, c: dict[str, Any], rs: dict[str, Any], cancel: Callable[[], bool], note: Callable[[str], None]) -> Iterator[Session]:
        if not c["meta"].get("demo") and (not c["path"] or not Path(c["path"]).is_file()):
            raise GaltonError("not_found", "model_file_gone", path=c["path"])
        info = placement.gguf_info(self.store, c, self.meta_reader)
        context = placement.choose_context(int(rs["context"] or self.settings.get("runner.context")), info)
        need = placement.estimate_mb(self.store, c, context, self.meta_reader)
        device = rs.get("device") or "auto"
        cpu_ok = placement.cpu_eligible(self.settings, c, device)
        grant = None
        if device != "cpu":
            note("waiting_gpu")
            # A small file that may run on the CPU does not queue for a GPU: it waits only as long as the run asked for (``wait_s``, default none).
            wait = (float(rs["wait_s"]) if rs.get("wait_s") is not None else 0.0) if cpu_ok else self._wait_s(rs, "runner.gpu_wait_s")
            try:
                grant = self.gpus.acquire(need, f"galton: {c['name']}"[:100], wait_s=wait, cancel=cancel, on_wait=lambda msg: log.info("%s: %s", c["name"], msg))
            except GaltonError as exc:
                if not (cpu_ok and exc.code == "no_gpu" and self._ram_fits(need) is None):
                    raise
                log.info("%s: no allowed GPU is free (%s); running on the CPU", c["name"], exc.message)
        if grant is None:
            problem = self._ram_fits(need)
            if problem is not None:
                raise problem
            with self._cpu_session(c, context, need, cancel, note) as session:
                yield session
            return
        try:
            note("running")
            before = self.gpus.used_mb(grant.gpus)
            handle = self.launcher.start(LaunchSpec(model_path=c["path"], name=c["name"], context=context, mmproj_path=c["mmproj"] or None, grant=grant), cancel=cancel)
            try:
                delta = self.gpus.used_mb(grant.gpus) - before
                vram, method = (delta, "nvidia-smi delta") if delta > 0 else (need, "estimate")
                backend = self.backend_factory("openai", handle.url, handle.alias)
                warnings = list(grant.warnings)
                if grant.tensor_split:
                    warnings.append(text("warn_split", gpus=grant.gpus, split=grant.tensor_split))
                reasons, how = self.detect_reasoning(c, url=handle.url, api="openai")
                try:
                    yield Session(backend=backend, runs_on=text("runs_on_own", gpus=",".join(str(g) for g in grant.gpus)), url=handle.url, context=context, load_ms=handle.load_ms,
                                  vram_mb=vram, vram_method=method, gpus=list(grant.gpus), device="gpu", warnings=warnings, reasons=reasons, reasons_how=how)
                finally:
                    backend.close()
            finally:
                handle.stop()
        finally:
            grant.release()

    def _ram_fits(self, need_mb: int) -> Optional[GaltonError]:
        """``None`` when the memory a CPU run needs is free (or the system does not say), else the error to raise."""
        free = self.ram_free_mb()
        if free is not None and need_mb > free:
            return GaltonError("busy", "cpu_no_ram", need_gb=f"{need_mb / 1024:.1f}", free_gb=f"{free / 1024:.1f}")
        return None

    @contextmanager
    def _cpu_session(self, c: dict[str, Any], context: int, need: int, cancel: Callable[[], bool], note: Callable[[str], None]) -> Iterator[Session]:
        """A llama-server of our own on the processor: no layers on a GPU, no GPU visible to it, no lease."""
        threads = int(self.cpu_threads())
        note("running")
        handle = self.launcher.start(LaunchSpec(model_path=c["path"], name=c["name"], context=context, mmproj_path=c["mmproj"] or None, grant=None, cpu=True, threads=threads),
                                     cancel=cancel)
        try:
            backend = self.backend_factory("openai", handle.url, handle.alias)
            reasons, how = self.detect_reasoning(c, url=handle.url, api="openai")
            try:
                yield Session(backend=backend, runs_on=text("runs_on_cpu", threads=threads), url=handle.url, context=context, load_ms=handle.load_ms, vram_mb=None, vram_method="cpu",
                              gpus=[], device="cpu", warnings=[text("warn_cpu", threads=threads)], reasons=reasons, reasons_how=how)
            finally:
                backend.close()
        finally:
            handle.stop()

    @contextmanager
    def _server_session(self, c: dict[str, Any], rs: dict[str, Any], cancel: Callable[[], bool], note: Callable[[str], None]) -> Iterator[Session]:
        api, url, model = c["api"] or "openai", c["url"], c["model"] or c["name"]
        warnings: list[str] = []
        grant = None
        loaded_by_us = False
        guard_id: Optional[IdentityGuard] = None
        with self.client_factory() as client:
            if api == "openai":
                probe = probe_openai(client, url)
                if not probe.get("up"):
                    raise GaltonError("unavailable", "server_no_answer", url=url)
                if probe.get("busy"):
                    self._wait_for_server(lambda: slots_busy(client, url), rs, cancel, note, url, busy_now=True)
                    probe = probe_openai(client, url)
                guard_id = self._identity_guard(c, identity.served_from_probe(probe, url), "openai", url)
                note("running")
                runs_on = text("runs_on_server", url=url)
                context = probe.get("context") or c.get("context")
                reasons, how = (probe["reasoning"], "props") if probe.get("reasoning") is not None else self._stored_reasoning(c)
                guard = LlamaGuard(self.client_factory, url) if probe.get("busy") is not None else None
                spill = lambda: ""  # noqa: E731
            else:
                served = identity.look_ollama(client, url)
                guard_id = self._identity_guard(c, served, "ollama", url) if served is not None else None
                resident = self._ollama_resident(client, url, model)
                if resident is None:
                    if not self.settings.get("runner.allow_ollama_load"):
                        raise GaltonError("unavailable", "ollama_not_loaded", name=c["name"])
                    need = int((c.get("size_bytes") or 0) * gguf_meta.FILE_FACTOR / (1024 * 1024)) + gguf_meta.HEADROOM_MB
                    note("waiting_gpu")
                    grant = self.gpus.acquire(need, f"galton: ollama {c['name']}"[:100], wait_s=self._wait_s(rs, "runner.gpu_wait_s"), cancel=cancel)
                    loaded_by_us = True
                    warnings.append(text("warn_ollama_gpu"))
                note("running")
                runs_on = text("runs_on_ollama_loaded", url=url) if loaded_by_us else text("runs_on_ollama", url=url)
                context = c.get("context")
                reasons, how = self.detect_reasoning(c, url=url, api="ollama", model=model, client=client)
                guard = OllamaGuard(self.client_factory, url)

                def spill() -> str:  # noqa: F811
                    with self.client_factory() as probe_client:
                        entry = self._ollama_resident(probe_client, url, model)
                    if entry and entry.get("size") and entry.get("size_vram") is not None and entry["size_vram"] < entry["size"] * 0.98:
                        pct = round(100 * (1 - entry["size_vram"] / entry["size"]))
                        return text("warn_spill", pct=pct)
                    return ""
            backend = self.backend_factory(api, url, model)
        try:
            yield Session(backend=backend, runs_on=runs_on, url=url, context=context, warnings=warnings, gpus=list(grant.gpus) if grant else [], spill=spill,
                          reasons=reasons, reasons_how=how, guard=guard, identity=guard_id)
        finally:
            backend.close()
            if loaded_by_us:
                try:
                    with self.client_factory() as client:
                        client.post(url.rstrip("/") + "/api/generate", json={"model": model, "keep_alive": 0}, timeout=15.0)
                except httpx.HTTPError:
                    log.warning("could not unload %s from Ollama", model)
            if grant is not None:
                grant.release()

    def _identity_guard(self, c: dict[str, Any], served: identity.Served, api: str, url: str) -> Optional[IdentityGuard]:
        """Before a shared server is used: does it serve what the contestant stands for? Raises ``server_changed`` when it serves another model (and
        marks the contestant as not served). Returns the guard that keeps checking while the run goes on, or ``None`` when the server says too little
        about itself for anything to be checked."""
        if identity.needs_pin(c):
            c = identity.pin(self.store, c, served)           # a model name given by hand: what the server serves the first time is what it stands for
        verdict, label = identity.judge(c, served)
        if verdict == "different":
            identity.mark_not_served(self.store, c, label, url, self.clock())
            raise identity.changed(url, label, c["name"])
        if verdict != "same":
            return None
        identity.clear_not_served(self.store, c)
        return IdentityGuard(url=url, expected_label=c["name"], look=lambda: identity.look(self.client_factory, api, url), verdict=lambda now: identity.judge(c, now),
                             clock=self.clock, on_change=lambda label: identity.mark_not_served(self.store, c, label, url, self.clock()))

    # ------------------------------------------------------------------ does the model think before it answers?
    @staticmethod
    def _stored_reasoning(c: dict[str, Any]) -> tuple[Optional[bool], str]:
        """What an earlier look at this model found out (discovery, or an answer that came with reasoning content)."""
        known = (c.get("meta") or {}).get("reasons")
        return (bool(known), str((c.get("meta") or {}).get("reasons_how") or "stored")) if known is not None else (None, "")

    def detect_reasoning(self, c: dict[str, Any], *, url: str, api: str, model: str = "", client: Optional[httpx.Client] = None) -> tuple[Optional[bool], str]:
        """``(reasons, how)``: llama-server says it in the chat template of ``/props`` (``enable_thinking`` or ``<think>``), Ollama in the
        capabilities of ``/api/show`` (``thinking``). Falls back to what was stored for this model; ``(None, "")`` while nothing is known."""
        try:
            if c["meta"].get("demo"):
                return self._stored_reasoning(c)
            if api == "ollama":
                found = self._ollama_capabilities(client, url, model or c["model"] or c["name"]) if client is not None else None
                caps = found if found is not None else list((c.get("meta") or {}).get("capabilities") or [])
                if caps:
                    return "thinking" in caps, "show"
            elif client is not None:
                probe = probe_openai(client, url)
                if probe.get("reasoning") is not None:
                    return bool(probe["reasoning"]), "props"
            else:
                with self.client_factory() as own:
                    probe = probe_openai(own, url)
                if probe.get("reasoning") is not None:
                    return bool(probe["reasoning"]), "props"
        except (httpx.HTTPError, ValueError, OSError):
            log.info("could not ask %s whether %s reasons", url, c["name"])
        return self._stored_reasoning(c)

    @staticmethod
    def _ollama_capabilities(client: httpx.Client, url: str, model: str) -> Optional[list[str]]:
        try:
            response = client.post(url.rstrip("/") + "/api/show", json={"model": model}, timeout=10.0)
            if response.status_code == 200 and isinstance(response.json(), dict):
                return [str(x) for x in response.json().get("capabilities") or []]
        except (httpx.HTTPError, ValueError):
            pass
        return None

    def learn_reasoning(self, contestant: dict[str, Any], session: Session) -> None:
        """An answer arrived with reasoning content: this model thinks, whatever its template says. Remembered for the next run."""
        session.reasons, session.reasons_how = True, "answer"
        try:
            fresh = self.store.contestant(contestant["id"])
            if not (fresh["meta"].get("reasons") and fresh["meta"].get("reasons_how") in ("answer", "props", "show")):
                self.store.update_contestant(fresh["id"], meta={**fresh["meta"], "reasons": True, "reasons_how": "answer"})
        except GaltonError:
            pass

    def _reasoning_warning(self, rs: dict[str, Any], session: Session) -> list[Any]:
        extra = self.reasoning_allowance(rs, session)
        return [text("warn_reasoning_budget", tokens=extra)] if extra else []

    def reasoning_allowance(self, rs: dict[str, Any], session: Session) -> int:
        """Tokens added to the answer budget so that thinking does not use it up: ``runner.reasoning_tokens`` (more when the run's effort asks for
        a bigger thinking budget), only for a model that reasons and only when the run's effort is not ``off``. 0 switches it off."""
        base = int(self.settings.get("runner.reasoning_tokens") or 0)
        if base <= 0 or not session.reasons or rs.get("effort") == "off":
            return 0
        return max(base, reasoning.budget_for(rs.get("effort")))

    @staticmethod
    def _ollama_resident(client: httpx.Client, url: str, model: str) -> Optional[dict[str, Any]]:
        try:
            response = client.get(url.rstrip("/") + "/api/ps")
            response.raise_for_status()
            for entry in response.json().get("models") or []:
                if model in (entry.get("name"), entry.get("model")):
                    return entry
        except (httpx.HTTPError, ValueError) as exc:
            raise GaltonError("unavailable", "ollama_no_answer", url=url) from exc
        return None

    # ------------------------------------------------------------------ judge wiring
    def _judge_for(self, contestant: dict[str, Any], session: Session, cancel: Callable[[], bool]) -> Optional[Callable[[dict[str, Any]], dict[str, Any]]]:
        """An inline judge when the judge model is reachable right now (it is this very model, or a server that is up); otherwise ``None``: grades wait."""
        ref = str(self.settings.get("judge.contestant") or "")
        judge = self.store.find_contestant(ref) if ref else None
        if judge is None or not judge["enabled"] or judge["missing"]:
            return None
        if judge["id"] == contestant["id"]:
            return make_ask(self.store, session.backend, judge, contestant["id"], cancel=cancel)
        if judge["kind"] == "server" and not (judge["remote"] and not judge["remote_ok"]):
            try:
                with self.client_factory() as client:
                    if judge["api"] == "ollama":
                        client.get(judge["url"].rstrip("/") + "/api/tags").raise_for_status()
                    elif not probe_openai(client, judge["url"]).get("up"):
                        return None
            except (httpx.HTTPError, ValueError):
                return None
            return make_ask(self.store, self.backend_factory(judge["api"] or "openai", judge["url"], judge["model"] or judge["name"]), judge, contestant["id"], cancel=cancel)
        return None

    def _judge_row(self) -> Optional[dict[str, Any]]:
        ref = str(self.settings.get("judge.contestant") or "")
        return self.store.find_contestant(ref) if ref else None

    @staticmethod
    def judge_shares_server(judge: dict[str, Any], contestant: dict[str, Any], session: Session) -> bool:
        """Is the judge the very model (or the very server) being measured? Its grading requests would then be queued behind, or run beside,
        the questions whose latency and first-token time are being recorded, so they wait until this contestant's cases are over."""
        if judge["id"] == contestant["id"]:
            return True
        if judge["kind"] != "server" or not session.url or not judge.get("url"):
            return False
        return _endpoint(judge["url"]) == _endpoint(session.url)

    # ------------------------------------------------------------------ checking one answer
    def check(self, case: dict[str, Any], completion: Completion, contestant_id: str, judge: Optional[Callable[[dict[str, Any]], dict[str, Any]]]) -> tuple[dict[str, Any], bool]:
        """``(verdict, judge_pending)`` for one answer."""
        output = ModelOutput(text=completion.text, reasoning=completion.reasoning, tool_calls=completion.tool_calls)
        pending = {"flag": False}
        needs_judge = has_judge(case["checker"])

        def ask(request: dict[str, Any]) -> dict[str, Any]:
            if judge is None:
                pending["flag"] = True
                return pending_reply(request)
            reply = judge(request)
            return reply

        prompt = case.get("prompt") or {}
        user_text = prompt.get("text") or " ".join(str(m.get("content", "")) for m in prompt.get("messages", []) if m.get("role") == "user")
        ctx = CheckContext(case={**case, "prompt_text": user_text}, allow_code=bool(self.settings.get("checks.allow_code_execution")),
                           code_timeout_s=float(self.settings.get("checks.code_timeout_s")), judge=ask if needs_judge else None, family_call=self.family_call, contestant_id=contestant_id)
        return run_checker(case["checker"], output, ctx), pending["flag"]

    def _store_result(self, run: dict[str, Any], contestant: dict[str, Any], suite: dict[str, Any], case: dict[str, Any], repeat: int, completion: Completion,
                      judge: Optional[Callable[[dict[str, Any]], dict[str, Any]]], rc_counter: Callable[[], None], cpu: bool = False) -> None:
        """Check one answer and store it. ``cpu``: the model ran on the processor, so the stored speed is flagged and never counted as a GPU speed."""
        base = dict(run_id=run["id"], contestant_id=contestant["id"], suite_id=suite["id"], case_id=case["id"], category=suite["category"], repeat=repeat,
                    weight=float(case.get("weight") or 1.0), digest=contestant["digest"], cpu=cpu)
        timing = dict(latency_ms=completion.latency_ms, ttft_ms=completion.ttft_ms, prompt_tokens=completion.prompt_tokens, completion_tokens=completion.completion_tokens,
                      decode_tps=completion.decode_tps, prompt_tps=completion.prompt_tps)
        try:
            if completion.error:
                self.store.add_result(**base, **timing, output=completion.text, reasoning=completion.reasoning, score=0.0, passed=False, error=completion.error,
                                      detail={"notes": completion.notes})
                return
            verdict, pending = self.check(case, completion, contestant["id"], judge)
            detail = {**verdict.get("detail", {}), "finish_reason": completion.finish_reason}
            if completion.max_tokens_sent:
                detail["max_tokens"] = completion.max_tokens_sent
            if completion.notes:
                detail["notes"] = completion.notes
            if not completion.text.strip() and not completion.tool_calls:
                detail["empty_answer"] = True
            if completion.truncated:
                detail["truncated"] = True
            self_judged = _find_flag(verdict.get("detail"), "self_judged")
            self.store.add_result(**base, **timing, output=completion.text, reasoning=completion.reasoning, tool_calls=completion.tool_calls, score=verdict["score"],
                                  passed=bool(verdict["passed"]) and not pending, detail=detail, unavailable=bool(verdict.get("unavailable")) and not pending, judge_pending=pending,
                                  self_judged=self_judged, confidence=0.5 if self_judged else 1.0, truncated=completion.truncated)
        except Exception as exc:  # noqa: BLE001 — one bad case must not end the run
            log.exception("storing a result failed")
            self.store.add_result(**base, **timing, score=0.0, passed=False, error=text("internal_error", detail=f"{type(exc).__name__}: {exc}"))
        finally:
            rc_counter()

    # ------------------------------------------------------------------ running a contestant
    def _skip_reason(self, case: dict[str, Any], session: Session) -> str:
        if session.context:
            needed = suite_lib.rough_tokens(case)
            if needed > session.context:
                return text("skip_context", needed=needed, available=session.context)
        return ""

    def build_request(self, case: dict[str, Any], suite: dict[str, Any], rs: dict[str, Any], repeat: int, session: Session) -> ChatRequest:
        built = suite_lib.materialize_case(case, self.images_dir)
        answer = int(case.get("max_tokens") or rs["max_tokens"] or suite["max_tokens"] or 512)
        extra = self.reasoning_allowance(rs, session)
        seed = (int(rs["seed"]) + repeat) if rs["seed"] is not None else None
        timeout = float(rs["timeout_s"] or self.settings.get("runner.timeout_s")) + suite_lib.rough_tokens(case) / 200.0 + extra * SECONDS_PER_REASONING_TOKEN
        return ChatRequest(messages=built["messages"], images=built["images"], tools=case.get("tools") or [], temperature=float(rs["temperature"]), top_p=rs["top_p"],
                           max_tokens=answer + extra, seed=seed, effort=rs["effort"], timeout_s=timeout, reasoning_tokens=extra,
                           context=rs["context"] if (session.runs_on.startswith("ollama") and rs["context"]) else None)

    def ask(self, contestant: dict[str, Any], session: Session, case: dict[str, Any], suite: dict[str, Any], rs: dict[str, Any], repeat: int,
            cancel: Callable[[], bool]) -> Completion:
        """One question to the model. A model that reasons gets room for its thinking in ``max_tokens``; one that was not known to reason and
        shows it (reasoning content, and the answer cut before it started) is asked again once with that room, and remembered."""
        request = self.build_request(case, suite, rs, repeat, session)
        completion = session.backend.chat(request, cancel)
        completion.max_tokens_sent = request.max_tokens
        if completion.reasoning and not session.reasons:
            self.learn_reasoning(contestant, session)
        if completion.truncated and not request.reasoning_tokens and self.reasoning_allowance(rs, session):
            retry = self.build_request(case, suite, rs, repeat, session)
            again = session.backend.chat(retry, cancel)
            again.max_tokens_sent = retry.max_tokens
            again.notes = [*completion.notes, *again.notes, text("note_reasoning_budget_retry", tokens=retry.reasoning_tokens)]
            return again
        return completion

    def _run_contestant(self, run: dict[str, Any], contestant: dict[str, Any], suites: list[dict[str, Any]], rs: dict[str, Any], pool: ThreadPoolExecutor) -> str:
        run_id, cid = run["id"], contestant["id"]
        cancel = lambda: self.is_cancelled(run_id)  # noqa: E731
        planned, notes = self.plan_cases(contestant, suites, rs)
        lock = threading.Lock()
        progress = {"done": 0}
        self.store.upsert_run_contestant(run_id, cid, state="running", started_ts=self.clock(), total=len(planned), done=0, digest=contestant["digest"], warnings=notes, error="")

        def tick() -> None:
            with lock:  # the write stays inside the lock: two checks finishing together must not store the counts out of order
                progress["done"] += 1
                self.store.upsert_run_contestant(run_id, cid, done=progress["done"])

        wait = {"since": None, "total": 0.0}

        def settle_wait() -> None:
            """Add the wait that is going on (if any) to the total waited and stop the clock."""
            if wait["since"] is not None:
                wait["total"] += max(0.0, self.clock() - wait["since"])
                wait["since"] = None
                self.store.upsert_run_contestant(run_id, cid, wait_since=None, waited_s=wait["total"])

        def note_state(state: str) -> None:
            if state == "waiting_server":
                if wait["since"] is None:         # the state is written once; the page works out how long from the time it began
                    wait["since"] = self.clock()
                    self.store.upsert_run_contestant(run_id, cid, state="waiting_server", wait_since=wait["since"])
                    self.store.update_run(run_id, state="waiting_server")
                return
            settle_wait()
            self.store.upsert_run_contestant(run_id, cid, state=state)
            self.store.update_run(run_id, state="waiting_gpu" if state == "waiting_gpu" else "running")

        dropped = {"n": 0}
        shown = list(notes)                       # the warnings the contestant shows so far
        try:
            with self.session(contestant, rs, cancel, note_state) as session:
                self.store.upsert_run_contestant(run_id, cid, state="running", runs_on=session.runs_on, load_ms=session.load_ms, vram_mb=session.vram_mb, vram_method=session.vram_method,
                                                 gpus=session.gpus, device=session.device, context=session.context, warnings=[*notes, *session.warnings, *self._reasoning_warning(rs, session)])
                started_with = [*notes, *session.warnings, *self._reasoning_warning(rs, session)]
                shown[:] = started_with
                judge = self._judge_for(contestant, session, cancel)
                # A judge that is this very server is not asked while the contestant is being measured: its grades wait (as pending) and are
                # given right after the last case, so they cannot delay a question whose timing is recorded.
                deferred = judge if judge is not None and self.judge_shares_server(self._judge_row(), contestant, session) else None
                inline = None if deferred is not None else judge
                futures: list[Future] = []
                guard_id = session.identity
                held: list[tuple[dict[str, Any], dict[str, Any], int, Completion]] = []

                def store_now(suite: dict[str, Any], case: dict[str, Any], repeat: int, completion: Completion) -> None:
                    futures.append(pool.submit(self._store_result, run, contestant, suite, case, repeat, completion, inline, tick, session.device == "cpu"))

                def release() -> None:
                    while held:
                        store_now(*held.pop(0))

                def drop() -> None:
                    dropped["n"] += len(held)
                    held.clear()

                def settle() -> None:
                    """On a shared server an answer is stored only once a look at the server, made after the answer, shows it is still the model of
                    the contestant. This is that last look: what was answered since the previous one is stored, or (the server changed, or cannot be
                    looked at) dropped."""
                    if guard_id is None or not held:
                        return
                    state = "unknown"
                    try:
                        for attempt in range(3):
                            state = guard_id.check(force=True)
                            if state == "fresh":
                                break
                            self.sleep(1.0)
                    except GaltonError:
                        drop()
                        raise
                    if state != "fresh":
                        n = len(held)
                        drop()
                        raise GaltonError("unavailable", "server_unverified", url=session.url, n=n)
                    release()

                def settle_quietly() -> None:
                    try:
                        settle()
                    except GaltonError:
                        log.info("run %s: %s: the last results could not be checked against the server", run_id, contestant["name"])

                try:
                    for suite, case, repeat in planned:
                        if cancel():
                            break
                        self.active_info = {"run": run_id, "contestant": contestant["name"], "case": case["title"], "suite": suite["name"]}
                        reason = self._skip_reason(case, session)
                        if reason:
                            self.store.add_result(run_id=run_id, contestant_id=cid, suite_id=suite["id"], case_id=case["id"], category=suite["category"], repeat=repeat, skipped=True,
                                                  error=reason, weight=float(case.get("weight") or 1.0), digest=contestant["digest"])
                            tick()
                            continue
                        self.yield_to_others(session, rs, cancel, note_state)
                        if cancel():
                            break
                        if guard_id is not None and guard_id.check() == "fresh":
                            release()                      # the server is what it was, and was looked at after the answers held so far
                        completion = self.ask(contestant, session, case, suite, rs, repeat, cancel)
                        if session.guard is not None:
                            session.guard.mark_own()
                        if guard_id is None:
                            store_now(suite, case, repeat, completion)
                        else:
                            guard_id.answer(completion.served_model)
                            held.append((suite, case, repeat, completion))
                except GaltonError as exc:
                    if exc.key == "server_changed":
                        drop()                              # nothing answered since the last look can be told from the new model's answers
                    else:
                        settle_quietly()
                    raise
                except BaseException:
                    settle_quietly()
                    raise
                settle()
                for future in futures:
                    future.result()
                if deferred is not None and not cancel():
                    self.active_info = {"run": run_id, "contestant": contestant["name"], "case": "judge", "suite": ""}
                    for _ in self._grade_rows([r for r in self.store.pending_judge(run_id) if r["contestant_id"] == cid], lambda _cid: deferred, cancel):
                        pass
                spill = session.spill()
                final = [*notes, *session.warnings, *self._reasoning_warning(rs, session), *([spill] if spill else [])]
                if spill or final != started_with:
                    self.store.upsert_run_contestant(run_id, cid, spill=spill, warnings=final)
            if cancel():
                self.store.upsert_run_contestant(run_id, cid, state="cancelled", finished_ts=self.clock())
                return "cancelled"
            self.store.upsert_run_contestant(run_id, cid, state="done", finished_ts=self.clock())
            return "done"
        except Cancelled:
            self.store.upsert_run_contestant(run_id, cid, state="cancelled", finished_ts=self.clock())
            return "cancelled"
        except GaltonError as exc:
            if cancel():
                self.store.upsert_run_contestant(run_id, cid, state="cancelled", finished_ts=self.clock())
                return "cancelled"
            extra = {"warnings": [*shown, text("warn_results_dropped", n=dropped["n"])]} if dropped["n"] else {}
            self.store.upsert_run_contestant(run_id, cid, state="failed", error=exc.message, finished_ts=self.clock(), **extra)
            return "failed"
        except Exception as exc:  # noqa: BLE001
            log.exception("run %s: %s failed", run_id, contestant["name"])
            self.store.upsert_run_contestant(run_id, cid, state="failed", error=f"{type(exc).__name__}: {exc}", finished_ts=self.clock())
            return "failed"
        finally:
            settle_wait()

    # ------------------------------------------------------------------ executing a run
    def execute(self, run_id: str) -> dict[str, Any]:
        run = self.store.run(run_id)
        if run["state"] != "queued":
            return run
        if self.is_cancelled(run_id) or run["cancel"]:
            return self.store.update_run(run_id, state="cancelled", finished_ts=self.clock())
        rs = normalise_settings(run["settings"])
        suites = [self.store.suite(s) for s in run["suites"]]
        self.store.update_run(run_id, state="running", started_ts=self.clock())
        self.active_run = run_id
        outcomes: dict[str, str] = {}
        pool = ThreadPoolExecutor(max_workers=max(1, int(self.settings.get("runner.check_threads"))), thread_name_prefix="galton-check")
        try:
            for cid in run["contestants"]:
                if self.is_cancelled(run_id):
                    break
                contestant = self.store.contestant(cid)
                outcomes[cid] = self._run_contestant(run, contestant, suites, rs, pool)
            if not self.is_cancelled(run_id):
                self.judge_pending(run_id)
        finally:
            pool.shutdown(wait=True)
            self.active_run, self.active_info = None, {}
        for cid in run["contestants"]:
            if cid not in outcomes:
                self.store.upsert_run_contestant(run_id, cid, state="cancelled")
        summary = self.summarise(run_id)
        if self.is_cancelled(run_id):
            state, error = "cancelled", ""
        elif outcomes and all(o == "failed" for o in outcomes.values()):
            state = "failed"
            error = text("run_all_failed")
        else:
            state, error = "done", ""
        final = self.store.update_run(run_id, state=state, finished_ts=self.clock(), error=error, summary=summary)
        self.emit(f"galton.run.{'done' if state == 'done' else state}", {"run": run_id, "state": state, "label": final["label"], "source": final["source"]})
        with self._lock:
            self._cancel.pop(run_id, None)
        return final

    def summarise(self, run_id: str) -> dict[str, Any]:
        out: dict[str, Any] = {"contestants": {}, "pending_judge": len(self.store.pending_judge(run_id))}
        for rc in self.store.run_contestants(run_id):
            counts = self.store.count_results(run_id, rc["contestant_id"])
            rows = self.store.scoring_rows(contestant_ids=[rc["contestant_id"]], run_ids=[run_id])
            weight = sum(r["weight"] for r in rows)
            mean = sum(r["score"] * r["weight"] for r in rows) / weight if weight else None
            out["contestants"][rc["contestant_id"]] = {"state": rc["state"], "checked": len(rows), "passed": counts.get("passed", 0), "skipped": counts.get("skipped", 0),
                                                       "errors": counts.get("errors", 0), "truncated": counts.get("truncated", 0),
                                                       "mean_score": round(mean, 4) if mean is not None else None}
        return out

    # ------------------------------------------------------------------ the judge phase
    def judge_pending(self, run_id: Optional[str] = None) -> dict[str, Any]:
        """Grade the answers that waited for the judge. Starts the judge if it is a GGUF. Anything still ungradable stays pending."""
        if run_id:
            pending = self.store.pending_judge(run_id)
        else:
            pending = self.store.pending_judge()
        if not pending:
            return {"graded": 0, "pending": 0}
        ref = str(self.settings.get("judge.contestant") or "")
        judge = self.store.find_contestant(ref) if ref else None
        if judge is None:
            return {"graded": 0, "pending": len(pending), "reason": text("judge_none")}
        if judge["missing"] or not judge["enabled"]:
            return {"graded": 0, "pending": len(pending), "reason": text("judge_unavailable", name=judge["name"])}
        rs = normalise_settings({})
        cancel = (lambda: self.is_cancelled(run_id)) if run_id else (lambda: False)
        graded = 0
        try:
            with self.session(judge, rs, cancel) as s:
                for _ in self._grade_rows(pending, lambda cid: make_ask(self.store, s.backend, judge, cid, cancel=cancel), cancel):
                    graded += 1
        except (GaltonError, Cancelled) as exc:
            reason = getattr(exc, "message", "cancelled")
            return {"graded": graded, "pending": len(pending) - graded, "reason": text("judge_not_started", reason=reason)}
        return {"graded": graded, "pending": len(pending) - graded}

    def _grade_rows(self, rows: list[dict[str, Any]], ask_for: Callable[[str], Callable[[dict[str, Any]], dict[str, Any]]],
                    cancel: Callable[[], bool]) -> Iterator[str]:
        """Grade stored answers that waited for the judge with ``ask_for(contestant_id)``, yielding each result id as it is graded (so a caller that is
        interrupted still knows how many were). What cannot be graded yet stays pending."""
        cases: dict[str, dict[str, Any]] = {}
        for row in rows:
            if cancel():
                break
            case = cases.get(row["case_id"]) or self.store.find_case(row["case_id"])
            if case is None:
                continue
            cases[row["case_id"]] = case
            completion = Completion(text=row["output"], reasoning=row["reasoning"], tool_calls=row["tool_calls"])
            verdict, still = self.check(case, completion, row["contestant_id"], ask_for(row["contestant_id"]))
            if still:
                continue
            self_judged = _find_flag(verdict.get("detail"), "self_judged")
            unavailable = bool(verdict.get("unavailable"))
            self.store.update_result(row["id"], score=verdict["score"], passed=bool(verdict["passed"]) and not unavailable, detail={**row["detail"], **verdict.get("detail", {})},
                                     unavailable=unavailable, judge_pending=False, self_judged=self_judged, confidence=0.5 if self_judged else 1.0)
            yield row["id"]

    # ------------------------------------------------------------------ one case, one model ("probar con...")
    def try_case(self, case: dict[str, Any], contestant_ref: Any, settings: Optional[dict[str, Any]] = None, suite: Optional[dict[str, Any]] = None) -> dict[str, Any]:
        """Run a single case (saved or a draft) against one model and return the answer and the check, saving nothing."""
        rs = normalise_settings(settings)
        contestant = self.resolve_contestants([contestant_ref])[0]
        suite = suite or {"id": "draft", "name": "draft", "category": "custom", "max_tokens": 512}
        problems = suite_lib.validate_case(case)
        if problems:
            raise GaltonError("invalid", "case_invalid", problems="; ".join(problems), problem_items=list(problems))
        if suite_lib.needs_vision(case) and not contestant["vision"]:
            raise GaltonError("unsupported", "no_vision", name=contestant["name"])
        event = threading.Event()
        with self.session(contestant, rs, event.is_set) as session:
            reason = self._skip_reason(case, session)
            if reason:
                return {"contestant": contestant["name"], "skipped": reason}
            completion = self.ask(contestant, session, case, suite, rs, 0, event.is_set)
            judge = self._judge_for(contestant, session, event.is_set)
            if completion.error:
                return {"contestant": contestant["name"], "error": completion.error, "notes": completion.notes}
            verdict, pending = self.check(case, completion, contestant["id"], judge)
        return {"contestant": contestant["name"], "output": completion.text, "reasoning": completion.reasoning, "tool_calls": completion.tool_calls, "verdict": verdict,
                "judge_pending": pending, "cpu": session.device == "cpu", "latency_ms": completion.latency_ms, "ttft_ms": completion.ttft_ms, "decode_tps": completion.decode_tps,
                "completion_tokens": completion.completion_tokens, "notes": completion.notes, "runs_on": session.runs_on}
