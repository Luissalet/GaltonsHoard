"""Runs a harness matrix: every model x harness x task x sample, stored trial by trial.

It reuses the run machinery instead of making a second one: models are opened with ``Runner.session`` (a server that is already up is probed, waited
for when somebody else uses it and watched by the identity guard; a GGUF file gets its own llama-server under a GPU lease on an allowed GPU), the matrix
is queued on the same GPU lane as the runs, and cancelling it never touches a shared server. A model that cannot be opened leaves its trials as
``none`` (the verifier never ran) instead of silently shrinking the matrix.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any, Callable, Optional

from ..backends import Cancelled, ChatRequest, Completion
from ..errors import GaltonError
from ..hoard_link import proc
from ..jobevents import RunJobEvents
from ..messages import text
from ..runner import RUN_DEFAULTS, SECONDS_PER_REASONING_TOKEN, TERMINAL, Runner, normalise_settings
from . import harnesses as hs
from .external import ID_PREFIX, ExternalSpec, parse_external, run_external
from .loop import Trial, run_trial
from .scoring import build_matrix, conclude
from .store import MatrixStore
from .tasks import KINDS, TASKS, TASKS_BY_ID, public, select_tasks
from .world import TOOL_SPECS

log = logging.getLogger("galton.harnessbench")

MAX_TRIALS = 20_000
MAX_CONSECUTIVE_INFRA = 4
DEFAULTS = {"max_turns": 8, "max_format_errors": 3}
BOUNDS = {"max_turns": (1, 30), "max_format_errors": (1, 10)}
DEFAULT_MAX_TOKENS = 1024


class MatrixJobEvents(RunJobEvents):
    """The same canonical job events as a run, with the link to the harness screen."""

    def url(self, run_id: str) -> str:
        base = (self._base_url() or "").rstrip("/")
        return f"{base}/#/arneses/{run_id}" if base else ""


def normalise_matrix_settings(raw: Optional[dict[str, Any]], k: int) -> dict[str, Any]:
    """The settings of a matrix: the shared run settings (token cap per reply, temperature, timeout, device...) plus ``max_turns`` and
    ``max_format_errors``. They are the same for every cell, which is what makes the cells comparable."""
    raw = dict(raw or {})
    if raw.pop("repeats", 1) not in (1, "1", None):       # the number of samples per task is ``k``; 1 is what an already normalised dict carries
        raise GaltonError("invalid", "run_unknown_settings", names=["repeats"], options=[*[x for x in RUN_DEFAULTS if x != "repeats"], *DEFAULTS])
    mine: dict[str, Any] = {}
    for key, (low, high) in BOUNDS.items():
        value = raw.pop(key, None)
        if value is None:
            mine[key] = DEFAULTS[key]
            continue
        try:
            number = int(float(value))
        except (TypeError, ValueError) as exc:
            raise GaltonError("invalid", "setting_number", setting=key) from exc
        if not low <= number <= high:
            raise GaltonError("invalid", "setting_range", setting=key, low=str(low), high=str(high))
        mine[key] = number
    if raw.get("max_tokens") is None:
        raw["max_tokens"] = DEFAULT_MAX_TOKENS
    if raw.get("temperature") is None:
        raw["temperature"] = 0.0 if k <= 1 else 0.7          # k samples of one task only differ when the temperature is above zero
    return {**normalise_settings(raw), **mine}


class MatrixRunner:
    def __init__(self, store: MatrixStore, runner: Runner, settings: Any, *, clock: Callable[[], float] = time.time,
                 emit: Optional[Callable[[str, dict[str, Any]], None]] = None, base_url: Callable[[], str] = lambda: ""):
        self.store, self.runner, self.settings, self.clock = store, runner, settings, clock
        self.jobs = MatrixJobEvents(emit or (lambda *_: None), clock=clock, base_url=base_url)
        self._cancel: dict[str, threading.Event] = {}
        self._lock = threading.Lock()
        self.active: Optional[str] = None

    # ------------------------------------------------------------------ what can be run
    def external_specs(self) -> list[ExternalSpec]:
        return parse_external(list(self.settings.get("harnessbench.external") or []))

    def catalog(self) -> dict[str, Any]:
        specs = self.external_specs()
        return {"harnesses": [hs.parse_harness_id(b).info() for b in hs.BASES], "variants": {"prompt": ["short", "long"], "compact": ["none", "trim"],
                "how": "Add .long and/or .trim to a harness id, for example json_text.long.trim."},
                "external": {"allowed": bool(self.settings.get("harnessbench.allow_external")), "configured": [s.info() for s in specs],
                             "how": "Write lines 'name = command' in the setting harnessbench.external; the harness id is cli:<name>."},
                "tasks": [public(t) for t in TASKS], "kinds": list(KINDS), "tools": {n: s.description for n, s in TOOL_SPECS.items()}}

    def resolve_harnesses(self, refs: Optional[list[str]]) -> list[Any]:
        refs = [str(r).strip() for r in (refs or []) if str(r).strip()] or list(hs.DEFAULT_HARNESSES)
        out, seen = [], set()
        configured = {s.name: s for s in self.external_specs()}
        for ref in refs:
            if ref.lower().startswith(ID_PREFIX):
                if not self.settings.get("harnessbench.allow_external"):
                    raise GaltonError("unavailable", "hb_external_off")
                name = ref[len(ID_PREFIX):].strip().lower()
                spec = configured.get(name)
                if spec is None:
                    raise GaltonError("not_found", "hb_external_unknown", name=name, options=", ".join(configured) or "none")
                if not spec.enabled:
                    raise GaltonError("unavailable", "hb_external_disabled", name=name)
                item: Any = spec
            else:
                item = hs.parse_harness_id(ref)
            if item.id not in seen:
                seen.add(item.id)
                out.append(item)
        return out

    @staticmethod
    def resolve_tasks(ids: Optional[list[str]], kinds: Optional[list[str]]) -> list[dict[str, Any]]:
        unknown = [t for t in (ids or []) if t not in TASKS_BY_ID]
        if unknown:
            raise GaltonError("not_found", "hb_no_task", task=unknown[0], options=", ".join(TASKS_BY_ID))
        chosen = select_tasks(ids or (), kinds or ())
        if not chosen:
            raise GaltonError("invalid", "hb_no_tasks")
        return chosen

    @staticmethod
    def planned(harness_list: list[Any], tasks: list[dict[str, Any]], k: int) -> list[tuple[Any, dict[str, Any], int]]:
        """``(harness, task, sample)``; a command-line harness only gets the tasks a shell agent can do."""
        return [(h, t, s) for h in harness_list for t in tasks if not (isinstance(h, ExternalSpec) and not t["external"]) for s in range(k)]

    # ------------------------------------------------------------------ creating a matrix
    def create(self, *, models: list[Any], harnesses: Optional[list[str]] = None, tasks: Optional[list[str]] = None, kinds: Optional[list[str]] = None, k: int = 1,
               settings: Optional[dict[str, Any]] = None, label: str = "", source: str = "ui", caller: str = "") -> dict[str, Any]:
        rs = normalise_matrix_settings(settings, k)
        rows = self.runner.resolve_contestants(models)
        if not rows:
            raise GaltonError("invalid", "no_model_chosen")
        for c in rows:
            if not c["enabled"]:
                raise GaltonError("invalid", "model_disabled", name=c["name"])
            if c["missing"]:
                raise GaltonError("not_found", "model_gone", name=c["name"])
            self.runner.ensure_served(c)
        harness_list = self.resolve_harnesses(harnesses)
        chosen = self.resolve_tasks(tasks, kinds)
        per_model = len(self.planned(harness_list, chosen, k))
        total = per_model * len(rows)
        if total > MAX_TRIALS:
            raise GaltonError("invalid", "hb_too_many", n=total, max=MAX_TRIALS)
        if total == 0:
            raise GaltonError("invalid", "hb_no_tasks")
        progress = {c["id"]: {"state": "queued", "done": 0, "total": per_model, "error": "", "runs_on": "", "warnings": []} for c in rows}
        matrix = self.store.create_matrix(label=label or text("label_matrix", models=len(rows), harnesses=len(harness_list), tasks=len(chosen)), state="queued",
                                          models=[c["id"] for c in rows], harnesses=[h.id for h in harness_list], tasks=[t["id"] for t in chosen], k=k, settings=rs,
                                          progress=progress, total=total, done=0, source=source, caller=caller)
        self.jobs.queued(matrix)
        return matrix

    def plan(self, *, models: list[Any], harnesses: Optional[list[str]] = None, tasks: Optional[list[str]] = None, kinds: Optional[list[str]] = None, k: int = 1,
             settings: Optional[dict[str, Any]] = None) -> dict[str, Any]:
        rs = normalise_matrix_settings(settings, k)
        rows = self.runner.resolve_contestants(models)
        harness_list = self.resolve_harnesses(harnesses)
        chosen = self.resolve_tasks(tasks, kinds)
        per_model = len(self.planned(harness_list, chosen, k))
        return {"models": [c["name"] for c in rows], "harnesses": [h.id for h in harness_list], "tasks": len(chosen), "k": k, "trials_per_model": per_model,
                "trials": per_model * len(rows), "limit": MAX_TRIALS, "settings": rs,
                "same_for_every_cell": {"max_tokens": rs["max_tokens"], "max_turns": rs["max_turns"], "max_format_errors": rs["max_format_errors"], "temperature": rs["temperature"]}}

    # ------------------------------------------------------------------ cancel
    def _event(self, mid: str) -> threading.Event:
        with self._lock:
            return self._cancel.setdefault(mid, threading.Event())

    def is_cancelled(self, mid: str) -> bool:
        return self._event(mid).is_set()

    def cancel(self, mid: str) -> dict[str, Any]:
        m = self.store.matrix(mid)
        if m["state"] in TERMINAL:
            return {"matrix": m["id"], "state": m["state"], "cancelled": False, "note": text("run_finished")}
        self._event(m["id"]).set()
        self.store.update_matrix(m["id"], cancel=True)
        if m["state"] == "queued":
            self.store.update_matrix(m["id"], state="cancelled", finished_ts=self.clock())
            self.jobs.finished(self.store.matrix(m["id"]), "cancelled")
            return {"matrix": m["id"], "state": "cancelled", "cancelled": True}
        shared = [c["url"] for c in (self.runner.store.find_contestant(i) for i in m["models"]) if c and c["kind"] != "gguf" and c["url"]]
        note = text("run_stopping_shared", urls=", ".join(dict.fromkeys(shared))) if shared else text("run_stopping_own")
        return {"matrix": m["id"], "state": m["state"], "cancelled": True, "note": note}

    # ------------------------------------------------------------------ progress bookkeeping
    def _set_progress(self, mid: str, cid: str, **fields: Any) -> None:
        with self._lock:
            m = self.store.matrix(mid)
            prog = dict(m["progress"])
            prog[cid] = {**prog.get(cid, {}), **fields}
            self.store.update_matrix(mid, progress=prog, done=sum(int(p.get("done") or 0) for p in prog.values()))

    def totals(self, mid: str) -> tuple[int, int]:
        m = self.store.matrix(mid)
        return int(m["done"]), int(m["total"])

    # ------------------------------------------------------------------ executing
    def execute(self, mid: str) -> dict[str, Any]:
        m = self.store.matrix(mid)
        if m["state"] != "queued":
            return m
        if self.is_cancelled(mid) or m["cancel"]:
            done = self.store.update_matrix(mid, state="cancelled", finished_ts=self.clock())
            self.jobs.finished(done, "cancelled")
            return done
        rs = m["settings"]
        try:
            harness_list = self.resolve_harnesses(m["harnesses"])
        except GaltonError as exc:               # a command-line harness was switched off between queueing and running
            failed = self.store.update_matrix(mid, state="failed", finished_ts=self.clock(), error=exc.message)
            self.jobs.finished(failed, "failed", str(exc.message))
            return failed
        tasks = [TASKS_BY_ID[t] for t in m["tasks"] if t in TASKS_BY_ID]
        self.store.update_matrix(mid, state="running", started_ts=self.clock())
        self.jobs.started(m)
        self.active = mid
        outcomes: dict[str, str] = {}
        try:
            for cid in m["models"]:
                if self.is_cancelled(mid):
                    break
                outcomes[cid] = self._run_model(m, self.runner.store.contestant(cid), rs, harness_list, tasks)
        finally:
            self.active = None
        for cid in m["models"]:
            if cid not in outcomes:
                self._set_progress(mid, cid, state="cancelled")
        if self.is_cancelled(mid):
            state, error = "cancelled", ""
        elif outcomes and all(o == "failed" for o in outcomes.values()):
            state, error = "failed", text("run_all_failed")
        else:
            state, error = "done", ""
        summary = self.summary(mid)
        final = self.store.update_matrix(mid, state=state, finished_ts=self.clock(), error=error, summary=summary)
        self.jobs.finished(final, state, error)
        with self._lock:
            self._cancel.pop(mid, None)
        return final

    def _run_model(self, m: dict[str, Any], c: dict[str, Any], rs: dict[str, Any], harness_list: list[Any], tasks: list[dict[str, Any]]) -> str:
        mid, cid = m["id"], c["id"]
        cancel = lambda: self.is_cancelled(mid)  # noqa: E731
        plan = self.planned(harness_list, tasks, int(m["k"]))
        stored = {"n": 0}
        waiting: dict[str, Any] = {"since": None}

        def note_state(state: str) -> None:
            if state == "waiting_server" and waiting["since"] is None:
                waiting["since"] = self.clock()
            if state == "running":
                waiting["since"] = None
            self._set_progress(mid, cid, state=state, wait_since=waiting["since"])
            self.store.update_matrix(mid, state={"waiting_gpu": "waiting_gpu", "waiting_server": "waiting_server"}.get(state, "running"))

        self._set_progress(mid, cid, state="running", done=0)
        held: list[Trial] = []

        def store_trial(trial: Trial) -> None:
            self.store.add_trial(mid, cid, c["digest"], trial.as_dict())
            stored["n"] += 1
            self._set_progress(mid, cid, done=stored["n"])
            self.jobs.progress(m, lambda: self.totals(mid))

        def release() -> None:
            while held:
                store_trial(held.pop(0))

        def none_for(rest: list[tuple[Any, dict[str, Any], int]], reason: str, detail: str) -> None:
            for h, task, sample in rest:
                store_trial(Trial(task_id=task["id"], harness=h.id, sample=sample, outcome="none", passed=None, reason=reason, detail=detail[:300],
                                  turns=None, calls=None, format_failures=None, extra_calls=None, extra_repeat=None, extra_after_solved=None, tokens_in=None, tokens_out=None))

        done_index = {"i": 0}
        dropped = {"n": 0}                  # trials answered but never verified against the server: they become none, not silent gaps
        try:
            with self.runner.session(c, rs, cancel, note_state) as session:
                self._set_progress(mid, cid, state="running", runs_on=session.runs_on, warnings=[str(w) for w in session.warnings])
                guard_id = session.identity
                api = "ollama" if c.get("api") == "ollama" else "openai"
                extra = self.runner.reasoning_allowance(rs, session)
                timeout = float(rs["timeout_s"] or self.settings.get("runner.timeout_s")) + extra * SECONDS_PER_REASONING_TOKEN
                current = {"sample": 0}

                def chat(messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> Completion:
                    seed = (int(rs["seed"]) + current["sample"]) if rs["seed"] is not None else None
                    req = ChatRequest(messages=messages, tools=tools, temperature=float(rs["temperature"]), max_tokens=int(rs["max_tokens"]) + extra, seed=seed, effort=rs["effort"],
                                      timeout_s=timeout, reasoning_tokens=extra, tools_fallback=False,
                                      context=rs["context"] if (session.runs_on.startswith("ollama") and rs["context"]) else None)
                    comp = session.backend.chat(req, cancel)
                    comp.max_tokens_sent = req.max_tokens
                    if comp.reasoning and not session.reasons:
                        self.runner.learn_reasoning(c, session)
                    if session.guard is not None:
                        session.guard.mark_own()
                    if guard_id is not None and not comp.error:
                        guard_id.answer(comp.served_model)
                    return comp

                def settle() -> None:
                    """On a shared server the trials are stored once a look made after them shows it is still the same model."""
                    if guard_id is None or not held:
                        return
                    state = "unknown"
                    for _ in range(3):
                        state = guard_id.check(force=True)
                        if state == "fresh":
                            break
                        self.runner.sleep(1.0)
                    if state != "fresh":
                        n = len(held)
                        dropped["n"] += n
                        held.clear()
                        raise GaltonError("unavailable", "server_unverified", url=session.url, n=n)
                    release()

                infra_in_a_row = 0
                try:
                    for index, (h, task, sample) in enumerate(plan):
                        done_index["i"] = index
                        if cancel():
                            break
                        self.runner.yield_to_others(session, rs, cancel, note_state)
                        if cancel():
                            break
                        if guard_id is not None and guard_id.check() == "fresh":
                            release()
                        current["sample"] = sample
                        if isinstance(h, ExternalSpec):
                            trial = run_external(h, task, base_url=session.url, model=getattr(session.backend, "model", "") or c["model"] or c["name"],
                                                 timeout_s=float(self.settings.get("harnessbench.external_timeout_s")), sample=sample, cancel=self._event(mid))
                        else:
                            trial = run_trial(task, h, chat, api=api, max_turns=int(rs["max_turns"]), max_format_errors=int(rs["max_format_errors"]), sample=sample)
                        infra_in_a_row = infra_in_a_row + 1 if trial.outcome == "none" else 0
                        if guard_id is None:
                            store_trial(trial)
                        else:
                            held.append(trial)
                        done_index["i"] = index + 1
                        if infra_in_a_row >= MAX_CONSECUTIVE_INFRA:
                            settle()
                            none_for(plan[index + 1:], "server_error", str(text("hb_infra_abort", n=infra_in_a_row)))
                            done_index["i"] = len(plan)
                            self._set_progress(mid, cid, error=str(text("hb_infra_abort", n=infra_in_a_row)))
                            break
                except GaltonError as exc:
                    if exc.key == "server_changed":
                        dropped["n"] += len(held)
                        held.clear()
                    else:
                        try:
                            settle()
                        except GaltonError:
                            log.info("matrix %s: %s: the last trials could not be checked against the server", mid, c["name"])
                    raise
                except BaseException:
                    try:
                        settle()
                    except GaltonError:
                        pass
                    raise
                settle()
            if cancel():
                self._set_progress(mid, cid, state="cancelled")
                return "cancelled"
            self._set_progress(mid, cid, state="done")
            return "done"
        except (Cancelled, proc.Cancelled):
            self._set_progress(mid, cid, state="cancelled")
            return "cancelled"
        except GaltonError as exc:
            if cancel():
                self._set_progress(mid, cid, state="cancelled")
                return "cancelled"
            self._fail_model(mid, cid, str(exc.message), plan[max(0, done_index["i"] - dropped["n"]):], none_for)
            return "failed"
        except Exception as exc:  # noqa: BLE001
            log.exception("matrix %s: %s failed", mid, c["name"])
            self._fail_model(mid, cid, f"{type(exc).__name__}: {exc}", plan[max(0, done_index["i"] - dropped["n"]):], none_for)
            return "failed"

    def _fail_model(self, mid: str, cid: str, error: str, rest: list[Any], none_for: Callable[..., None]) -> None:
        """The model could not be used (or stopped being the same model): its remaining trials are ``none``, with the reason."""
        none_for(rest, "session", str(text("hb_session_failed", error=error[:240])))
        self._set_progress(mid, cid, state="failed", error=error)

    # ------------------------------------------------------------------ reading
    def names(self, ids: list[str]) -> dict[str, str]:
        out = {}
        for cid in ids:
            c = self.runner.store.find_contestant(cid)
            out[cid] = c["name"] if c else cid
        return out

    def harness_info(self, hid: str) -> dict[str, Any]:
        if hid.startswith(ID_PREFIX):
            spec = next((s for s in self.external_specs() if s.id == hid), None)
            return spec.info() if spec else {"id": hid, "base": "cli", "external": True}
        return hs.parse_harness_id(hid).info()

    def card(self, m: dict[str, Any]) -> dict[str, Any]:
        names = self.names(m["models"])
        models = [{"id": cid, "name": names[cid], **(m["progress"].get(cid) or {})} for cid in m["models"]]
        return {"id": m["id"], "label": m["label"], "state": m["state"], "k": m["k"], "total": m["total"], "done": m["done"], "settings": m["settings"], "models": models,
                "harnesses": m["harnesses"], "tasks": len(m["tasks"]), "source": m["source"], "created_ts": m["created_ts"], "started_ts": m["started_ts"],
                "finished_ts": m["finished_ts"], "error": m["error"], "active": m["id"] == self.active, "cancel": m["cancel"]}

    def result(self, mid: str) -> dict[str, Any]:
        m = self.store.matrix(mid)
        trials = self.store.trials(m["id"])
        matrix = build_matrix(trials, m["models"], m["harnesses"], int(m["k"]))
        names = self.names(m["models"])
        conclusion = conclude(matrix, names, m["harnesses"])
        digests = self.store.digests(m["id"])
        models = []
        for cid in m["models"]:
            c = self.runner.store.find_contestant(cid)
            models.append({"id": cid, "name": names[cid], "digest": digests.get(cid, ""), "stale": bool(c and digests.get(cid) and c["digest"] != digests[cid]), "removed": c is None})
        return {"matrix": self.card(m), "models": models, "harnesses": [self.harness_info(h) for h in m["harnesses"]], **matrix, "conclusion": conclusion,
                "markdown": markdown_table(models, m["harnesses"], matrix), "trials": len(trials)}

    def summary(self, mid: str) -> dict[str, Any]:
        out = self.result(mid)
        c = out["conclusion"]
        return {"verdict": c["verdict"], "harness_spread_points": c["harness_spread_points"], "model_spread_points": c["model_spread_points"], "best_harness_by_model": c["best_harness_by_model"],
                "weak": c["weak"], "none_total": c["none_total"], "summary": c["summary"], "trials": out["trials"]}

    def cell(self, mid: str, model: str, harness: str, task: str = "") -> dict[str, Any]:
        m = self.store.matrix(mid)
        c = self.runner.store.find_contestant(model)
        cid = c["id"] if c and c["id"] in m["models"] else next((i for i, n in self.names(m["models"]).items() if n == model or i == model), "")
        if not cid:
            raise GaltonError("not_found", "hb_no_model_in_matrix", model=model, id=m["id"])
        hid = harness if harness in m["harnesses"] else next((h for h in m["harnesses"] if h.lower() == harness.lower()), "")
        if not hid:
            raise GaltonError("not_found", "hb_unknown_harness", ref=harness, options=", ".join(m["harnesses"]))
        trials = self.store.trials(m["id"], contestant_id=cid, harness=hid, task_id=task, trace=bool(task))
        if not trials:
            raise GaltonError("not_found", "hb_no_cell", id=m["id"], model=model, harness=harness)
        light = self.store.trials(m["id"], contestant_id=cid, harness=hid)
        from .scoring import cell_stats
        by_task: dict[str, list[dict[str, Any]]] = {}
        for t in light:
            by_task.setdefault(t["task_id"], []).append(t)
        rows = []
        for tid, group in by_task.items():
            spec = TASKS_BY_ID.get(tid, {"kind": "", "prompt": ""})
            rows.append({"task": tid, "kind": spec["kind"], "samples": len(group), "outcomes": [g["outcome"] for g in group], "passed": sum(1 for g in group if g["outcome"] == "pass"),
                         "format_failures": sum(g["format_failures"] or 0 for g in group), "extra_calls": sum(g["extra_calls"] or 0 for g in group),
                         "turns": [g["turns"] for g in group], "reasons": sorted({g["reason"] for g in group if g["reason"]}), "detail": next((g["detail"] for g in group if g["detail"]), "")})
        out = {"matrix": m["id"], "model": self.names([cid])[cid], "model_id": cid, "harness": self.harness_info(hid), "stats": cell_stats(light, int(m["k"])), "tasks": rows}
        if task:
            out["task"] = public(TASKS_BY_ID[task]) if task in TASKS_BY_ID else {"id": task}
            out["trials"] = trials
        return out


def markdown_table(models: list[dict[str, Any]], harness_ids: list[str], matrix: dict[str, Any]) -> str:
    def cell(c: dict[str, Any]) -> str:
        if c["trials"] == 0:
            return "—"
        if c["pass_at_1"] is None:
            return f"— (none {c['none_n']})"
        parts = [f"{round(c['pass_at_1'] * 100)} % (n={c['valid']})"]
        if c.get("format_fail_rate"):
            parts.append(f"fmt {round(c['format_fail_rate'] * 100)} %")
        if c.get("extra_calls_mean"):
            parts.append(f"extra {c['extra_calls_mean']:g}")
        if c["none_n"]:
            parts.append(f"none {c['none_n']}")
        return ", ".join(parts)

    lines = ["| model | " + " | ".join(harness_ids) + " | mean |", "|---|" + "---|" * (len(harness_ids) + 1)]
    for m in models:
        row = matrix["cells"][m["id"]]
        mean = matrix["by_model"][m["id"]]
        lines.append(f"| {m['name']} | " + " | ".join(cell(row[h]) for h in harness_ids) + f" | {'—' if mean is None else str(round(mean * 100)) + ' %'} |")
    lines.append("| mean | " + " | ".join("—" if matrix["by_harness"][h] is None else f"{round(matrix['by_harness'][h] * 100)} %" for h in harness_ids) + " | |")
    return "\n".join(lines)
