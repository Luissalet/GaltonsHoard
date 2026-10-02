"""Canonical job events for runs (galton.job.*) and the published routing file as the hub's reader sees it."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from conftest import tool
from helpers import add_gguf, run_inline
from galton_hoard.hoard_link import family
from galton_hoard.hoard_link.routes import load_routes
from galton_hoard.jobevents import RunJobEvents


@pytest.fixture
def bus(monkeypatch):
    events: list[tuple[str, dict]] = []
    monkeypatch.setattr(family, "emit", lambda type_, data=None, **kw: events.append((type_, dict(data or {}))) or True)
    return events


def names(bus):
    return [t for t, _ in bus if t.startswith("galton.job.")]


def test_a_run_emits_queued_started_progress_done_with_the_canonical_fields(svc, bus, clock):
    add_gguf(svc, "alfa-q4")
    svc.runner.jobs.min_interval_s = 0.0
    run = run_inline(svc, ["razonamiento"], ["alfa-q4"])
    assert run["state"] == "done"
    seq = names(bus)
    assert seq[0] == "galton.job.queued" and seq[1] == "galton.job.started" and seq[-1] == "galton.job.done"
    assert "galton.job.progress" in seq
    assert not [t for t, _ in bus if t.startswith("galton.run.")]            # the legacy name is no longer emitted
    queued = dict(bus)["galton.job.queued"]
    assert queued["job_id"] == run["id"] and queued["kind"] == "run" and queued["title"] and queued["progress"] == 0.0
    assert queued["url"] == f"http://127.0.0.1:{svc.config.port}/#/ejecutar/{run['id']}"
    progress = [d for t, d in bus if t == "galton.job.progress"]
    assert all(0.0 <= d["progress"] <= 1.0 for d in progress) and progress[-1]["progress"] >= progress[0]["progress"]
    done = dict(bus)["galton.job.done"]
    assert done["progress"] == 1.0 and done["job_id"] == run["id"] and "error" not in done


def test_the_gpu_of_the_session_is_reported(svc, bus):
    add_gguf(svc, "alfa-q4")
    svc.runner.jobs.min_interval_s = 0.0
    run_inline(svc, ["rapida"], ["alfa-q4"])
    gpus = [d.get("gpu") for t, d in bus if t == "galton.job.progress"]
    assert gpus and all(isinstance(g, int) for g in gpus)


def test_progress_is_throttled():
    sent: list = []
    now = [100.0]
    jobs = RunJobEvents(lambda t, d: sent.append((t, d)), clock=lambda: now[0], min_interval_s=5.0)
    run = {"id": "r_1", "label": "x", "source": "test"}
    jobs.started(run)
    calls = []

    def totals():
        calls.append(1)
        return 5, 10

    jobs.progress(run, totals)                       # too soon after start
    now[0] += 6
    jobs.progress(run, totals)
    jobs.progress(run, totals)                       # same instant: throttled
    assert [t for t, _ in sent] == ["galton.job.started", "galton.job.progress"] and len(calls) == 1
    assert sent[-1][1]["progress"] == 0.5 and sent[-1][1]["eta_s"] == 6


def test_a_failing_emit_never_reaches_the_run():
    def boom(*_):
        raise RuntimeError("bus down")
    jobs = RunJobEvents(boom)
    jobs.queued({"id": "r_1", "label": "x"})
    jobs.finished({"id": "r_1", "label": "x"}, "failed", "e")


def test_a_cancelled_queued_run_and_a_failed_run_emit_their_event(svc, bus):
    add_gguf(svc, "alfa-q4")
    created = svc.runner.create(suites=["razonamiento"], contestants=["alfa-q4"], label="x")
    svc.runner.cancel(created["id"])
    assert names(bus) == ["galton.job.queued", "galton.job.cancelled"]
    bus.clear()
    created = svc.runner.create(suites=["razonamiento"], contestants=["alfa-q4"], label="y")
    svc.store.update_run(created["id"], state="running")
    svc._abandon_stale_runs()
    failed = dict(bus)["galton.job.failed"]
    assert failed["job_id"] == created["id"] and failed["error"]


def test_a_run_where_every_contestant_fails_emits_failed_with_the_error(svc, bus, monkeypatch):
    add_gguf(svc, "alfa-q4")
    monkeypatch.setattr(svc.runner, "_run_contestant", lambda *a, **k: "failed")
    run = run_inline(svc, ["rapida"], ["alfa-q4"])
    assert run["state"] == "failed"
    failed = dict(bus)["galton.job.failed"]
    assert failed["job_id"] == run["id"] and failed["error"] and failed["kind"] == "run"


def test_the_published_routes_file_is_read_by_the_family_library(svc, measured, tmp_path, monkeypatch):
    path = tmp_path / "hoard-home" / "routes.json"
    monkeypatch.setenv("HOARD_ROUTES_FILE", str(path))
    svc.config.routes_file = None                          # resolved at publish time, as in production
    out = tool(svc, "routes_publish")
    assert out["path"] == str(path) and path.is_file()
    raw = json.loads(path.read_text(encoding="utf-8"))
    assert raw["schema"] == 1 and raw["source"] == "galton" and raw["updated_at"]
    routes = load_routes()                                   # the reader finds it through the same variable
    assert routes.problem is None and routes.source == "galton" and not routes.empty
    general = routes.preferences("llm", "general")
    assert general and general[0] == raw["tasks"]["general"]["prefer"][0]["names"][0]
    pref = routes.tasks["general"].prefer[0]
    assert pref.score is not None and pref.ci and pref.n and pref.tok_s
    assert routes.preferences("llm") == [n for p in routes.capabilities["llm"] for n in p.names]


def test_the_default_routes_location_is_the_one_the_reader_uses(svc, tmp_path, monkeypatch):
    svc.config.routes_file = None
    monkeypatch.delenv("HOARD_ROUTES_FILE", raising=False)
    assert svc.routes.path() == Path(load_routes().path) and svc.routes.path().parent.name == ".hoard"
    monkeypatch.setenv("HOARD_HOME", str(tmp_path / "hh"))
    assert svc.routes.path() == Path(load_routes().path) == tmp_path / "hh" / "routes.json"
    monkeypatch.setenv("HOARD_ROUTES_FILE", str(tmp_path / "x.json"))
    assert svc.routes.path() == Path(load_routes().path) == tmp_path / "x.json"
