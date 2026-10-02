"""Regression watch (idle tracking, eligibility, quiet hours, regressions), the blind arena and the scheduler lanes."""

from __future__ import annotations

import threading
import time
from datetime import datetime

import httpx
import pytest

from galton_hoard.errors import GaltonError
from galton_hoard.fakes import reference_responder
from galton_hoard.scheduler import Scheduler
from galton_hoard.watch import SMOKE_SUITE, Watch
from helpers import Clock, add_gguf, json_response, mock_client_factory, references, run_inline


def noon(clock):
    clock.now = datetime(2026, 3, 4, 12, 0).timestamp()


def add_server(svc, name="srv", url="http://127.0.0.1:9999", api="openai"):
    return svc.store.create_contestant(key=f"server:{url}:{name}", kind="server", name=name, url=url, api=api, model=name, provider="llamacpp", aliases=[name],
                                       source="manual", enabled=True)


def watch_with(svc, handler):
    return Watch(svc.store, svc.settings, svc.runner, svc.gpus, clock=svc.clock, submit_run=svc.submit_run, emit=svc.emit,
                 client_factory=mock_client_factory(handler), offline=False)


# ---- watch: idle tracking ---------------------------------------------------------------------------------------------------------------------

def test_a_server_is_idle_only_after_it_stopped_being_busy(svc, clock):
    noon(clock)
    server = add_server(svc)
    state = {"busy": False}

    def handler(req):
        if req.url.path == "/v1/models":
            return json_response({"data": [{"id": "srv"}]})
        if req.url.path == "/slots":
            return json_response([{"is_processing": state["busy"]}])
        return httpx.Response(404)
    watch = watch_with(svc, handler)
    assert watch.idle_for(server["id"]) is None
    assert watch.probe() == {"servers_up": 1}
    clock.advance(300)
    assert watch.idle_for(server["id"]) == 300
    state["busy"] = True
    watch.probe()
    assert watch.idle_for(server["id"]) == 0
    state["busy"] = False
    clock.advance(120)
    watch.probe()
    assert watch.idle_for(server["id"]) == 120


def test_a_server_that_goes_down_is_forgotten(svc, clock):
    server = add_server(svc)
    up = {"v": True}
    watch = watch_with(svc, lambda req: json_response({"data": [{"id": "srv"}]}) if up["v"] and req.url.path == "/v1/models" else httpx.Response(500))
    watch.probe()
    assert watch.idle_for(server["id"]) is not None
    up["v"] = False
    watch.probe()
    assert watch.idle_for(server["id"]) is None


def test_ollama_activity_is_seen_when_expiry_moves(svc, clock):
    server = add_server(svc, name="qwen3:8b", url="http://127.0.0.1:11434", api="ollama")
    stamp = {"v": "t1"}
    watch = watch_with(svc, lambda req: json_response({"models": [{"name": "qwen3:8b", "expires_at": stamp["v"]}]}))
    watch.probe()
    clock.advance(500)
    watch.probe()
    assert watch.idle_for(server["id"]) == 500
    stamp["v"] = "t2"          # somebody used the model: the keep-alive moved
    watch.probe()
    assert watch.idle_for(server["id"]) == 0


def test_offline_watch_does_not_touch_the_network(svc):
    assert Watch(svc.store, svc.settings, svc.runner, svc.gpus, clock=svc.clock, submit_run=svc.submit_run, offline=True).probe() == {"offline": True}


# ---- watch: eligibility and ticks -------------------------------------------------------------------------------------------------------------

def test_candidates_are_models_not_yet_measured_on_the_quick_suite(svc, clock):
    add_gguf(svc, "nuevo")
    b = add_gguf(svc, "viejo")
    run_inline(svc, ["rapida"], [b["id"]])
    assert [c["name"] for c in svc.watch.candidates()] == ["nuevo"]
    svc.store.update_contestant(b["id"], digest="changed")
    assert {c["name"] for c in svc.watch.candidates()} == {"nuevo", "viejo"}


def test_server_candidates_wait_until_they_are_loaded_and_idle(svc, clock):
    server = add_server(svc)
    ok, why = svc.watch._eligible(server)
    assert not ok and "not loaded" in why
    svc.watch._seen[server["id"]] = clock()
    ok, why = svc.watch._eligible(server)
    assert not ok and "busy recently" in why
    clock.advance(11 * 60)
    assert svc.watch._eligible(server) == (True, "loaded and idle")


def test_a_gguf_that_cannot_fit_is_not_eligible(svc):
    huge = add_gguf(svc, "inmenso", size_gb=80.0)
    ok, why = svc.watch._eligible(svc.store.contestant(huge["id"]))
    assert not ok and "GB" in why


def test_tick_queues_one_quick_run_for_a_gguf_with_room(svc, clock):
    noon(clock)
    m = add_gguf(svc, "nuevo")
    decision = svc.watch.tick()
    assert decision["queued"]["contestant"] == "nuevo"
    run = svc.store.run(decision["queued"]["run"])
    assert run["source"] == "watch" and run["state"] == "done" and run["suites"] == [SMOKE_SUITE]
    assert svc.watch.candidates() == []           # measured now, and tried within the interval
    assert m["id"] in run["contestants"]


def test_tick_runs_at_most_one_model_at_a_time(svc, clock):
    noon(clock)
    watch = Watch(svc.store, svc.settings, svc.runner, svc.gpus, clock=clock, submit_run=lambda _id: None, offline=True)
    add_gguf(svc, "uno"), add_gguf(svc, "dos")
    first = watch.tick()
    assert "queued" in first
    second = watch.tick()
    assert second["skipped"] == "a run is already queued or running"


@pytest.mark.parametrize("setting,value,reason", [("watch.auto_smoke", False, "auto smoke is off"), ("watch.enabled", False, "auto smoke is off"),
                                                  ("scheduler.paused", True, "the scheduler is paused")])
def test_tick_respects_the_switches(svc, clock, setting, value, reason):
    noon(clock)
    add_gguf(svc, "nuevo")
    svc.settings.set_many({setting: value})
    assert svc.watch.tick()["skipped"] == reason


def test_tick_never_runs_in_quiet_hours(svc, clock):
    clock.now = datetime(2026, 3, 4, 3, 0).timestamp()
    add_gguf(svc, "nuevo")
    assert svc.watch.tick()["skipped"] == "quiet hours"
    clock.now = datetime(2026, 3, 4, 8, 30).timestamp()
    assert "queued" in svc.watch.tick()


def test_tick_reports_why_nobody_was_queued(svc, clock):
    noon(clock)
    add_gguf(svc, "inmenso", size_gb=80.0)
    decision = svc.watch.tick()
    assert "queued" not in decision and "inmenso" in decision["waiting"]


# ---- watch: regressions -----------------------------------------------------------------------------------------------------------------------

def test_a_model_that_got_worse_after_its_file_changed_raises_a_notice_and_an_event(svc, clock):
    events = []
    svc.emit = lambda t, d: events.append((t, d))
    svc.watch.emit = svc.emit
    m = add_gguf(svc, "cambia", digest="v1")
    run_inline(svc, ["rapida"], [m["id"]])
    clock.advance(3600)
    svc.store.update_contestant(m["id"], digest="v2")
    svc.fake_world.register("cambia", reference_responder(references(), accuracy=0.0))
    run = run_inline(svc, ["rapida"], [m["id"]])
    found = svc.watch.detect_regressions(run["id"])
    assert len(found) == 1 and found[0]["name"] == "cambia" and found[0]["diff"] > 0.5
    notice = svc.store.notices(kind="regression")[0]
    assert notice["severity"] == "high" and "cambia" in notice["title"]
    assert events[-1][0] == "galton.regression"
    svc.watch.detect_regressions(run["id"])
    assert len(svc.store.notices(kind="regression")) == 1       # the notice is not repeated


def test_an_improvement_is_noted_but_not_reported_as_a_regression(svc, clock):
    m = add_gguf(svc, "mejora", accuracy=0.0, digest="v1")
    run_inline(svc, ["rapida"], [m["id"]])
    clock.advance(3600)
    svc.store.update_contestant(m["id"], digest="v2")
    svc.fake_world.register("mejora", reference_responder(references(), accuracy=1.0))
    run = run_inline(svc, ["rapida"], [m["id"]])
    assert svc.watch.detect_regressions(run["id"]) == []
    assert svc.store.notices(kind="improvement") and not svc.store.notices(kind="regression")


def test_no_previous_version_means_no_comparison(svc):
    m = add_gguf(svc, "primero")
    run = run_inline(svc, ["rapida"], [m["id"]])
    assert svc.watch.detect_regressions(run["id"]) == [] and svc.store.notices() == []


# ---- arena -----------------------------------------------------------------------------------------------------------------------------------

def test_next_pair_hides_names_and_offers_two_different_answers(svc, measured):
    pair = svc.arena.next_pair()
    assert pair and pair["a"]["text"] != pair["b"]["text"] and pair["prompt"]
    assert set(pair) == {"pair", "case", "title", "category", "prompt", "a", "b"}
    assert all("grande" not in str(pair[k]) for k in ("a", "b"))


def test_a_vote_is_recorded_once_and_the_pair_does_not_come_back(svc, measured):
    pair = svc.arena.next_pair()
    out = svc.arena.vote(pair["pair"], "a")
    assert out["vote"] == "a" and out["a"] and out["b"]
    with pytest.raises(GaltonError) as exc:
        svc.arena.vote(pair["pair"], "b")
    assert exc.value.code == "conflict"
    seen = {(pair["case"], tuple(sorted((v["a"], v["b"])))) for v in svc.store.votes()}
    for _ in range(30):
        nxt = svc.arena.next_pair()
        if nxt is None:
            break
        stored = svc.store.arena_pair(nxt["pair"])
        assert (nxt["case"], tuple(sorted((stored["a_contestant"], stored["b_contestant"])))) not in seen
        svc.arena.vote(nxt["pair"], "tie")


def test_bad_votes_and_unknown_pairs_are_refused(svc, measured):
    pair = svc.arena.next_pair()
    with pytest.raises(GaltonError):
        svc.arena.vote(pair["pair"], "maybe")
    with pytest.raises(GaltonError) as exc:
        svc.arena.vote("pair_nope", "a")
    assert exc.value.code == "not_found"


def test_the_arena_is_empty_when_models_gave_identical_answers(svc):
    a, b = add_gguf(svc, "gemelo-a"), add_gguf(svc, "gemelo-b")
    run_inline(svc, ["rapida"], [a["id"], b["id"]])
    assert svc.arena.next_pair() is None


def test_ratings_follow_the_votes_and_need_a_minimum(svc, measured):
    big, small = measured["big"]["id"], measured["small"]["id"]
    svc.settings.set_many({"arena.min_votes": 3})
    for i in range(4):
        svc.store.add_vote(pair_id=f"p{i}", case_id="c", category="reasoning", a=big, b=small, vote="a")
    rows = svc.arena.ratings("reasoning")
    names = [r["name"] for r in rows]
    assert names[0] == "grande-q4" and set(names) == {"grande-q4", "pequeno-q8"}
    assert svc.arena.categories_with_votes() == ["reasoning"]


def test_ratings_are_empty_without_votes(svc):
    assert svc.arena.ratings() == [] and svc.arena.categories_with_votes() == []


def test_the_category_filter_limits_the_pairs(svc, measured):
    pair = svc.arena.next_pair("math")
    assert pair is None or pair["category"] == "math"


# ---- scheduler -------------------------------------------------------------------------------------------------------------------------------

def make_scheduler(handlers=None, **kw):
    class Store:
        def __init__(self):
            self.activity = []

        def add_activity(self, *a):
            self.activity.append(a)
    store = Store()
    sched = Scheduler(handlers or {k: (lambda ref, k=k: k) for k in ("run", "judge", "refresh", "watch", "housekeeping")}, store, clock=kw.pop("clock", Clock()), **kw)
    return sched, store


def test_without_running_lanes_run_now_executes_inline():
    sched, _ = make_scheduler({"run": lambda ref: f"ran {ref}"})
    assert sched.run_now("run", "r1") == "ran r1"


def test_run_now_waits_for_a_running_lane_and_reraises_failures():
    def bad(ref):
        raise ValueError("boom")
    sched, store = make_scheduler({"run": lambda ref: ref.upper(), "judge": bad})
    sched.start()
    try:
        assert sched.run_now("run", "abc", timeout=5) == "ABC"
        with pytest.raises(RuntimeError, match="ValueError: boom"):
            sched.run_now("judge", "", timeout=5)
        assert store.activity and store.activity[0][0] == "judge" and store.activity[0][2] is False
        assert sched.run_now("run", "again", timeout=5) == "AGAIN"          # the lane survived the failure
    finally:
        sched.stop()


def test_the_same_job_is_not_queued_twice():
    gate = threading.Event()
    sched, _ = make_scheduler({"run": lambda ref: gate.wait(5)})
    sched.start()
    try:
        first = sched.submit("run", "r1")
        assert first is not None and sched.submit("run", "r1") is None
        assert sched.submit("run", "r2") is not None
        gate.set()
        assert first.done.wait(5)
    finally:
        gate.set()
        sched.stop()


def test_runs_and_judging_share_one_lane_and_do_not_overlap():
    active, peak, lock = [0], [0], threading.Lock()

    def work(ref):
        with lock:
            active[0] += 1
            peak[0] = max(peak[0], active[0])
        time.sleep(0.05)
        with lock:
            active[0] -= 1
    sched, _ = make_scheduler({"run": work, "judge": work})
    sched.start()
    try:
        jobs = [sched.submit("run", "a"), sched.submit("judge", ""), sched.submit("run", "b")]
        assert all(j.done.wait(5) for j in jobs)
        assert peak[0] == 1
    finally:
        sched.stop()


def test_a_slow_job_reports_that_it_is_still_running():
    gate = threading.Event()
    sched, _ = make_scheduler({"run": lambda ref: gate.wait(5)})
    sched.start()
    try:
        out = sched.run_now("run", "slow", timeout=0.1)
        assert out["queued"] is True
        assert sched.status()["lanes"]["gpu"]["current"]["ref"] == "slow"
    finally:
        gate.set()
        sched.stop()


def _ran(sched, *kinds):
    """The shared scheduler remembers when a job last *ran* (not when it was queued): run the queued ones inline, as the lane would."""
    sched._lane._pending.clear()
    for kind in kinds:
        sched.run_now(kind)


def test_enqueue_due_follows_the_intervals():
    clock = Clock()
    sched, _ = make_scheduler(clock=clock, refresh_every_h=lambda: 6.0)
    assert sched.enqueue_due(clock()) == 0                      # nothing is due right at start
    clock.advance(100)
    assert sched.enqueue_due(clock()) == 3                      # refresh, watch and housekeeping have waited long enough
    assert sched.enqueue_due(clock()) == 0                      # and are pending
    _ran(sched, "refresh", "watch", "housekeeping")
    clock.advance(61)
    assert sched.enqueue_due(clock()) == 1                      # only the one-minute watch is due again
    _ran(sched, "watch")
    clock.advance(6 * 3600)
    assert sched.enqueue_due(clock()) == 3


def test_watch_can_be_switched_off_independently():
    clock = Clock()
    sched, _ = make_scheduler(clock=clock, watch_enabled=lambda: False)
    clock.advance(100)
    sched.enqueue_due(clock())
    assert "watch" not in sched._lane._pending and "refresh" in sched._lane._pending


def test_status_shape_and_pause_flag():
    sched, _ = make_scheduler(paused=lambda: True)
    status = sched.status()
    assert status["paused"] is True and set(status["lanes"]) == {"gpu", "io"} and status["running"] is False


def test_start_is_idempotent_and_stop_ends_the_lanes():
    sched, _ = make_scheduler()
    sched.start()
    threads = list(sched._lane._threads)
    sched.start()
    assert sched._lane._threads == threads and sched.status()["running"] is True
    sched.stop()
    assert sched.status()["running"] is False


def test_unknown_job_kind_is_an_error():
    sched, _ = make_scheduler({"run": lambda ref: 1})
    with pytest.raises(ValueError):
        sched.run_now("refresh", "")


def test_runs_left_queued_by_a_restart_are_run_when_the_app_starts_again(svc):
    from helpers import add_gguf
    model = add_gguf(svc, "cola-q4")
    run = svc.runner.create(suites=["rapida"], contestants=[model["id"]])   # queued in the database; the old process's queue is gone
    assert run["state"] == "queued"
    svc.start()
    try:
        deadline = time.monotonic() + 20
        while time.monotonic() < deadline and svc.store.run(run["id"])["state"] not in ("done", "failed", "cancelled"):
            time.sleep(0.05)
        assert svc.store.run(run["id"])["state"] == "done"
    finally:
        svc.scheduler.stop()


def test_publishing_routes_retries_while_a_reader_holds_the_file(svc, measured, monkeypatch):
    from galton_hoard.hoard_link import atomic
    real, calls = atomic.os.replace, []

    def flaky(src, dst):
        calls.append(src)
        if len(calls) < 3:
            raise PermissionError(13, "in use")
        real(src, dst)

    monkeypatch.setattr(atomic.os, "replace", flaky)
    monkeypatch.setattr(atomic.time, "sleep", lambda s: None)
    out = svc.routes.publish()
    assert len(calls) == 3 and svc.config.routes_path().is_file() and out["path"] == str(svc.config.routes_path())
    assert not list(svc.config.routes_path().parent.glob("*.tmp"))        # no temp file is left behind
