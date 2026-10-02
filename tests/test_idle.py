"""Sharing a server politely: a run waits while somebody else uses it, for as long as the settings say, and yields between questions.

Time is a fake clock moved by the sleep hook; the servers are mock HTTP worlds whose answers depend on that clock."""

from __future__ import annotations

import httpx
import pytest

from galton_hoard.agent_tools import call_tool
from galton_hoard.errors import GaltonError
from galton_hoard.fakes import FakeWorld
from galton_hoard.idle import POLL_S, LlamaGuard, OllamaGuard, wait_until_idle
from galton_hoard.settings import SPECS
from helpers import Clock, build_services, json_response, mock_client_factory, run_inline


# ------------------------------------------------------------------ wait_until_idle
def scripted(clock, busy_windows):
    """busy() true while the clock is inside one of the (start, end) windows."""
    return lambda: any(a <= clock.now < b for a, b in busy_windows)


def wait(clock, busy, **kw):
    args = dict(grace_s=20, max_s=3600, cancel=lambda: False, clock=clock, sleep=clock.advance, url="http://x")
    args.update(kw)
    return wait_until_idle(busy, **args)


def test_an_idle_server_is_not_waited_for_and_no_time_passes():
    clock = Clock()
    start = clock.now
    assert wait(clock, lambda: False) == 0.0
    assert clock.now == start


def test_a_busy_server_is_waited_for_until_it_has_been_quiet_for_the_grace():
    clock = Clock()
    start = clock.now
    waited = wait(clock, scripted(clock, [(start, start + 100)]))
    # busy until +100 (seen at the first poll at or after it), then 20 s without being busy
    assert 100 + 20 <= waited <= 100 + 20 + 2 * POLL_S


def test_a_blip_of_activity_during_the_grace_starts_the_grace_again():
    clock = Clock()
    start = clock.now
    waited = wait(clock, scripted(clock, [(start, start + 30), (start + 40, start + 45)]))
    assert waited >= 42 + 20, "the grace runs from the last time it was seen busy"


def test_the_wait_ends_with_server_busy_after_the_longest_wait():
    clock = Clock()
    with pytest.raises(GaltonError) as raised:
        wait(clock, lambda: True, max_s=60, url="http://srv:8080")
    assert raised.value.code == "busy" and raised.value.key == "server_busy" and raised.value.params["url"] == "http://srv:8080" and raised.value.params["limit"] == "60"
    assert 60 <= clock.now - 1_760_000_000.0 <= 60 + POLL_S


def test_a_longest_wait_of_zero_does_not_wait_at_all():
    clock = Clock()
    with pytest.raises(GaltonError):
        wait(clock, lambda: True, max_s=0)
    assert clock.now == 1_760_000_000.0


def test_no_longest_wait_means_for_ever():
    clock = Clock()
    start = clock.now
    waited = wait(clock, scripted(clock, [(start, start + 10_000)]), max_s=None)
    assert waited >= 10_020


def test_a_cancelled_run_stops_waiting_at_once():
    clock = Clock()
    assert wait(clock, lambda: True, cancel=lambda: True) == 0.0


def test_the_wait_reports_how_long_it_has_waited():
    clock = Clock()
    start = clock.now
    seen = []
    wait(clock, scripted(clock, [(start, start + 12)]), on_wait=seen.append)
    assert seen[0] == 0.0 and seen == sorted(seen) and len(seen) >= 4


def test_busy_now_makes_the_caller_wait_even_if_the_reading_is_idle_now():
    clock = Clock()
    waited = wait(clock, lambda: False, busy_now=True)
    assert waited >= 20


# ------------------------------------------------------------------ guards
def slots_client(state):
    def handler(req):
        if req.url.path == "/slots":
            return json_response([{"is_processing": state["busy"]}])
        return json_response({}, 404)
    return mock_client_factory(handler)


def test_llama_guard_reads_the_slots():
    state = {"busy": False}
    guard = LlamaGuard(slots_client(state), "http://x")
    assert guard.others_busy() is False
    state["busy"] = True
    assert guard.others_busy() is True


def test_llama_guard_says_nothing_when_the_server_does_not_list_slots_or_does_not_answer():
    assert LlamaGuard(mock_client_factory(lambda req: json_response({}, 404)), "http://x").others_busy() is None

    def down(req):
        raise httpx.ConnectError("down")
    assert LlamaGuard(mock_client_factory(down), "http://x").others_busy() is None


def ps_client(state):
    def handler(req):
        if req.url.path == "/api/ps":
            return json_response({"models": [{"name": "m:1", "expires_at": state["expires"]}]})
        return json_response({}, 404)
    return mock_client_factory(handler)


def test_ollama_guard_sees_somebody_elses_use_as_a_moved_expiry_and_absorbs_its_own():
    state = {"expires": "t1"}
    guard = OllamaGuard(ps_client(state), "http://x")
    assert guard.others_busy() is False
    state["expires"] = "t2"                      # our own question moved it
    guard.mark_own()
    assert guard.others_busy() is False
    state["expires"] = "t3"                      # somebody else's
    assert guard.others_busy() is True
    assert guard.others_busy() is False          # the new picture is the baseline: only one pause per use, then the grace decides


def test_ollama_guard_without_an_answer_is_unknown():
    def down(req):
        raise httpx.ConnectError("down")
    guard = OllamaGuard(mock_client_factory(down), "http://x")
    assert guard.others_busy() is None


# ------------------------------------------------------------------ settings
def test_the_wait_settings_have_their_defaults_and_bounds(svc):
    assert svc.settings.get("runner.idle_grace_s") == 20
    assert svc.settings.get("runner.wait_idle_max_s") == 3600
    assert "runner.wait_idle_s" not in SPECS
    with pytest.raises(GaltonError):
        svc.settings.set_many({"runner.idle_grace_s": -1})
    svc.settings.set_many({"runner.wait_idle_max_s": 0})
    assert svc.settings.get("runner.wait_idle_max_s") == 0


# ------------------------------------------------------------------ runs
def server_contestant(svc, name, model, url="http://127.0.0.1:8080"):
    return svc.store.create_contestant(key=f"server:{name}", kind="server", name=name, url=url, api="openai", model=model, provider="openai_compat", aliases=[name, model],
                                       digest=f"d-{name}", source="manual", context=8192)


class Shared:
    """A llama-server shared with somebody else: busy while the clock is in a busy window, plus whatever the test adds."""

    def __init__(self, clock):
        self.clock = clock
        self.windows: list[tuple[float, float]] = []
        self.slot_reads = 0
        self.has_slots = True

    def busy(self):
        return any(a <= self.clock.now < b for a, b in self.windows)

    def client(self):
        def handler(req):
            path = req.url.path
            if path == "/v1/models":
                return json_response({"data": [{"id": "modelo"}]})
            if path == "/props":
                return json_response({"default_generation_settings": {"n_ctx": 8192}})
            if path == "/slots":
                if not self.has_slots:
                    return json_response({}, 404)
                self.slot_reads += 1
                return json_response([{"is_processing": self.busy()}])
            return json_response({}, 404)
        return mock_client_factory(handler)


def build(tmp_path, clock, shared, on_sleep=None):
    world = FakeWorld()
    world.register("modelo", lambda req: "Una respuesta.")

    def sleep(seconds):
        clock.advance(seconds)
        if on_sleep:
            on_sleep()
    svc = build_services(tmp_path, clock=clock, world=world, client_factory=shared.client(), sleep=sleep)
    return svc, world


def test_a_busy_server_is_waited_for_beyond_the_old_two_minutes_and_the_run_finishes(tmp_path):
    clock = Clock()
    shared = Shared(clock)
    shared.windows = [(clock.now, clock.now + 300)]
    seen = []
    holder = {}

    def on_sleep():
        run = holder["svc"].store.runs()[0] if holder["svc"].store.runs() else None
        if run:
            card = holder["svc"].run_card(run)
            seen.append((card["state"], card["contestants"][0]["state"], card["contestants"][0]["waiting_s"]))

    svc, _ = build(tmp_path, clock, shared, on_sleep)
    holder["svc"] = svc
    m = server_contestant(svc, "compartido", "modelo")
    run = run_inline(svc, ["rapida"], [m["id"]])
    assert run["state"] == "done", run
    rows = svc.store.results(run_id=run["id"])
    assert len(rows) == 12 and not any(r["error"] for r in rows)
    waiting = [s for s in seen if s[1] == "waiting_server"]
    assert waiting and all(s[0] == "waiting_server" for s in waiting)
    times = [s[2] for s in waiting]
    assert times == sorted(times) and times[-1] > 120, "it kept waiting past the old limit"
    card = svc.run_card(svc.store.run(run["id"]))
    c = card["contestants"][0]
    assert c["state"] == "done" and c["waiting_s"] is None and 300 + 20 - POLL_S <= c["waited_s"] <= 300 + 20 + 2 * POLL_S + 1
    svc.db.close()


def test_an_idle_shared_server_starts_at_once_without_any_waiting(tmp_path):
    clock = Clock()
    start = clock.now
    shared = Shared(clock)
    svc, _ = build(tmp_path, clock, shared)
    m = server_contestant(svc, "libre", "modelo")
    run = run_inline(svc, ["rapida"], [m["id"]])
    assert run["state"] == "done"
    assert clock.now == start, "no sleeping at all"
    assert svc.run_card(svc.store.run(run["id"]))["contestants"][0]["waited_s"] == 0
    svc.db.close()


def test_the_run_gives_up_after_the_longest_wait_and_says_for_how_long(tmp_path):
    clock = Clock()
    shared = Shared(clock)
    shared.windows = [(clock.now, clock.now + 10 ** 9)]
    svc, _ = build(tmp_path, clock, shared)
    svc.settings.set_many({"runner.wait_idle_max_s": 600})
    m = server_contestant(svc, "ocupado", "modelo")
    run = run_inline(svc, ["rapida"], [m["id"]])
    card = svc.run_card(svc.store.run(run["id"]))
    c = card["contestants"][0]
    assert c["state"] == "failed" and svc.store.results(run_id=run["id"]) == []
    assert getattr(c["error"], "key", "") == "server_busy" and c["error"].params["limit"] in (600, "600") and "600 s" in str(c["error"]) and c["hint"]
    assert 600 <= c["waited_s"] <= 600 + POLL_S + 1
    svc.db.close()


def test_the_runs_own_wait_overrides_the_setting(tmp_path):
    clock = Clock()
    shared = Shared(clock)
    shared.windows = [(clock.now, clock.now + 10 ** 9)]
    svc, _ = build(tmp_path, clock, shared)
    m = server_contestant(svc, "ocupado", "modelo")
    run = run_inline(svc, ["rapida"], [m["id"]], wait_s=30)
    c = svc.run_card(svc.store.run(run["id"]))["contestants"][0]
    assert c["state"] == "failed" and 30 <= c["waited_s"] <= 30 + POLL_S + 1
    start = clock.now
    run = run_inline(svc, ["rapida"], [m["id"]], wait_s=0)
    c = svc.run_card(svc.store.run(run["id"]))["contestants"][0]
    assert c["state"] == "failed" and clock.now == start, "wait_s 0: do not wait at all"
    svc.db.close()


def test_a_setting_of_zero_waits_for_ever(tmp_path):
    clock = Clock()
    shared = Shared(clock)
    shared.windows = [(clock.now, clock.now + 20_000)]      # longer than any default
    svc, _ = build(tmp_path, clock, shared)
    svc.settings.set_many({"runner.wait_idle_max_s": 0})
    m = server_contestant(svc, "eterno", "modelo")
    run = run_inline(svc, ["rapida"], [m["id"]])
    assert run["state"] == "done"
    assert svc.run_card(svc.store.run(run["id"]))["contestants"][0]["waited_s"] >= 20_000
    svc.db.close()


def test_between_questions_the_run_pauses_while_somebody_else_is_using_the_server(tmp_path):
    clock = Clock()
    shared = Shared(clock)
    calls = {"n": 0}
    world = FakeWorld()

    def answer(req):
        calls["n"] += 1
        if calls["n"] == 3:                                   # somebody starts a chat right after our third question
            shared.windows.append((clock.now, clock.now + 200))
        return "Una respuesta."
    world.register("modelo", answer)
    svc = build_services(tmp_path, clock=clock, world=world, client_factory=shared.client(), sleep=clock.advance)
    m = server_contestant(svc, "amable", "modelo")
    marks = []
    inner = svc.runner.ask

    def ask(*a, **kw):
        marks.append(clock.now)
        return inner(*a, **kw)
    svc.runner.ask = ask
    run = run_inline(svc, ["rapida"], [m["id"]])
    assert run["state"] == "done"
    gaps = [b - a for a, b in zip(marks, marks[1:])]
    long_gaps = [g for g in gaps if g > 100]
    assert len(long_gaps) == 1 and 200 + 20 - POLL_S <= long_gaps[0] <= 200 + 20 + 2 * POLL_S + 2, gaps
    assert len(svc.store.results(run_id=run["id"])) == 12
    c = svc.run_card(svc.store.run(run["id"]))["contestants"][0]
    assert c["waited_s"] >= 200
    svc.db.close()


def test_results_already_measured_are_kept_when_a_later_wait_gives_up(tmp_path):
    clock = Clock()
    shared = Shared(clock)
    calls = {"n": 0}
    world = FakeWorld()

    def answer(req):
        calls["n"] += 1
        if calls["n"] == 4:
            shared.windows.append((clock.now, clock.now + 10 ** 9))
        return "Una respuesta."
    world.register("modelo", answer)
    svc = build_services(tmp_path, clock=clock, world=world, client_factory=shared.client(), sleep=clock.advance)
    svc.settings.set_many({"runner.wait_idle_max_s": 120})
    m = server_contestant(svc, "parcial", "modelo")
    run = run_inline(svc, ["rapida"], [m["id"]])
    c = svc.run_card(svc.store.run(run["id"]))["contestants"][0]
    assert c["state"] == "failed" and 120 <= c["waited_s"] <= 120 + POLL_S + 2
    assert len(svc.store.results(run_id=run["id"])) == 4
    svc.db.close()


def test_a_server_that_lists_no_slots_is_taken_as_idle(tmp_path):
    clock = Clock()
    start = clock.now
    shared = Shared(clock)
    shared.has_slots = False
    shared.windows = [(clock.now, clock.now + 10 ** 9)]    # nobody can tell
    svc, _ = build(tmp_path, clock, shared)
    m = server_contestant(svc, "mudo", "modelo")
    run = run_inline(svc, ["rapida"], [m["id"]])
    assert run["state"] == "done" and clock.now == start, "nothing can be known, so nothing is waited for"
    svc.db.close()


def test_cancelling_while_waiting_ends_the_run_without_results(tmp_path):
    clock = Clock()
    shared = Shared(clock)
    shared.windows = [(clock.now, clock.now + 10 ** 9)]
    holder = {"cancelled": False}

    def sleep(seconds):
        clock.advance(seconds)
        if clock.now - 1_760_000_000.0 > 30 and not holder["cancelled"]:
            holder["cancelled"] = True
            call_tool(holder["svc"], "run_cancel", {})

    world = FakeWorld()
    world.register("modelo", lambda req: "x")
    svc = build_services(tmp_path, clock=clock, world=world, client_factory=shared.client(), sleep=sleep)
    holder["svc"] = svc
    m = server_contestant(svc, "cancelado", "modelo")
    out = svc.start_run(suites=["rapida"], contestants=[m["id"]], source="test")
    run = svc.store.run(out["run"]["id"])
    assert holder["cancelled"] and run["state"] == "cancelled", run["state"]
    assert svc.store.results(run_id=run["id"]) == []
    assert clock.now - 1_760_000_000.0 < 60, "it stopped waiting"
    svc.db.close()


def test_run_status_shows_the_waiting_time_to_an_agent_and_the_run_page(tmp_path):
    clock = Clock()
    shared = Shared(clock)
    shared.windows = [(clock.now, clock.now + 100)]
    seen = []
    holder = {}

    def sleep(seconds):
        clock.advance(seconds)
        if holder.get("svc") and holder["svc"].store.runs():
            seen.append(call_tool(holder["svc"], "run_status", {})["run"])

    world = FakeWorld()
    world.register("modelo", lambda req: "x")
    svc = build_services(tmp_path, clock=clock, world=world, client_factory=shared.client(), sleep=sleep)
    holder["svc"] = svc
    m = server_contestant(svc, "visible", "modelo")
    out = svc.start_run(suites=["rapida"], contestants=[m["id"]], source="test")
    waiting = [r for r in seen if r["state"] == "waiting_server"]
    assert waiting, "the run itself says it is waiting for the server"
    contestant = waiting[-1]["contestants"][0]
    assert contestant["state"] == "waiting_server" and contestant["waiting_s"] >= 90 and contestant["waited_s"] == contestant["waiting_s"]
    final = call_tool(svc, "run_status", {"run": out["run"]["id"]})["run"]
    assert final["state"] == "done" and final["contestants"][0]["waiting_s"] is None and final["contestants"][0]["waited_s"] >= 100
    svc.db.close()


def test_a_run_that_is_waiting_for_a_server_counts_as_active(tmp_path):
    from galton_hoard.store import ACTIVE_RUN_STATES
    assert "waiting_server" in ACTIVE_RUN_STATES
    clock = Clock()
    shared = Shared(clock)
    shared.windows = [(clock.now, clock.now + 50)]
    states = []
    holder = {}

    def sleep(seconds):
        clock.advance(seconds)
        if holder.get("svc") and holder["svc"].store.runs():
            states.append(holder["svc"].overview()["running"]["state"] if holder["svc"].overview().get("running") else None)

    world = FakeWorld()
    world.register("modelo", lambda req: "x")
    svc = build_services(tmp_path, clock=clock, world=world, client_factory=shared.client(), sleep=sleep)
    holder["svc"] = svc
    m = server_contestant(svc, "activo", "modelo")
    svc.start_run(suites=["rapida"], contestants=[m["id"]], source="test")
    assert "waiting_server" in states
    svc.db.close()


def test_an_ollama_server_used_by_somebody_else_between_two_questions_is_waited_for(tmp_path):
    clock = Clock()
    state = {"expires": "start"}

    def handler(req):
        path = req.url.path
        if path == "/api/ps":
            return json_response({"models": [{"name": "m:1", "model": "m:1", "expires_at": state["expires"], "size": 100, "size_vram": 100}]})
        if path == "/api/show":
            return json_response({"capabilities": ["completion"]})
        return json_response({}, 404)

    world = FakeWorld()
    calls = {"n": 0}

    def answer(req):
        calls["n"] += 1
        state["expires"] = f"own-{calls['n']}"              # every answer of ours moves the expiry
        return "Una respuesta."
    world.register("m:1", answer)
    svc = build_services(tmp_path, clock=clock, world=world, client_factory=mock_client_factory(handler), sleep=clock.advance)
    m = svc.store.create_contestant(key="server:ollama", kind="server", name="ollama-m", url="http://127.0.0.1:11434", api="ollama", model="m:1", provider="ollama",
                                    aliases=["m:1"], digest="d-ollama", source="manual", context=8192)
    marks = []
    inner_ask = svc.runner.ask
    inner_yield = svc.runner.yield_to_others

    def ask(*a, **kw):
        marks.append(clock.now)
        return inner_ask(*a, **kw)

    def yield_to_others(session, rs, cancel, note):
        if calls["n"] == 3 and not state.get("used"):
            state["used"] = True
            state["expires"] = "somebody-else"             # a chat of somebody else moved it before our fourth question
        return inner_yield(session, rs, cancel, note)
    svc.runner.ask, svc.runner.yield_to_others = ask, yield_to_others
    run = run_inline(svc, ["rapida"], [m["id"]])
    assert run["state"] == "done" and len(svc.store.results(run_id=run["id"])) == 12
    gaps = [b - a for a, b in zip(marks, marks[1:])]
    pauses = [g for g in gaps if g > 0]
    assert len(pauses) == 1 and 20 <= pauses[0] <= 20 + 2 * POLL_S, gaps       # our own answers never made it wait; the other chat made it wait for the grace
    svc.db.close()
