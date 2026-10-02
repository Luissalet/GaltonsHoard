"""Cancelling a run says what it really does for each kind of contestant, and never touches a server that somebody else runs: only the
llama-server that Galton started itself for a GGUF file is stopped."""

from __future__ import annotations

import httpx
import pytest

from galton_hoard import procs, servers
from galton_hoard.agent_tools import call_tool
from galton_hoard.fakes import FakeWorld
from galton_hoard.messages import CodedText
from helpers import Clock, add_gguf, build_services, json_response, mock_client_factory, run_inline
from test_server_identity import SharedServer, llama_contestant


@pytest.fixture
def kills(monkeypatch):
    """Every way a process can be killed from this code base, recorded instead of done."""
    seen = []
    for module in (procs, servers):
        for name in ("kill_tree", "kill_pid_tree"):
            if hasattr(module, name):
                monkeypatch.setattr(module, name, lambda *a, _n=name, **k: seen.append(_n))
    return seen


def cancel_on(svc, nth, notes):
    """A responder that cancels the run through the tool on its ``nth`` answer."""
    state = {"n": 0}

    def answer(req):
        state["n"] += 1
        if state["n"] == nth:
            notes.append(call_tool(svc, "run_cancel", {}))
        return "x"
    return answer


def test_cancelling_a_run_on_a_shared_llama_server_says_so_and_stops_nothing(tmp_path, kills):
    server = SharedServer()
    world = FakeWorld()
    notes = []
    svc = build_services(tmp_path, world=world, client_factory=server.factory())
    world.register("qwen-q4", cancel_on(svc, 3, notes))
    c = llama_contestant(svc)
    run = run_inline(svc, ["rapida"], [c["id"]])
    assert run["state"] == "cancelled"
    out = notes[0]
    assert out["cancelled"] is True and isinstance(out["note"], CodedText) and out["note"].key == "run_stopping_shared" and out["note"].params == {"urls": "http://127.0.0.1:8081"}
    assert "the request in flight is dropped" in out["note"] and "not touched and keeps running" in out["note"] and "is stopped" not in out["note"]
    assert world.stopped == [] and kills == [], "nothing was stopped or killed"
    assert {m for m, _p in server.requests} == {"GET"}, "the shared server was only ever read"
    assert len(svc.store.results(run_id=run["id"])) == 3, "what was measured is kept"
    svc.db.close()


def test_cancelling_a_run_on_a_resident_ollama_model_does_not_unload_it(tmp_path, kills):
    requests = []

    def handler(req):
        requests.append((req.method, req.url.path))
        if req.url.path == "/api/ps":
            return json_response({"models": [{"name": "qwen3:8b", "expires_at": "t"}]})
        return json_response({}, 404)
    world = FakeWorld()
    notes = []
    svc = build_services(tmp_path, world=world, client_factory=mock_client_factory(handler))
    world.register("qwen3:8b", cancel_on(svc, 2, notes))
    c = svc.store.create_contestant(key="server:ollama:qwen3:8b", kind="server", name="qwen3:8b", url="http://127.0.0.1:11434", api="ollama", model="qwen3:8b", provider="ollama",
                                    ollama_ref="qwen3:8b", digest="d" * 64, source="ollama", context=8192)
    run = run_inline(svc, ["rapida"], [c["id"]])
    assert run["state"] == "cancelled" and notes[0]["note"].key == "run_stopping_shared"
    assert world.stopped == [] and kills == []
    assert not [r for r in requests if r[0] != "GET" and r[1] != "/api/show"], "the only POST is the capabilities question of /api/show: nothing is unloaded (/api/generate with keep_alive 0)"
    assert ("POST", "/api/generate") not in requests
    svc.db.close()


def test_cancelling_a_run_on_a_file_says_that_galtons_own_server_is_stopped(tmp_path, kills):
    world = FakeWorld()
    notes = []
    svc = build_services(tmp_path, world=world)
    c = add_gguf(svc, "alfa-q4", responder=lambda req: "x")
    svc.fake_world.register("alfa-q4", cancel_on(svc, 2, notes))
    run = run_inline(svc, ["rapida"], [c["id"]])
    assert run["state"] == "cancelled"
    assert notes[0]["note"].key == "run_stopping_own" and "llama-server that Galton started is stopped" in notes[0]["note"] and "shared" not in notes[0]["note"]
    assert svc.fake_world.stopped == ["alfa-q4"], "the server Galton started for the file is the one that was stopped"
    svc.db.close()


def test_a_run_with_both_kinds_says_which_is_stopped_and_which_is_left_alone(tmp_path, kills):
    server = SharedServer()
    svc = build_services(tmp_path, client_factory=server.factory())
    notes = []
    mine = add_gguf(svc, "alfa-q4")
    svc.fake_world.register("alfa-q4", cancel_on(svc, 2, notes))
    svc.fake_world.register("qwen-q4", lambda req: "x")
    shared = llama_contestant(svc)
    run = run_inline(svc, ["rapida"], [mine["id"], shared["id"]])
    assert run["state"] == "cancelled"
    note = notes[0]["note"]
    assert note.key == "run_stopping_both" and note.params["urls"] == "http://127.0.0.1:8081" and "llama-server that Galton started is stopped" in note and "keeps running" in note
    assert svc.fake_world.stopped == ["alfa-q4"] and kills == [] and server.requests == [], "the shared server was never even looked at: its turn never came"
    svc.db.close()


def test_the_note_only_counts_contestants_that_have_not_finished(tmp_path, kills):
    server = SharedServer()
    svc = build_services(tmp_path, client_factory=server.factory())
    notes = []
    mine = add_gguf(svc, "alfa-q4", responder=lambda req: "x")
    svc.fake_world.register("qwen-q4", cancel_on(svc, 2, notes))
    shared = llama_contestant(svc)
    run = run_inline(svc, ["rapida"], [mine["id"], shared["id"]])      # the file finishes first; the cancel comes while the shared server is being asked
    assert run["state"] == "cancelled" and notes[0]["note"].key == "run_stopping_shared"
    svc.db.close()


def test_cancelling_a_queued_run_has_nothing_to_stop(tmp_path, kills):
    svc = build_services(tmp_path)
    c = add_gguf(svc, "alfa-q4")
    run = svc.runner.create(suites=["rapida"], contestants=[c["id"]], source="test")
    out = call_tool(svc, "run_cancel", {"run": run["id"]})
    assert out == {"run": run["id"], "state": "cancelled", "cancelled": True}
    assert svc.fake_world.stopped == [] and kills == []
    svc.db.close()


def test_the_tool_and_the_catalogue_do_not_promise_that_a_server_is_stopped():
    from galton_hoard.agent_tools import TOOLS_BY_NAME
    from galton_hoard.messages import TEXTS
    description = TOOLS_BY_NAME["run_cancel"].description
    assert "shared servers untouched" in description and "never stopped, killed or unloaded" in description
    assert "run_stopping" not in TEXTS, "the single sentence that said the server is stopped is gone"
    for key in ("run_stopping_shared", "run_stopping_both"):
        assert "not touched" in TEXTS[key]
