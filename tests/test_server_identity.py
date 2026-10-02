"""A shared server can be restarted by somebody else with another model on the same address. A run must notice, stop that contestant with a clear
error, store nothing that was answered after the change, and the entry of the old model must be marked as not served (never measured under the
new model's name). The servers here are mock HTTP worlds that swap their model on command; time is a fake clock moved by the answers."""

from __future__ import annotations

import httpx

from galton_hoard.agent_tools import call_tool
from galton_hoard.backends import ChatRequest, OllamaBackend, OpenAIBackend
from galton_hoard.discovery import Discovery
from galton_hoard.errors import GaltonError
from galton_hoard.fakes import FakeWorld
from galton_hoard.identity import CHECK_TTL_S, Served, judge, look_openai
from galton_hoard.messages import CodedText
from helpers import Clock, build_services, json_response, mock_client_factory, run_inline
import pytest

URL = "http://127.0.0.1:8081"
BLOB_A, BLOB_B = "/models/blobs/sha256-f5f1.gguf", "/models/blobs/sha256-2bb2.gguf"


class SharedServer:
    """A llama-server on 8081 that other people can restart with another model."""

    def __init__(self, alias="qwen-q4", path=BLOB_A):
        self.alias, self.path = alias, path
        self.requests: list[tuple[str, str]] = []
        self.down = False

    def swap(self, alias: str, path: str) -> None:
        self.alias, self.path = alias, path

    def count(self, path: str) -> int:
        return sum(1 for _m, p in self.requests if p == path)

    def handler(self, req: httpx.Request) -> httpx.Response:
        self.requests.append((req.method, req.url.path))
        if self.down or req.url.port != 8081:
            raise httpx.ConnectError("refused")
        if req.url.path == "/v1/models":
            return json_response({"data": [{"id": self.alias}]})
        if req.url.path == "/props":
            return json_response({"model_alias": self.alias, "model_path": self.path, "default_generation_settings": {"n_ctx": 8192}})
        if req.url.path == "/slots":
            return json_response([{"is_processing": False}])
        if req.url.path == "/v1/chat/completions":
            return json_response({"choices": [{"message": {"content": "Hi"}}]})
        return json_response({}, 404)

    def factory(self):
        return mock_client_factory(self.handler)


def build(tmp_path, server, responder, *, clock=None):
    clock = clock or Clock()
    world = FakeWorld()
    world.register("qwen-q4", responder)
    svc = build_services(tmp_path, clock=clock, world=world, client_factory=server.factory())
    return svc, world, clock


def llama_contestant(svc, alias="qwen-q4", path=BLOB_A, **extra):
    return svc.store.create_contestant(key=f"server:llamacpp:{alias}", kind="server", name=alias, url=URL, api="openai", model=alias, provider="llamacpp", aliases=[alias],
                                       digest="sha256:" + "a" * 64, path=path, source="llamacpp", context=8192, meta={"up": True}, **extra)


def first_contestant(svc, run):
    return svc.run_card(svc.store.run(run["id"]))["contestants"][0]


# ------------------------------------------------------------------ a swap in the middle of a run
def test_a_model_swapped_mid_run_stops_the_contestant_and_nothing_after_the_change_is_stored(tmp_path):
    server = SharedServer()
    state = {"n": 0}
    holder = {}

    def answer(req):
        state["n"] += 1
        holder["clock"].advance(CHECK_TTL_S + 5)             # every answer is older than the check, so each case looks at the server again
        if state["n"] == 5:                                  # somebody restarts the server while the fifth question is answered
            server.swap("qwen-q8", BLOB_B)
        return f"R{state['n']}"
    clock = Clock()
    holder["clock"] = clock
    svc, world, _ = build(tmp_path, server, answer, clock=clock)
    c = llama_contestant(svc)
    run = run_inline(svc, ["rapida"], [c["id"]])
    card = first_contestant(svc, run)
    assert run["state"] == "failed" and card["state"] == "failed"
    assert isinstance(card["error"], CodedText) and card["error"].key == "server_changed"
    assert card["error"].params["url"] == URL and card["error"].params["model"] == "qwen-q8" and "refresh the models" in str(card["error"]) and card["hint"]
    rows = svc.store.results(run_id=run["id"])
    assert sorted(r["output"] for r in rows) == ["R1", "R2", "R3", "R4"], "the fifth answer came from the new model: it is not stored; the four before it were checked after they were given"
    assert all(r["digest"] == c["digest"] for r in rows)
    assert state["n"] == 5, "no sixth question was asked of the new model"
    assert any(getattr(w, "key", "") == "warn_results_dropped" and w.params["n"] == 1 for w in card["warnings"])
    fresh = svc.store.contestant(c["id"])
    assert fresh["meta"]["not_served"]["model"] == "qwen-q8" and svc.store.notices(kind="not_served")
    svc.db.close()


def test_a_server_already_swapped_when_the_run_starts_is_refused_before_any_question(tmp_path):
    server = SharedServer()
    svc, world, _ = build(tmp_path, server, lambda req: "x")
    c = llama_contestant(svc)
    server.swap("qwen-q8", BLOB_B)                        # the run is started later, against the id of the old model
    run = run_inline(svc, ["rapida"], [c["id"]])
    card = first_contestant(svc, run)
    assert card["state"] == "failed" and card["error"].key == "server_changed" and card["error"].params["model"] == "qwen-q8"
    assert svc.store.results(run_id=run["id"]) == [] and world.calls == [], "not one question reached the new model"
    assert svc.store.contestant(c["id"])["meta"]["not_served"]["url"] == URL
    svc.db.close()


def test_the_same_alias_with_another_file_is_another_model(tmp_path):
    server = SharedServer()
    svc, world, _ = build(tmp_path, server, lambda req: "x")
    c = llama_contestant(svc)
    server.swap("qwen-q4", BLOB_B)                        # the alias was kept, the file is not the one that was measured
    run = run_inline(svc, ["rapida"], [c["id"]])
    assert first_contestant(svc, run)["error"].key == "server_changed" and world.calls == []
    svc.db.close()


def test_the_model_field_of_a_reply_catches_a_swap_between_two_looks(tmp_path):
    server = SharedServer()
    state = {"n": 0}

    def answer(req):
        state["n"] += 1
        return {"text": f"R{state['n']}", "served_model": "qwen-q4" if state["n"] < 5 else "qwen-q8"}     # the clock never moves: no look at the server is due
    svc, world, _ = build(tmp_path, server, answer)
    c = llama_contestant(svc)
    run = run_inline(svc, ["rapida"], [c["id"]])
    card = first_contestant(svc, run)
    assert card["state"] == "failed" and card["error"].key == "server_changed" and card["error"].params["model"] == "qwen-q8"
    assert svc.store.results(run_id=run["id"]) == [], "nothing since the last look at the server can be told from the new model's answers"
    assert any(getattr(w, "key", "") == "warn_results_dropped" and w.params["n"] == 4 for w in card["warnings"])
    assert state["n"] == 5
    svc.db.close()


def test_a_reply_that_keeps_saying_the_same_model_changes_nothing(tmp_path):
    server = SharedServer()
    svc, _, _ = build(tmp_path, server, lambda req: {"text": "x", "served_model": "qwen-q4"})
    c = llama_contestant(svc)
    run = run_inline(svc, ["rapida"], [c["id"]])
    assert run["state"] == "done" and len(svc.store.results(run_id=run["id"])) == 12
    svc.db.close()


# ------------------------------------------------------------------ how often the server is looked at
def test_the_server_is_looked_at_once_at_the_start_and_once_after_the_last_answer_when_no_time_passes(tmp_path):
    server = SharedServer()
    svc, _, _ = build(tmp_path, server, lambda req: "x")
    c = llama_contestant(svc)
    run = run_inline(svc, ["rapida"], [c["id"]])
    assert run["state"] == "done" and len(svc.store.results(run_id=run["id"])) == 12
    assert server.count("/v1/models") == 2, "the start (the probe that was already made) and the final check; the 20 s cache covers the cases in between"
    svc.db.close()


def test_the_cache_is_20_seconds_so_slow_cases_look_at_the_server_every_time(tmp_path):
    server = SharedServer()
    clock = Clock()

    def answer(req):
        clock.advance(CHECK_TTL_S + 1)
        return "x"
    svc, _, _ = build(tmp_path, server, answer, clock=clock)
    c = llama_contestant(svc)
    run = run_inline(svc, ["rapida"], [c["id"]])
    assert run["state"] == "done" and len(svc.store.results(run_id=run["id"])) == 12
    assert server.count("/v1/models") >= 12 and server.count("/props") == server.count("/v1/models")
    svc.db.close()


def test_results_are_stored_only_after_a_look_that_comes_after_them(tmp_path):
    server = SharedServer()
    clock = Clock()
    seen = []

    def answer(req):
        clock.advance(CHECK_TTL_S + 1)
        seen.append(len(svc.store.results(run_id=svc.store.runs()[0]["id"])) if svc.store.runs() else 0)
        return "x"
    svc, _, _ = build(tmp_path, server, answer, clock=clock)
    c = llama_contestant(svc)
    run_inline(svc, ["rapida"], [c["id"]])
    assert seen[0] == 0 and seen[1] <= 1 and seen[-1] >= 9, "while question k is being answered the results of at most the last one are not stored yet"
    svc.db.close()


def test_a_server_that_cannot_be_looked_at_after_the_last_answer_loses_the_unchecked_results_and_says_so(tmp_path):
    server = SharedServer()
    state = {"n": 0}

    def answer(req):
        state["n"] += 1
        if state["n"] == 12:
            server.down = True                              # it dies after the last answer
        return "x"
    svc, _, _ = build(tmp_path, server, answer)
    c = llama_contestant(svc)
    run = run_inline(svc, ["rapida"], [c["id"]])
    card = first_contestant(svc, run)
    assert card["state"] == "failed" and card["error"].key == "server_unverified" and card["error"].params["n"] == 12 and card["hint"]
    assert svc.store.results(run_id=run["id"]) == []
    svc.db.close()


def test_a_server_that_restarts_with_the_same_model_is_the_same_server(tmp_path):
    server = SharedServer()
    state = {"n": 0}
    clock = Clock()

    def answer(req):
        state["n"] += 1
        clock.advance(CHECK_TTL_S + 1)
        server.down = state["n"] == 4                       # a moment without an answer, then the same model again
        return "x"
    svc, _, _ = build(tmp_path, server, answer, clock=clock)
    c = llama_contestant(svc)
    run = run_inline(svc, ["rapida"], [c["id"]])
    assert run["state"] == "done" and len(svc.store.results(run_id=run["id"])) == 12
    assert "not_served" not in svc.store.contestant(c["id"])["meta"]
    svc.db.close()


# ------------------------------------------------------------------ names given by hand
def test_a_model_name_given_by_hand_is_pinned_on_first_use_and_checked_afterwards(tmp_path):
    server = SharedServer()
    svc, world, _ = build(tmp_path, server, lambda req: "x")
    world.register("etiqueta", lambda req: "x")
    spec = {"kind": "server", "url": URL, "model": "etiqueta"}       # a label that the server does not list
    run = run_inline(svc, ["rapida"], [spec])
    assert run["state"] == "done"
    pinned = svc.store.find_contestant("etiqueta")
    assert pinned["meta"]["identity"]["alias"] == "qwen-q4" and pinned["meta"]["identity"]["path"] == BLOB_A
    server.swap("qwen-q8", BLOB_B)
    run = run_inline(svc, ["rapida"], [spec])
    assert first_contestant(svc, run)["error"].key == "server_changed"
    server.swap("qwen-q4", BLOB_A)
    assert run_inline(svc, ["rapida"], [spec])["state"] == "done", "it serves what it served: the run goes on"
    svc.db.close()


def test_a_server_that_says_nothing_about_itself_is_not_checked(tmp_path):
    def silent(req):
        if req.url.path == "/v1/models":
            return json_response({"data": [{"id": "x"}]})
        return json_response({}, 404)
    world = FakeWorld()
    world.register("x", lambda req: "ok")
    svc = build_services(tmp_path, world=world, client_factory=mock_client_factory(silent))
    c = svc.store.create_contestant(key="server:manual:x", kind="server", name="x", url=URL, api="openai", model="", source="manual", context=8192)
    run = run_inline(svc, ["rapida"], [c["id"]])
    assert run["state"] == "done" and len(svc.store.results(run_id=run["id"])) == 12
    svc.db.close()


# ------------------------------------------------------------------ Ollama: the digest of the tag
def test_an_ollama_tag_that_now_has_another_digest_is_not_the_model_that_was_measured(tmp_path):
    state = {"digest": "d1" * 32}

    def handler(req):
        if req.url.path == "/api/tags":
            return json_response({"models": [{"name": "qwen3:8b", "digest": state["digest"]}]})
        if req.url.path == "/api/ps":
            return json_response({"models": [{"name": "qwen3:8b", "expires_at": "t"}]})
        return json_response({}, 404)
    world = FakeWorld()
    world.register("qwen3:8b", lambda req: "x")
    svc = build_services(tmp_path, world=world, client_factory=mock_client_factory(handler))
    c = svc.store.create_contestant(key="server:ollama:qwen3:8b", kind="server", name="qwen3:8b", url="http://127.0.0.1:11434", api="ollama", model="qwen3:8b", provider="ollama",
                                    ollama_ref="qwen3:8b", digest="d1" * 32, source="ollama", context=8192)
    assert run_inline(svc, ["rapida"], [c["id"]])["state"] == "done"
    state["digest"] = "d2" * 32                          # the tag was pulled again: other weights under the same name
    run = run_inline(svc, ["rapida"], [c["id"]])
    card = first_contestant(svc, run)
    assert card["state"] == "failed" and card["error"].key == "server_changed" and "qwen3:8b" in card["error"].params["model"]
    svc.db.close()


# ------------------------------------------------------------------ models_refresh
def llama_world(server):
    def handler(req):
        if req.url.port == 8081:
            return server.handler(req)
        raise httpx.ConnectError("refused")
    return handler


def test_refresh_marks_the_old_model_as_not_served_makes_the_new_one_and_gives_the_old_one_back(tmp_path):
    server = SharedServer()
    svc, world, _ = build(tmp_path, server, lambda req: "x")
    svc.discovery = Discovery(svc.store, svc.settings, client_factory=mock_client_factory(llama_world(server)), clock=svc.clock, offline=False)
    first = call_tool(svc, "models_refresh", {})
    old = svc.store.contestant_by_key("server:llamacpp:qwen-q4")
    assert first["new"] == ["qwen-q4"] and first["not_served"] == [] and old["path"] == BLOB_A
    run = run_inline(svc, ["rapida"], [old["id"]])
    assert run["state"] == "done"
    before = svc.store.counts()["results"]

    server.swap("qwen-q8", BLOB_B)
    second = call_tool(svc, "models_refresh", {})
    new = svc.store.contestant_by_key("server:llamacpp:qwen-q8")
    old = svc.store.contestant(old["id"])
    assert second["new"] == ["qwen-q8"] and second["not_served"] == ["qwen-q4"] and second["served_again"] == []
    assert old["meta"]["not_served"]["model"] == "qwen-q8" and old["enabled"] is True and old["missing"] is False, "kept, with its history"
    assert "not_served" not in new["meta"] and svc.store.counts()["results"] == before
    card = svc.contestant_card(old)
    assert card["not_served"] is True and card["not_served_reason"].key == "model_not_served" and "qwen-q8" in str(card["not_served_reason"])
    assert [m["name"] for m in call_tool(svc, "models_list", {})["models"] if m["not_served"]] == ["qwen-q4"]
    assert svc.store.notices(kind="not_served")

    with pytest.raises(GaltonError) as raised:
        svc.start_run(suites=["rapida"], contestants=[old["id"]], source="test")
    assert raised.value.key == "model_not_served" and raised.value.code == "unavailable" and raised.value.params["model"] == "qwen-q8"
    plan = svc.plan_run(["rapida"], [old["id"], new["id"]])
    assert [p["key"] for p in plan["problems"]] == ["model_not_served"] and plan["problems"][0]["name"] == "qwen-q4"
    assert run_inline(svc, ["rapida"], [new["id"]])["state"] == "done", "the model that is served now can be measured"

    server.swap("qwen-q4", BLOB_A)                          # somebody starts the first model again
    third = call_tool(svc, "models_refresh", {})
    assert third["served_again"] == ["qwen-q4"] and third["not_served"] == ["qwen-q8"]
    assert "not_served" not in svc.store.contestant(old["id"])["meta"] and svc.store.contestant(new["id"])["meta"]["not_served"]["model"] == "qwen-q4"
    assert run_inline(svc, ["rapida"], [old["id"]])["state"] == "done"
    svc.db.close()


def test_a_mark_that_is_out_of_date_is_checked_against_the_server_before_a_run_is_refused(tmp_path):
    server = SharedServer()
    svc, _, _ = build(tmp_path, server, lambda req: "x")
    c = llama_contestant(svc)
    svc.store.update_contestant(c["id"], meta={"up": True, "not_served": {"since": 1.0, "model": "otro", "url": URL}})
    run = run_inline(svc, ["rapida"], [c["id"]])
    assert run["state"] == "done" and "not_served" not in svc.store.contestant(c["id"])["meta"], "the server serves it again: the mark went and the run started"
    svc.db.close()


def test_measure_new_and_the_watch_leave_a_not_served_entry_alone(tmp_path):
    server = SharedServer()
    svc, _, _ = build(tmp_path, server, lambda req: "x")
    c = llama_contestant(svc)
    svc.store.update_contestant(c["id"], meta={"up": True, "not_served": {"since": 1.0, "model": "otro", "url": URL}})
    assert svc.measure_new()["started"] is False
    assert svc.watch.candidates() == []
    svc.db.close()


def test_refresh_never_calls_a_remote_endpoint_to_check_what_it_serves(tmp_path):
    calls = []

    def handler(req):
        calls.append(str(req.url))
        raise httpx.ConnectError("refused")
    svc = build_services(tmp_path)
    svc.store.create_contestant(key="server:manual:r", kind="server", name="r", url="https://api.example.com", api="openai", model="m", provider="remote", remote=True, remote_ok=False,
                                source="manual")
    Discovery(svc.store, svc.settings, client_factory=mock_client_factory(handler), clock=svc.clock, offline=False).refresh()
    assert not any("example.com" in c for c in calls)
    svc.db.close()


# ------------------------------------------------------------------ the pieces
def test_judging_a_server_by_its_alias_its_file_and_its_ids():
    c = {"model": "qwen-q4", "path": BLOB_A, "digest": "", "meta": {}, "source": "llamacpp"}
    same = Served(url=URL, api="openai", models=("qwen-q4",), alias="qwen-q4", path=BLOB_A)
    assert judge(c, same)[0] == "same"
    assert judge(c, Served(url=URL, api="openai", models=("qwen-q4",), alias="qwen-q4", path=BLOB_A.upper().replace("/", "\\")))[0] == "same", "separators and case do not matter"
    assert judge(c, Served(url=URL, api="openai", models=("qwen-q8",), alias="qwen-q8", path=BLOB_B)) == ("different", "qwen-q8")
    assert judge(c, Served(url=URL, api="openai", models=("qwen-q4",), alias="qwen-q4", path=BLOB_B))[0] == "different"
    assert judge(c, Served(url=URL, api="openai", models=("qwen-q8",), alias="", path=""))[0] == "different"
    assert judge(c, Served(url=URL, api="openai"))[0] == "unknown"
    assert judge({**c, "path": ""}, Served(url=URL, api="openai", models=("a", "qwen-q4")))[0] == "same", "a server with several models"


def test_look_openai_reads_models_and_props_and_gives_up_quietly():
    server = SharedServer()
    with mock_client_factory(server.handler)() as client:
        seen = look_openai(client, URL)
    assert seen.models == ("qwen-q4",) and seen.alias == "qwen-q4" and seen.path == BLOB_A and seen.label == "qwen-q4"
    server.down = True
    with mock_client_factory(server.handler)() as client:
        assert look_openai(client, URL) is None


def sse(*chunks):
    return "".join(f"data: {c}\n\n" for c in chunks) + "data: [DONE]\n\n"


def test_the_backends_keep_the_model_field_of_the_reply():
    body = sse('{"model":"qwen-q4","choices":[{"delta":{"content":"hola"}}]}', '{"model":"qwen-q4","choices":[{"delta":{},"finish_reason":"stop"}]}')
    backend = OpenAIBackend(URL, "qwen-q4", transport=httpx.MockTransport(lambda r: httpx.Response(200, content=body)))
    done = backend.chat(ChatRequest(messages=[{"role": "user", "content": "hi"}]))
    assert done.text == "hola" and done.served_model == "qwen-q4"
    lines = '{"model":"qwen3:8b","message":{"content":"hola"},"done":false}\n{"model":"qwen3:8b","message":{"content":""},"done":true,"done_reason":"stop"}\n'
    ollama = OllamaBackend(URL, "qwen3:8b", transport=httpx.MockTransport(lambda r: httpx.Response(200, content=lines)))
    assert ollama.chat(ChatRequest(messages=[{"role": "user", "content": "hi"}])).served_model == "qwen3:8b"
