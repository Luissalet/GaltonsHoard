"""Discovery against a mocked network and disk, the two chat backends over a mock transport, and the llama-server launcher with fake processes."""

from __future__ import annotations

import json

import httpx
import pytest

from galton_hoard import backends
from galton_hoard.backends import Cancelled, ChatRequest, OllamaBackend, OpenAIBackend, make_backend, tools_prompt
from galton_hoard.discovery import Discovery, host_port, name_variants
from galton_hoard.errors import GaltonError
from galton_hoard.gpus import GpuGrant
from galton_hoard.port import can_listen, find_available_port, free_port
from galton_hoard.servers import Launcher, LaunchSpec, PORT_RANGE, probe_openai
from helpers import Clock, json_response, mock_client_factory
from test_gguf import LLAMA, make_store, write_gguf


# ---- discovery -------------------------------------------------------------------------------------------------------------------------------

def make_discovery(svc, handler, clock=None):
    return Discovery(svc.store, svc.settings, client_factory=mock_client_factory(handler), clock=clock or svc.clock, offline=False)


def ollama_handler(models, resident=(), caps=()):
    def handler(req):
        path = req.url.path
        if req.url.port == 11434 and path == "/api/tags":
            return json_response({"models": models})
        if req.url.port == 11434 and path == "/api/ps":
            return json_response({"models": list(resident)})
        if req.url.port == 11434 and path == "/api/show":
            return json_response({"details": {"family": "qwen3"}, "capabilities": list(caps), "model_info": {"general.architecture": "qwen3", "qwen3.context_length": 40960}})
        raise httpx.ConnectError("refused")
    return handler


TAG = {"name": "qwen3:8b", "digest": "sha256:" + "a" * 64, "size": 5_000_000_000, "details": {"family": "qwen3", "parameter_size": "8.2B", "quantization_level": "Q4_K_M"}}


def test_name_variants_and_host_port():
    assert name_variants("qwen3:latest") == ["qwen3:latest", "qwen3"]
    assert name_variants("D:\\models\\Tiny-8B-Q4_K_M.gguf") == ["Tiny-8B-Q4_K_M"], "a path gives its file stem, never the path"
    assert host_port("http://LOCALHOST:8080/v1") == "localhost:8080" and host_port("127.0.0.1:7000") == "127.0.0.1:7000"


def test_offline_discovery_does_nothing(svc):
    out = Discovery(svc.store, svc.settings, offline=True).refresh()
    assert out["offline"] is True and out["new"] == []


def test_ollama_models_become_server_contestants(svc, tmp_path):
    svc.settings.set_many({"ollama.models_dir": str(tmp_path / "none")})
    out = make_discovery(svc, ollama_handler([TAG], resident=[{"name": "qwen3:8b", "size_vram": 6 * 1024 ** 3}], caps=["vision"])).refresh()
    assert out["sources"]["ollama"]["ok"] and out["sources"]["ollama"]["models"] == 1 and len(out["new"]) == 1
    c = svc.store.contestant_by_key("server:ollama:qwen3:8b")
    assert c["kind"] == "server" and c["api"] == "ollama" and c["params_b"] == 8.2 and c["quant"] == "Q4_K_M" and c["context"] == 40960
    assert c["vision"] is True and c["meta"]["resident"] is True and c["enabled"] is True
    assert "qwen3:8b" in c["aliases"] and c["digest"] == TAG["digest"]
    assert svc.store.notices(kind="new_model")


def test_an_ollama_model_with_a_blob_on_disk_also_gets_a_llama_cpp_twin(svc, tmp_path):
    root = make_store(tmp_path)
    svc.settings.set_many({"ollama.models_dir": str(root)})
    out = make_discovery(svc, ollama_handler([TAG])).refresh()
    assert len(out["new"]) == 2
    server, twin = svc.store.contestant_by_key("server:ollama:qwen3:8b"), svc.store.contestant_by_key("gguf:ollama:qwen3:8b")
    assert twin["kind"] == "gguf" and twin["name"] == "qwen3:8b (llama.cpp)" and twin["path"].endswith("sha256-" + "a" * 64) and twin["meta"]["blob"] is True
    assert server["enabled"] is False and twin["enabled"] is True      # the twin is the one to test, the server would load it into Ollama


def test_refreshing_twice_updates_instead_of_duplicating(svc, tmp_path):
    svc.settings.set_many({"ollama.models_dir": str(tmp_path / "none")})
    d = make_discovery(svc, ollama_handler([TAG]))
    d.refresh()
    second = d.refresh()
    assert second["new"] == [] and len(second["updated"]) == 1 and len(svc.store.contestants()) == 1


def test_a_changed_digest_marks_the_model_changed_and_stale(svc, tmp_path):
    svc.settings.set_many({"ollama.models_dir": str(tmp_path / "none")})
    make_discovery(svc, ollama_handler([TAG])).refresh()
    out = make_discovery(svc, ollama_handler([{**TAG, "digest": "sha256:" + "f" * 64}])).refresh()
    c = svc.store.contestant_by_key("server:ollama:qwen3:8b")
    assert len(out["changed"]) == 1 and c["meta"]["previous_digest"] == TAG["digest"] and c["digest"].endswith("f" * 64)
    assert svc.store.notices(kind="changed_model")


def test_a_model_that_disappears_is_marked_missing_not_deleted(svc, tmp_path):
    svc.settings.set_many({"ollama.models_dir": str(tmp_path / "none")})
    make_discovery(svc, ollama_handler([TAG])).refresh()
    out = make_discovery(svc, ollama_handler([])).refresh()
    c = svc.store.contestant_by_key("server:ollama:qwen3:8b")
    assert out["missing"] == [c["id"]] and c["missing"] is True
    make_discovery(svc, ollama_handler([TAG])).refresh()
    assert svc.store.contestant_by_key("server:ollama:qwen3:8b")["missing"] is False


def test_nothing_is_marked_missing_when_ollama_could_not_be_asked(svc, tmp_path):
    svc.settings.set_many({"ollama.models_dir": str(tmp_path / "none")})
    make_discovery(svc, ollama_handler([TAG])).refresh()
    out = make_discovery(svc, lambda req: (_ for _ in ()).throw(httpx.ConnectError("refused"))).refresh()
    assert out["sources"]["ollama"]["ok"] is False and out["missing"] == []


def test_manual_and_adhoc_models_are_never_marked_missing(svc):
    svc.store.create_contestant(key="gguf:/gone.gguf", kind="gguf", name="manual", path="/gone.gguf", source="manual")
    out = make_discovery(svc, ollama_handler([])).refresh()
    assert out["missing"] == []


def test_a_user_chosen_name_survives_a_refresh(svc, tmp_path):
    svc.settings.set_many({"ollama.models_dir": str(tmp_path / "none")})
    d = make_discovery(svc, ollama_handler([TAG]))
    d.refresh()
    c = svc.store.contestant_by_key("server:ollama:qwen3:8b")
    svc.store.update_contestant(c["id"], name="Mi Qwen", meta={**c["meta"], "renamed": True})
    d.refresh()
    assert svc.store.contestant(c["id"])["name"] == "Mi Qwen"


def test_llama_servers_on_the_shared_ports_are_found(svc):
    def handler(req):
        if req.url.port == 8082 and req.url.path == "/v1/models":
            return json_response({"data": [{"id": "gemma-3-12b-it-Q4_K_M"}]})
        if req.url.port == 8082 and req.url.path == "/props":
            return json_response({"model_alias": "gemma-3-12b-it-Q4_K_M", "default_generation_settings": {"n_ctx": 16384}, "modalities": {"vision": True}})
        if req.url.port == 8082 and req.url.path == "/slots":
            return json_response([{"is_processing": False}])
        if req.url.port == 11434:
            raise httpx.ConnectError("refused")
        raise httpx.ConnectError("refused")
    out = make_discovery(svc, handler).refresh()
    assert out["sources"]["llamacpp"] == {"ok": True, "servers": 1}
    c = svc.store.contestant_by_key("server:llamacpp:gemma-3-12b-it-q4-k-m")
    assert c["url"] == "http://127.0.0.1:8082" and c["context"] == 16384 and c["vision"] is True and c["params_b"] == 12.0 and c["quant"] == "Q4_K_M" and c["meta"]["up"] is True


def test_a_llama_server_that_stops_answering_is_down_not_gone(svc):
    state = {"up": True}

    def handler(req):
        if state["up"] and req.url.port == 8080 and req.url.path == "/v1/models":
            return json_response({"data": [{"id": "m"}]})
        raise httpx.ConnectError("refused")
    d = make_discovery(svc, handler)
    d.refresh()
    state["up"] = False
    out = d.refresh()
    c = svc.store.contestant_by_key("server:llamacpp:m")
    assert c["meta"]["up"] is False and c["missing"] is False and out["missing"] == []


def test_gguf_folders_are_scanned_with_shards_and_projectors(svc, tmp_path):
    folder = tmp_path / "gguf"
    (folder / "sub").mkdir(parents=True)
    write_gguf(folder / "Tiny-8B-Q4_K_M.gguf", LLAMA)
    write_gguf(folder / "Big-70B-Q4_K_M-00001-of-00002.gguf", {**LLAMA, "general.size_label": "70B"})
    write_gguf(folder / "Big-70B-Q4_K_M-00002-of-00002.gguf", {**LLAMA, "general.size_label": "70B"})
    write_gguf(folder / "sub" / "Vis-4B-Q8_0.gguf", {**LLAMA, "general.size_label": "4B"})
    (folder / "sub" / "mmproj-Vis-4B-f16.gguf").write_bytes(b"GGUF")      # not a model: skipped by name
    svc.settings.set_many({"gguf.folders": [str(folder)]})
    out = Discovery(svc.store, svc.settings, client_factory=mock_client_factory(lambda r: (_ for _ in ()).throw(httpx.ConnectError("x"))), clock=svc.clock, offline=False).refresh()
    names = sorted(c["name"] for c in svc.store.contestants(kind="gguf"))
    assert names == ["Big-70B-Q4_K_M", "Tiny-8B-Q4_K_M", "Vis-4B-Q8_0"] and out["sources"]["folders"]["ok"]
    vis = svc.store.contestant_by_key(f"gguf:{folder / 'sub' / 'Vis-4B-Q8_0.gguf'}")
    assert vis["vision"] is True and vis["mmproj"].endswith("mmproj-Vis-4B-f16.gguf")
    big = svc.store.contestant_by_key(f"gguf:{folder / 'Big-70B-Q4_K_M-00001-of-00002.gguf'}")
    assert big["size_bytes"] == sum(p.stat().st_size for p in folder.glob("Big-*.gguf"))


def test_a_gguf_that_cannot_be_read_is_reported_not_fatal(svc, tmp_path):
    folder = tmp_path / "g"
    folder.mkdir()
    (folder / "roto.gguf").write_bytes(b"not a gguf")
    write_gguf(folder / "bueno-8B-Q4_K_M.gguf", LLAMA)
    svc.settings.set_many({"gguf.folders": [str(folder)]})
    out = Discovery(svc.store, svc.settings, client_factory=mock_client_factory(lambda r: (_ for _ in ()).throw(httpx.ConnectError("x"))), clock=svc.clock, offline=False).refresh()
    assert any("roto.gguf" in e for e in out["errors"]) and [c["name"] for c in svc.store.contestants(kind="gguf")] == ["bueno-8B-Q4_K_M"]


def test_a_deleted_gguf_file_becomes_missing(svc, tmp_path):
    folder = tmp_path / "g"
    folder.mkdir()
    f = write_gguf(folder / "tmp-8B-Q4_K_M.gguf", LLAMA)
    svc.settings.set_many({"gguf.folders": [str(folder)]})
    d = Discovery(svc.store, svc.settings, client_factory=mock_client_factory(lambda r: (_ for _ in ()).throw(httpx.ConnectError("x"))), clock=svc.clock, offline=False)
    d.refresh()
    f.unlink()
    out = d.refresh()
    assert len(out["missing"]) == 1


def test_faustus_registry_adds_local_servers_and_ignores_cloud_ones(svc):
    items = [{"name": "Mistral local", "model": "mistral", "url": "http://127.0.0.1:8085", "backend": "llamacpp", "category": "local", "model_type": "llm"},
             {"name": "Nube", "model": "x", "url": "https://api.example.com", "category": "cloud", "model_type": "llm"},
             {"name": "Imagen", "model": "i", "url": "http://127.0.0.1:8188", "category": "local", "model_type": "image"}]

    def handler(req):
        if req.url.port == 7000 and req.url.path == "/api/models":
            assert req.headers.get("authorization") == "Bearer tok"
            return json_response({"items": items})
        raise httpx.ConnectError("refused")
    svc.settings.set_many({"faustus.token": "tok"})
    out = make_discovery(svc, handler).refresh()
    assert out["sources"]["faustus"] == {"ok": True, "items": 3, "added": 1}
    c = svc.store.contestant_by_key("server:faustus:mistral-local")
    assert c["url"] == "http://127.0.0.1:8085" and c["api"] == "openai" and c["source"] == "faustus"


def test_a_faustus_server_already_known_only_gains_an_alias(svc):
    server = svc.store.create_contestant(key="server:x", kind="server", name="x", url="http://127.0.0.1:8085", api="openai", model="m", aliases=["x"], source="manual")
    items = [{"name": "Alias nuevo", "model": "m", "url": "http://127.0.0.1:8085", "category": "local", "model_type": "llm"}]
    out = make_discovery(svc, lambda req: json_response({"items": items}) if req.url.port == 7000 else (_ for _ in ()).throw(httpx.ConnectError("x"))).refresh()
    assert out["sources"]["faustus"]["added"] == 0 and "Alias nuevo" in svc.store.contestant(server["id"])["aliases"]


# ---- backends: OpenAI-compatible -------------------------------------------------------------------------------------------------------------

def sse(*chunks):
    body = "".join(f"data: {json.dumps(c)}\n\n" if not isinstance(c, str) else f"data: {c}\n\n" for c in chunks)
    return httpx.Response(200, content=body.encode(), headers={"content-type": "text/event-stream"})


def openai(handler):
    return OpenAIBackend("http://x:1", "m", transport=httpx.MockTransport(handler))


def delta(text=None, **extra):
    return {"choices": [{"delta": {"content": text, **extra}, "finish_reason": None}]}


def test_openai_streaming_collects_text_usage_and_llama_timings():
    seen = {}

    def handler(req):
        seen["body"] = json.loads(req.content)
        return sse(delta("Hola "), delta("mundo"), {"choices": [{"delta": {}, "finish_reason": "stop"}],
                                                     "timings": {"prompt_n": 12, "predicted_n": 40, "predicted_per_second": 55.55, "prompt_per_second": 900.0}}, "[DONE]")
    c = openai(handler).chat(ChatRequest(messages=[{"role": "user", "content": "hi"}], temperature=0.2, seed=7, top_p=0.9, max_tokens=64))
    assert c.text == "Hola mundo" and c.error == "" and c.finish_reason == "stop"
    assert (c.prompt_tokens, c.completion_tokens, c.decode_tps, c.prompt_tps) == (12, 40, 55.55, 900.0) and c.ttft_ms is not None
    body = seen["body"]
    assert body["stream"] is True and body["seed"] == 7 and body["top_p"] == 0.9 and body["max_tokens"] == 64 and body["temperature"] == 0.2 and body["model"] == "m"


def test_openai_reasoning_content_is_kept_apart_and_think_tags_are_split():
    c = openai(lambda req: sse(delta(None, reasoning_content="pienso"), delta("<think>más</think>respuesta"), "[DONE]")).chat(ChatRequest(messages=[{"role": "user", "content": "x"}]))
    assert c.text == "respuesta" and "pienso" in c.reasoning and "más" in c.reasoning


def test_openai_tool_calls_are_assembled_from_fragments():
    def handler(req):
        return sse({"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"name": "get_", "arguments": '{"ci'}}]}}]},
                   {"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"name": "weather", "arguments": 'ty": "Madrid"}'}}]}, "finish_reason": "tool_calls"}]}, "[DONE]")
    c = openai(handler).chat(ChatRequest(messages=[{"role": "user", "content": "x"}], tools=[{"type": "function", "function": {"name": "get_weather"}}]))
    assert c.tool_calls == [{"name": "get_weather", "arguments": {"city": "Madrid"}}] and c.finish_reason == "tool_calls"


def test_openai_images_go_into_the_last_user_message():
    seen = {}
    openai(lambda req: (seen.update(body=json.loads(req.content)), sse(delta("ok"), "[DONE]"))[1]).chat(ChatRequest(messages=[{"role": "user", "content": "mira"}], images=[b"\x89PNG"]))
    content = seen["body"]["messages"][-1]["content"]
    assert content[0] == {"type": "text", "text": "mira"} and content[1]["image_url"]["url"].startswith("data:image/png;base64,")


def test_openai_falls_back_to_tools_in_the_prompt_when_the_server_refuses_them():
    calls = []

    def handler(req):
        body = json.loads(req.content)
        calls.append(body)
        if body.get("tools"):
            return httpx.Response(400, text='{"error": "tools are not supported"}')
        return sse(delta('{"name": "f", "arguments": {}}'), "[DONE]")
    c = openai(handler).chat(ChatRequest(messages=[{"role": "user", "content": "x"}], tools=[{"type": "function", "function": {"name": "f", "description": "d", "parameters": {}}}]))
    assert len(calls) == 2 and "tools" not in calls[1] and "f: d" in calls[1]["messages"][0]["content"]
    assert any(n.startswith("tools_in_prompt") for n in c.notes) and c.text.startswith("{")


def test_openai_retries_without_reasoning_fields_when_they_are_rejected():
    calls = []

    def handler(req):
        body = json.loads(req.content)
        calls.append(body)
        if len(calls) == 1:
            return httpx.Response(400, text='{"error": "unknown field reasoning_effort"}')
        return sse(delta("ok"), "[DONE]")
    c = openai(handler).chat(ChatRequest(messages=[{"role": "user", "content": "x"}], effort="high"))
    assert c.text == "ok" and len(calls) == 2 and any("reasoning" in n for n in c.notes)
    assert not any(k in calls[1] for k in ("reasoning_effort", "reasoning", "chat_template_kwargs"))


def test_openai_http_errors_and_connection_errors_become_completion_errors():
    c = openai(lambda req: httpx.Response(500, text="kaput")).chat(ChatRequest(messages=[{"role": "user", "content": "x"}]))
    assert "HTTP 500" in c.error and "kaput" in c.error and c.text == ""

    def refuse(req):
        raise httpx.ConnectError("refused")
    c = openai(refuse).chat(ChatRequest(messages=[{"role": "user", "content": "x"}]))
    assert c.error.startswith("connection error")

    def slow(req):
        raise httpx.ReadTimeout("slow")
    assert openai(slow).chat(ChatRequest(messages=[{"role": "user", "content": "x"}], timeout_s=5)).error.startswith("timeout")


def test_openai_stream_error_chunk_is_an_error():
    c = openai(lambda req: sse({"error": {"message": "out of memory"}})).chat(ChatRequest(messages=[{"role": "user", "content": "x"}]))
    assert "out of memory" in c.error


def test_cancel_raises_cancelled_mid_stream():
    with pytest.raises(Cancelled):
        openai(lambda req: sse(delta("a"), delta("b"), "[DONE]")).chat(ChatRequest(messages=[{"role": "user", "content": "x"}]), cancel=lambda: True)


def test_decode_speed_is_estimated_from_usage_when_the_server_gives_no_timings(monkeypatch):
    ticks = iter([0.0, 0.5, 2.5, 2.5, 2.5, 2.5])
    monkeypatch.setattr(backends.time, "monotonic", lambda: next(ticks, 3.0))
    c = openai(lambda req: sse(delta("a"), {"choices": [], "usage": {"prompt_tokens": 3, "completion_tokens": 21}}, "[DONE]")).chat(ChatRequest(messages=[{"role": "user", "content": "x"}]))
    assert c.completion_tokens == 21 and c.decode_tps is not None and c.decode_tps > 0


# ---- backends: Ollama ------------------------------------------------------------------------------------------------------------------------

def ndjson(*chunks):
    return httpx.Response(200, content=("\n".join(json.dumps(c) for c in chunks) + "\n").encode(), headers={"content-type": "application/x-ndjson"})


def ollama(handler):
    return OllamaBackend("http://x:11434", "qwen3:8b", transport=httpx.MockTransport(handler))


def test_ollama_streaming_reads_durations_in_nanoseconds():
    seen = {}

    def handler(req):
        seen["body"] = json.loads(req.content)
        return ndjson({"message": {"content": "Ho"}}, {"message": {"content": "la"}},
                      {"done": True, "done_reason": "stop", "prompt_eval_count": 10, "prompt_eval_duration": 500_000_000, "eval_count": 50, "eval_duration": 1_000_000_000,
                       "load_duration": 2_000_000_000, "message": {"content": ""}})
    c = ollama(handler).chat(ChatRequest(messages=[{"role": "user", "content": "hi"}], max_tokens=33, context=4096, seed=3))
    assert c.text == "Hola" and c.decode_tps == 50.0 and c.prompt_tps == 20.0 and c.load_ms == 2000.0 and c.finish_reason == "stop"
    options = seen["body"]["options"]
    assert options["num_predict"] == 33 and options["num_ctx"] == 4096 and options["seed"] == 3 and seen["body"]["stream"] is True


def test_ollama_thinking_and_tool_calls():
    c = ollama(lambda req: ndjson({"message": {"thinking": "mmm"}}, {"message": {"content": "", "tool_calls": [{"function": {"name": "f", "arguments": {"a": 1}}}]}}, {"done": True})
               ).chat(ChatRequest(messages=[{"role": "user", "content": "x"}]))
    assert c.reasoning == "mmm" and c.tool_calls == [{"name": "f", "arguments": {"a": 1}}]


def test_ollama_images_and_think_retry():
    calls = []

    def handler(req):
        body = json.loads(req.content)
        calls.append(body)
        if "think" in body:
            return httpx.Response(400, text='{"error": "model does not support think"}')
        return ndjson({"message": {"content": "ok"}}, {"done": True})
    c = ollama(handler).chat(ChatRequest(messages=[{"role": "user", "content": "x"}], images=[b"img"], effort="high"))
    assert c.text == "ok" and len(calls) == 2 and "think" not in calls[1] and calls[1]["messages"][-1]["images"]


def test_ollama_errors():
    assert "HTTP 404" in ollama(lambda req: httpx.Response(404, text="model not found")).chat(ChatRequest(messages=[{"role": "user", "content": "x"}])).error
    assert "disk" in ollama(lambda req: ndjson({"error": "disk"})).chat(ChatRequest(messages=[{"role": "user", "content": "x"}])).error


def test_make_backend_and_tools_prompt():
    assert isinstance(make_backend("ollama", "http://x", "m"), OllamaBackend) and isinstance(make_backend("openai", "http://x", "m"), OpenAIBackend)
    with pytest.raises(GaltonError):
        make_backend("grpc", "http://x", "m")
    assert "get_time: hora" in tools_prompt([{"function": {"name": "get_time", "description": "hora", "parameters": {}}}])


# ---- servers: the launcher ---------------------------------------------------------------------------------------------------------------------

class FakeProc:
    _next = 4000

    def __init__(self, exits_with=None):
        FakeProc._next += 1
        self.pid = FakeProc._next
        self.exits_with = exits_with
        self.killed = False

    def poll(self):
        return -9 if self.killed else self.exits_with

    def kill(self):
        self.killed = True

    def wait(self, timeout=None):
        return 0


@pytest.fixture
def launcher_parts(tmp_path, svc, monkeypatch):
    # the fake child has a made-up pid: never let a test signal whatever process happens to own that number
    monkeypatch.setattr("galton_hoard.servers.kill_tree", lambda proc, **kw: proc.kill())
    binary = tmp_path / "llama-server"
    binary.write_text("#!/bin/sh\n")
    svc.settings.set_many({"llama.server_path": str(binary)})
    procs = []
    cmds = []

    def popen(cmd, **kw):
        cmds.append((cmd, kw))
        proc = FakeProc(exits_with=procs[0] if procs else None)
        return proc
    clock = Clock(0.0)
    health = {"ok": True}
    client = mock_client_factory(lambda req: json_response({"status": "ok"}) if health["ok"] else httpx.Response(503, json={"status": "loading"}))
    launcher = Launcher(svc.settings, tmp_path / "logs", tmp_path / "servers.json", popen=popen, which=lambda n: None, client_factory=client, port_free=lambda p: True,
                        sleep=lambda s: clock.advance(s), clock=clock)
    return launcher, cmds, health, procs, clock


def test_the_command_has_the_right_flags_split_and_extra_args(launcher_parts, svc, tmp_path):
    launcher, *_ = launcher_parts
    svc.settings.set_many({"llama.extra_args": "--threads 4 --no-mmap"})
    spec = LaunchSpec(model_path="/m/x.gguf", name="x", context=4096, mmproj_path="/m/proj.gguf", grant=GpuGrant(gpus=[2, 3], shares_mb={2: 6000, 3: 4000}, via="test"), extra_args=["--ubatch-size", "256"])
    cmd = launcher.command(spec, 8091, "x")
    assert cmd[1:5] == ["-m", "/m/x.gguf", "-c", "4096"] and "--host" in cmd and cmd[cmd.index("--host") + 1] == "127.0.0.1" and cmd[cmd.index("--port") + 1] == "8091"
    assert cmd[cmd.index("--mmproj") + 1] == "/m/proj.gguf" and cmd[cmd.index("--tensor-split") + 1] == "0.6,0.4" and "--no-mmap" in cmd and cmd[-2:] == ["--ubatch-size", "256"]


def test_start_picks_a_port_in_8091_to_8099_sets_the_visible_gpus_and_records_the_child(launcher_parts):
    launcher, cmds, *_ = launcher_parts
    handle = launcher.start(LaunchSpec(model_path="/m/x.gguf", name="Mi Modelo", grant=GpuGrant(gpus=[3], shares_mb={3: 8000}, via="test")), timeout_s=30)
    assert handle.port in PORT_RANGE and handle.url == f"http://127.0.0.1:{handle.port}" and handle.alias == "mi-modelo" and handle.alive()
    assert cmds[0][1]["env"]["CUDA_VISIBLE_DEVICES"] == "3"
    assert [e["port"] for e in launcher._read_registry()] == [handle.port]
    handle.stop()
    assert handle.alive() is False and launcher._read_registry() == []


def test_two_servers_get_different_ports_and_a_full_range_is_an_error(launcher_parts):
    launcher, *_ = launcher_parts
    a = launcher.start(LaunchSpec(model_path="/m/a.gguf", name="a"), timeout_s=30)
    b = launcher.start(LaunchSpec(model_path="/m/b.gguf", name="b"), timeout_s=30)
    assert a.port != b.port and {a.port, b.port} <= set(PORT_RANGE)
    launcher._port_free = lambda p: False
    with pytest.raises(GaltonError) as exc:
        launcher.pick_port()
    assert exc.value.code == "busy"


def test_a_process_that_dies_while_loading_is_an_honest_error_with_the_log_tail(launcher_parts, tmp_path):
    launcher, cmds, health, procs, clock = launcher_parts
    procs.append(1)     # the fake process exits with code 1
    with pytest.raises(GaltonError) as exc:
        launcher.start(LaunchSpec(model_path="/m/x.gguf", name="x"), timeout_s=30)
    assert exc.value.code == "unavailable" and "exited with code 1" in exc.value.message
    assert launcher._read_registry() == []


def test_a_server_that_never_gets_ready_times_out_and_is_stopped(launcher_parts):
    launcher, cmds, health, procs, clock = launcher_parts
    health["ok"] = False
    with pytest.raises(GaltonError) as exc:
        launcher.start(LaunchSpec(model_path="/m/x.gguf", name="x"), timeout_s=5)
    assert "did not become ready" in exc.value.message and launcher._read_registry() == []


def test_cancelling_while_loading_stops_the_child(launcher_parts):
    launcher, cmds, health, *_ = launcher_parts
    health["ok"] = False
    with pytest.raises(GaltonError):
        launcher.start(LaunchSpec(model_path="/m/x.gguf", name="x"), cancel=lambda: True, timeout_s=30)
    assert launcher._read_registry() == []


def test_a_missing_binary_is_reported_with_where_it_looked(svc, tmp_path):
    svc.settings.set_many({"llama.server_path": str(tmp_path / "no-such")})
    launcher = Launcher(svc.settings, tmp_path / "logs", tmp_path / "servers.json", which=lambda n: None)
    with pytest.raises(GaltonError) as exc:
        launcher.binary()
    assert exc.value.code == "unavailable" and "llama-server" in exc.value.message
    launcher = Launcher(svc.settings, tmp_path / "logs", tmp_path / "servers.json", which=lambda n: "/usr/bin/llama-server" if n == "llama-server" else None)
    assert launcher.binary() == "/usr/bin/llama-server"


def test_popen_failure_is_reported(launcher_parts):
    launcher, *_ = launcher_parts

    def broken(cmd, **kw):
        raise OSError("exec format error")
    launcher._popen = broken
    with pytest.raises(GaltonError) as exc:
        launcher.start(LaunchSpec(model_path="/m/x.gguf", name="x"), timeout_s=30)
    assert "exec format error" in exc.value.message


def test_reap_orphans_kills_only_llama_processes(launcher_parts, monkeypatch):
    launcher, *_ = launcher_parts
    launcher._write_registry([{"pid": 111, "port": 8091}, {"pid": 222, "port": 8092}, {"pid": 333}])
    import galton_hoard.servers as servers_mod
    killed = []
    monkeypatch.setattr(servers_mod, "process_name", lambda pid: "llama-server.exe" if pid == 111 else "python")
    monkeypatch.setattr(servers_mod, "kill_pid_tree", lambda pid: killed.append(pid))
    assert launcher.reap_orphans() == [111] and killed == [111] and launcher._read_registry() == []


def test_a_corrupt_registry_reads_as_empty(launcher_parts):
    launcher, *_ = launcher_parts
    launcher.registry.write_text("not json", encoding="utf-8")
    assert launcher._read_registry() == []
    launcher.registry.write_text('{"a": 1}', encoding="utf-8")
    assert launcher._read_registry() == []


def test_probe_openai_reads_models_props_and_busy_slots():
    def handler(req):
        return {"/v1/models": json_response({"data": [{"id": "m"}]}), "/props": json_response({"model_alias": "m", "default_generation_settings": {"n_ctx": 4096},
                                                                                         "modalities": {"vision": False}, "chat_template": "x"}),
                "/slots": json_response([{"is_processing": True}])}[req.url.path]
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        info = probe_openai(client, "http://h:1/")
    assert info["up"] and info["models"] == ["m"] and info["context"] == 4096 and info["busy"] is True and info["has_template"] is True
    with httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(500))) as client:
        assert probe_openai(client, "http://h:1")["up"] is False


# ---- ports -----------------------------------------------------------------------------------------------------------------------------------

def test_ports():
    port = free_port()
    assert can_listen(port) and find_available_port(port) == port
    import socket
    s = socket.socket()
    s.bind(("127.0.0.1", port))
    s.listen(1)
    try:
        assert not can_listen(port) and find_available_port(port) > port
    finally:
        s.close()
    with pytest.raises(RuntimeError):
        find_available_port(65536, attempts=3)


def test_the_child_numbers_gpus_like_nvidia_smi_and_sees_none_without_a_grant(launcher_parts):
    launcher, cmds, *_ = launcher_parts
    launcher.start(LaunchSpec(model_path="/m/x.gguf", name="x", grant=GpuGrant(gpus=[2], shares_mb={2: 8000}, via="test")), timeout_s=30)
    env = cmds[-1][1]["env"]
    assert env["CUDA_DEVICE_ORDER"] == "PCI_BUS_ID" and env["CUDA_VISIBLE_DEVICES"] == "2"
    launcher.start(LaunchSpec(model_path="/m/y.gguf", name="y"), timeout_s=30)
    assert cmds[-1][1]["env"]["CUDA_VISIBLE_DEVICES"] == "" and cmds[-1][1]["env"]["CUDA_DEVICE_ORDER"] == "PCI_BUS_ID"


# ---- the same weights served two ways -------------------------------------------------------------------------------------------------------

BLOB = "sha256-" + "a" * 64


def combined_handler(models, server_path, *, port=8082, alias="qwen3-blob"):
    """Ollama on 11434 plus a llama-server on ``port`` whose /props names ``server_path`` as its model file."""
    ollama = ollama_handler(models)

    def handler(req):
        if req.url.port == port:
            if req.url.path == "/v1/models":
                return json_response({"data": [{"id": alias}]})
            if req.url.path == "/props":
                return json_response({"model_alias": alias, "model_path": server_path, "default_generation_settings": {"n_ctx": 8192}, "chat_template": "{{x}}"})
            if req.url.path == "/slots":
                return json_response([{"is_processing": False}])
        return ollama(req)
    return handler


def served_store(tmp_path):
    root = make_store(tmp_path)
    write_gguf(root / "blobs" / BLOB, LLAMA)
    return root


def test_a_server_that_loads_an_ollama_blob_shares_names_with_the_tag(svc, tmp_path):
    root = served_store(tmp_path)
    svc.settings.set_many({"ollama.models_dir": str(root)})
    out = make_discovery(svc, combined_handler([TAG], str(root / "blobs" / BLOB))).refresh()
    assert out["same_weights"] == 3
    server = svc.store.contestant_by_key("server:llamacpp:qwen3-blob")
    ollama, twin = svc.store.contestant_by_key("server:ollama:qwen3:8b"), svc.store.contestant_by_key("gguf:ollama:qwen3:8b")
    assert "qwen3:8b" in server["aliases"], "the tag is an alias of the server"
    assert "qwen3-blob" in ollama["aliases"] and "qwen3-blob" in twin["aliases"], "the server alias is an alias of the Ollama entries"
    assert server["meta"]["same_weights"] == sorted([ollama["id"], twin["id"]])
    assert ollama["meta"]["same_weights"] == sorted([server["id"], twin["id"]]) and twin["meta"]["same_weights"] == sorted([server["id"], ollama["id"]])
    assert not any(a.endswith("(llama.cpp)") or BLOB in a for a in ollama["aliases"] + twin["aliases"]), "no file blobs or labels leak into the Ollama names"
    assert not any(a.endswith("(llama.cpp)") for a in server["aliases"])


def test_a_windows_style_blob_path_is_recognised(svc, tmp_path):
    root = served_store(tmp_path)
    svc.settings.set_many({"ollama.models_dir": str(root)})
    make_discovery(svc, combined_handler([TAG], "D:\\LocalAI\\ollama-models\\blobs\\" + BLOB)).refresh()
    server = svc.store.contestant_by_key("server:llamacpp:qwen3-blob")
    assert server["meta"]["model_digest"] == "sha256:" + "a" * 64 and "qwen3:8b" in server["aliases"]


def test_a_server_with_another_file_is_not_linked(svc, tmp_path):
    root = served_store(tmp_path)
    svc.settings.set_many({"ollama.models_dir": str(root)})
    other = tmp_path / "other.gguf"
    write_gguf(other, LLAMA)
    out = make_discovery(svc, combined_handler([TAG], str(other))).refresh()
    assert out["same_weights"] == 0
    server = svc.store.contestant_by_key("server:llamacpp:qwen3-blob")
    assert "qwen3:8b" not in server["aliases"] and "same_weights" not in server["meta"]


def test_a_blob_the_manifests_do_not_name_is_not_linked(svc, tmp_path):
    root = served_store(tmp_path)
    svc.settings.set_many({"ollama.models_dir": str(root)})
    stranger = "sha256-" + "9" * 64
    write_gguf(root / "blobs" / stranger, LLAMA)
    assert make_discovery(svc, combined_handler([TAG], str(root / "blobs" / stranger))).refresh()["same_weights"] == 0


def test_the_link_needs_the_manifest_store_to_be_found(svc, tmp_path):
    """Without a readable store there is nothing to say which tag owns the blob: no guess is made."""
    root = served_store(tmp_path)
    svc.settings.set_many({"ollama.models_dir": str(tmp_path / "none")})
    assert make_discovery(svc, combined_handler([TAG], str(root / "blobs" / BLOB))).refresh()["same_weights"] == 0


def test_the_link_is_recomputed_and_dropped_when_the_server_moves_to_another_file(svc, tmp_path):
    root = served_store(tmp_path)
    svc.settings.set_many({"ollama.models_dir": str(root)})
    make_discovery(svc, combined_handler([TAG], str(root / "blobs" / BLOB))).refresh()
    other = tmp_path / "other.gguf"
    write_gguf(other, LLAMA)
    out = make_discovery(svc, combined_handler([TAG], str(other))).refresh()
    assert out["same_weights"] == 0
    for key in ("server:llamacpp:qwen3-blob", "server:ollama:qwen3:8b", "gguf:ollama:qwen3:8b"):
        assert "same_weights" not in svc.store.contestant_by_key(key)["meta"], key


def test_model_detail_lists_the_same_weights_siblings(svc, tmp_path):
    from galton_hoard.agent_tools import call_tool
    root = served_store(tmp_path)
    svc.settings.set_many({"ollama.models_dir": str(root)})
    make_discovery(svc, combined_handler([TAG], str(root / "blobs" / BLOB))).refresh()
    server = svc.store.contestant_by_key("server:llamacpp:qwen3-blob")
    detail = call_tool(svc, "model_get", {"model": server["id"]})
    siblings = {s["name"]: s for s in detail["same_weights"]}
    assert set(siblings) == {"qwen3:8b", "qwen3:8b (llama.cpp)"} and all(s["relation"] == "same_weights" for s in siblings.values())
    assert siblings["qwen3:8b"]["source"] == "ollama" and detail["model"]["same_weights"] == sorted(s["id"] for s in siblings.values())


def test_routes_publish_every_name_of_the_shared_weights(svc, tmp_path):
    from galton_hoard.routes import names_for
    root = served_store(tmp_path)
    svc.settings.set_many({"ollama.models_dir": str(root)})
    make_discovery(svc, combined_handler([TAG], str(root / "blobs" / BLOB))).refresh()
    server = svc.store.contestant_by_key("server:llamacpp:qwen3-blob")
    assert "qwen3:8b" in names_for(server) and "qwen3-blob" in names_for(svc.store.contestant_by_key("gguf:ollama:qwen3:8b"))


# ---- aliases are names a server or an app can report -----------------------------------------------------------------------------------------

def test_usable_alias_and_clean_aliases():
    from galton_hoard.util import clean_aliases, usable_alias
    sha = "sha256-" + "a" * 64
    assert usable_alias("qwen3:8b") and usable_alias("hf.co/org/model:q4") and usable_alias("Tiny-8B-Q4_K_M") and usable_alias("Tiny-8B-Q4_K_M.gguf")
    for bad in ("", "  ", None, "D:\\models\\Tiny.gguf", "C:/models/Tiny.gguf", "\\\\nas\\share\\Tiny.gguf", "/home/me/Tiny.gguf", "~/models/Tiny.gguf", "./Tiny.gguf",
                "models/sub/Tiny.gguf", sha, sha + ".gguf", "SHA256-" + "B" * 64):
        assert not usable_alias(bad), bad
    assert clean_aliases(["Qwen3", "D:\\m\\q.gguf", sha], ["qwen3", "Qwen3:latest", ""]) == ["Qwen3", "Qwen3:latest"]
    assert name_variants(sha) == [] and name_variants("/var/lib/ollama/blobs/" + sha) == []
    assert name_variants("C:\\Users\\me\\models\\Gemma-3-12B-Q4_K_M.gguf") == ["Gemma-3-12B-Q4_K_M"]


def test_a_llama_server_that_reports_paths_gets_names_not_paths(svc):
    path = "C:\\Users\\me\\.ollama\\models\\blobs\\sha256-" + "c" * 64

    def handler(req):
        if req.url.port == 8082 and req.url.path == "/v1/models":
            return json_response({"data": [{"id": path}]})
        if req.url.port == 8082 and req.url.path == "/props":
            return json_response({"model_path": path, "default_generation_settings": {"n_ctx": 4096}})
        raise httpx.ConnectError("refused")
    make_discovery(svc, handler).refresh()
    c = next(x for x in svc.store.contestants() if x["kind"] == "server" and x["url"].endswith(":8082"))
    assert c["aliases"] == [] and c["name"] == "llama-server:8082"
    assert c["path"] == path, "the path stays in its own field"
    assert not any("sha256" in a or "\\" in a for a in c["aliases"])


def test_a_llama_server_with_a_gguf_path_keeps_the_file_stem_as_its_name(svc):
    path = "D:\\models\\Gemma-3-12B-it-Q4_K_M.gguf"

    def handler(req):
        if req.url.port == 8082 and req.url.path == "/v1/models":
            return json_response({"data": [{"id": path}]})
        if req.url.port == 8082 and req.url.path == "/props":
            return json_response({"model_path": path, "default_generation_settings": {"n_ctx": 4096}})
        raise httpx.ConnectError("refused")
    make_discovery(svc, handler).refresh()
    c = next(x for x in svc.store.contestants() if x["kind"] == "server" and x["url"].endswith(":8082"))
    assert c["name"] == "Gemma-3-12B-it-Q4_K_M" and c["aliases"] == ["Gemma-3-12B-it-Q4_K_M"] and c["path"] == path


def test_an_ollama_model_with_a_blob_twin_has_no_digest_alias(svc, tmp_path):
    root = tmp_path / "ollama"
    blob = root / "blobs" / ("sha256-" + "a" * 64)
    blob.parent.mkdir(parents=True)
    blob.write_bytes(b"GGUF")
    svc.settings.set_many({"ollama.models_dir": str(root)})
    make_discovery(svc, ollama_handler([TAG])).refresh()
    for c in svc.store.contestants():
        assert all(not a.startswith("sha256") and "/" not in a.replace("hf.co/", "") and "\\" not in a for a in c["aliases"]), (c["name"], c["aliases"])


def test_aliases_stored_by_older_versions_are_cleaned(svc):
    sha = "sha256-" + "d" * 64
    c = svc.store.create_contestant(key="gguf:/x/viejo.gguf", kind="gguf", name="viejo", path="/x/viejo.gguf", provider="llamacpp",
                                    aliases=["viejo", "/x/viejo.gguf", "D:\\m\\viejo.gguf", sha, "Viejo", "viejo.gguf"], source="manual")
    other = svc.store.create_contestant(key="gguf:/x/limpio.gguf", kind="gguf", name="limpio", path="/x/limpio.gguf", provider="llamacpp", aliases=["limpio"], source="manual")
    assert svc.store.clean_aliases() == 1
    assert svc.store.contestant(c["id"])["aliases"] == ["viejo", "viejo.gguf"] and svc.store.contestant(other["id"])["aliases"] == ["limpio"]
    assert svc.store.clean_aliases() == 0


def test_starting_the_app_cleans_the_stored_aliases(tmp_path):
    from helpers import build_services
    first = build_services(tmp_path)
    first.store.create_contestant(key="gguf:/x/a.gguf", kind="gguf", name="a", path="/x/a.gguf", provider="llamacpp", aliases=["a", "/x/a.gguf", "sha256-" + "e" * 64], source="manual")
    first.db.close()
    second = build_services(tmp_path)
    try:
        assert second.aliases_cleaned == 1 and second.store.contestant_by_key("gguf:/x/a.gguf")["aliases"] == ["a"]
    finally:
        second.db.close()


def test_an_unregistered_gguf_spec_gets_only_names(svc, tmp_path):
    from galton_hoard import adhoc
    blob = tmp_path / "blobs" / ("sha256-" + "f" * 64)
    blob.parent.mkdir()
    blob.write_bytes(b"GGUF" + b"\0" * 64)
    named = tmp_path / "Afinado-8B-Q4_K_M.gguf"
    named.write_bytes(b"GGUF" + b"\0" * 64)
    reader = lambda p: {"architecture": "llama", "params_b": 8.0, "quant": "Q4_K_M", "size_bytes": 64, "is_projector": False, "context_length": 4096}  # noqa: E731
    one = adhoc.resolve_one(svc.store, svc.settings, {"kind": "gguf", "path": str(blob)}, meta_reader=reader, digest_fn=lambda p: "d1")
    two = adhoc.resolve_one(svc.store, svc.settings, {"kind": "gguf", "path": str(named)}, meta_reader=reader, digest_fn=lambda p: "d2")
    assert one["aliases"] == [] and one["path"] == str(blob)
    assert two["aliases"] == ["Afinado-8B-Q4_K_M.gguf", "Afinado-8B-Q4_K_M"] or set(two["aliases"]) == {"Afinado-8B-Q4_K_M", "Afinado-8B-Q4_K_M.gguf"}
    assert all("/" not in a for a in two["aliases"])


def test_model_update_and_the_routes_only_keep_names(svc):
    from galton_hoard.routes import names_for
    from conftest import tool
    from helpers import add_gguf
    m = add_gguf(svc, "beta-q4")
    out = tool(svc, "model_update", model="beta-q4", add_aliases=["beta:latest", "D:\\x\\beta.gguf", "sha256-" + "1" * 64, "beta:latest"])
    assert out["model"]["aliases"].count("beta:latest") == 1 and not any("\\" in a or "sha256" in a for a in out["model"]["aliases"])
    out = tool(svc, "model_update", model="beta-q4", aliases=["beta", "/x/beta.gguf"])
    assert out["model"]["aliases"] == ["beta"]
    dirty = {**svc.store.contestant(m["id"]), "aliases": ["beta", "/x/beta.gguf", "sha256-" + "2" * 64], "model": "", "ollama_ref": ""}
    assert names_for(dirty) == ["beta", "beta-q4"]


def test_a_cpu_launch_has_no_layers_on_a_gpu_sets_the_threads_and_shows_no_gpu_to_the_child(launcher_parts, svc):
    launcher, cmds, *_ = launcher_parts
    spec = LaunchSpec(model_path="/m/x.gguf", name="x", context=4096, cpu=True, threads=10, grant=GpuGrant(gpus=[2, 3], shares_mb={2: 600, 3: 400}, via="test"))
    cmd = launcher.command(spec, 8091, "x")
    assert cmd[cmd.index("-ngl") + 1] == "0" and cmd[cmd.index("-t") + 1] == "10" and "--tensor-split" not in cmd
    launcher.start(spec, timeout_s=30)
    assert cmds[0][1]["env"]["CUDA_VISIBLE_DEVICES"] == "" and cmds[0][1]["env"]["CUDA_DEVICE_ORDER"] == "PCI_BUS_ID"


def test_a_gpu_launch_is_unchanged_by_the_cpu_fields(launcher_parts):
    launcher, *_ = launcher_parts
    cmd = launcher.command(LaunchSpec(model_path="/m/x.gguf", name="x", grant=GpuGrant(gpus=[2], shares_mb={2: 600}, via="test")), 8091, "x")
    assert cmd[cmd.index("-ngl") + 1] == "99" and "-t" not in cmd
