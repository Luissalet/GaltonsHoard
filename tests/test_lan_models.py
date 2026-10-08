"""Models served by other machines of the person's own network (a cluster of GPU machines next to the PC): they become contestants, the automatic watch
measures them where they are, and it never loads a model on this PC by itself unless ``watch.load_local`` says so. Everything here is a fake world:
no network, no GPU, no model."""

from __future__ import annotations

import types
from datetime import datetime

import httpx
import pytest

from galton_hoard.agent_tools import call_tool
from galton_hoard.discovery import LAN_MAX_SERVERS, Discovery, lan_base, lan_key
from galton_hoard.errors import GaltonError
from galton_hoard.fakes import FakeWorld, reference_responder
from galton_hoard.servers import metrics_busy, slots_busy
from galton_hoard.watch import SMOKE_SUITE, Watch, failure_of
from helpers import Clock, add_gguf, build_services, json_response, mock_client_factory, references, run_inline

NODE = "http://192.168.7.10:8003"
NODE_MODEL = "qwen3.8-27b-nvfp4"
GLM = "http://192.168.7.11:8002"
GLM_MODEL = "glm-5.3-flash-nvfp4"
ENDPOINT = "Cluster · qwen-27b-1m"


def faustus_items():
    return [
        {"url": NODE + "/v1/chat/completions", "models": [NODE_MODEL], "models_vision": [NODE_MODEL], "endpoint_name": ENDPOINT, "category": "local",
         "model_type": "llm", "backend": "unknown"},
        {"url": GLM + "/v1/chat/completions", "models": [GLM_MODEL], "endpoint_name": "Cluster · glm", "category": "local", "model_type": "llm", "backend": "unknown"},
        {"url": "http://127.0.0.1:8081/v1/chat/completions", "models": ["local-llama"], "endpoint_name": "llama.cpp (local)", "category": "local", "model_type": "llm",
         "backend": "llamacpp"},
        {"url": "https://api.example.com/v1/chat/completions", "models": ["cloud-model"], "endpoint_name": "Cloud", "category": "cloud", "model_type": "llm"},
        {"url": "http://192.168.0.50:8188/v1/images", "models": ["flux"], "endpoint_name": "Images", "category": "local", "model_type": "image"},
        {"url": "http://192.168.0.60:9000/v1/chat/completions", "models": ["lan-but-cloud"], "endpoint_name": "Odd", "category": "cloud", "model_type": "llm"},
    ]


class World:
    """Faustus's registry, Prometheus's Hoard and the servers of the local network, as one mock HTTP world. Anything else refuses the connection."""

    def __init__(self):
        self.faustus: object = faustus_items()          # a list of registry items, or None when Faustus is down
        self.prometheus: object = None                  # a list of endpoints, or None when Prometheus's Hoard is down
        self.serving = {NODE: [NODE_MODEL], GLM: []}  # server base -> model ids it serves now ([]: not answering)
        self.busy = False
        self.calls: list[str] = []

    def handler(self, req: httpx.Request) -> httpx.Response:
        base = f"{req.url.scheme}://{req.url.host}:{req.url.port}"
        self.calls.append(f"{req.url.port}{req.url.path}")
        if req.url.port == 7000 and req.url.path == "/api/models":
            if self.faustus is None:
                raise httpx.ConnectError("refused")
            return json_response({"items": self.faustus})
        if req.url.port == 5205 and req.url.path == "/api/endpoints":
            if self.prometheus is None:
                raise httpx.ConnectError("refused")
            return json_response({"endpoints": self.prometheus})
        models = self.serving.get(base)
        if models:                                       # a vLLM-like server: /v1/models, no /props, no /slots, gauges on /metrics
            if req.url.path == "/v1/models":
                return json_response({"data": [{"id": m, "max_model_len": 1048576} for m in models]})
            if req.url.path == "/metrics":
                return httpx.Response(200, text=f'# HELP x\nvllm:num_requests_running{{engine="0"}} {1.0 if self.busy else 0.0}\nvllm:num_requests_waiting{{engine="0"}} 0.0\n')
            if req.url.path == "/v1/chat/completions":
                return json_response({"choices": [{"message": {"content": "Hi"}}]})
            return httpx.Response(404)
        raise httpx.ConnectError("refused")

    def factory(self):
        return mock_client_factory(self.handler)


def discovery(svc, world):
    return Discovery(svc.store, svc.settings, client_factory=world.factory(), clock=svc.clock, offline=False)


def lan_models(svc):
    return svc.store.contestants(kind="server", include_missing=True, include_adhoc=True)


@pytest.fixture
def world():
    return World()


# ---- discovery -----------------------------------------------------------------------------------------------------------------------------

def test_the_address_of_a_listed_model_is_the_root_of_its_server():
    assert lan_base("http://192.168.7.10:8003/v1/chat/completions") == NODE
    assert lan_base("http://node1.local/v1") == "http://node1.local"
    assert lan_base("http://[fd00::5]:8000/v1/x") == "http://[fd00::5]:8000"
    assert lan_base("") == "" and lan_base("not a url") == ""
    assert lan_key(NODE, NODE_MODEL) == f"server:lan:192.168.7.10:8003:{NODE_MODEL}"


def test_a_model_served_on_the_network_becomes_a_server_contestant(svc, world):
    out = discovery(svc, world).refresh()
    assert out["sources"]["lan"] == {"ok": True, "via": "faustus", "servers": 1, "asked": 2, "models": 1, "added": 1}
    assert out["sources"]["faustus"]["ok"] is True and out["sources"]["faustus"]["items"] == 6
    c = svc.store.contestant_by_key(f"server:lan:192.168.7.10:8003:{NODE_MODEL}")
    assert (c["kind"], c["api"], c["url"], c["model"], c["name"]) == ("server", "openai", NODE, NODE_MODEL, NODE_MODEL)
    assert c["remote"] is False and c["remote_ok"] is False and c["enabled"] is True and c["missing"] is False and c["adhoc"] is False
    assert c["source"] == "lan" and c["meta"]["network"] == "lan" and c["meta"]["up"] is True and c["meta"]["via"] == "faustus"
    assert c["context"] == 1048576 and c["vision"] is True and c["params_b"] == 27.0 and c["quant"] == "NVFP4"
    assert NODE_MODEL in c["aliases"] and ENDPOINT in c["aliases"], "Faustus knows it by the name of the endpoint too"
    names = {x["name"] for x in lan_models(svc)}
    assert GLM_MODEL not in names, "the registry lists it but the server does not serve it"
    assert not names & {"cloud-model", "flux", "lan-but-cloud"}, "only local language models"


def test_a_remote_flag_is_never_set_for_the_person_s_own_network(svc, world):
    discovery(svc, world).refresh()
    card = svc.contestant_card(svc.store.contestant_by_key(lan_key(NODE, NODE_MODEL)))
    assert card["remote"] is False and card["network"] == "lan"


def test_refreshing_again_updates_the_same_contestant(svc, world):
    d = discovery(svc, world)
    d.refresh()
    first = svc.store.contestant_by_key(lan_key(NODE, NODE_MODEL))
    out = d.refresh()
    assert out["sources"]["lan"]["added"] == 0 and svc.store.contestant_by_key(lan_key(NODE, NODE_MODEL))["id"] == first["id"]
    assert len([c for c in lan_models(svc) if c["source"] == "lan"]) == 1


def test_a_lan_server_that_stops_is_down_not_gone_and_returns_as_the_same_contestant(svc, world):
    d = discovery(svc, world)
    d.refresh()
    first = svc.store.contestant_by_key(lan_key(NODE, NODE_MODEL))
    svc.store.update_contestant(first["id"], enabled=False, aliases=[*first["aliases"], "mi alias"])      # what the person decided stays
    world.serving[NODE] = []
    out = d.refresh()
    down = svc.store.contestant(first["id"])
    assert down["meta"]["up"] is False and down["missing"] is False and out["missing"] == [] and down["enabled"] is False
    assert out["sources"]["lan"]["servers"] == 0
    world.serving[NODE] = [NODE_MODEL]
    d.refresh()
    back = svc.store.contestant(first["id"])
    assert back["id"] == first["id"] and back["meta"]["up"] is True and back["enabled"] is False and "mi alias" in back["aliases"]
    assert len([c for c in lan_models(svc) if c["source"] == "lan"]) == 1


def test_a_down_server_does_not_hang_the_refresh(svc, world):
    class Slow(World):
        def handler(self, req):
            if req.url.host == "192.168.7.10":
                raise httpx.ReadTimeout("timed out")
            return super().handler(req)
    slow = Slow()
    out = discovery(svc, slow).refresh()
    assert out["sources"]["lan"]["ok"] is True and out["sources"]["lan"]["models"] == 0 and not [c for c in lan_models(svc) if c["source"] == "lan"]


def test_at_most_a_fixed_number_of_servers_is_asked(svc, world):
    items = [{"url": f"http://192.168.1.{i}:8000/v1/chat/completions", "models": [f"m{i}"], "category": "local", "model_type": "llm", "endpoint_name": f"s{i}"} for i in range(1, 30)]
    world.faustus = items
    world.serving = {f"http://192.168.1.{i}:8000": [f"m{i}"] for i in range(1, 30)}
    out = discovery(svc, world).refresh()
    assert out["sources"]["lan"]["asked"] == LAN_MAX_SERVERS == out["sources"]["lan"]["servers"]
    assert len([c for c in lan_models(svc) if c["source"] == "lan"]) == LAN_MAX_SERVERS


def test_prometheus_is_the_source_when_faustus_is_down(svc, world):
    world.faustus = None
    world.prometheus = [{"recipe": "qwen-27b-1m", "title": "qwen-27b-1m", "base_url": NODE + "/v1", "models": [NODE_MODEL], "max_model_len": 1048576, "nodes": ["node3"],
                         "detected": False},
                        {"recipe": "cloudish", "base_url": "https://api.example.com/v1", "models": ["x"]}]
    out = discovery(svc, world).refresh()
    assert out["sources"]["faustus"]["ok"] is False and out["sources"]["lan"]["via"] == "prometheus" and out["sources"]["lan"]["models"] == 1
    c = svc.store.contestant_by_key(lan_key(NODE, NODE_MODEL))
    assert c["url"] == NODE and c["remote"] is False and c["meta"]["via"] == "prometheus" and "qwen-27b-1m" in c["aliases"]
    assert [x["name"] for x in lan_models(svc)] == [NODE_MODEL]


def test_prometheus_is_not_asked_while_faustus_lists_the_servers(svc, world):
    world.prometheus = []
    discovery(svc, world).refresh()
    assert not [c for c in world.calls if c.startswith("5205")]


def test_when_nobody_can_say_what_is_served_the_contestants_are_left_as_they_are(svc, world):
    d = discovery(svc, world)
    d.refresh()
    world.faustus, world.prometheus = None, None
    out = d.refresh()
    assert out["sources"]["lan"]["ok"] is False
    assert svc.store.contestant_by_key(lan_key(NODE, NODE_MODEL))["meta"]["up"] is True


def test_a_server_restarted_with_another_model_keeps_the_old_entry_marked_not_served(svc, world):
    d = discovery(svc, world)
    d.refresh()
    world.serving[NODE] = [GLM_MODEL]                 # Faustus still announces the old model
    out = d.refresh()
    old = svc.store.contestant_by_key(lan_key(NODE, NODE_MODEL))
    assert old["meta"]["up"] is False and old["meta"]["not_served"]["model"] == GLM_MODEL and out["not_served"] == [old["id"]]
    assert svc.store.contestant_by_key(lan_key(NODE, GLM_MODEL)) is None, "the registry has not announced it: no entry yet"
    world.faustus = [{**faustus_items()[0], "models": [GLM_MODEL], "models_vision": []}]
    d.refresh()
    assert svc.store.contestant_by_key(lan_key(NODE, GLM_MODEL))["meta"]["up"] is True
    world.serving[NODE] = [NODE_MODEL]
    world.faustus = faustus_items()
    d.refresh()
    assert svc.store.contestant(old["id"])["meta"]["up"] is True and "not_served" not in svc.store.contestant(old["id"])["meta"]


def test_the_lan_models_are_published_to_the_family_under_their_names(svc, world):
    discovery(svc, world).refresh()
    c = svc.store.contestant_by_key(lan_key(NODE, NODE_MODEL))
    assert call_tool(svc, "models_list", {})["models"] and NODE_MODEL in {m["name"] for m in call_tool(svc, "models_list", {})["models"]}
    assert c["aliases"][0] == NODE_MODEL


# ---- how busy is a vLLM server ---------------------------------------------------------------------------------------------------------------

def test_request_gauges_say_whether_a_server_without_slots_is_busy():
    assert metrics_busy('# TYPE vllm:num_requests_running gauge\nvllm:num_requests_running{engine="0"} 0.0\nvllm:num_requests_waiting{engine="0"} 0.0\n') is False
    assert metrics_busy('vllm:num_requests_running{engine="0"} 2.0\n') is True
    assert metrics_busy('vllm:num_requests_waiting{engine="0"} 1.0\nvllm:num_requests_running{engine="0"} 0.0\n') is True
    assert metrics_busy("nothing: 1\n") is None and metrics_busy("") is None


def test_slots_busy_falls_back_to_the_metrics_page(world):
    with world.factory()() as client:
        assert slots_busy(client, NODE) is False
        world.busy = True
        assert slots_busy(client, NODE) is True
        assert slots_busy(client, "http://192.168.0.99:1") is None, "a server that answers neither page says nothing"


# ---- the watch never loads a model on this PC by default ---------------------------------------------------------------------------------------

def noon(clock):
    clock.now = datetime(2026, 3, 4, 12, 0).timestamp()


def watch_with(svc, handler):
    return Watch(svc.store, svc.settings, svc.runner, svc.gpus, clock=svc.clock, submit_run=svc.submit_run, emit=svc.emit,
                 client_factory=mock_client_factory(handler), offline=False)


def nowhere(req):
    raise httpx.ConnectError("refused")


def test_load_local_is_a_documented_setting_and_off_by_default(svc):
    spec = {s["key"]: s for s in svc.settings.describe()}["watch.load_local"]
    assert spec["kind"] == "bool" and spec["default"] is False and spec["group"] == "watch"
    assert svc.settings.get("watch.load_local") is False
    out = call_tool(svc, "settings_set", {"values": {"watch.load_local": True}})
    assert svc.settings.get("watch.load_local") is True and out
    call_tool(svc, "settings_set", {"values": {"watch.load_local": False}})
    assert svc.settings.get("watch.load_local") is False


def test_the_watch_never_calls_the_loader_with_load_local_off(svc, clock):
    noon(clock)
    m = add_gguf(svc, "nuevo")
    decision = svc.watch.tick()
    assert "queued" not in decision and svc.fake_world.started == [] and svc.store.runs() == []
    assert decision["held_back"] == {"setting": "watch.load_local", "models": ["nuevo"]}
    assert svc.watch.candidates() == []
    ok, why = svc.watch._eligible(svc.store.contestant(m["id"]))
    assert not ok and "watch.load_local" in why
    for _ in range(5):                               # the same on every minute of the day
        clock.advance(3600)
        svc.watch.tick()
    assert svc.fake_world.started == []


def test_with_load_local_on_the_watch_loads_as_before(svc, clock):
    noon(clock)
    svc.settings.set_many({"watch.load_local": True})
    add_gguf(svc, "nuevo")
    decision = svc.watch.tick()
    assert decision["queued"]["contestant"] == "nuevo" and len(svc.fake_world.started) == 1
    assert svc.store.run(decision["queued"]["run"])["settings"]["load_local"] is True


def test_a_run_that_must_not_load_refuses_a_gguf_and_an_unloaded_ollama_model(svc):
    m = add_gguf(svc, "archivo")
    run = run_inline(svc, ["rapida"], [m["id"]], load_local=False)
    rc = svc.store.run_contestants(run["id"])[0]
    assert run["state"] == "failed" and rc["error"].key == "load_local_off" and svc.fake_world.started == []
    assert failure_of(svc.store.contestant(m["id"])) is None, "that is not a failure of the model: nothing marks it"
    with pytest.raises(GaltonError):
        run_inline(svc, ["rapida"], [m["id"]], load_local="maybe")


def test_manual_runs_still_load_locally(svc):
    assert svc.settings.get("watch.load_local") is False
    m = add_gguf(svc, "archivo")
    run = run_inline(svc, ["rapida"], [m["id"]])
    assert run["state"] == "done" and len(svc.fake_world.started) == 1 and run["settings"]["load_local"] is True
    out = svc.measure_new()
    assert out["started"] is False                                       # it was measured; and the button path is the same code as run_start
    n = add_gguf(svc, "otro")
    assert svc.measure_new()["started"] is True and len(svc.fake_world.started) == 2


def test_an_ollama_model_that_is_not_resident_is_not_loaded_by_a_run_that_must_not_load(tmp_path, clock):
    def ollama(req):
        if req.url.path == "/api/ps":
            return json_response({"models": []})
        if req.url.path == "/api/tags":
            return json_response({"models": [{"name": "qwen3:8b", "digest": "sha256:" + "a" * 64}]})
        raise httpx.ConnectError("refused")
    svc = build_services(tmp_path, clock=clock, client_factory=mock_client_factory(ollama))
    svc.settings.set_many({"runner.allow_ollama_load": True})
    c = svc.store.create_contestant(key="server:ollama:qwen3:8b", kind="server", name="qwen3:8b", url="http://127.0.0.1:11434", api="ollama", model="qwen3:8b",
                                    provider="ollama", aliases=["qwen3:8b"], digest="sha256:" + "a" * 64, source="ollama", enabled=True)
    svc.fake_world.register("qwen3:8b", reference_responder(references()))
    run = run_inline(svc, ["rapida"], [c["id"]], load_local=False, wait_s=0)
    assert svc.store.run_contestants(run["id"])[0]["error"].key == "ollama_not_loaded"
    svc.db.close()


# ---- a model the watch could not measure is not tried again by itself ------------------------------------------------------------------------

def failing_loader(svc):
    attempts: list[str] = []

    def start(spec, **_):
        attempts.append(spec.name)
        raise GaltonError("unavailable", "server_exited", exit_code=1, log_tail="CUDA out of memory", log="llama-x.log")
    svc.runner.launcher = types.SimpleNamespace(start=start, reap_orphans=lambda: [])
    return attempts


def test_a_model_that_fails_to_load_is_not_retried_until_its_file_changes(svc, clock):
    noon(clock)
    svc.settings.set_many({"watch.load_local": True})
    attempts = failing_loader(svc)
    m = add_gguf(svc, "roto", digest="v1")
    first = svc.watch.tick()
    assert first["queued"]["contestant"] == "roto" and attempts == ["roto"]
    failed = svc.store.contestant(m["id"])
    mark = failure_of(failed)
    assert mark and mark["key"] == "server_exited" and "exited with code 1" in mark["error"] and mark["digest"] == "v1"
    card = svc.contestant_card(failed)
    assert card["watch_gave_up"] is True and card["watch_gave_up_reason"].key == "watch_gave_up" and "exited with code 1" in str(card["watch_gave_up_reason"])
    assert len(svc.store.notices(kind="run_failed")) == 1
    for _ in range(4):                                # many cycles later: nothing is started, nothing is announced
        clock.advance(7 * 3600 if clock() < datetime(2026, 3, 4, 18).timestamp() else 3600)
        decision = svc.watch.tick()
        assert "queued" not in decision, decision
    assert attempts == ["roto"] and len(svc.store.notices(kind="run_failed")) == 1 and svc.watch.candidates() == []
    svc.store.update_contestant(m["id"], digest="v2")      # the file changed
    clock.now = datetime(2026, 3, 5, 12, 0).timestamp()
    assert svc.watch.tick()["queued"]["contestant"] == "roto" and attempts == ["roto", "roto"]
    assert failure_of(svc.store.contestant(m["id"]))["digest"] == "v2"
    assert len(svc.store.notices(kind="run_failed")) == 2, "a new file is a new failure, announced once"


def test_asking_for_the_model_by_hand_lets_the_watch_try_again(svc, clock):
    noon(clock)
    svc.settings.set_many({"watch.load_local": True})
    failing_loader(svc)
    m = add_gguf(svc, "roto", digest="v1")
    svc.watch.tick()
    assert failure_of(svc.store.contestant(m["id"]))
    svc.runner.launcher = svc.fake_world                      # the loader works now
    run = run_inline(svc, ["rapida"], [m["id"]])
    assert run["state"] == "done" and failure_of(svc.store.contestant(m["id"])) is None


def test_a_failure_of_a_manual_run_also_stops_the_watch(svc, clock):
    noon(clock)
    svc.settings.set_many({"watch.load_local": True})
    attempts = failing_loader(svc)
    m = add_gguf(svc, "roto", digest="v1")
    run_inline(svc, ["rapida"], [m["id"]])
    assert failure_of(svc.store.contestant(m["id"])) and attempts == ["roto"]
    clock.advance(8 * 3600)
    assert "queued" not in svc.watch.tick() and attempts == ["roto"]


def test_a_failure_that_says_nothing_about_the_model_marks_nothing(svc, clock):
    m = add_gguf(svc, "ocupado")
    run = svc.runner.create(suites=[SMOKE_SUITE], contestants=[m["id"]], source="watch", caller="watch")
    busy = GaltonError("busy", "server_busy", url="http://x", limit="0")
    svc.store.upsert_run_contestant(run["id"], m["id"], state="failed", error=busy.message)
    assert svc.watch.record_outcome(run["id"]) == [] and failure_of(svc.store.contestant(m["id"])) is None
    dead = GaltonError("unavailable", "server_exited", exit_code=3, log_tail="x", log="y")
    svc.store.upsert_run_contestant(run["id"], m["id"], state="failed", error=dead.message)
    assert svc.watch.record_outcome(run["id"]) == [m["id"]] and failure_of(svc.store.contestant(m["id"]))["key"] == "server_exited"


def test_a_server_that_comes_back_may_be_tried_again(svc, clock):
    c = svc.store.create_contestant(key="server:lan:h:1:m", kind="server", name="m", url="http://192.168.0.9:1", api="openai", model="m", provider="openai_compat",
                                    aliases=["m"], source="lan", enabled=True, meta={"up": False, "network": "lan"})
    svc.store.update_contestant(c["id"], meta={**c["meta"], "watch_failed": {"ts": 1, "error": "boom", "key": "", "digest": "", "path": "", "url": c["url"], "model": "m"}})
    assert failure_of(svc.store.contestant(c["id"]))
    watch_with(svc, lambda req: json_response({"data": [{"id": "m"}]}) if req.url.path == "/v1/models" else httpx.Response(404)).probe()
    back = svc.store.contestant(c["id"])
    assert back["meta"]["up"] is True and "watch_failed" not in back["meta"]


# ---- the watch measures the models of the network where they are ---------------------------------------------------------------------------

def lan_setup(tmp_path, world, clock):
    fake = FakeWorld()
    fake.register(NODE_MODEL, reference_responder(references()))
    svc = build_services(tmp_path, clock=clock, world=fake, client_factory=world.factory())
    discovery(svc, world).refresh()
    return svc, svc.store.contestant_by_key(lan_key(NODE, NODE_MODEL))


def test_a_new_lan_model_is_measured_by_the_watch_once_its_server_is_idle(tmp_path, clock, world):
    noon(clock)
    svc, c = lan_setup(tmp_path, world, clock)
    watch = watch_with(svc, world.handler)
    first = watch.tick()
    assert "queued" not in first and "busy recently" in first["waiting"][NODE_MODEL]
    clock.advance(11 * 60)
    world.busy = True                                    # somebody is chatting with it: the gauge says so
    busy = watch.tick()
    assert "queued" not in busy and "busy recently" in busy["waiting"][NODE_MODEL]
    world.busy = False
    clock.advance(11 * 60)
    decision = watch.tick()
    assert decision["queued"]["contestant"] == NODE_MODEL
    run = svc.store.run(decision["queued"]["run"])
    assert run["source"] == "watch" and run["state"] == "done" and run["suites"] == [SMOKE_SUITE] and svc.fake_world.started == [], "nothing was loaded here"
    card = next(x for x in svc.run_card(run)["contestants"])
    assert card["state"] == "done" and "192.168.7.10" in str(card["runs_on"])
    assert [r["name"] for r in call_tool(svc, "leaderboard", {"suite": "rapida"})["rows"]] == [NODE_MODEL]
    assert watch.candidates() == []
    svc.db.close()


def test_a_lan_model_that_is_down_is_not_measured_and_comes_back_with_its_results(tmp_path, clock, world):
    noon(clock)
    svc, c = lan_setup(tmp_path, world, clock)
    watch = watch_with(svc, world.handler)
    watch.tick()
    clock.advance(11 * 60)
    watch.tick()                                         # measured
    n = len(svc.store.results(contestant_id=c["id"]))
    assert n > 0
    world.serving[NODE] = []
    watch.probe()
    assert svc.store.contestant(c["id"])["meta"]["up"] is False and watch.idle_for(c["id"]) is None
    d = discovery(svc, world)
    d.refresh()
    world.serving[NODE] = [NODE_MODEL]
    d.refresh()
    again = svc.store.contestant_by_key(lan_key(NODE, NODE_MODEL))
    assert again["id"] == c["id"] and again["meta"]["up"] is True and len(svc.store.results(contestant_id=c["id"])) == n
    svc.db.close()


def test_the_watch_does_not_load_anything_while_it_measures_the_network(tmp_path, clock, world):
    noon(clock)
    svc, c = lan_setup(tmp_path, world, clock)
    add_gguf(svc, "local-gguf")
    watch = watch_with(svc, world.handler)
    watch.tick()
    clock.advance(11 * 60)
    decision = watch.tick()
    assert decision["queued"]["contestant"] == NODE_MODEL and svc.fake_world.started == []
    assert decision["probe"] == {"servers_up": 1}
    svc.db.close()
