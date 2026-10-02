"""What comes from Hoard Link: the one-instance start, the stable token, the shared bridge and agent router (the caller of a call still labels its
runs), one error envelope, ULID ids next to old ids, JSON settings, the secrets of .env, the atomic registry and the launcher script."""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
from pathlib import Path

import httpx
import pytest
from helpers import add_gguf, build_services

from galton_hoard import config as config_module
from galton_hoard.config import Config
from galton_hoard.hoard_link import net

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def running_app(tmp_path):
    """The real app (``python -m galton_hoard``) in a child process on a free port."""
    port = net.free_port()
    env = {**os.environ, "GALTON_DATA_DIR": str(tmp_path / "data"), "GALTON_PORT": str(port), "GALTON_OFFLINE": "1", "GALTON_SCHEDULER": "0",
           "HOARD_NO_BROWSER": "1", "HOARD_ROUTES_FILE": str(tmp_path / "routes.json"), "HOARD_HUB_AUTOSTART": "0", "PYTHONPATH": str(ROOT)}
    command = [sys.executable, "-m", "galton_hoard"]
    child = subprocess.Popen(command, cwd=ROOT, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        assert net.wait_healthy(f"http://127.0.0.1:{port}", "galton-hoard", timeout=40), "the app did not start"
        yield {"port": port, "env": env, "command": command, "data": tmp_path / "data"}
    finally:
        child.terminate()
        try:
            child.wait(timeout=15)
        except subprocess.TimeoutExpired:
            child.kill()


def test_second_start_exits_cleanly_and_keeps_the_token(running_app):
    token_file = running_app["data"] / "mcp-token"
    token = token_file.read_text(encoding="utf-8")
    second = subprocess.run(running_app["command"], cwd=ROOT, env=running_app["env"], capture_output=True, text=True, timeout=40)
    assert second.returncode == 0 and "already running" in second.stdout
    assert token_file.read_text(encoding="utf-8") == token and len(token) >= 32
    health = httpx.get(f"http://127.0.0.1:{running_app['port']}/api/health", trust_env=False).json()
    assert health["service"] == "galton-hoard" and health["dataDirConfigured"] is True and health["offline"] is True and "hoard_link" in health
    assert {"counts", "scheduler", "demo"} <= set(health)
    assert (running_app["data"] / "logs").is_dir()                                   # the shared launcher logs to <data>/logs
    assert (running_app["data"] / "url").read_text(encoding="utf-8") == f"http://127.0.0.1:{running_app['port']}"


def test_the_shared_bridge_lists_and_calls_the_tools_of_the_running_app(running_app, monkeypatch):
    from galton_hoard.hoard_link.bridge import CatalogBridge

    monkeypatch.setenv("GALTON_DATA_DIR", str(running_app["data"]))
    monkeypatch.setenv("GALTON_URL", f"http://127.0.0.1:{running_app['port']}")
    monkeypatch.setenv("GALTON_BRIDGE_AUTOSTART", "0")
    bridge = CatalogBridge(app="galton", service="galton-hoard", package="galton_hoard", default_port=5201, data_dir_env="GALTON_DATA_DIR",
                           title="Galton's Hoard", root=str(ROOT / "mcp_server.py"), default_timeout=660.0)

    async def go():
        tools = await bridge.tools()
        overview = await bridge.call("galton_overview", {})
        missing = await bridge.call("model_get", {"model": "nada"})
        return tools, overview, missing

    tools, overview, missing = asyncio.run(go())
    waits = {t["name"]: t.get("x-timeout-s") for t in tools}
    assert len(tools) >= 38 and waits["run_start"] == waits["galton_run"] == waits["judge_run"] == 660.0 and waits["models_list"] is None
    assert not overview.is_error and overview.body["counts"]["suites"] == 13
    assert missing.is_error and missing.body["code"] == "not_found" and missing.body["hint"] and missing.body["key"]


def test_the_caller_of_an_agent_call_labels_the_runs_it_starts(client):
    add_gguf(client.svc, "cola-q4")
    first = client.post("/api/agent/call", headers=client.bearer, json={"name": "run_start", "caller": "hypatia", "arguments": {"suites": ["rapida"], "contestants": ["cola-q4"]}})
    assert first.status_code == 200, first.text
    run = client.svc.store.run(first.json()["run"]["id"])
    assert run["caller"] == "hypatia" and run["source"] == "assistant"
    again = client.post("/api/agent/call", headers=client.bearer, json={"name": "run_start", "arguments": {"suites": ["rapida"], "contestants": ["cola-q4"]}})
    assert client.svc.store.run(again.json()["run"]["id"])["caller"] == ""
    ui = client.post("/api/ui/call", json={"name": "run_start", "arguments": {"suites": ["rapida"], "contestants": ["cola-q4"]}})
    assert client.svc.store.run(ui.json()["run"]["id"])["source"] == "ui"        # the person, through the interface


def test_errors_share_one_envelope_with_a_code_and_the_ui_gets_the_translatable_parts(client):
    assert client.get("/api/nothing-here").json()["code"] == "not_found"
    assert client.post("/api/agent/call", json={"name": "models_list"}).status_code == 401
    assistant = client.post("/api/agent/call", headers=client.bearer, json={"name": "model_get", "arguments": {"model": "nada"}})
    assert assistant.status_code == 404 and assistant.json()["code"] == "not_found" and assistant.json()["error"] and assistant.json()["hint"]
    assert isinstance(assistant.json()["key"], str) and isinstance(assistant.json()["params"], dict)      # plain text for the assistant, with the parts as before
    unknown = client.post("/api/agent/call", headers=client.bearer, json={"name": "no_such_tool"})
    assert unknown.status_code == 404 and unknown.json()["code"] == "unknown_tool"
    invalid = client.post("/api/agent/call", headers=client.bearer, json={"name": "run_start", "arguments": {"suites": []}})
    assert invalid.status_code == 400 and invalid.json()["code"] == "invalid_arguments" and invalid.json()["issues"]
    ui = client.post("/api/ui/call", json={"name": "model_get", "arguments": {"model": "nada"}})
    assert ui.status_code == 404 and ui.json()["code"] == "not_found"
    asked_twice = client.post("/api/ui/call", json={"name": "run_start", "arguments": {"suites": [], "contestants": ["x"]}})
    assert asked_twice.status_code == 400 and asked_twice.json()["code"] == "invalid_arguments"


def test_galton_errors_are_app_errors_with_their_statuses():
    from galton_hoard.errors import GaltonError
    from galton_hoard.hoard_link.agentkit import AppError

    error = GaltonError("invalid", "no_suite", ref="x")
    assert isinstance(error, AppError) and error.status == 400 and error.key == "no_suite" and error.params == {"ref": "x"} and error.hint
    assert error.to_dict()["key"] == "no_suite" and error.to_dict()["code"] == "invalid" and str(error) == error.message
    assert GaltonError("no_gpu", "free text").status == 409 and GaltonError("unavailable", "x").status == 503 and GaltonError("x", "y", status=418).status == 418
    assert GaltonError("conflict", "free text", "do this", extra=[1]).to_dict() == {"extra": [1], "error": "free text", "code": "conflict", "hint": "do this"}


def test_the_new_ids_are_ordered_ulids_and_the_old_ones_still_work(svc):
    old = svc.store.create_contestant(id="c_0abcdefghjkm", key="gguf:/old.gguf", kind="gguf", name="vieja", path="/old.gguf")
    new = [add_gguf(svc, f"nueva-{i}") for i in range(3)]
    ids = [m["id"] for m in new]
    assert ids == sorted(ids) and all(i.startswith("c_") and len(i) == 28 and i == i.lower() for i in ids)
    assert svc.store.find_contestant("c_0abcdefghjkm")["id"] == old["id"] == "c_0abcdefghjkm"
    assert svc.store.contestant(old["id"])["name"] == "vieja"


def test_settings_are_json_values_and_old_text_rows_still_read(svc):
    svc.db.execute("INSERT INTO settings(key, value) VALUES ('arena.min_votes', '7'), ('gpus.allowed', '[1, 4]'), ('watch.enabled', 'perhaps')")   # the old writer: spaces, plain text
    assert svc.settings.get("arena.min_votes") == 7 and svc.settings.get("gpus.allowed") == [1, 4]
    assert svc.settings.get("watch.enabled") is True                      # a text that is not a boolean: the default
    svc.settings.set_many({"arena.min_votes": 9, "gpus.allowed": [2, 3], "faustus.token": "abcdefgh12345678"})
    assert svc.settings.get("arena.min_votes") == 9 and svc.settings.get("gpus.allowed") == [2, 3]
    assert svc.db.get_setting("arena.min_votes") == 9 and svc.db.one("SELECT value FROM settings WHERE key = 'gpus.allowed'")["value"] == "[2,3]"
    svc.settings.reset("arena.min_votes")
    assert svc.settings.get("arena.min_votes") == 5


def test_the_secrets_of_dotenv_are_read_but_never_exported(tmp_path, monkeypatch):
    (tmp_path / ".env").write_text("GALTON_FAUSTUS_TOKEN=secreto-del-archivo\nGALTON_OTHER=otro\n", encoding="utf-8")
    monkeypatch.setattr(config_module, "REPO_ROOT", tmp_path)
    monkeypatch.setenv("GALTON_OTHER", "real")
    config = Config.from_env()
    assert config.secret("FAUSTUS_TOKEN") == "secreto-del-archivo" and config.secret("OTHER") == "real"      # the real environment wins
    assert "GALTON_FAUSTUS_TOKEN" not in os.environ and os.environ["GALTON_OTHER"] == "real"                    # llama-server children never see the file's secrets


def test_the_server_registry_is_written_atomically(svc, tmp_path):
    from galton_hoard.servers import Launcher

    launcher = Launcher(svc.settings, tmp_path / "logs", tmp_path / "servers.json")
    launcher._write_registry([{"pid": 1, "port": 8091, "name": "x", "started": 0}])
    assert json.loads((tmp_path / "servers.json").read_text(encoding="utf-8"))[0]["port"] == 8091
    assert launcher._read_registry()[0]["pid"] == 1 and not list(tmp_path.glob("*.tmp"))


def test_a_run_that_outlasts_wait_s_says_it_is_still_running(svc, monkeypatch):
    add_gguf(svc, "lento-q4")
    monkeypatch.setattr(svc, "submit_run", lambda run_id, wait_s=0.0: None)          # the lane has not started it when the wait ends
    waited = svc.start_run(suites=["rapida"], contestants=["lento-q4"], wait_s=1)
    assert waited["still_running"] is True and waited["run"]["state"] == "queued"
    assert "still_running" not in svc.start_run(suites=["rapida"], contestants=["lento-q4"], wait_s=0)


def test_the_launcher_script_imports_the_shared_net_helpers():
    import importlib.util

    spec = importlib.util.spec_from_file_location("galton_launch_script", ROOT / "scripts" / "launch.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)                          # a deleted helper module would fail here, not on the user's double click
    assert module.SERVICE == "galton-hoard" and callable(module.main)
