"""The stdio MCP bridge against a real server process: initialize, tools/list, tools/call."""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import httpx
import pytest

ROOT = Path(__file__).resolve().parent.parent


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture
def server(tmp_path):
    port = free_port()
    env = {**os.environ, "GALTON_DATA_DIR": str(tmp_path / "data"), "GALTON_PORT": str(port), "PORT_STRICT": "1", "PYTHONUNBUFFERED": "1",
           "GALTON_SCHEDULER": "0", "GALTON_OFFLINE": "1", "HOARD_ROUTES_FILE": str(tmp_path / "routes.json")}
    proc = subprocess.Popen([sys.executable, "-m", "galton_hoard"], cwd=ROOT, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    url = f"http://127.0.0.1:{port}"
    try:
        for _ in range(150):
            try:
                if httpx.get(f"{url}/api/health", timeout=1, trust_env=False).status_code == 200:
                    break
            except httpx.HTTPError:
                time.sleep(0.2)
        else:
            pytest.fail("server did not start")
        yield url, tmp_path / "data", env
    finally:
        proc.terminate()
        try:
            proc.wait(10)
        except subprocess.TimeoutExpired:
            proc.kill()


def rpc(proc, payload):
    proc.stdin.write((json.dumps(payload) + "\n").encode())
    proc.stdin.flush()
    return json.loads(proc.stdout.readline())


def call(proc, rid, name, arguments):
    reply = rpc(proc, {"jsonrpc": "2.0", "id": rid, "method": "tools/call", "params": {"name": name, "arguments": arguments}})
    return json.loads(reply["result"]["content"][0]["text"])


def test_bridge_roundtrip(server):
    url, data, env = server
    benv = {**env, "GALTON_URL": url, "GALTON_TOKEN_FILE": str(data / "mcp-token"), "GALTON_BRIDGE_AUTOSTART": "0"}
    proc = subprocess.Popen([sys.executable, str(ROOT / "mcp_server.py")], cwd=ROOT, env=benv, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    try:
        init = rpc(proc, {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"protocolVersion": "2024-11-05", "capabilities": {},
                                                                                       "clientInfo": {"name": "t", "version": "0"}}})
        assert init["result"]["serverInfo"]["name"] == "galton-hoard"
        proc.stdin.write((json.dumps({"jsonrpc": "2.0", "method": "notifications/initialized"}) + "\n").encode())
        proc.stdin.flush()
        tools = rpc(proc, {"jsonrpc": "2.0", "id": 2, "method": "tools/list"})["result"]["tools"]
        names = {t["name"] for t in tools}
        assert {"galton_overview", "models_list", "run_start", "leaderboard", "compare", "recommend", "routes_publish", "gpu_status"} <= names
        assert len(names) >= 38
        overview = call(proc, 3, "galton_overview", {})
        assert overview["counts"]["suites"] == 13
        made = call(proc, 4, "suite_create", {"name": "Puente", "category": "custom", "cases": [{"title": "Uno", "prompt": "Di hola", "expected": "hola"}]})
        assert made["suite"]["cases"] == 1
        assert call(proc, 5, "suites_list", {"builtin": False})["count"] == 1
        err = call(proc, 6, "model_get", {"model": "nada"})
        assert err["code"] == "not_found" and err["hint"]
        bad = call(proc, 7, "run_start", {"suites": [], "contestants": ["x"]})
        assert "error" in bad
    finally:
        proc.terminate()
        proc.wait(10)


def test_bridge_refuses_a_non_local_url(server):
    url, data, env = server
    benv = {**env, "GALTON_URL": "http://example.com:5201", "GALTON_TOKEN_FILE": str(data / "mcp-token"), "GALTON_BRIDGE_AUTOSTART": "0"}
    out = subprocess.run([sys.executable, str(ROOT / "mcp_server.py")], cwd=ROOT, env=benv, capture_output=True, text=True, timeout=30, input="")
    assert out.returncode != 0 and "local server" in (out.stderr + out.stdout)
