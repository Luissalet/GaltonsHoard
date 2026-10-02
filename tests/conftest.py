from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from galton_hoard import config as config_module, cpu as cpu_module, ollama_models, servers, settings as settings_module  # noqa: E402
from helpers import Clock, add_gguf, build_services  # noqa: E402

# Environment variables that change what the app finds on the machine: a test that wants one sets it itself.
HOST_ENV_PREFIXES = ("GALTON_", "HOARD_", "OLLAMA_", "HF_", "FAUSTUS_", "CUDA_")
HOST_ENV_NAMES = ("PORT", "PORT_STRICT", "LOCALAPPDATA", "APPDATA", "PROGRAMFILES", "PROGRAMFILES(X86)", "PROGRAMDATA")


@pytest.fixture(autouse=True)
def hermetic_host(tmp_path_factory, monkeypatch):
    """No test depends on the computer it runs on. The real defaults stay in the product (the Windows model folders, ``OLLAMA_MODELS`` and
    ``~/.ollama/models``, llama-server on the PATH, the repository's ``.env``...) and are looked up by the same functions as always; this
    fixture only points those lookups at a place where nothing exists, unless a test builds something there."""
    nowhere = tmp_path_factory.getbasetemp() / "no-such-host"          # never created
    home = tmp_path_factory.mktemp("home")                             # an empty profile folder: ~/.ollama/models and ~/.hoard are not the real ones
    for name in list(os.environ):
        if name.upper().startswith(HOST_ENV_PREFIXES) or name.upper() in HOST_ENV_NAMES:
            monkeypatch.delenv(name)
    for name in ("HOMEDRIVE", "HOMEPATH"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))
    for name in ("LOCALAPPDATA", "APPDATA", "ProgramFiles", "ProgramFiles(x86)", "ProgramData"):
        monkeypatch.setenv(name, str(nowhere / name))
    monkeypatch.setenv("HOARD_HUB_AUTOSTART", "0")                     # never start a real hub
    monkeypatch.setattr(settings_module, "DEFAULT_GGUF_FOLDER", str(nowhere / "models"))
    monkeypatch.setattr(settings_module, "DEFAULT_LLAMA_SERVER", str(nowhere / "llama.cpp" / "llama-server.exe"))
    monkeypatch.setattr(ollama_models, "CANDIDATES_WINDOWS", ())
    monkeypatch.setattr(ollama_models, "REGISTRY_LOCATIONS", ())          # the Windows registry of the host: no place to look in
    monkeypatch.setattr(servers, "which_on_path", lambda name: None)
    monkeypatch.setattr(cpu_module, "physical_cores", lambda: 12)       # the processor and the free memory of the host are not the tests' business
    monkeypatch.setattr(cpu_module, "ram_free_mb", lambda: None)
    monkeypatch.setattr(config_module, "REPO_ROOT", nowhere / "repo")  # no .env of the real checkout


@pytest.fixture
def clock():
    return Clock()


@pytest.fixture
def svc(tmp_path, clock):
    service = build_services(tmp_path, clock=clock)
    yield service
    service.db.close()


@pytest.fixture
def measured(svc):
    """Three models of different quality that have run the quick suite plus two scored suites."""
    from helpers import run_inline
    big = add_gguf(svc, "grande-q4", accuracy=0.95, tps=30.0, size_gb=17.0)
    mid = add_gguf(svc, "medio-q5", accuracy=0.7, tps=80.0, size_gb=6.0, seed=1)
    small = add_gguf(svc, "pequeno-q8", accuracy=0.4, tps=140.0, size_gb=3.0, seed=2)
    run = run_inline(svc, ["razonamiento", "matematicas", "instrucciones", "rapida"], [big["id"], mid["id"], small["id"]])
    assert run["state"] == "done", run
    return {"big": big, "mid": mid, "small": small, "run": run}


def tool(svc, tool_name: str, /, **arguments):
    from galton_hoard.agent_tools import call_tool
    return call_tool(svc, tool_name, arguments)


@pytest.fixture
def client(tmp_path, clock):
    from fastapi.testclient import TestClient
    from galton_hoard.main import create_app

    services = build_services(tmp_path, clock=clock)
    app = create_app(services.config, services=services)
    with TestClient(app, base_url="http://127.0.0.1") as c:
        c.svc = services
        c.bearer = {"Authorization": f"Bearer {services.token}"}
        yield c
