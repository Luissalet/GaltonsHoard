"""The tests do not depend on the computer they run on, and the real defaults still work: the machine's defaults (the Windows model folders,
``OLLAMA_MODELS``, ``~/.ollama/models``, llama-server on the PATH, GALTON_*/OLLAMA_*/HOARD_* variables) are neutralised by the ``hermetic_host``
fixture in conftest, and these tests check both halves: nothing of the host is visible, and each default is still honoured when something
really is there."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from galton_hoard import ollama_models, servers, settings as settings_module
from galton_hoard.errors import GaltonError
from galton_hoard.servers import Launcher
from test_discovery_backends_servers import TAG, make_discovery, ollama_handler
from test_gguf import make_store


# ------------------------------------------------------------------------------------------------ nothing of the host is visible
def test_no_variable_of_the_host_reaches_the_tests():
    leaked = [k for k in os.environ if k.upper().startswith(("GALTON_", "OLLAMA_", "HF_", "FAUSTUS_", "CUDA_")) or k.upper() in ("PORT", "PORT_STRICT")]
    assert leaked == []
    assert os.environ["HOARD_HUB_AUTOSTART"] == "0" and not Path(os.environ["LOCALAPPDATA"]).exists()


def test_the_profile_folder_is_an_empty_stand_in():
    assert Path.home() == Path(os.environ["HOME"]) and list(Path.home().iterdir()) == []


def test_no_model_store_of_the_host_is_found(tmp_path):
    assert ollama_models.candidate_dirs("") == [Path.home() / ".ollama" / "models"]
    assert ollama_models.models_dir("") is None and ollama_models.models_dir(str(tmp_path / "none")) is None


def test_no_default_folder_or_program_of_the_host_is_used(svc):
    assert svc.settings.get("gguf.folders") == []
    assert not Path(svc.settings.get("llama.server_path")).exists()
    assert servers.which_on_path("llama-server") is None
    with pytest.raises(GaltonError) as exc:
        Launcher(svc.settings, svc.config.logs_dir, svc.config.servers_file).binary()
    assert exc.value.code == "unavailable"


def test_the_checkout_dotenv_is_not_read():
    from galton_hoard.config import Config
    assert Config.from_env().secrets == {}


# ------------------------------------------------------------------------------------------------ the real defaults still apply
def test_the_model_folder_is_looked_up_in_order(tmp_path, monkeypatch):
    configured, from_env, windows = (make_store(tmp_path / n) for n in ("a", "b", "c"))
    home_store = Path.home() / ".ollama" / "models"
    (home_store / "manifests" / "registry.ollama.ai" / "library" / "m").mkdir(parents=True)
    (home_store / "manifests" / "registry.ollama.ai" / "library" / "m" / "1b").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(ollama_models, "CANDIDATES_WINDOWS", (str(windows),))
    assert ollama_models.models_dir("") == home_store, "the profile folder comes before the Windows default"
    monkeypatch.setenv("OLLAMA_MODELS", str(from_env))
    assert ollama_models.models_dir("") == from_env, "OLLAMA_MODELS comes before the profile folder"
    assert ollama_models.models_dir(str(configured)) == configured, "the setting comes first"
    assert ollama_models.models_dir(str(tmp_path / "none")) == from_env, "a setting that is not a store falls through to the next place"
    monkeypatch.delenv("OLLAMA_MODELS")
    (home_store / "manifests" / "registry.ollama.ai" / "library" / "m" / "1b").unlink()
    assert ollama_models.models_dir("") == windows, "the Windows default is the last resort, and an empty store does not hide it"


def test_discovery_finds_the_store_of_the_host_when_the_setting_does_not_name_one(svc, tmp_path, monkeypatch):
    """The failure seen on a Windows computer with models installed: the tests set a folder that is not a store and still found the real one."""
    svc.settings.set_many({"ollama.models_dir": str(tmp_path / "none")})
    out = make_discovery(svc, ollama_handler([TAG])).refresh()
    assert out["sources"]["ollama"]["manifests"] is None and len(out["new"]) == 1
    root = make_store(tmp_path / "windows-default")
    monkeypatch.setattr(ollama_models, "CANDIDATES_WINDOWS", (str(root),))
    out = make_discovery(svc, ollama_handler([TAG])).refresh()
    assert out["sources"]["ollama"]["manifests"] == str(root) and svc.store.contestant_by_key("gguf:ollama:qwen3:8b") is not None


def test_the_default_gguf_folder_is_used_only_when_it_exists(tmp_path, monkeypatch, svc):
    folder = tmp_path / "models"
    monkeypatch.setattr(settings_module, "DEFAULT_GGUF_FOLDER", str(folder))
    assert svc.settings.get("gguf.folders") == []
    folder.mkdir()
    assert svc.settings.get("gguf.folders") == [str(folder)]
    svc.settings.set_many({"gguf.folders": [str(tmp_path / "mine")]})
    assert svc.settings.get("gguf.folders") == [str(tmp_path / "mine")], "a saved setting beats the default"


def test_the_llama_server_default_is_the_windows_install(tmp_path, monkeypatch, svc):
    exe = tmp_path / "llama.cpp" / "llama-server.exe"
    monkeypatch.setattr(settings_module, "DEFAULT_LLAMA_SERVER", str(exe))
    launcher = Launcher(svc.settings, tmp_path / "logs", tmp_path / "servers.json")
    with pytest.raises(GaltonError):
        launcher.binary()
    exe.parent.mkdir()
    exe.write_bytes(b"")
    assert svc.settings.get("llama.server_path") == str(exe) and launcher.binary() == str(exe)


def test_llama_server_is_found_on_the_path_when_the_setting_is_not_a_file(tmp_path, monkeypatch, svc):
    monkeypatch.setattr(servers, "which_on_path", lambda name: str(tmp_path / "bin" / name) if name == "llama-server" else None)
    svc.settings.set_many({"llama.server_path": str(tmp_path / "no-such")})
    assert Launcher(svc.settings, tmp_path / "logs", tmp_path / "servers.json").binary() == str(tmp_path / "bin" / "llama-server")
    with pytest.raises(GaltonError):
        Launcher(svc.settings, tmp_path / "logs", tmp_path / "servers.json", which=lambda n: None).binary()
