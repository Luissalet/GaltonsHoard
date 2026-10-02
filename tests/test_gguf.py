"""GGUF reader (a tiny file built in the test), Ollama manifests, memory estimates and file digests."""

from __future__ import annotations

import json
import struct
import sys
from pathlib import Path

import pytest

from galton_hoard import gguf_meta, ollama_models
from galton_hoard.errors import GaltonError


# ------------------------------------------------------------------ a minimal GGUF writer
def _s(text: str) -> bytes:
    raw = text.encode()
    return struct.pack("<Q", len(raw)) + raw


def _kv(key: str, value) -> bytes:
    out = _s(key)
    if isinstance(value, bool):
        return out + struct.pack("<I?", 7, value)
    if isinstance(value, int):
        return out + struct.pack("<II", 4, value)
    if isinstance(value, float):
        return out + struct.pack("<If", 6, value)
    if isinstance(value, str):
        return out + struct.pack("<I", 8) + _s(value)
    if isinstance(value, list) and value and isinstance(value[0], str):
        return out + struct.pack("<IIQ", 9, 8, len(value)) + b"".join(_s(v) for v in value)
    if isinstance(value, list):
        return out + struct.pack("<IIQ", 9, 4, len(value)) + b"".join(struct.pack("<I", v) for v in value)
    raise TypeError(value)


def write_gguf(path: Path, pairs: dict, version: int = 3, pad: int = 0) -> Path:
    body = b"".join(_kv(k, v) for k, v in pairs.items())
    path.write_bytes(b"GGUF" + struct.pack("<IQQ", version, 7, len(pairs)) + body + b"\0" * pad)
    return path


LLAMA = {"general.architecture": "llama", "general.name": "Tiny", "general.file_type": 15, "general.size_label": "8B", "llama.context_length": 8192,
         "llama.block_count": 32, "llama.embedding_length": 4096, "llama.attention.head_count": 32, "llama.attention.head_count_kv": 8,
         "tokenizer.chat_template": "{{ messages }}"}


@pytest.fixture
def tiny(tmp_path):
    return write_gguf(tmp_path / "Tiny-8B-Q4_K_M.gguf", LLAMA)


def test_reads_the_header_and_summarises(tiny):
    meta = gguf_meta.read_meta(tiny)
    assert meta["architecture"] == "llama" and meta["context_length"] == 8192 and meta["block_count"] == 32
    assert meta["head_count_kv"] == 8 and meta["head_dim"] == 128 and meta["quant"] == "Q4_K_M" and meta["params_b"] == 8.0
    assert meta["has_chat_template"] is True and meta["is_projector"] is False and meta["size_bytes"] == tiny.stat().st_size


def test_long_arrays_are_counted_not_kept(tmp_path):
    path = write_gguf(tmp_path / "a.gguf", {**LLAMA, "tokenizer.ggml.tokens": [f"t{i}" for i in range(100)], "x.nums": list(range(5))})
    kv = gguf_meta.read_kv(path)
    assert kv["tokenizer.ggml.tokens"] == {"array": 100} and kv["x.nums"] == [0, 1, 2, 3, 4]
    assert gguf_meta.summarize(kv)["vocab_size"] == 100


def test_per_layer_arrays_use_the_largest_entry(tmp_path):
    path = write_gguf(tmp_path / "a.gguf", {**LLAMA, "llama.attention.head_count_kv": [8, 8, 4, 2]})
    assert gguf_meta.read_meta(path)["head_count_kv"] == 8


def test_projector_is_recognised(tmp_path):
    path = write_gguf(tmp_path / "mmproj.gguf", {"general.architecture": "clip"})
    assert gguf_meta.read_meta(path)["is_projector"] is True


def test_bad_magic_truncation_and_version_are_honest_errors(tmp_path):
    bad = tmp_path / "bad.gguf"
    bad.write_bytes(b"NOPE" + b"\0" * 40)
    with pytest.raises(gguf_meta.GgufError, match="bad magic"):
        gguf_meta.read_kv(bad)
    cut = tmp_path / "cut.gguf"
    cut.write_bytes((tmp_path / "bad.gguf").read_bytes()[:2])
    with pytest.raises(GaltonError):
        gguf_meta.read_kv(cut)
    old = write_gguf(tmp_path / "v1.gguf", LLAMA, version=1)
    with pytest.raises(gguf_meta.GgufError, match="version"):
        gguf_meta.read_kv(old)
    with pytest.raises(GaltonError) as e:
        gguf_meta.read_kv(tmp_path / "missing.gguf")
    assert e.value.code == "not_found"


def test_truncated_after_the_header(tmp_path):
    full = write_gguf(tmp_path / "f.gguf", LLAMA)
    cut = tmp_path / "t.gguf"
    cut.write_bytes(full.read_bytes()[:60])
    with pytest.raises(gguf_meta.GgufError):
        gguf_meta.read_kv(cut)


@pytest.mark.parametrize("name,quant", [("qwen3-27b-Q4_K_M.gguf", "Q4_K_M"), ("model.IQ3_XXS.gguf", "IQ3_XXS"), ("x-bf16.gguf", "BF16"), ("plain.gguf", ""), ("q8_0-model", "Q8_0")])
def test_quant_from_name(name, quant):
    assert gguf_meta.quant_from_name(name) == quant


@pytest.mark.parametrize("text,b", [("Qwen3-27B-Instruct", 27.0), ("gemma-2b", 2.0), ("tiny-600M", 0.6), ("llama-3.1-8B", 8.0), ("nothing", None)])
def test_params_from_text(text, b):
    assert gguf_meta.params_from_text(text) == b


def test_digest_changes_with_content_and_size(tmp_path):
    a = write_gguf(tmp_path / "a.gguf", LLAMA)
    b = write_gguf(tmp_path / "b.gguf", {**LLAMA, "general.name": "Other"})
    c = write_gguf(tmp_path / "c.gguf", LLAMA, pad=10)
    assert gguf_meta.file_digest(a) == gguf_meta.file_digest(a)
    assert len({gguf_meta.file_digest(p) for p in (a, b, c)}) == 3


def test_kv_cache_is_exact_with_layer_and_head_counts(tiny):
    meta = gguf_meta.read_meta(tiny)
    # 2 (K and V) * 32 layers * 8192 tokens * 8 heads * 128 dim * 2 bytes = 1 GiB
    assert gguf_meta.kv_cache_mb(meta, 8192) == pytest.approx(1024.0)


def test_kv_cache_falls_back_to_a_rule_of_thumb():
    assert gguf_meta.kv_cache_mb({"params_b": 27.0}, 8192) == pytest.approx(0.15 * 8.192 * 1024, rel=1e-6)
    assert gguf_meta.kv_cache_mb({"params_b": 7.0}, 8192) < gguf_meta.kv_cache_mb({"params_b": 27.0}, 8192)


def test_estimate_adds_weights_cache_projector_and_headroom(tiny):
    meta = gguf_meta.read_meta(tiny)
    size = 4 * 1024 ** 3
    base = gguf_meta.estimate_vram_mb(size, meta, 8192)
    assert base == pytest.approx(size * 1.08 / 1024 ** 2 + 1024 + 600, abs=2)
    assert gguf_meta.estimate_vram_mb(size, meta, 8192, mmproj_bytes=1024 ** 3) > base + 1000


# ------------------------------------------------------------------ Ollama manifests
def make_store(tmp_path, name="qwen3", tag="8b", projector=False, blobs=True):
    root = tmp_path / "ollama"
    manifest_dir = root / "manifests" / "registry.ollama.ai" / "library" / name
    manifest_dir.mkdir(parents=True)
    layers = [{"mediaType": "application/vnd.ollama.image.model", "digest": "sha256:" + "a" * 64, "size": 1234},
              {"mediaType": "application/vnd.ollama.image.template", "digest": "sha256:" + "b" * 64, "size": 10}]
    if projector:
        layers.append({"mediaType": "application/vnd.ollama.image.projector", "digest": "sha256:" + "c" * 64, "size": 99})
    (manifest_dir / tag).write_text(json.dumps({"schemaVersion": 2, "layers": layers}), encoding="utf-8")
    (root / "blobs").mkdir()
    if blobs:
        (root / "blobs" / ("sha256-" + "a" * 64)).write_bytes(b"x")
        if projector:
            (root / "blobs" / ("sha256-" + "c" * 64)).write_bytes(b"y")
    return root


@pytest.mark.parametrize("ref,expected", [
    ("qwen3:8b", ("registry.ollama.ai", "library", "qwen3", "8b")),
    ("qwen3", ("registry.ollama.ai", "library", "qwen3", "latest")),
    ("user/model:q4", ("registry.ollama.ai", "user", "model", "q4")),
    ("host.io/ns/name:t", ("host.io", "ns", "name", "t")),
])
def test_parse_ref(ref, expected):
    assert ollama_models.parse_ref(ref) == expected


def test_display_name_round_trips():
    assert ollama_models.display_name(*ollama_models.parse_ref("qwen3:8b")) == "qwen3:8b"
    assert ollama_models.display_name(*ollama_models.parse_ref("user/model")) == "user/model:latest"


def test_resolve_finds_the_blob_and_the_projector(tmp_path):
    root = make_store(tmp_path, projector=True)
    out = ollama_models.resolve("qwen3:8b", root)
    assert out["model_path"].endswith("sha256-" + "a" * 64) and out["mmproj_path"].endswith("sha256-" + "c" * 64)
    assert out["digest"] == "sha256:" + "a" * 64 and out["size_bytes"] == 1234 and out["missing"] == [] and out["ref"] == "qwen3:8b"


def test_resolve_reports_missing_blobs(tmp_path):
    root = make_store(tmp_path, blobs=False)
    assert len(ollama_models.resolve("qwen3:8b", root)["missing"]) == 1


def test_resolve_unknown_model_or_folder(tmp_path):
    root = make_store(tmp_path)
    assert ollama_models.resolve("nope:1b", root) is None
    assert ollama_models.resolve("qwen3:8b", None) is None


def test_resolve_manifest_without_model_layer(tmp_path):
    root = make_store(tmp_path)
    path = root / "manifests" / "registry.ollama.ai" / "library" / "qwen3" / "8b"
    path.write_text(json.dumps({"layers": [{"mediaType": "other", "digest": "sha256:z"}]}), encoding="utf-8")
    assert ollama_models.resolve("qwen3:8b", root) is None


def test_corrupt_manifest_is_ignored(tmp_path):
    root = make_store(tmp_path)
    (root / "manifests" / "registry.ollama.ai" / "library" / "qwen3" / "8b").write_text("{not json", encoding="utf-8")
    assert ollama_models.resolve("qwen3:8b", root) is None


def test_list_models_walks_the_manifests(tmp_path):
    root = make_store(tmp_path)
    other = root / "manifests" / "registry.ollama.ai" / "library" / "gemma"
    other.mkdir(parents=True)
    (other / "2b").write_text("{}", encoding="utf-8")
    assert ollama_models.list_models(root) == ["gemma:2b", "qwen3:8b"]
    assert ollama_models.list_models(None) == []


def test_models_dir_uses_the_setting_first(tmp_path, monkeypatch):
    root = make_store(tmp_path)
    monkeypatch.setenv("OLLAMA_MODELS", str(tmp_path / "elsewhere"))
    assert ollama_models.models_dir(str(root)) == root
    assert ollama_models.candidate_dirs(str(root))[0] == root


def test_models_dir_from_the_environment(tmp_path, monkeypatch):
    root = make_store(tmp_path)
    monkeypatch.setenv("OLLAMA_MODELS", str(root))
    assert ollama_models.models_dir("") == root


def test_per_layer_kv_heads_of_a_hybrid_model_survive_and_size_the_cache(tmp_path):
    # 80 layers (more than a short array keeps), one in four with a KV cache: the estimate counts 20 layers, not 80
    per_layer = [8 if i % 4 == 3 else 0 for i in range(80)]
    pairs = {**LLAMA, "llama.block_count": 80, "llama.attention.head_count_kv": per_layer}
    meta = gguf_meta.read_meta(write_gguf(tmp_path / "hybrid.gguf", pairs))
    assert meta["head_count_kv"] == 8 and meta["kv_layers"] == 20 and meta["block_count"] == 80
    dense = {**meta, "kv_layers": None}
    assert gguf_meta.kv_cache_mb(meta, 8192) == pytest.approx(gguf_meta.kv_cache_mb(dense, 8192) / 4)


def test_arrays_of_arrays_are_read_not_refused(tmp_path):
    inner = struct.pack("<IQ", 4, 2) + struct.pack("<II", 1, 2)
    nested = _s("x.pairs") + struct.pack("<IIQ", 9, 9, 3) + inner * 3
    body = nested + _kv("general.architecture", "llama")
    path = tmp_path / "nested.gguf"
    path.write_bytes(b"GGUF" + struct.pack("<IQQ", 3, 0, 2) + body)
    kv = gguf_meta.read_kv(path)
    assert kv["x.pairs"] == [[1, 2], [1, 2], [1, 2]] and kv["general.architecture"] == "llama"


def test_a_hugging_face_namespace_model_resolves_and_lists_under_its_full_name(tmp_path):
    root = make_store(tmp_path)
    folder = root / "manifests" / "hf.co" / "Org" / "Repo-GGUF"
    folder.mkdir(parents=True)
    layer = {"mediaType": ollama_models.MODEL_LAYER, "digest": "sha256:" + "a" * 64, "size": 99}
    (folder / "Q8_0").write_text(json.dumps({"layers": [layer]}), encoding="utf-8")
    out = ollama_models.resolve("hf.co/Org/Repo-GGUF:Q8_0", root)
    assert out["ref"] == "hf.co/Org/Repo-GGUF:Q8_0" and out["model_path"].endswith("sha256-" + "a" * 64) and out["missing"] == []
    assert "hf.co/Org/Repo-GGUF:Q8_0" in ollama_models.list_models(root)


# ------------------------------------------------------------------ finding the real store on Windows
def test_an_empty_profile_store_does_not_hide_the_real_one(tmp_path, monkeypatch):
    """The failure on the real computer: ~/.ollama/models/manifests exists but is empty, the models are in the folder OLLAMA_MODELS names."""
    empty = Path.home() / ".ollama" / "models"
    (empty / "manifests" / "registry.ollama.ai" / "library").mkdir(parents=True)
    real = make_store(tmp_path / "real")
    monkeypatch.setenv("OLLAMA_MODELS", str(real))
    assert ollama_models.models_dir("") == real
    assert ollama_models.store_dirs("") == [real, empty], "the empty store stays as the last resort"


def test_a_store_with_manifests_wins_over_an_earlier_empty_one_even_from_the_windows_default(tmp_path, monkeypatch):
    empty = Path.home() / ".ollama" / "models"
    (empty / "manifests").mkdir(parents=True)
    windows = make_store(tmp_path / "w")
    monkeypatch.setattr(ollama_models, "CANDIDATES_WINDOWS", (str(windows),))
    assert ollama_models.models_dir("") == windows


def test_only_empty_stores_still_give_the_first_one(tmp_path):
    empty = Path.home() / ".ollama" / "models"
    (empty / "manifests").mkdir(parents=True)
    assert ollama_models.models_dir("") == empty


def test_resolve_tries_every_store_until_the_manifest_exists(tmp_path):
    a = make_store(tmp_path / "a", name="other")
    b = make_store(tmp_path / "b", name="qwen3")
    assert ollama_models.resolve("qwen3:8b", a) is None
    found = ollama_models.resolve("qwen3:8b", [a, b])
    assert found and found["model_path"].startswith(str(b))
    assert ollama_models.resolve("qwen3:8b", [b, a])["model_path"].startswith(str(b))
    assert ollama_models.resolve("nope:1b", [a, b]) is None and ollama_models.resolve("qwen3:8b", []) is None


def test_ollama_models_is_read_from_the_registry_when_the_process_lacks_it(tmp_path, monkeypatch):
    real = make_store(tmp_path / "real")
    monkeypatch.setattr(ollama_models, "IS_WINDOWS", True)
    asked = []
    monkeypatch.setattr(ollama_models, "registry_variable", lambda name: asked.append(name) or str(real))
    assert ollama_models.environment_variable("OLLAMA_MODELS") == str(real) and asked == ["OLLAMA_MODELS"]
    assert ollama_models.models_dir("") == real


def test_the_process_environment_beats_the_registry(tmp_path, monkeypatch):
    mine, theirs = make_store(tmp_path / "mine"), make_store(tmp_path / "theirs")
    monkeypatch.setattr(ollama_models, "IS_WINDOWS", True)
    monkeypatch.setattr(ollama_models, "registry_variable", lambda name: str(theirs))
    monkeypatch.setenv("OLLAMA_MODELS", str(mine))
    assert ollama_models.models_dir("") == mine


def test_the_registry_is_not_asked_off_windows(monkeypatch):
    monkeypatch.setattr(ollama_models, "IS_WINDOWS", False)
    monkeypatch.setattr(ollama_models, "registry_variable", lambda name: pytest.fail("the registry does not exist here"))
    assert ollama_models.environment_variable("OLLAMA_MODELS") == ""


class FakeWinreg:
    """The slice of ``winreg`` the reader uses, over a dict of {(hive, subkey): {name: (value, kind)}}."""
    HKEY_CURRENT_USER, HKEY_LOCAL_MACHINE, REG_SZ, REG_EXPAND_SZ = "HKCU", "HKLM", 1, 2

    def __init__(self, tables):
        self.tables = tables

    def OpenKey(self, hive, subkey):  # noqa: N802
        table = self.tables.get((hive, subkey))
        if table is None:
            raise FileNotFoundError(subkey)
        return _Key(table)

    def QueryValueEx(self, key, name):  # noqa: N802
        if name not in key.table:
            raise FileNotFoundError(name)
        return key.table[name]


class _Key:
    def __init__(self, table):
        self.table = table

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def test_the_registry_reader_prefers_the_user_then_the_machine_and_expands(monkeypatch):
    user_key = ("HKCU", "Environment")
    machine_key = ("HKLM", r"SYSTEM\CurrentControlSet\Control\Session Manager\Environment")
    monkeypatch.setattr(ollama_models, "IS_WINDOWS", True)
    monkeypatch.setattr(ollama_models, "REGISTRY_LOCATIONS", (("HKEY_CURRENT_USER", "Environment"), ("HKEY_LOCAL_MACHINE", machine_key[1])))
    monkeypatch.setenv("LOCALAI_ROOT", "D:\\LocalAI")
    fake = FakeWinreg({user_key: {"OLLAMA_MODELS": ("%LOCALAI_ROOT%\\ollama-models", FakeWinreg.REG_EXPAND_SZ)}, machine_key: {"OLLAMA_MODELS": ("C:\\machine", 1), "ONLY_MACHINE": ("x", 1)}})
    monkeypatch.setitem(sys.modules, "winreg", fake)
    assert ollama_models.registry_variable("OLLAMA_MODELS") == "D:\\LocalAI\\ollama-models"
    assert ollama_models.registry_variable("ONLY_MACHINE") == "x"
    assert ollama_models.registry_variable("NOT_THERE") is None
    fake.tables[user_key] = {"OLLAMA_MODELS": ("   ", 1)}
    assert ollama_models.registry_variable("OLLAMA_MODELS") == "C:\\machine", "a blank user value falls through to the machine"
