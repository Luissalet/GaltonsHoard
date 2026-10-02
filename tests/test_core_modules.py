"""Smaller modules: ad hoc contestants, the judge, placement, processes, configuration, the database and the store."""

from __future__ import annotations

import json
import os
import sqlite3
import subprocess
import sys
import time

import pytest

from galton_hoard import adhoc, placement
from galton_hoard.backends import Completion
from galton_hoard.config import Config, default_routes_file, load_dotenv
from galton_hoard.db import MIGRATIONS, Database
from galton_hoard.errors import GaltonError
from galton_hoard.judging import has_judge, make_ask, pending_reply
from galton_hoard.hoard_link import proc as hl_proc
from galton_hoard.procs import IS_WINDOWS, kill_pid_tree, kill_tree, process_name
from helpers import add_gguf
from test_gguf import LLAMA, make_store, write_gguf


# ---- ad hoc contestants ----------------------------------------------------------------------------------------------------------------------

def resolve(svc, item):
    return adhoc.resolve_one(svc.store, svc.settings, item, meta_reader=svc.meta_reader, digest_fn=svc.digest_fn)


def test_a_name_resolves_to_a_registered_model(svc):
    m = add_gguf(svc, "qwen-test")
    assert resolve(svc, "qwen-test")["id"] == m["id"] and resolve(svc, "QWEN-TEST.gguf")["id"] == m["id"]
    with pytest.raises(GaltonError) as exc:
        resolve(svc, "nadie")
    assert exc.value.code == "not_found"


def test_a_gguf_spec_registers_an_evaluation_only_model(svc, tmp_path):
    f = write_gguf(tmp_path / "Afinado-8B-Q4_K_M.gguf", LLAMA)
    c = resolve(svc, {"kind": "gguf", "path": str(f)})
    assert c["adhoc"] is True and c["kind"] == "gguf" and c["name"] == "Afinado-8B-Q4_K_M" and c["quant"] == "Q4_K_M" and c["source"] == "spec"
    again = resolve(svc, {"kind": "gguf", "path": str(f)})
    assert again["id"] == c["id"] and len(svc.store.contestants(include_adhoc=True)) == 1


def test_gguf_spec_errors(svc, tmp_path):
    for spec, code in [({"kind": "gguf"}, "invalid"), ({"kind": "gguf", "path": "relativo.gguf"}, "invalid"), ({"kind": "gguf", "path": str(tmp_path / "no.gguf")}, "not_found")]:
        with pytest.raises(GaltonError) as exc:
            resolve(svc, spec)
        assert exc.value.code == code
    f = write_gguf(tmp_path / "m.gguf", LLAMA)
    with pytest.raises(GaltonError) as exc:
        resolve(svc, {"kind": "gguf", "path": str(f), "mmproj": str(tmp_path / "ausente.gguf")})
    assert exc.value.code == "not_found"
    proj = write_gguf(tmp_path / "proj.gguf", {"general.architecture": "clip", "clip.has_vision_encoder": True})
    with pytest.raises(GaltonError) as exc:
        resolve(svc, {"kind": "gguf", "path": str(proj)})
    assert "projector" in exc.value.message


def test_server_spec_flags_remote_hosts_and_never_trusts_them_by_default(svc):
    local = resolve(svc, {"kind": "server", "url": "http://127.0.0.1:8080/", "model": "m"})
    remote = resolve(svc, {"kind": "server", "url": "https://api.example.com", "model": "m"})
    assert local["remote"] is False and local["api"] == "openai" and remote["remote"] is True and remote["remote_ok"] is False and remote["provider"] == "remote"
    assert resolve(svc, {"kind": "server", "url": "http://127.0.0.1:8080", "model": "m"})["id"] == local["id"]
    assert resolve(svc, {"kind": "server", "url": "http://localhost:11434", "model": "q"})["api"] == "ollama"
    with pytest.raises(GaltonError):
        resolve(svc, {"kind": "server", "url": "http://127.0.0.1:1"})


def test_ollama_spec_uses_a_known_model_or_the_blob_on_disk(svc, tmp_path):
    with pytest.raises(GaltonError) as exc:
        resolve(svc, {"kind": "ollama", "model": "qwen3:8b"})
    assert exc.value.code == "not_found"
    root = make_store(tmp_path)
    blob = root / "blobs" / ("sha256-" + "a" * 64)
    write_gguf(blob, LLAMA)
    svc.settings.set_many({"ollama.models_dir": str(root)})
    c = resolve(svc, {"kind": "ollama", "model": "qwen3:8b"})
    assert c["kind"] == "gguf" and c["name"] == "qwen3:8b (llama.cpp)" and c["adhoc"] is True
    known = add_gguf(svc, "ya-registrado")
    svc.store.update_contestant(known["id"], ollama_ref="otro:1b", aliases=["otro:1b"])
    assert resolve(svc, {"kind": "ollama", "model": "otro:1b"})["id"] == known["id"]


def test_unknown_kinds_and_non_objects_are_refused(svc):
    for bad in ({"kind": "cloud"}, 42, ["x"]):
        with pytest.raises(GaltonError):
            resolve(svc, bad)


def test_resolve_many_deduplicates(svc):
    m = add_gguf(svc, "uno")
    out = adhoc.resolve_many(svc.store, svc.settings, ["uno", m["id"], "UNO"], meta_reader=svc.meta_reader, digest_fn=svc.digest_fn)
    assert [c["id"] for c in out] == [m["id"]]


@pytest.mark.parametrize("url,remote", [("http://127.0.0.1:1", False), ("http://localhost", False), ("http://[::1]:5", False), ("192.168.1.5:8080", True), ("https://host.example", True)])
def test_is_remote(url, remote):
    assert adhoc.is_remote(url) is remote


# ---- judge -----------------------------------------------------------------------------------------------------------------------------------

class StubBackend:
    def __init__(self, text='{"score": 8, "reasons": "bien"}', error=""):
        self.text, self.error, self.calls = text, error, 0

    def chat(self, req, cancel=lambda: False):
        self.calls += 1
        return Completion(text=self.text, error=self.error)


JUDGE = {"id": "c_judge", "name": "Juez", "digest": "d"}
REQUEST = {"prompt": "p", "rubric": "r", "answer": "a"}


def test_the_judge_grades_and_caches_by_what_it_saw(svc):
    backend = StubBackend()
    ask = make_ask(svc.store, backend, JUDGE, "c_other")
    first = ask(REQUEST)
    assert first["score"] == 8 and first["judge"] == "Juez" and first["self_judged"] is False and "cached" not in first
    second = ask(REQUEST)
    assert second["cached"] is True and backend.calls == 1
    assert ask({**REQUEST, "answer": "otra"})["score"] == 8 and backend.calls == 2


def test_the_judge_marks_its_own_answers_as_self_judged(svc):
    assert make_ask(svc.store, StubBackend(), JUDGE, "c_judge")(REQUEST)["self_judged"] is True


def test_judge_failures_are_reported_not_cached(svc):
    assert make_ask(svc.store, StubBackend(error="HTTP 500"), JUDGE, "x")(REQUEST) == {"error": "HTTP 500"}
    bad = StubBackend(text="no sé")
    assert "error" in make_ask(svc.store, bad, JUDGE, "x")(REQUEST)
    good = StubBackend()
    assert make_ask(svc.store, good, JUDGE, "x")(REQUEST)["score"] == 8 and good.calls == 1


def test_has_judge_looks_inside_combinators():
    assert has_judge({"type": "judge"}) and has_judge({"type": "all", "checks": [{"type": "exact"}, {"type": "any", "checks": [{"type": "judge"}]}]})
    assert not has_judge({"type": "all", "checks": [{"type": "exact"}]}) and not has_judge("x")
    assert pending_reply({}) == {"error": "judge not live", "pending": True}


# ---- placement -------------------------------------------------------------------------------------------------------------------------------

def test_context_is_capped_by_the_trained_context():
    assert placement.choose_context(32768, {"context_length": 8192}) == 8192
    assert placement.choose_context(4096, {"context_length": 8192}) == 4096 and placement.choose_context(4096, {}) == 4096


def test_describe_a_gguf_that_fits_one_card(svc):
    m = svc.store.contestant(add_gguf(svc, "cabe", size_gb=5.0)["id"])
    out = placement.describe(svc.store, m, 8192, svc.gpus, svc.meta_reader)
    assert out["runs_on"] == "llama.cpp" and out["fits_one_16gb"] is True and out["vram_mb"] > 5 * 1024 and out["where"].startswith("GPU ") and out["gpus"]


def test_describe_a_gguf_that_needs_two_cards_and_one_that_fits_nowhere(svc):
    two = placement.describe(svc.store, svc.store.contestant(add_gguf(svc, "doble", size_gb=24.0)["id"]), 8192, svc.gpus, svc.meta_reader)
    assert two["fits_one_16gb"] is False and len(two["gpus"]) == 2
    nowhere = placement.describe(svc.store, svc.store.contestant(add_gguf(svc, "enorme", size_gb=80.0)["id"]), 8192, svc.gpus, svc.meta_reader)
    assert nowhere["warnings"] and "GB" in nowhere["where"]


def test_describe_a_server_and_an_unknown_size(svc):
    server = svc.store.create_contestant(key="s", kind="server", name="s", url="http://127.0.0.1:1", api="openai", model="m", meta={"resident": True})
    out = placement.describe(svc.store, server, 8192, svc.gpus)
    assert out["runs_on"] == "server" and "resident" in out["where"] and out["vram_mb"] is None
    nosize = svc.store.create_contestant(key="g", kind="gguf", name="g", path="/no/such.gguf")
    out = placement.describe(svc.store, nosize, 8192, svc.gpus)
    assert out["vram_mb"] is None and out["warnings"]


# ---- processes -------------------------------------------------------------------------------------------------------------------------------

def test_children_start_in_their_own_group_through_the_shared_launcher():
    child = hl_proc.popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        if not IS_WINDOWS:
            assert os.getpgid(child.pid) == child.pid                   # its own session: killing the group never touches this process
    finally:
        kill_tree(child)
    assert child.poll() is not None


@pytest.mark.skipif(IS_WINDOWS, reason="uses a POSIX shell")
def test_kill_tree_kills_the_children_too(tmp_path):
    marker = tmp_path / "child.pid"
    proc = hl_proc.popen(["sh", "-c", f"sleep 60 & echo $! > {marker}; wait"])
    for _ in range(50):
        if marker.exists() and marker.read_text().strip():
            break
        time.sleep(0.05)
    child = int(marker.read_text().strip())
    kill_tree(proc)
    assert proc.poll() is not None
    for _ in range(50):
        if "sleep" not in process_name(child):
            break
        time.sleep(0.05)
    assert "sleep" not in process_name(child)


def test_kill_tree_of_nothing_or_an_ended_process_is_harmless():
    kill_tree(None)
    kill_pid_tree(2 ** 22 + 12345)          # a pid that does not exist
    proc = subprocess.Popen([sys.executable, "-c", "pass"])
    proc.wait()
    kill_tree(proc)


@pytest.mark.skipif(IS_WINDOWS, reason="reads /proc")
def test_process_name_reads_the_command_line():
    import os
    assert "python" in process_name(os.getpid()) and process_name(2 ** 22 + 12345) == ""


# ---- configuration ---------------------------------------------------------------------------------------------------------------------------

def test_dotenv_reader(tmp_path):
    f = tmp_path / ".env"
    f.write_text("# c\nA=1\nB = 'dos'\nC=\"tres\"\n\nsin igual\nD=a=b\n", encoding="utf-8")
    assert load_dotenv(f) == {"A": "1", "B": "dos", "C": "tres", "D": "a=b"}
    assert load_dotenv(tmp_path / "missing") == {}


def test_from_env_reads_the_prefixed_variables(monkeypatch, tmp_path):
    for k, v in {"GALTON_DATA_DIR": str(tmp_path), "GALTON_PORT": "6000", "GALTON_OFFLINE": "1", "GALTON_FAKE": "1", "GALTON_SCHEDULER": "0", "GALTON_HTTP_TIMEOUT_S": "7"}.items():
        monkeypatch.setenv(k, v)
    c = Config.from_env()
    assert (c.data_dir, c.port, c.offline, c.fake, c.scheduler, c.http_timeout_s, c.data_dir_configured) == (tmp_path, 6000, True, True, False, 7.0, True)


def test_from_env_ignores_bad_values(monkeypatch):
    monkeypatch.setenv("GALTON_PORT", "no")
    monkeypatch.setenv("GALTON_HTTP_TIMEOUT_S", "9999")
    monkeypatch.delenv("PORT", raising=False)
    c = Config.from_env()
    assert c.port == 5201 and c.http_timeout_s == 10.0


def test_secret_prefers_the_environment_over_dotenv(monkeypatch):
    c = Config(secrets={"GALTON_FAUSTUS_TOKEN": "de-env-file"})
    assert c.secret("FAUSTUS_TOKEN") == "de-env-file"
    monkeypatch.setenv("GALTON_FAUSTUS_TOKEN", "del-entorno")
    assert c.secret("FAUSTUS_TOKEN") == "del-entorno"


def test_routes_file_follows_the_shared_variable(monkeypatch, tmp_path):
    monkeypatch.setenv("HOARD_ROUTES_FILE", str(tmp_path / "r.json"))
    assert default_routes_file() == tmp_path / "r.json" and Config().routes_path() == tmp_path / "r.json"
    monkeypatch.delenv("HOARD_ROUTES_FILE")
    assert default_routes_file().name == "routes.json" and default_routes_file().parent.name == ".hoard"
    assert Config(routes_file=tmp_path / "x.json").routes_path() == tmp_path / "x.json"


def test_data_paths_live_under_the_data_dir(tmp_path):
    c = Config(data_dir=tmp_path)
    assert all(p.parent == tmp_path for p in (c.db_path, c.token_path, c.url_path, c.logs_dir, c.images_dir, c.cache_dir, c.servers_file))


# ---- database and store ----------------------------------------------------------------------------------------------------------------------

def test_the_database_migrates_once_and_uses_wal(tmp_path):
    db = Database(tmp_path / "x.db")
    assert db.version() == len(MIGRATIONS) and db.query("PRAGMA journal_mode")[0][0] == "wal" and db.query("PRAGMA foreign_keys")[0][0] == 1
    db.close()
    again = Database(tmp_path / "x.db")
    assert again.version() == len(MIGRATIONS) and again.query("SELECT COUNT(*) FROM schema_version")[0][0] == len(MIGRATIONS)
    again.close()


def test_answers_graded_after_waiting_lose_the_old_waiting_note(tmp_path):
    raw = sqlite3.connect(str(tmp_path / "x.db"))
    raw.execute("CREATE TABLE schema_version (version INTEGER NOT NULL)")
    for index, sql in enumerate(MIGRATIONS[:6], start=1):
        raw.executescript(sql)
        raw.execute("INSERT INTO schema_version(version) VALUES (?)", (index,))
    raw.execute("INSERT INTO runs(id, created_ts) VALUES ('r_1', 0)")
    rows = [('{"reason": "the judge could not grade: judge not live", "judge_score": 0.0, "reasons": "x"}', 0),   # graded later: the note goes
            ('{"reason": "the judge could not grade: judge not live"}', 1),                                     # still waiting: it stays
            ('{"reason": "wrong number", "judge_score": 0.5}', 0)]                                               # another reason: untouched
    for detail, pending in rows:
        raw.execute("INSERT INTO results(run_id, contestant_id, suite_id, case_id, detail, judge_pending, ts) VALUES ('r_1', 'c', 's', 'k', ?, ?, 0)", (detail, pending))
    raw.commit()
    raw.close()
    db = Database(tmp_path / "x.db")
    details = [json.loads(r["detail"]) for r in db.query("SELECT detail FROM results ORDER BY id")]
    assert "reason" not in details[0] and details[0]["reasons"] == "x"
    assert details[1]["reason"].startswith("the judge could not grade") and details[2]["reason"] == "wrong number"
    db.close()


def test_transactions_commit_and_roll_back(tmp_path):
    db = Database(tmp_path / "x.db")
    with db.transaction():
        db.execute("INSERT INTO settings(key, value) VALUES ('a', '1')")
    with pytest.raises(RuntimeError):
        with db.transaction():
            db.execute("INSERT INTO settings(key, value) VALUES ('b', '2')")
            raise RuntimeError("boom")
    assert db.get_setting("a") == "1" and db.get_setting("b") is None and db.get_setting("b", "d") == "d"
    db.set_setting("a", "3")
    assert db.get_setting("a") == "3"
    db.close()


def test_contestant_round_trip_and_lookup_by_name_alias_and_prefix(svc):
    m = add_gguf(svc, "Qwen3-8B-Instruct")
    assert svc.store.find_contestant(m["id"])["id"] == m["id"]
    assert svc.store.find_contestant("qwen3-8b-instruct")["id"] == m["id"]
    assert svc.store.find_contestant("QWEN3-8B-INSTRUCT.GGUF")["id"] == m["id"]
    assert svc.store.find_contestant("") is None and svc.store.find_contestant("zzz") is None
    assert m["enabled"] is True and m["aliases"] and m["meta"]["demo"] is True


def test_duplicate_contestant_keys_are_refused_by_the_database(svc):
    add_gguf(svc, "uno")
    with pytest.raises(Exception):
        svc.store.create_contestant(key="gguf:/test/uno.gguf", kind="gguf", name="otro")


def test_notices_dedupe_and_can_be_marked_seen(svc):
    a = svc.store.add_notice(kind="new_model", params={"name": "t"}, dedupe="x")
    assert a and svc.store.add_notice(kind="new_model", params={"name": "t"}, dedupe="x") is None
    svc.store.add_notice(kind="new_model", params={"name": "u"})
    assert len(svc.store.notices(unseen=True)) == 2
    assert svc.store.mark_notices_seen() == 2 and svc.store.notices(unseen=True) == []


def test_activity_log_keeps_the_latest_entries(svc):
    for i in range(5):
        svc.store.add_activity("refresh", str(i), i % 2 == 0, i, "d")
    rows = svc.store.activity("refresh", limit=3)
    assert len(rows) == 3
