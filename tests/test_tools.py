"""Every tool of the catalogue, through the same entry point the assistant and the UI use."""

from __future__ import annotations

import json

import pytest

from conftest import tool
from helpers import add_gguf, run_inline
from galton_hoard.agent_tools import TOOLS, TOOLS_BY_NAME, uncapped
from galton_hoard.errors import GaltonError

CALLED: set[str] = set()


def call(svc, name, /, **args):
    CALLED.add(name)
    return tool(svc, name, **args)


def test_the_catalogue_has_unique_names_and_valid_schemas():
    names = [t.name for t in TOOLS]
    assert len(names) == len(set(names)) >= 38
    for t in TOOLS:
        schema = t.input_model.model_json_schema()
        assert schema["type"] == "object"
        assert "Sinónimos:" in t.description
        assert len(t.description.splitlines()[0]) <= 110
        assert {"readOnlyHint", "destructiveHint", "idempotentHint", "openWorldHint"} <= set(t.annotations)


def test_destructive_tools_are_annotated_and_need_confirm():
    destructive = {t.name for t in TOOLS if t.annotations["destructiveHint"]}
    assert destructive == {"model_remove", "suite_remove", "case_remove"}
    discard = TOOLS_BY_NAME["run_discard"]
    assert not discard.annotations["destructiveHint"] and "confirm" in discard.input_model.model_fields and "reason" in discard.input_model.model_fields, "reversible, but it asks for confirm"
    for t in TOOLS:
        if t.name in destructive:
            assert "confirm" in t.input_model.model_fields


def test_overview_on_an_empty_bench_suggests_the_first_step(svc):
    o = call(svc, "galton_overview")
    assert o["counts"]["suites"] == 13 and o["counts"]["cases"] >= 220 and o["counts"]["models"] == 0
    assert any("models_refresh" in s for s in o["next_steps"])
    assert o["allowed_gpus"] == [2, 3]


def test_status_reports_every_part(svc):
    s = call(svc, "galton_status")
    for key in ("scheduler", "gpus", "llama_server", "settings", "counts", "routes_path", "watch"):
        assert key in s, key
    assert s["settings"]["gpus.allowed"] == [2, 3]


# ------------------------------------------------------------------ models
def test_models_list_filters_and_flags(svc):
    a = add_gguf(svc, "alfa-q4")
    add_gguf(svc, "beta-vl-q8", vision=True)
    add_gguf(svc, "gamma-off", enabled=False)
    everything = call(svc, "models_list")
    assert everything["total"] == 3 and all(m["never_measured"] for m in everything["models"])
    assert [m["name"] for m in call(svc, "models_list", vision=True)["models"]] == ["beta-vl-q8"]
    assert [m["name"] for m in call(svc, "models_list", enabled=False)["models"]] == ["gamma-off"]
    assert call(svc, "models_list", text="ALFA")["models"][0]["id"] == a["id"]
    assert call(svc, "models_list", measured="fresh")["total"] == 0
    assert call(svc, "models_list", kind="server")["total"] == 0


def test_model_get_shows_scores_and_where_it_runs(svc, measured):
    out = call(svc, "model_get", model="grande-q4")
    assert out["model"]["measured_n"] > 0 and out["scores"]["score"] is not None
    assert out["where"]["runs_on"] == "llama.cpp" and out["where"]["vram_mb"] > 17 * 1024
    assert out["measurements"] and out["measurements"][0]["vram_mb"]
    assert call(svc, "model_get", model="grande-q4.gguf")["model"]["id"] == measured["big"]["id"]
    with pytest.raises(GaltonError) as e:
        call(svc, "model_get", model="no-existe")
    assert e.value.code == "not_found"


def test_model_add_gguf_validates_the_file(svc, tmp_path):
    with pytest.raises(GaltonError) as e:
        call(svc, "model_add", path=str(tmp_path / "missing.gguf"))
    assert e.value.code == "not_found"
    with pytest.raises(GaltonError) as e:
        call(svc, "model_add")
    assert e.value.code == "invalid"
    with pytest.raises(GaltonError):
        call(svc, "model_add", url="http://127.0.0.1:9", path="/x.gguf")


def test_model_update_remove(svc):
    a = add_gguf(svc, "alfa-q4")
    out = call(svc, "model_update", model="alfa-q4", enabled=False, add_aliases=["alfa:latest"], name="Alfa 4 bits")
    assert out["model"]["enabled"] is False and "alfa:latest" in out["model"]["aliases"] and out["model"]["name"] == "Alfa 4 bits"
    with pytest.raises(GaltonError) as e:
        call(svc, "model_update", model=a["id"], remote_ok=True)
    assert e.value.code == "invalid"
    with pytest.raises(GaltonError) as e:
        call(svc, "model_remove", model=a["id"])
    assert e.value.code == "confirm_required"
    removed = call(svc, "model_remove", model=a["id"], confirm=True)
    assert removed["id"] == a["id"] and call(svc, "models_list", include_missing=True)["total"] == 0


def test_models_refresh_offline_returns_a_clear_summary(svc):
    out = call(svc, "models_refresh")
    assert out["offline"] is True and out["new"] == []


# ------------------------------------------------------------------ suites and cases
def test_suites_list_and_get(svc):
    out = call(svc, "suites_list")
    assert out["count"] == 13 and {s["category"] for s in out["suites"]} >= {"code", "vision", "long_context", "rag"}
    assert [s["id"] for s in call(svc, "suites_list", category="math")["suites"]] == ["s_matematicas"]
    assert all(s["builtin"] for s in call(svc, "suites_list", builtin=True)["suites"])
    got = call(svc, "suite_get", suite="matematicas", limit=5)
    assert got["total_cases"] == 20 and len(got["cases"]) == 5 and got["cases"][0]["checker_label"]
    generated = call(svc, "suite_get", suite="Visión")["suite"] if svc.store.find_suite("Visión") else call(svc, "suite_get", suite="vision")["suite"]
    assert generated["generated"] is True
    case = call(svc, "case_get", case=got["cases"][0]["id"])["case"]
    assert case["reference"] and case["prompt"]["text"]


def test_user_suite_lifecycle(svc):
    s = call(svc, "suite_create", name="Mis pruebas", category="reasoning", description="Cosas mías")["suite"]
    assert s["builtin"] is False and s["cases"] == 0
    c = call(svc, "case_add", suite=s["id"], title="Capital", prompt="¿Capital de Francia?", expected="París", tags=["geo"])
    assert c["case"]["checker"]["type"] == "contains" and c["case"]["source"] == "assistant"
    n = call(svc, "case_add", suite=s["id"], prompt="Suma 2+2. Responde solo con el número.", expected="4")
    assert n["case"]["checker"] == {"type": "number", "expected": 4.0}
    free = call(svc, "case_add", suite=s["id"], title="Abierta", prompt="Cuéntame un chiste", checker="judge:Es gracioso y claro")
    assert free["case"]["checker"]["type"] == "judge" and any("judge" in n for n in free["notes"])
    raw = call(svc, "case_add", suite=s["id"], title="Sin comprobador", prompt="Habla del mar")
    assert raw["case"]["checker"]["type"] == "none" and raw["notes"]
    with pytest.raises(GaltonError) as e:
        call(svc, "case_add", suite=s["id"], title="Capital", prompt="otra", expected="x")
    assert e.value.code == "conflict"
    upd = call(svc, "case_update", case=c["case"]["id"], expected="Paris", weight=2.0)
    assert upd["case"]["weight"] == 2.0 and upd["case"]["checker"]["all"] == ["Paris"]
    renamed = call(svc, "suite_update", suite=s["id"], name="Mis pruebas 2")
    assert renamed["suite"]["name"] == "Mis pruebas 2"
    with pytest.raises(GaltonError) as e:
        call(svc, "case_remove", case=c["case"]["id"])
    assert e.value.code == "confirm_required"
    call(svc, "case_remove", case=c["case"]["id"], confirm=True)
    assert call(svc, "suite_get", suite=s["id"])["total_cases"] == 3
    with pytest.raises(GaltonError):
        call(svc, "suite_remove", suite=s["id"])
    gone = call(svc, "suite_remove", suite=s["id"], confirm=True)
    assert gone["removed"] == s["id"] and svc.store.find_suite(s["id"]) is None


def test_builtin_suites_are_read_only_but_can_be_copied(svc):
    for name, args in (("case_add", {"suite": "s_matematicas", "prompt": "x", "expected": "1"}), ("suite_update", {"suite": "s_matematicas", "name": "otro"}),
                       ("suite_remove", {"suite": "s_matematicas", "confirm": True}), ("cases_import", {"suite": "s_matematicas", "text": '{"prompt":"x"}'})):
        with pytest.raises(GaltonError) as e:
            call(svc, name, **args)
        assert e.value.code == "builtin_readonly", name
    copy = call(svc, "suite_duplicate", suite="s_matematicas", name="Mates mías")["suite"]
    assert copy["cases"] == 20 and copy["builtin"] is False and copy["id"] != "s_matematicas"
    first = call(svc, "suite_get", suite=copy["id"], limit=1)["cases"][0]
    call(svc, "case_update", case=first["id"], weight=3.0)
    assert svc.store.case(first["id"])["weight"] == 3.0
    assert svc.store.cases("s_matematicas")[0]["weight"] == 1.0


def test_case_update_warns_when_results_measured_the_old_version(svc):
    add_gguf(svc, "alfa-q4")
    s = call(svc, "suite_create", name="Pocas", category="reasoning", cases=[{"title": "Uno", "prompt": "Di hola", "expected": "hola"}])["suite"]
    case_id = svc.store.cases(s["id"])[0]["id"]
    run_inline(svc, [s["id"]], ["alfa-q4"])
    assert svc.db_count("results", "case_id", case_id) == 1
    out = call(svc, "case_update", case=case_id, prompt="Di adiós", expected="adiós")
    assert any("old version" in n for n in out["notes"]) and svc.db_count("results", "case_id", case_id) == 1
    out = call(svc, "case_update", case=case_id, prompt="Di buenas", expected="buenas", forget_results=True)
    assert svc.db_count("results", "case_id", case_id) == 0 and any("deleted" in n for n in out["notes"])


def test_cases_import_jsonl_and_csv(svc, tmp_path):
    s = call(svc, "suite_create", name="Importadas", category="custom")["suite"]
    jsonl = "\n".join([json.dumps({"title": "Suma", "prompt": "2+3", "expected": "5"}),
                       json.dumps({"pregunta": "Capital de Italia", "respuesta": "Roma", "checker": "exact:Roma"}),
                       json.dumps({"prompt": "Dame un número par", "checker": "regex:^[0-9]*[02468]$"}),
                       "esto no es json", json.dumps({"title": "Sin prompt"})])
    out = call(svc, "cases_import", suite=s["id"], text=jsonl)
    assert out["added"] == 3 and out["error_count"] == 2 and {e["row"] for e in out["errors"]} == {4, 5}
    csv_path = tmp_path / "casos.csv"
    csv_path.write_text("title;prompt;expected;weight;tags\nCaso A;¿Cuánto es 10/4?;2,5;2;mates;aritmética\nSuma;otra;7;1;\n", encoding="utf-8")
    out2 = call(svc, "cases_import", suite=s["id"], path=str(csv_path))
    assert out2["added"] == 1 and out2["duplicates_skipped"] == ["Suma"]
    stored = {k["title"]: k for k in svc.store.cases(s["id"])}
    assert stored["Caso A"]["checker"] == {"type": "number", "expected": 2.5} and stored["Caso A"]["weight"] == 2.0
    assert stored["Capital de Italia"]["source"] == "import"
    with pytest.raises(GaltonError):
        call(svc, "cases_import", suite=s["id"])
    with pytest.raises(GaltonError):
        call(svc, "cases_import", suite=s["id"], path="relative.csv")


def test_case_try_runs_one_case_and_stores_nothing(svc):
    add_gguf(svc, "alfa-q4")
    case = svc.store.cases("s_razonamiento")[0]
    out = call(svc, "case_try", model="alfa-q4", case=case["id"])
    assert out["verdict"]["passed"] is True and out["runs_on"].startswith("llama.cpp") and svc.store.counts()["results"] == 0
    draft = call(svc, "case_try", model="alfa-q4", prompt="Di hola", expected="hola", settings={"temperature": 0.2})
    assert draft["case"] == "(draft)" and draft["verdict"]["passed"] is False
    with pytest.raises(GaltonError):
        call(svc, "case_try", model="alfa-q4")


# ------------------------------------------------------------------ runs
def test_run_plan_then_start_status_results(svc):
    add_gguf(svc, "alfa-q4")
    add_gguf(svc, "beta-vl", vision=True)
    plan = call(svc, "run_plan", suites=["vision", "razonamiento"], contestants=["alfa-q4", "beta-vl"])
    by = {c["name"]: c for c in plan["contestants"]}
    assert by["alfa-q4"]["cases"] == 20 and by["beta-vl"]["cases"] == 40 and plan["total_cases"] == 60
    assert by["alfa-q4"]["where"].startswith("GPU") and by["alfa-q4"]["vram_mb"] > 5000 and any("vision" in n for n in by["alfa-q4"]["notes"])
    started = call(svc, "run_start", suites=["razonamiento"], contestants=["alfa-q4"], label="prueba")
    rid = started["run"]["id"]
    status = call(svc, "run_status", run=rid)["run"]
    assert status["state"] == "done" and status["progress"]["done"] == 20 and status["contestants"][0]["runs_on"].startswith("llama.cpp (own) on GPU")
    assert call(svc, "run_status")["run"]["id"] == rid
    res = call(svc, "run_results", run=rid, limit=5)
    assert res["count"] == 5 and res["results"][0]["output"] and res["counts"]["n"] == 20
    fails = call(svc, "run_results", run=rid, only_failed=True, include_output=False)
    assert all(not r["passed"] for r in fails["results"]) and "output" not in (fails["results"] or [{}])[0]
    assert call(svc, "runs_list")["runs"][0]["id"] == rid
    assert call(svc, "runs_list", state="running")["count"] == 0
    assert call(svc, "run_cancel", run=rid)["cancelled"] is False


def test_run_discard_takes_a_run_out_of_every_statistic_and_run_restore_puts_it_back(svc, measured):
    rid = measured["run"]["id"]
    board = call(svc, "leaderboard")
    routes = call(svc, "routes_get")["tasks"]
    assert board["count"] == 3 and any(r["winner"] for r in routes)
    base = {r["name"]: (r["score"], r["n"]) for r in board["rows"]}
    cmp_before = call(svc, "compare", a="grande-q4", b="medio-q5")
    assert cmp_before["n"] > 0
    with pytest.raises(GaltonError) as unconfirmed:
        call(svc, "run_discard", run=rid, reason="the model behind the server changed")
    assert unconfirmed.value.key == "confirm_discard" and unconfirmed.value.code == "confirm_required"
    assert call(svc, "leaderboard")["count"] == 3, "nothing was discarded without confirm"
    with pytest.raises(Exception):
        call(svc, "run_discard", run=rid, reason="", confirm=True)                 # a reason is needed
    out = call(svc, "run_discard", run=rid, reason="the model behind the server changed", confirm=True)
    assert out["discarded"] is True and out["run"]["discarded"] is True and out["run"]["discard_reason"] == "the model behind the server changed" and out["results"] > 0
    assert out["note"].key == "run_discard_note" and out["run"]["discarded_ts"] == svc.clock()
    # no statistic, ranking, route, comparison or "never measured" flag sees the run any more
    assert call(svc, "leaderboard")["count"] == 0
    assert all(r["winner"] is None for r in call(svc, "routes_get")["tasks"])
    assert call(svc, "compare", a="grande-q4", b="medio-q5")["verdict"] == "no_data"
    assert {m["name"] for m in call(svc, "models_list", measured="never")["models"]} == {"grande-q4", "medio-q5", "pequeno-q8"}
    assert svc.store.counts()["results"] == 0 and svc.store.counts()["runs_discarded"] == 1
    assert svc.store.scoring_rows() == [] and svc.store.last_measured() == {} and svc.store.measurements(measured["big"]["id"]) == []
    assert svc.store.speed_rows(measured["big"]["id"]) == []
    assert call(svc, "galton_overview")["never_measured"], "the attention list asks for a new measurement"
    # it stays in the history, with its answers, flagged
    listed = [r for r in call(svc, "runs_list")["runs"] if r["id"] == rid][0]
    assert listed["discarded"] is True and listed["discard_reason"]
    assert call(svc, "run_results", run=rid, limit=3)["count"] == 3 and call(svc, "run_status", run=rid)["run"]["discarded"] is True
    # measure_new measures again what only the discarded run had measured
    assert call(svc, "measure_new")["started"] is True
    assert call(svc, "leaderboard")["count"] == 3
    # restoring brings the results back next to the new ones
    back = call(svc, "run_restore", run=rid)
    assert back["restored"] is True and back["run"]["discarded"] is False and back["run"]["discard_reason"] == "" and back["note"].key == "run_restore_note"
    after = {r["name"]: (r["score"], r["n"]) for r in call(svc, "leaderboard", suite="rapida")["rows"]}
    assert after and svc.store.counts()["runs_discarded"] == 0
    again = call(svc, "run_restore", run=rid)
    assert again["restored"] is False and again["note"].key == "run_not_discarded_note"
    assert base and cmp_before


def test_discarding_a_run_does_not_touch_the_others_and_a_second_discard_updates_the_reason(svc):
    a = add_gguf(svc, "alfa-q4", accuracy=1.0)
    first = run_inline(svc, ["razonamiento"], [a["id"]])
    second = run_inline(svc, ["razonamiento"], [a["id"]])
    call(svc, "run_discard", run=first["id"], reason="primera razón", confirm=True)
    assert call(svc, "leaderboard")["rows"][0]["n"] == 20, "the other run still counts"
    assert call(svc, "run_discard", run=first["id"], reason="otra razón", confirm=True)["run"]["discard_reason"] == "otra razón"
    assert svc.store.run(second["id"])["discarded"] is False


def test_run_resume_queues_a_continuation_of_an_interrupted_run(svc, measured):
    big = measured["big"]
    run = svc.store.update_run(measured["run"]["id"], state="failed", error="Galton was stopped while this run was in progress.")
    for row in svc.store.results(run_id=run["id"], contestant_id=big["id"], with_output=False)[3:]:
        svc.db.execute("DELETE FROM results WHERE id = ?", (row["id"],))
    out = call(svc, "run_resume", run=run["id"])
    assert out["continues"] == run["id"] and out["run"]["continues"] == run["id"] and out["run"]["state"] == "done" and out["warnings"] == []
    assert out["run"]["label"].endswith(" (continued)") and len(out["run"]["label"]) <= 80 and str(out["run"]["label"]).startswith(str(run["label"])[:50])
    assert svc.store.run(out["run"]["id"])["source"] == "assistant"
    again = svc.store.run(out["run"]["id"])
    with pytest.raises(GaltonError) as raised:
        call(svc, "run_resume", run=again["id"])
    assert raised.value.key == "run_not_resumable" and raised.value.params["state"] == "done"
    with pytest.raises(GaltonError):
        call(svc, "run_resume", run="r_nope")
    annotations = TOOLS_BY_NAME["run_resume"].annotations
    assert annotations["destructiveHint"] is False and annotations["idempotentHint"] is False and annotations["readOnlyHint"] is False


def test_a_run_that_is_still_going_cannot_be_discarded(svc):
    a = add_gguf(svc, "alfa-q4")
    run = svc.runner.create(suites=["razonamiento"], contestants=[a["id"]], source="test")
    with pytest.raises(GaltonError) as raised:
        call(svc, "run_discard", run=run["id"], reason="no vale", confirm=True)
    assert raised.value.key == "run_not_finished" and raised.value.params["state"] == "queued" and raised.value.code == "conflict"
    with pytest.raises(GaltonError):
        call(svc, "run_discard", run="r_nope", reason="no vale", confirm=True)
    assert svc.store.run(run["id"])["discarded"] is False


def test_a_discarded_run_leaves_the_arena_the_regression_check_and_the_judge_queue(svc, clock):
    a, b = add_gguf(svc, "alfa-q4", accuracy=1.0), add_gguf(svc, "beta-q4", accuracy=0.3, seed=3)
    run = run_inline(svc, ["razonamiento"], [a["id"], b["id"]])
    pair = svc.arena.next_pair()
    assert pair is not None
    svc.arena.vote(pair["pair"], "a")
    assert call(svc, "arena_ratings", category=pair["category"])["votes"] == 1
    call(svc, "run_discard", run=run["id"], reason="mal", confirm=True)
    assert svc.arena.next_pair() is None, "no pair is made from the answers of a discarded run"
    assert call(svc, "arena_ratings", category=pair["category"])["votes"] == 0, "a vote on its answers no longer counts"
    call(svc, "run_restore", run=run["id"])
    assert call(svc, "arena_ratings", category=pair["category"])["votes"] == 1
    # the regression check compares only results that count
    from galton_hoard.fakes import reference_responder
    from helpers import references
    m = add_gguf(svc, "cambia", digest="v1")
    run_inline(svc, ["rapida"], [m["id"]])
    clock.advance(3600)
    svc.store.update_contestant(m["id"], digest="v2")
    svc.fake_world.register("cambia", reference_responder(references(), accuracy=0.0))
    worse = run_inline(svc, ["rapida"], [m["id"]])
    call(svc, "run_discard", run=worse["id"], reason="measured the wrong server", confirm=True)
    assert svc.watch.detect_regressions(worse["id"]) == [] and not svc.store.notices(kind="regression")


def test_galton_run_measures_the_named_models_and_lists_the_unknown(svc):
    add_gguf(svc, "alfa-q4")
    out = call(svc, "galton_run", models=["alfa-q4", "no-existe:1b"])
    assert out["models"] == ["alfa-q4"] and out["not_found"] == ["no-existe:1b"]
    run = svc.store.run(out["run"]["id"])
    assert run["state"] == "done" and run["suites"] == ["s_rapida"] and run["label"]
    again = call(svc, "galton_run", models=["alfa-q4"], suites=["razonamiento"], label="mía")
    assert svc.store.run(again["run"]["id"])["label"] == "mía"
    with pytest.raises(GaltonError):
        call(svc, "galton_run", models=["tampoco:7b"])
    with pytest.raises(Exception):
        call(svc, "galton_run", models=[])


def test_galton_run_rediscovers_once_for_a_name_it_has_not_seen(svc):
    seen = []

    def refresh():
        seen.append(1)
        add_gguf(svc, "recien-publicado")
        return {"new": [], "changed": [], "missing": []}

    svc.discovery.refresh = refresh
    out = call(svc, "galton_run", models=["recien-publicado", "otro-desconocido"])
    assert out["models"] == ["recien-publicado"] and out["not_found"] == ["otro-desconocido"] and len(seen) == 1


def test_run_start_with_specs_for_an_unregistered_file(svc, tmp_path):
    path = tmp_path / "tuned.gguf"
    path.write_bytes(b"GGUF" + b"\0" * 64)
    svc.meta_reader = lambda p: {"architecture": "llama", "block_count": 32, "head_count_kv": 8, "head_dim": 128, "context_length": 8192, "params_b": 7.0,
                                 "quant": "Q4_K_M", "size_bytes": 4 * 1024 ** 3, "is_projector": False, "has_chat_template": True}
    svc.runner.meta_reader = svc.meta_reader
    svc.fake_world.register("tuned", lambda req: "x")
    out = call(svc, "run_start", suites=["razonamiento"], contestants=[{"kind": "gguf", "path": str(path)}])
    c = svc.store.find_contestant(out["run"]["contestants"][0]["id"])
    assert c["adhoc"] is True and c["path"] == str(path) and out["run"]["state"] == "done"


def test_run_start_rejects_bad_input(svc):
    add_gguf(svc, "alfa-q4")
    for args in ({"suites": ["no-existe"], "contestants": ["alfa-q4"]}, {"suites": ["razonamiento"], "contestants": ["no-existe"]},
                 {"suites": ["razonamiento"], "contestants": ["alfa-q4"], "settings": {"temperature": 9}},
                 {"suites": ["razonamiento"], "contestants": ["alfa-q4"], "settings": {"bogus": 1}},
                 {"suites": ["razonamiento"], "contestants": [{"kind": "gguf", "path": "relativa.gguf"}]}):
        with pytest.raises(GaltonError):
            call(svc, "run_start", **args)
    assert svc.store.counts()["runs"] == 0


def test_measure_new_only_takes_what_needs_it(svc):
    add_gguf(svc, "alfa-q4")
    first = call(svc, "measure_new")
    assert first["started"] is True and first["models"] == ["alfa-q4"] and first["run"]["state"] == "done"
    again = call(svc, "measure_new")
    assert again["started"] is False
    svc.store.update_contestant(svc.store.find_contestant("alfa-q4")["id"], digest="d-new")
    third = call(svc, "measure_new")
    assert third["started"] is True and third["models"] == ["alfa-q4"]
    call(svc, "measure_new", include_stale=False)


def test_judge_run_without_a_judge_leaves_cases_pending(svc):
    add_gguf(svc, "alfa-q4")
    s = call(svc, "suite_create", name="Juez", category="custom", cases=[{"title": "Abierta", "prompt": "Cuenta algo", "checker": "judge:Es claro"}])["suite"]
    run = run_inline(svc, [s["id"]], ["alfa-q4"])
    assert run["summary"]["pending_judge"] == 1
    out = call(svc, "judge_run")
    assert out["graded"] == 0 and out["pending"] == 1 and "judge" in out["reason"]


# ------------------------------------------------------------------ reading the numbers
def test_leaderboard_ranks_by_lower_bound_and_flags_memory(svc, measured):
    board = call(svc, "leaderboard", category="reasoning")
    names = [r["name"] for r in board["rows"]]
    assert names == ["grande-q4", "medio-q5", "pequeno-q8"] and board["rows"][0]["rank"] == 1
    top = board["rows"][0]
    assert top["ci"][0] <= top["score"] <= top["ci"][1] and top["n"] == 20 and top["decode_tps"] == pytest.approx(30.0)
    assert top["fits_16gb"] is False and board["rows"][1]["fits_16gb"] is True
    assert call(svc, "leaderboard", suite="matematicas", limit=1)["count"] == 1
    assert call(svc, "leaderboard")["rows"][0]["by_category"].keys() >= {"reasoning", "math", "instruction"}
    with pytest.raises(GaltonError):
        call(svc, "leaderboard", category="nonsense")


def test_compare_gives_a_verdict_with_warnings(svc, measured):
    out = call(svc, "compare", a="grande-q4", b="pequeno-q8", suite="razonamiento")
    assert out["verdict"] == "better" and out["n"] == 20 and out["wins"] > out["losses"] and out["p_value"] < 0.05
    assert "better than" in out["sentence"] and out["per_case"][0]["title"]
    weak = call(svc, "compare", a="grande-q4", b="medio-q5", suite="razonamiento")
    assert any("fewer than 20" in w for w in weak["warnings"]) or weak["n"] >= 20
    with pytest.raises(GaltonError):
        call(svc, "compare", a="grande-q4", b="grande-q4")
    empty = call(svc, "compare", a="grande-q4", b="pequeno-q8", suite="codigo-python")
    assert empty["verdict"] == "no_data" and empty["warnings"]


def test_recommend_explains_or_admits_it_lacks_data(svc, measured):
    r = call(svc, "recommend", task="necesito razonar problemas de lógica", category="general")
    assert r["enough_data"] is True and r["recommendation"]["name"] == "grande-q4" and r["runner_up"] and "wins general" in r["why"]
    guess = call(svc, "recommend", task="Traduce estos textos al inglés")
    assert guess["category"] == "translation" and guess["guess"]["guessed"] and guess["recommendation"] is None and guess["next"]
    few = call(svc, "suite_create", name="Poco código", category="code",
               cases=[{"title": f"c{i}", "prompt": f"pregunta {i}", "expected": "x"} for i in range(4)])["suite"]
    run_inline(svc, [few["id"]], ["grande-q4"])
    prov = call(svc, "recommend", task="escribe una función", category="code")
    assert prov["category"] == "code" and prov["enough_data"] is False and prov["recommendation"]["provisional"] is True
    assert any(x["reason"] == "too_few" for x in prov["excluded"])
    with pytest.raises(GaltonError):
        call(svc, "recommend", task="lo que sea", category="nope")


def test_routes_get_and_publish_write_the_schema(svc, measured, tmp_path):
    before = call(svc, "routes_get")
    assert before["diff"]["has_changes"] is True and before["published_updated_at"] is None
    assert {t["category"] for t in before["tasks"]} == {"general", "writing_es", "code", "extraction", "tool_use", "long_context", "rag", "vision", "summary", "translation", "math"}
    out = call(svc, "routes_publish", note="primera")
    doc = json.loads((tmp_path / "routes.json").read_text(encoding="utf-8"))
    assert doc["schema"] == 1 and doc["source"] == "galton" and "general" in doc["tasks"] and doc["capabilities"]["llm"]["prefer"]
    assert out["tasks"]["general"] == "grande-q4"
    after = call(svc, "routes_get")
    assert after["diff"]["has_changes"] is False and after["history"][0]["note"] == "primera"
    with uncapped():
        full = call(svc, "routes_get")
    assert full["published"]["tasks"].keys() == full["proposed"]["tasks"].keys() and full["detail"]["general"]["ranked"]


def test_routes_publish_refuses_when_nothing_is_measured(svc):
    with pytest.raises(GaltonError) as e:
        call(svc, "routes_publish")
    assert e.value.code == "invalid"


# ------------------------------------------------------------------ arena
def test_arena_flow(svc, measured):
    pair = call(svc, "arena_next", category="reasoning")
    assert pair["pair"] and pair["a"]["text"] != pair["b"]["text"] and "contestant" not in json.dumps(pair)
    voted = call(svc, "arena_vote", pair=pair["pair"], vote="a")
    assert voted["vote"] == "a" and voted["ratings"]
    with pytest.raises(GaltonError) as e:
        call(svc, "arena_vote", pair=pair["pair"], vote="b")
    assert e.value.code == "conflict"
    for _ in range(12):
        nxt = call(svc, "arena_next", category="reasoning")
        if not nxt["pair"]:
            break
        call(svc, "arena_vote", pair=nxt["pair"], vote="tie")
    ratings = call(svc, "arena_ratings")
    assert ratings["votes"] >= 2 and "reasoning" in ratings["by_category"]
    assert call(svc, "arena_ratings", category="reasoning")["overall"]


def test_arena_without_answers_says_so(svc):
    out = call(svc, "arena_next")
    assert out["pair"] is None and out["note"]


# ------------------------------------------------------------------ settings, GPUs, notices, housekeeping
def test_settings_get_set_and_the_reserved_gpus(svc):
    got = call(svc, "settings_get")
    assert got["values"]["gpus.allowed"] == [2, 3] and any(s["key"] == "routes.policy" for s in got["spec"])
    out = call(svc, "settings_set", values={"runner.timeout_s": 60, "watch.quiet_from": 2})
    assert out["settings"]["runner.timeout_s"] == 60
    with pytest.raises(GaltonError) as e:
        call(svc, "settings_set", values={"gpus.allowed": [1, 2]})
    assert e.value.code == "confirm_required" and svc.settings.get("gpus.allowed") == [2, 3]
    ok = call(svc, "settings_set", values={"gpus.allowed": [1, 2]}, confirm_reserved=True)
    assert ok["settings"]["gpus.allowed"] == [1, 2]
    with pytest.raises(GaltonError):
        call(svc, "settings_set", values={"nope": 1})
    with pytest.raises(GaltonError):
        call(svc, "settings_set", values={"runner.timeout_s": "abc"})


def test_settings_judge_is_stored_as_an_id(svc):
    a = add_gguf(svc, "juez-q4")
    out = call(svc, "settings_set", values={"judge.contestant": "juez-q4"})
    assert out["settings"]["judge.contestant"] == a["id"]
    with pytest.raises(GaltonError):
        call(svc, "settings_set", values={"judge.contestant": "no-existe"})


def test_gpu_status_lists_allowed_and_reserved(svc):
    out = call(svc, "gpu_status")
    by = {g["index"]: g for g in out["gpus"]}
    assert by[2]["allowed"] and by[3]["allowed"] and by[0]["reserved"] and by[1]["reserved"] and not by[0]["allowed"]
    assert out["allowed"] == [2, 3] and out["llama_server"]["found"] is True


def test_notices_and_housekeeping(svc):
    svc.store.add_notice(kind="new_model", params={"name": "Nuevo"}, severity="low")
    svc.store.add_notice(kind="regression", params={"name": "Peor", "n": 12, "diff": "0.30", "p": 0.01}, severity="high")
    out = call(svc, "notices_list", unseen=True)
    assert out["count"] == 2
    assert call(svc, "notices_list", kind="regression")["count"] == 1
    assert call(svc, "notices_list", mark_seen=True)["marked_seen"] == 2
    assert call(svc, "notices_list", unseen=True)["count"] == 0
    hk = call(svc, "housekeeping_run")
    assert {"images_removed", "judge_cache_removed"} <= set(hk)


def test_assistant_results_are_capped_but_the_ui_is_not(svc):
    add_gguf(svc, "alfa-q4")
    s = call(svc, "suite_create", name="Larga", category="custom",
             cases=[{"title": f"Caso {i} " + "x" * 80, "prompt": f"pregunta {i} " + "y" * 200, "expected": "z"} for i in range(150)])["suite"]
    capped = call(svc, "suite_get", suite=s["id"], limit=150)
    assert capped["total_cases"] == 150
    with uncapped():
        full = call(svc, "suite_get", suite=s["id"], limit=150)
    assert len(full["cases"]) == 150 and "truncated" not in full
    if "truncated" in capped:
        assert len(capped["cases"]) < 150


def test_benchmark_import_builds_a_suite_from_a_pinned_download(tmp_path, monkeypatch):
    import ifmt_data
    from helpers import build_services, mock_client_factory
    from galton_hoard import ifmtbench
    files = ifmt_data.files()
    monkeypatch.setattr(ifmtbench, "PINS", ifmt_data.pins(files))
    svc = build_services(tmp_path, offline=False, client_factory=mock_client_factory(ifmt_data.server(files)))
    out = call(svc, "benchmark_import", per_type=2, multi=3, seed=9)
    assert out["cases"] == 15 and out["suite"]["name"] == "IFMTBench" and "CC BY 4.0" in out["suite"]["description"]
    assert call(svc, "suite_get", suite=out["suite"]["id"], limit=1)["total_cases"] == 15
    with pytest.raises(GaltonError):
        call(svc, "benchmark_import", per_type=2, multi=3, seed=9)


def test_unknown_tool_and_bad_arguments(svc):
    from galton_hoard.agent_tools import call_tool
    with pytest.raises(KeyError):
        call_tool(svc, "nope", {})
    with pytest.raises(Exception):
        call_tool(svc, "run_start", {"suites": [], "contestants": ["x"]})


def test_every_tool_was_called():
    missing = sorted(set(TOOLS_BY_NAME) - CALLED)
    assert not missing, f"tools never called in this module: {missing}"
