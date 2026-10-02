"""Typed settings (coercion, bounds, reserved GPUs, masked secrets) and the routing table (policy, exclusions, schema, diff, atomic publish)."""

from __future__ import annotations

import json

import pytest

from galton_hoard.errors import GaltonError
from galton_hoard.routes import SCHEMA, TASKS, Routes, names_for
from galton_hoard.settings import ROUTES_POLICY_DEFAULT, SPEC_BY_KEY, SPECS
from helpers import add_gguf, run_inline


# ---- settings --------------------------------------------------------------------------------------------------------------------------------

def test_every_setting_has_a_default_that_passes_its_own_validation(svc):
    for spec in SPECS:
        value = svc.settings.get(spec.key)
        if spec.kind == "secret":
            continue
        assert svc.settings.validate({spec.key: value})[spec.key] == value, spec.key


def test_defaults_reserve_gpus_0_and_1_and_allow_2_and_3(svc):
    assert svc.settings.get("gpus.reserved") == [0, 1] and svc.settings.get("gpus.allowed") == [2, 3]


@pytest.mark.parametrize("value,expected", [("true", True), ("Sí", True), ("0", False), ("off", False), (True, True), (0, False)])
def test_bool_coercion(svc, value, expected):
    svc.settings.set_many({"scheduler.paused": value})
    assert svc.settings.get("scheduler.paused") is expected


def test_bad_bool_text_is_rejected(svc):
    with pytest.raises(GaltonError):
        svc.settings.set_many({"scheduler.paused": "quizá"})


@pytest.mark.parametrize("key,value", [("runner.timeout_s", 1), ("runner.timeout_s", 99999), ("runner.timeout_s", "mucho"), ("watch.quiet_from", 25),
                                       ("ui.language", "fr"), ("gpus.allowed", "a,b")])
def test_out_of_range_or_wrong_type_values_are_rejected(svc, key, value):
    with pytest.raises(GaltonError) as exc:
        svc.settings.set_many({key: value})
    assert exc.value.code == "invalid"
    assert svc.settings.get(key) == SPEC_BY_KEY[key].default_value()


def test_unknown_setting_is_rejected_and_lists_the_known_ones(svc):
    with pytest.raises(GaltonError) as exc:
        svc.settings.set_many({"nope.nothing": 1})
    assert "gpus.allowed" in exc.value.hint


def test_int_list_accepts_text_and_sorts_and_deduplicates(svc):
    svc.settings.set_many({"gpus.allowed": "3; 2, 3"})
    assert svc.settings.get("gpus.allowed") == [2, 3]


def test_str_list_splits_text_on_lines(svc):
    svc.settings.set_many({"gguf.folders": "D:\\a\n\n D:\\b "})
    assert svc.settings.get("gguf.folders") == ["D:\\a", "D:\\b"]


def test_reserved_gpus_need_an_explicit_confirmation(svc):
    with pytest.raises(GaltonError) as exc:
        svc.settings.set_many({"gpus.allowed": [0, 2]})
    assert exc.value.code == "confirm_required" and exc.value.details["reserved"] == [0]
    assert svc.settings.get("gpus.allowed") == [2, 3]
    svc.settings.set_many({"gpus.allowed": [0, 2]}, confirm_reserved=True)
    assert svc.settings.get("gpus.allowed") == [0, 2]


def test_changing_the_reserved_list_in_the_same_call_is_taken_into_account(svc):
    svc.settings.set_many({"gpus.reserved": [1], "gpus.allowed": [0, 2]})
    assert svc.settings.get("gpus.allowed") == [0, 2]


def test_secrets_are_masked_in_listings_but_stored(svc):
    svc.settings.set_many({"faustus.token": "abcdefgh12345678"})
    shown = svc.settings.all()["faustus.token"]
    assert shown["configured"] is True and "abcdefgh" not in json.dumps(shown) and shown["value"].endswith("5678")
    assert svc.settings.all(mask_secrets=False)["faustus.token"] == "abcdefgh12345678"
    svc.settings.set_many({"faustus.token": {"configured": True, "value": "…5678"}})   # the UI sends the masked value back: clears nothing it cannot know
    assert svc.settings.get("faustus.token") == ""


def test_short_secrets_are_fully_masked(svc):
    svc.settings.set_many({"faustus.token": "abc"})
    assert svc.settings.all()["faustus.token"]["value"] == "****"


def test_policy_merges_known_fields_and_rejects_unknown_ones(svc):
    svc.settings.set_many({"routes.policy": {"days": 7}})
    policy = svc.settings.get("routes.policy")
    assert policy["days"] == 7 and policy["min_cases"] == ROUTES_POLICY_DEFAULT["min_cases"]
    with pytest.raises(GaltonError):
        svc.settings.set_many({"routes.policy": {"colour": 1}})
    with pytest.raises(GaltonError):
        svc.settings.set_many({"routes.policy": {"top_k": 99}})
    with pytest.raises(GaltonError):
        svc.settings.set_many({"routes.policy": "x"})


def test_reset_goes_back_to_the_default(svc):
    svc.settings.set_many({"runner.timeout_s": 300})
    svc.settings.reset("runner.timeout_s")
    assert svc.settings.get("runner.timeout_s") == 120


def test_defaults_are_not_shared_between_reads(svc):
    svc.settings.get("gpus.allowed").append(9)
    svc.settings.get("routes.policy")["days"] = 1
    assert svc.settings.get("gpus.allowed") == [2, 3] and svc.settings.get("routes.policy")["days"] == 30


def test_describe_lists_every_setting_with_its_kind(svc):
    described = {d["key"]: d for d in svc.settings.describe()}
    assert set(described) == set(SPEC_BY_KEY) and described["ui.language"]["choices"] == ["es", "en"]


# ---- routes ----------------------------------------------------------------------------------------------------------------------------------

def test_names_for_lists_each_name_once_and_skips_server_labels():
    c = {"ollama_ref": "qwen3:8b", "model": "qwen3:8b", "aliases": ["Qwen3", "qwen3:8B", "x (llama.cpp)"], "name": "Qwen3"}
    assert names_for(c) == ["qwen3:8b", "Qwen3"]


def test_nothing_is_published_without_enough_checked_cases(svc):
    add_gguf(svc, "solo")
    with pytest.raises(GaltonError) as exc:
        svc.routes.publish()
    assert exc.value.code == "invalid" and not svc.config.routes_path().exists()


def test_the_published_schema_is_exact(svc, measured):
    out = svc.routes.publish(note="test")
    doc = json.loads(svc.config.routes_path().read_text(encoding="utf-8"))
    assert doc == out["doc"]
    assert set(doc) == {"schema", "source", "updated_at", "tasks", "capabilities"} and doc["schema"] == SCHEMA == 1 and doc["source"] == "galton"
    assert "general" in doc["tasks"] and doc["capabilities"]["llm"]["prefer"] == doc["tasks"]["general"]["prefer"]
    for name, task in doc["tasks"].items():
        assert name in TASKS and set(task) == {"capability", "prefer", "explain"} and task["explain"].endswith(".")
        for item in task["prefer"]:
            assert set(item) == {"names", "score", "ci", "n", "tok_s", "vram_gb"} and item["names"] and item["ci"][0] <= item["score"] <= item["ci"][1]


def test_ranking_uses_the_lower_bound_so_the_better_model_leads(svc, measured):
    prefer = svc.routes.publish()["doc"]["tasks"]["general"]["prefer"]
    assert prefer[0]["names"][0] == "grande-q4"
    assert [p["score"] for p in prefer] == sorted((p["score"] for p in prefer), reverse=True)


def test_top_k_limits_the_table(svc, measured):
    svc.settings.set_many({"routes.policy": {"top_k": 1}})
    assert len(svc.routes.publish()["doc"]["tasks"]["general"]["prefer"]) == 1


def test_a_category_below_min_cases_is_not_published_and_the_reason_is_given(svc, measured):
    svc.settings.set_many({"routes.policy": {"min_cases": 5000}})
    built = svc.routes.build()
    assert built["doc"]["tasks"] == {}
    assert {e["reason"] for e in built["detail"]["general"]["excluded"]} == {"too_few"}


def test_disabled_remote_adhoc_and_stale_models_are_excluded_with_a_reason(svc, measured):
    store = svc.store
    store.update_contestant(measured["mid"]["id"], enabled=False)
    store.update_contestant(measured["small"]["id"], digest="another-digest")
    reasons = {e["name"]: e["reason"] for e in svc.routes.build()["detail"]["general"]["excluded"]}
    assert reasons == {"medio-q5": "disabled", "pequeno-q8": "stale"}
    names = [x["name"] for x in svc.routes.build()["detail"]["general"]["ranked"]]
    assert names == ["grande-q4"]


def test_slow_and_big_models_are_excluded_by_policy(svc, measured):
    svc.settings.set_many({"routes.policy": {"min_tok_s": 100}})
    detail = svc.routes.build()["detail"]["general"]
    assert {e["name"]: e["reason"] for e in detail["excluded"]} == {"grande-q4": "too_slow", "medio-q5": "too_slow"}
    svc.settings.reset("routes.policy")
    svc.settings.set_many({"routes.policy": {"max_vram_gb": 5}})
    detail = svc.routes.build()["detail"]["general"]
    assert "grande-q4" in {e["name"] for e in detail["excluded"] if e["reason"] == "too_big"}


def test_old_results_fall_out_of_the_window(svc, measured, clock):
    clock.advance(40 * 86400)
    assert svc.routes.build()["doc"]["tasks"] == {}
    svc.settings.set_many({"routes.policy": {"days": 90}})
    assert "general" in svc.routes.build()["doc"]["tasks"]


def test_speed_weight_can_reorder_the_table(svc, measured):
    svc.settings.set_many({"routes.policy": {"weight_quality": 0.1, "weight_speed": 5.0}})
    first = svc.routes.build()["doc"]["tasks"]["general"]["prefer"][0]["names"][0]
    assert first == "pequeno-q8"


def test_diff_reports_added_removed_changed_and_unchanged():
    old = {"tasks": {"a": {"prefer": [{"names": ["x"], "score": 0.5}]}, "b": {"prefer": [{"names": ["y"]}]}, "c": {"prefer": [{"names": ["z"], "score": 1}]},
                     "d": {"prefer": [{"names": ["w"], "score": 1}]}}}
    new = {"tasks": {"a": {"prefer": [{"names": ["q"], "score": 0.6}]}, "c": {"prefer": [{"names": ["z"], "score": 0.9}]}, "d": {"prefer": [{"names": ["w"], "score": 1}]},
                     "e": {"prefer": [{"names": ["n"]}]}}}
    d = Routes.diff(old, new)
    assert d["added"] == ["e"] and d["removed"] == ["b"] and d["unchanged"] == 1 and d["has_changes"]
    assert {"task": "a", "from": "x", "to": "q"} in d["changed"]
    assert {"task": "c", "from": "z", "to": "z", "numbers_only": True} in d["changed"]
    assert Routes.diff(None, {"tasks": {}})["has_changes"] is False


def test_publishing_twice_reports_no_changes_the_second_time(svc, measured):
    first = svc.routes.publish()
    assert first["diff"]["added"] and first["diff"]["has_changes"]
    second = svc.routes.publish()
    assert second["diff"]["has_changes"] is False


def test_publish_emits_the_update_event_and_keeps_history(svc, measured):
    events = []
    svc.routes.emit = lambda t, d: events.append((t, d))
    svc.routes.publish(note="primera")
    assert events and events[0][0] == "galton.routes.updated" and "general" in events[0][1]["tasks"]
    assert svc.routes.get()["history"][0]["note"] == "primera"


def test_publish_is_atomic_and_leaves_no_temporary_files(svc, measured):
    path = svc.config.routes_path()
    path.write_text("{}", encoding="utf-8")
    svc.routes.publish()
    assert [p.name for p in path.parent.iterdir() if p.name.startswith(".routes-")] == []
    assert json.loads(path.read_text(encoding="utf-8"))["schema"] == 1


def test_a_failed_write_keeps_the_previous_file(svc, measured, monkeypatch):
    svc.routes.publish()
    path = svc.config.routes_path()
    before = path.read_text(encoding="utf-8")
    import galton_hoard.routes as routes_mod
    monkeypatch.setattr(routes_mod.os, "replace", lambda *a, **k: (_ for _ in ()).throw(OSError("disk full")))
    svc.store.update_contestant(measured["mid"]["id"], enabled=False)
    with pytest.raises(OSError):
        svc.routes.publish()
    assert path.read_text(encoding="utf-8") == before
    assert [p.name for p in path.parent.iterdir() if p.name.startswith(".routes-")] == []


def test_a_corrupt_published_file_reads_as_nothing_published(svc):
    path = svc.config.routes_path()
    path.write_text("not json", encoding="utf-8")
    assert svc.routes.published() is None
    path.write_text("[1]", encoding="utf-8")
    assert svc.routes.published() is None


def test_remote_models_are_routed_only_when_the_policy_and_the_check_allow(svc):
    m = add_gguf(svc, "remoto")
    svc.store.update_contestant(m["id"], remote=True, remote_ok=True)
    run_inline(svc, ["razonamiento", "matematicas", "instrucciones"], [m["id"]])
    assert svc.routes.build()["doc"]["tasks"] == {}
    assert svc.routes.build()["detail"]["general"]["excluded"][0]["reason"] == "remote"
    svc.settings.set_many({"routes.policy": {"include_remote": True}})
    assert "general" in svc.routes.build()["doc"]["tasks"]
