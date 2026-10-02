"""Reading the measurements: leaderboard, stale results, memory figures, comparison and recommendation."""

from __future__ import annotations

import pytest

from galton_hoard.board import memory_gb, suite_ids_for
from galton_hoard.errors import GaltonError
from helpers import add_gguf, run_inline


def rows_by_name(board):
    return {r["name"]: r for r in board["rows"]}


def test_the_leaderboard_orders_by_the_lower_bound_and_numbers_the_ranks(svc, measured):
    board = svc.board.leaderboard()
    names = [r["name"] for r in board["rows"]]
    assert names == ["grande-q4", "medio-q5", "pequeno-q8"] and [r["rank"] for r in board["rows"]] == [1, 2, 3]
    assert all(r["lower"] <= r["score"] <= r["ci"][1] for r in board["rows"]) and "lower bound" in board["note"]


def test_the_leaderboard_reports_speed_memory_and_the_16_gb_badge(svc, measured):
    rows = rows_by_name(svc.board.leaderboard())
    assert rows["grande-q4"]["decode_tps"] == pytest.approx(30.0, rel=0.2) and rows["pequeno-q8"]["decode_tps"] > rows["grande-q4"]["decode_tps"]
    assert rows["grande-q4"]["memory_method"] == "measured" and rows["grande-q4"]["memory_gb"] > rows["pequeno-q8"]["memory_gb"]
    assert rows["pequeno-q8"]["fits_16gb"] is True


def test_models_never_measured_are_listed_apart(svc, measured):
    add_gguf(svc, "sin-medir")
    board = svc.board.leaderboard()
    assert [u["name"] for u in board["unmeasured"]] == ["sin-medir"] and "sin-medir" not in rows_by_name(board)


def test_a_category_or_a_suite_limits_the_scope(svc, measured):
    math = svc.board.leaderboard(category="math")
    assert math["scope"] == "math" and all(set(r["by_category"]) == {"math"} for r in math["rows"])
    one = svc.board.leaderboard(suite="matematicas")
    assert one["suites"] == [svc.store.find_suite("matematicas")["id"]]
    with pytest.raises(GaltonError):
        svc.board.leaderboard(category="astrologia")
    with pytest.raises(GaltonError):
        svc.board.leaderboard(suite="no-existe")


def test_stale_results_are_not_ranked_unless_asked(svc, measured):
    svc.store.update_contestant(measured["mid"]["id"], digest="nuevo")
    default = rows_by_name(svc.board.leaderboard())
    assert default["medio-q5"]["stale"] is True and default["medio-q5"]["rank"] is None and default["medio-q5"]["stale_n"] > 0
    with_stale = rows_by_name(svc.board.leaderboard(include_stale=True))
    assert with_stale["medio-q5"]["rank"] is not None


def test_disabled_models_are_hidden_unless_asked(svc, measured):
    svc.store.update_contestant(measured["small"]["id"], enabled=False)
    assert "pequeno-q8" not in rows_by_name(svc.board.leaderboard())
    assert "pequeno-q8" in rows_by_name(svc.board.leaderboard(include_disabled=True))


def test_the_days_window_drops_old_results(svc, measured, clock):
    clock.advance(10 * 86400)
    assert svc.board.leaderboard(days=5)["rows"] == []
    assert len(svc.board.leaderboard(days=30)["rows"]) == 3


def test_memory_prefers_a_measurement_then_the_reported_size_then_the_file_estimate(svc):
    measured_c = svc.store.contestant(add_gguf(svc, "a", size_gb=10.0)["id"])
    gb, how = memory_gb(svc.store, measured_c)
    assert how == "estimated" and gb > 10.0
    resident = svc.store.create_contestant(key="s1", kind="server", name="s1", meta={"size_vram": 8 * 1024 ** 3})
    assert memory_gb(svc.store, resident) == (8.0, "measured")
    nothing = svc.store.create_contestant(key="s2", kind="server", name="s2")
    assert memory_gb(svc.store, nothing) == (None, "")


def test_suite_ids_for_maps_tasks_and_categories(svc):
    ids, label = suite_ids_for(svc.store, category="general")
    assert label == "general" and len(ids) >= 3
    code_ids, _ = suite_ids_for(svc.store, category="code")
    assert code_ids == [svc.store.find_suite("codigo-python")["id"]]
    all_ids, label = suite_ids_for(svc.store)
    assert label == "all" and len(all_ids) == len(svc.store.suites())


def test_compare_gives_a_verdict_and_a_sentence(svc, measured):
    out = svc.board.compare("grande-q4", "pequeno-q8")
    assert out["verdict"] == "better" and out["a"]["name"] == "grande-q4" and out["diff"] > 0 and out["sentence"].startswith("grande-q4 is better than pequeno-q8")
    assert out["per_case"] and "title" in out["per_case"][0] and abs(out["per_case"][0]["diff"]) >= abs(out["per_case"][-1]["diff"])
    reverse = svc.board.compare("pequeno-q8", "grande-q4")
    assert reverse["verdict"] == "worse"


def test_compare_refuses_the_same_model_twice_and_unknown_names(svc, measured):
    with pytest.raises(GaltonError) as exc:
        svc.board.compare("grande-q4", "grande-q4.gguf")
    assert exc.value.code == "invalid"
    with pytest.raises(GaltonError) as exc:
        svc.board.compare("grande-q4", "fantasma")
    assert exc.value.code == "not_found"


def test_compare_without_shared_cases_says_so(svc):
    a, b = add_gguf(svc, "sola"), add_gguf(svc, "otra")
    run_inline(svc, ["matematicas"], [a["id"]])
    run_inline(svc, ["razonamiento"], [b["id"]])
    out = svc.board.compare("sola", "otra")
    assert out["verdict"] == "no_data" and any("checked results" in w for w in out["warnings"])


def test_compare_warns_when_a_model_changed(svc, measured):
    svc.store.update_contestant(measured["small"]["id"], digest="nuevo")
    out = svc.board.compare("grande-q4", "pequeno-q8")
    assert any("older version" in w for w in out["warnings"]) and out["verdict"] == "no_data"
    assert svc.board.compare("grande-q4", "pequeno-q8", include_stale=True)["verdict"] == "better"


def test_recommend_names_the_winner_the_runner_up_and_why(svc, measured):
    out = svc.board.recommend("tengo que razonar con lógica")
    assert out["enough_data"] is True and out["recommendation"]["name"] == "grande-q4" and out["runner_up"]["name"] == "medio-q5" and "wins" in out["why"]
    forced = svc.board.recommend("lo que sea", category="math")
    assert forced["category"] == "math" and forced["guess"]["guessed"] is False


def test_recommend_is_provisional_when_the_policy_wants_more_evidence(svc, measured):
    svc.settings.set_many({"routes.policy": {"min_cases": 5000}})
    out = svc.board.recommend("razonar", category="general")
    assert out["enough_data"] is False and out["recommendation"]["provisional"] is True and "Provisional" in out["why"]


def test_recommend_admits_when_nothing_was_measured(svc):
    add_gguf(svc, "virgen")
    out = svc.board.recommend("escribe código en python")
    assert out["category"] == "code" and out["recommendation"] is None and out["available_models"] == ["virgen"] and out["next"]


def test_recommend_rejects_unknown_categories(svc):
    with pytest.raises(GaltonError):
        svc.board.recommend("x", category="astrologia")
