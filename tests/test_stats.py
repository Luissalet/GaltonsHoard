"""Statistics with values worked out by hand."""

from __future__ import annotations

import math

import pytest

from galton_hoard import stats


def rows(scores, case_prefix="c", weight=1.0, passed=None):
    return [{"case_id": f"{case_prefix}{i}", "score": s, "passed": (s >= 0.5) if passed is None else passed[i], "weight": weight, "confidence": 1.0}
            for i, s in enumerate(scores)]


def test_wilson_known_value():
    lo, hi = stats.wilson(8, 10)
    assert lo == pytest.approx(0.4902, abs=1e-3) and hi == pytest.approx(0.9433, abs=1e-3)


def test_wilson_extremes_stay_inside_the_unit_interval():
    assert stats.wilson(0, 10)[0] == pytest.approx(0.0, abs=1e-12)
    assert stats.wilson(10, 10)[1] == pytest.approx(1.0)
    assert stats.wilson(10, 10)[0] > 0.6


def test_wilson_with_nothing_measured_is_all_doubt():
    assert stats.wilson(0, 0) == (0.0, 1.0)


def test_wilson_narrows_with_more_data():
    small, large = stats.wilson(8, 10), stats.wilson(80, 100)
    assert (large[1] - large[0]) < (small[1] - small[0])


def test_weighted_mean():
    assert stats.weighted_mean([1, 0], [3, 1]) == 0.75
    assert stats.weighted_mean([], []) == 0.0


def test_bootstrap_is_seeded_and_cached():
    a = stats.bootstrap_ci([1, 0, 1, 1, 0, 1, 1, 1], seed=7)
    b = stats.bootstrap_ci([1, 0, 1, 1, 0, 1, 1, 1], seed=7)
    assert a == b and a[0] == 0.75 and a[1] < a[0] < a[2]


def test_bootstrap_of_one_value_is_all_doubt():
    assert stats.bootstrap_ci([1.0]) == (1.0, 0.0, 1.0)


def test_bootstrap_of_nothing():
    assert stats.bootstrap_ci([]) == (0.0, 0.0, 1.0)


def test_bootstrap_constant_values_have_no_spread():
    mean, lo, hi = stats.bootstrap_ci([0.5] * 20)
    assert mean == lo == hi == 0.5


def test_per_case_averages_repeats_and_scales_weights():
    out = stats.per_case([{"case_id": "a", "score": 1.0, "passed": True, "weight": 2.0, "confidence": 1.0},
                          {"case_id": "a", "score": 0.0, "passed": False, "weight": 2.0, "confidence": 0.5}])
    assert out["a"]["score"] == 0.5 and out["a"]["passed"] == 0.5 and out["a"]["weight"] == pytest.approx(1.5) and out["a"]["n"] == 2


def test_summarize_counts_cases_not_repeats():
    summary = stats.summarize(rows([1, 1, 0, 1]) + rows([1, 1, 0, 1]))
    assert summary["n"] == 4 and summary["n_results"] == 8 and summary["score"] == 0.75 and summary["pass_rate"] == 0.75
    assert summary["pass_ci"][0] < 0.75 < summary["pass_ci"][1]


def test_summarize_nothing():
    assert stats.summarize([])["score"] is None


def test_summarize_respects_case_weights():
    heavy = [{"case_id": "a", "score": 1.0, "passed": True, "weight": 9.0}, {"case_id": "b", "score": 0.0, "passed": False, "weight": 1.0}]
    assert stats.summarize(heavy)["score"] == 0.9


@pytest.mark.parametrize("k,n,p", [(0, 0, 1.0), (5, 10, 1.0), (0, 10, 0.002), (1, 10, 0.0215), (2, 10, 0.1094)])
def test_binomial_two_sided(k, n, p):
    assert stats.binomial_two_sided(k, n) == pytest.approx(p, abs=1e-3)


def test_compare_clear_win():
    a = rows([1] * 18 + [0, 0])
    b = rows([0] * 12 + [1] * 8)
    out = stats.compare(a, b)
    assert out["verdict"] == "better" and out["wins"] == 12 and out["losses"] == 2 and out["ties"] == 6 and out["p_value"] < 0.05


def test_compare_is_antisymmetric():
    a, b = rows([1] * 18 + [0, 0]), rows([0] * 12 + [1] * 8)
    assert stats.compare(b, a)["verdict"] == "worse"


def test_compare_identical_models_have_no_difference():
    out = stats.compare(rows([1, 0] * 10), rows([1, 0] * 10))
    assert out["verdict"] == "no_clear_difference" and out["wins"] == out["losses"] == 0 and out["p_value"] == 1.0


def test_compare_small_samples_warn_and_never_claim():
    out = stats.compare(rows([1, 1, 1]), rows([0, 0, 0]))
    assert out["verdict"] == "no_clear_difference" and any("fewer than" in w for w in out["warnings"])


def test_compare_without_shared_cases():
    out = stats.compare(rows([1] * 5, "x"), rows([0] * 5, "y"))
    assert out["verdict"] == "no_data" and out["only_a"] == 5 and out["only_b"] == 5 and out["n"] == 0


def test_compare_only_uses_the_shared_cases():
    out = stats.compare(rows([1] * 10), rows([0] * 10) + rows([1] * 4, "z"))
    assert out["n"] == 10 and out["only_b"] == 4


def test_speed_summary():
    data = [{"decode_tps": v, "prompt_tps": v * 10, "ttft_ms": 100 + v} for v in (10, 20, 30, 40, 50)]
    s = stats.speed_summary(data)
    assert s["decode_tps_median"] == 30.0 and s["decode_tps_p90"] == 46.0 and s["prompt_tps_median"] == 300.0 and s["ttft_ms_median"] == 130.0


def test_speed_summary_without_data():
    assert stats.speed_summary([])["decode_tps_median"] is None


def test_bradley_terry_orders_by_wins():
    votes = [("a", "b", "a")] * 8 + [("a", "b", "b")] * 2
    r = stats.bradley_terry(votes)
    assert r["a"] > r["b"]
    assert sum(r.values()) / 2 == pytest.approx(1000, abs=1.0)


def test_bradley_terry_ties_are_even_and_both_bad_is_ignored():
    r = stats.bradley_terry([("a", "b", "tie")] * 6 + [("a", "b", "both_bad")] * 20)
    assert r["a"] == pytest.approx(r["b"], abs=1.0)
    assert stats.bradley_terry([("a", "b", "both_bad")]) == {}


def test_bradley_terry_transitive_chain():
    votes = [("a", "b", "a")] * 6 + [("b", "c", "a")] * 6 + [("a", "c", "a")] * 6
    r = stats.bradley_terry(votes)
    assert r["a"] > r["b"] > r["c"]


def test_arena_ratings_hide_models_with_few_votes():
    votes = [("a", "b", "a")] * 6 + [("a", "c", "a")]
    out = {r["contestant"]: r for r in stats.arena_ratings(votes, min_votes=5)}
    assert out["a"]["shown"] and out["a"]["rating"] is not None and out["a"]["ci"][0] <= out["a"]["rating"] <= out["a"]["ci"][1]
    assert not out["c"]["shown"] and out["c"]["rating"] is None


def test_arena_ratings_are_sorted_best_first():
    votes = [("a", "b", "a")] * 10
    out = stats.arena_ratings(votes, min_votes=5)
    assert [r["contestant"] for r in out] == ["a", "b"]


def test_arena_ratings_no_votes():
    assert stats.arena_ratings([]) == []
    assert not math.isnan(sum(r["rating"] or 0 for r in stats.arena_ratings([("a", "b", "tie")] * 5)))
