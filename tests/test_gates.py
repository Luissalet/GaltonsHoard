"""Gate correlation: synthetic outcome matrices (independent, identical, anti-correlated, small, with abstentions), extraction from stored
results, the tool and the HTTP route. No model is called: every result is written by hand."""

from __future__ import annotations

import random

from conftest import tool
from helpers import add_gguf
from galton_hoard import gates
from galton_hoard.errors import GaltonError

import pytest

P, F, X = gates.PASS, gates.FAIL, gates.ABSTAIN


def matrix(rows, names=("a", "b")):
    """rows: list of tuples of outcomes, one per answer, in the order of ``names``."""
    units = {i: dict(zip(names, row)) for i, row in enumerate(rows)}
    return {"gates": sorted(names), "units": units, "answers": len(rows)}


def cells(n11, n10, n01, n00):
    return [(F, F)] * n11 + [(F, P)] * n10 + [(P, F)] * n01 + [(P, P)] * n00


def report(rows, **kw):
    return gates.analyze(matrix(rows), resamples=300, **kw)


def only_pair(rep):
    assert len(rep["pairs"]) == 1
    return rep["pairs"][0]


# ------------------------------------------------------------------ the numbers
def test_independent_gates_have_phi_near_zero_lift_one_and_no_redundancy():
    rep = report(cells(50, 50, 50, 50))
    p = only_pair(rep)
    assert p["n"] == 200 and p["reading"] == "independent"
    assert abs(p["phi"]) < 1e-9 and abs(p["kappa"]) < 1e-9 and p["lift"] == 1.0
    assert p["phi_ci"][0] < 0 < p["phi_ci"][1]
    assert p["p_a_given_b"] == 0.5 and p["p_b_given_a"] == 0.5
    assert p["joint_fail_rate"] == p["expected_joint"] == 0.25
    assert rep["recommendations"] == []
    assert "independiente" in next(f for f in rep["findings"] if f["kind"] == "independent")["es"]


def test_identical_gates_never_disagree_and_one_is_dropped():
    rng = random.Random(4)
    rows = [(F, F) if rng.random() < 0.3 else (P, P) for _ in range(120)]
    rep = report(rows)
    p = only_pair(rep)
    assert p["phi"] == 1.0 and p["kappa"] == 1.0 and p["p_a_given_b"] == 1.0 and p["p_b_given_a"] == 1.0
    assert p["reading"] == "never_disagree"
    f = next(f for f in rep["findings"] if f["kind"] == "never_disagree")
    assert f["recommend_drop"] == "b" and f["keep"] == "a", "a tie is broken alphabetically"
    assert rep["recommendations"] == [{"gate": "b", "keep": "a", "kind": "never_disagree", "phi": 1.0}]
    assert "sobra" in rep["summary"]["es"]


def test_nearly_identical_gates_are_redundant_with_a_tight_interval():
    rep = report(cells(30, 1, 0, 89))
    p = only_pair(rep)
    assert p["phi"] > 0.95 and p["phi_ci"][0] >= 0.5 and p["reading"] == "redundant"
    assert p["lift"] > 3
    # b catches no failure that a misses, while a catches one that b misses: b is the redundant one
    drop = rep["recommendations"][0]
    assert drop["gate"] == "b" and drop["keep"] == "a" and drop["kind"] == "redundant"


def test_anticorrelated_gates_are_complementary_and_never_redundant():
    rep = report(cells(0, 40, 40, 20))
    p = only_pair(rep)
    assert p["phi"] < -0.5 and p["reading"] == "complementary"
    assert p["lift"] == 0.0 and p["p_a_given_b"] == 0.0
    assert rep["recommendations"] == []
    assert any(f["kind"] == "complementary" for f in rep["findings"])


def test_moderate_correlation_on_a_small_sample_is_only_probably_redundant():
    rep = report(cells(5, 1, 1, 23))          # phi about 0.72 but n = 30: the interval reaches below 0.5
    p = only_pair(rep)
    assert p["phi"] >= 0.7 and p["phi_ci"][0] < 0.5
    assert p["reading"] == "probably_redundant"
    assert rep["recommendations"] == [], "an uncertain pair is mentioned, not recommended"
    f = next(f for f in rep["findings"] if f["kind"] == "probably_redundant")
    assert f["recommend_drop"] is None and f["candidate"] in ("a", "b")


def test_a_small_sample_draws_no_conclusion():
    rep = report(cells(3, 0, 0, 5))           # 8 shared answers, identical
    p = only_pair(rep)
    assert p["n"] == 8 and p["reading"] == "insufficient" and p["phi"] == 1.0
    assert rep["recommendations"] == []
    kinds = {f["kind"] for f in rep["findings"]}
    assert "pair_insufficient" in kinds and "few_samples" in kinds
    assert "faltan datos" in rep["summary"]["es"]
    # the guard is a parameter
    lower = report(cells(3, 0, 0, 5), min_samples=8)
    assert only_pair(lower)["reading"] == "never_disagree"


def test_pairs_that_never_share_an_answer_make_one_finding_not_one_each():
    """Gates of different cases never meet; listing every such pair buried the real findings."""
    units = {}
    for i in range(30):                                  # case 1: gates a and b
        units[i] = {"a": F if i % 3 == 0 else P, "b": F if i % 2 == 0 else P}
    for i in range(30, 60):                              # case 2: gates c, d, e
        units[i] = {"c": F if i % 4 == 0 else P, "d": F if i % 5 == 0 else P, "e": P if i % 3 else F}
    rep = gates.analyze({"gates": ["a", "b", "c", "d", "e"], "units": units, "answers": 60}, resamples=100)
    zero = [p for p in rep["pairs"] if p["n"] == 0]
    assert len(zero) == 6                                # a,b x c,d,e
    kinds = [f["kind"] for f in rep["findings"]]
    assert kinds.count("pairs_disjoint") == 1
    assert "pair_insufficient" not in kinds, "no per-pair line for pairs with nothing in common"
    f = next(f for f in rep["findings"] if f["kind"] == "pairs_disjoint")
    assert f["es"].startswith("6 pares") and f["en"].startswith("6 gate pairs")
    assert sorted(map(tuple, f["pairs"])) == sorted((p["a"], p["b"]) for p in zero)
    assert f["gates"] == ["a", "b", "c", "d", "e"]


def test_the_disjoint_finding_names_a_bounded_list(monkeypatch):
    monkeypatch.setattr(gates, "DISJOINT_LISTED", 2)
    units = {i: {"a": P if i % 2 else F} for i in range(10)}
    units.update({i: {"b": F if i % 2 else P, "c": P, "d": F} for i in range(10, 20)})
    rep = gates.analyze({"gates": ["a", "b", "c", "d"], "units": units, "answers": 20}, resamples=50)
    f = next(f for f in rep["findings"] if f["kind"] == "pairs_disjoint")
    assert len(f["pairs"]) == 3, "every pair stays in the data"
    assert f["es"].startswith("3 pares") and f["es"].endswith("y 1 más.")
    assert f["en"].endswith("and 1 more.")
    assert f["es"].count("·") == 2, "only DISJOINT_LISTED pairs are named"
    # a pair with some (but too few) shared answers keeps its own line
    small = report(cells(3, 0, 0, 5))
    assert "pair_insufficient" in {x["kind"] for x in small["findings"]}
    assert "pairs_disjoint" not in {x["kind"] for x in small["findings"]}


def test_abstentions_leave_a_pair_but_not_the_other_gates():
    rows = [(F, F)] * 20 + [(P, P)] * 40 + [(F, X)] * 10 + [(X, P)] * 30    # b abstains 10 times, a abstains 30
    names = ("a", "b")
    rep = gates.analyze(matrix(rows, names), resamples=200)
    p = only_pair(rep)
    assert p["n"] == 60, "only answers where both gates gave a verdict"
    single = {g["gate"]: g for g in rep["gates"]}
    assert single["a"]["n"] == 70 and single["a"]["abstained"] == 30 and single["a"]["abstain_rate"] == 0.3
    assert single["b"]["n"] == 90 and single["b"]["abstained"] == 10
    assert any(f["kind"] == "often_abstains" and f["gates"] == ["a"] for f in rep["findings"])
    assert not any(f["kind"] == "often_abstains" and f["gates"] == ["b"] for f in rep["findings"])


def test_a_gate_that_never_fails_has_no_phi_and_is_flagged():
    rep = report([(P, P)] * 30 + [(P, F)] * 10)
    p = only_pair(rep)
    assert p["phi"] is None and p["lift"] is None and p["phi_ci"] is None and p["reading"] == "mixed"
    assert p["p_a_given_b"] == 0.0 and p["p_b_given_a"] is None
    f = next(f for f in rep["findings"] if f["kind"] == "never_fails")
    assert f["gates"] == ["a"]
    both = report([(P, P)] * 40)
    assert only_pair(both)["reading"] == "never_fail" and not rep["recommendations"]


def test_a_gate_that_always_fails_is_flagged():
    rep = report([(F, P)] * 20 + [(F, F)] * 20)
    assert any(f["kind"] == "always_fails" and f["gates"] == ["a"] for f in rep["findings"])


def test_ties_between_redundant_gates_drop_the_costlier_one():
    rows = [(F, F)] * 20 + [(P, P)] * 60
    rep = gates.analyze(matrix(rows, ("judge", "regex")), resamples=100)
    assert rep["recommendations"][0]["gate"] == "judge" and rep["recommendations"][0]["keep"] == "regex"


def test_the_gate_with_fewer_unique_catches_is_dropped_even_when_it_is_cheap():
    rows = [(F, F)] * 15 + [(F, P)] * 1 + [(P, F)] * 6 + [(P, P)] * 60       # (judge, regex): the judge catches 1 alone, the regex 6
    rep = gates.analyze(matrix(rows, ("judge", "regex")), resamples=100, strong_phi=0.5)
    assert [d["gate"] for d in rep["recommendations"]] == ["judge"], rep["pairs"]


def test_the_bootstrap_is_deterministic_and_the_interval_contains_the_estimate():
    one, two = report(cells(20, 8, 6, 66)), report(cells(20, 8, 6, 66))
    assert one == two
    p = only_pair(one)
    for key in ("phi", "kappa", "p_a_given_b", "p_b_given_a"):
        lo, hi = p[f"{key}_ci"]
        assert lo <= p[key] <= hi, key


def test_three_gates_give_three_pairs_and_a_gate_that_agrees_with_all_is_named():
    rows = [(F, F, F)] * 15 + [(P, P, P)] * 50
    rep = gates.analyze(matrix(rows, ("a", "b", "c")), resamples=100)
    assert len(rep["pairs"]) == 3
    assert {f["gates"][0] for f in rep["findings"] if f["kind"] == "never_disagrees_with_any"} == {"a", "b", "c"}
    assert len(rep["recommendations"]) >= 1


def test_the_report_without_data_or_with_one_gate_says_so():
    assert "No hay respuestas" in gates.analyze({"gates": [], "units": {}, "answers": 0})["summary"]["es"]
    single = gates.analyze({"gates": ["a"], "units": {i: {"a": P} for i in range(30)}, "answers": 30})
    assert single["pairs"] == [] and "una comprobación" in single["summary"]["es"]


# ------------------------------------------------------------------ extraction
def verdict(passed, *, unavailable=False, checks=None):
    return {"passed": passed, "unavailable": unavailable, "detail": {"checks": checks} if checks is not None else {}}


def test_leaves_open_combinators_and_keep_the_type_of_each_gate():
    spec = {"type": "all", "checks": [{"type": "regex", "pattern": "x"}, {"type": "any", "checks": [{"type": "contains"}, {"type": "judge"}]}]}
    v = verdict(False, checks=[verdict(True), verdict(False, checks=[verdict(False), verdict(False, unavailable=True)])])
    assert gates.leaves(spec, v) == [("regex", P), ("contains", F), ("judge", X)]


def test_leaves_of_a_plain_checker_and_of_malformed_detail():
    assert gates.leaves({"type": "exact"}, verdict(True)) == [("exact", P)]
    assert gates.leaves({"type": "none"}, verdict(True)) == []
    assert gates.leaves({}, verdict(True)) == []
    mismatched = {"type": "all", "checks": [{"type": "regex"}, {"type": "contains"}]}
    assert gates.leaves(mismatched, verdict(False, checks=[verdict(True)])) == [("all", F)], "detail that does not line up is one gate"
    assert gates.leaves({"type": "judge"}, {"passed": False, "judge_pending": True, "detail": {}}) == [("judge", X)]


def test_repeats_collapse_to_a_majority_and_the_same_gate_twice_fails_if_either_fails():
    spec = {"type": "all", "checks": [{"type": "regex"}, {"type": "regex"}, {"type": "judge"}]}

    def row(c, repeat, a, b, judge_unavailable=False):
        return {"contestant_id": "m", "case_id": c, "digest": "d", "passed": False, "unavailable": False, "judge_pending": False,
                "detail": {"checks": [verdict(a), verdict(b), verdict(True, unavailable=judge_unavailable)]}, "repeat": repeat}
    rows = [row("c1", 0, True, False), row("c1", 1, True, True), row("c1", 2, True, True), row("c2", 0, True, True, True)]
    m = gates.build_matrix(rows, {"c1": spec, "c2": spec})
    assert m["answers"] == 4 and m["gates"] == ["judge", "regex"]
    by_case = {k[2]: v for k, v in m["units"].items()}
    # c1 repeat 0: the second regex fails, so the gate fails (either one failing counts); repeats 1 and 2 pass: 1 of 3 failed -> pass
    assert by_case["c1"] == {"regex": P, "judge": P}
    assert by_case["c2"] == {"regex": P, "judge": X}


# ------------------------------------------------------------------ stored results, the tool and the route
def seed(svc, model, rows, *, spec=None, run=None, digest="", suite=None):
    """One case per row with a two-check ``all`` checker (regex, contains); each row is ``(a_outcome, b_outcome)`` of P / F / X."""
    suite = suite or svc.store.create_suite(name=f"Gates {model['name']} {len(svc.store.suites())}", description="", category="custom", builtin=False, max_tokens=200, position=100)
    run = run or svc.store.create_run(label="gates", state="done", suites=[suite["id"]], contestants=[model["id"]])
    spec = spec or {"type": "all", "checks": [{"type": "regex", "pattern": "a"}, {"type": "contains", "values": ["b"]}]}
    for i, (a, b) in enumerate(rows):
        case = svc.store.create_case(suite_id=suite["id"], position=len(svc.store.cases(suite["id"])), source="user", title=f"c{i}-{model['name']}-{run['id']}",
                                     prompt={"text": "q"}, checker=spec)
        part = lambda o: {"passed": o == P, "score": 1.0 if o == P else 0.0, "detail": {}, **({"unavailable": True} if o == X else {})}
        svc.store.add_result(run_id=run["id"], contestant_id=model["id"], suite_id=suite["id"], case_id=case["id"], category="custom", repeat=0, output="x",
                             score=0.0, passed=all(o == P for o in (a, b)), detail={"checks": [part(a), part(b)]}, digest=digest or model["digest"])
    return suite, run


def test_stored_results_become_a_report_through_the_tool(svc):
    m = add_gguf(svc, "alfa-q4")
    seed(svc, m, [(F, F)] * 12 + [(P, P)] * 30 + [(F, P)] * 8)
    out = tool(svc, "gates_correlation")
    assert [g["gate"] for g in out["gates"]] == ["contains", "regex"]
    p = out["pairs"][0]
    assert p["n"] == 50 and p["counts"] == {"fail_both": 12, "fail_a_only": 0, "fail_b_only": 8, "pass_both": 30}, p
    assert p["reading"] == "mixed" and out["answers"] == 50, "phi about 0.6: related, not redundant"
    assert [f["kind"] for f in out["findings"]] == ["partial_overlap"]
    assert out["scope"]["models"] == ["alfa-q4"] and out["summary"]["es"] and out["findings"]
    assert all({"es", "en", "kind", "gates"} <= set(f) for f in out["findings"])


def test_several_runs_are_pooled_and_a_run_filter_restricts(svc):
    m = add_gguf(svc, "alfa-q4")
    suite, run1 = seed(svc, m, [(F, F)] * 10 + [(P, P)] * 10)
    _, run2 = seed(svc, m, [(F, P)] * 10 + [(P, F)] * 10)
    assert tool(svc, "gates_correlation", runs=[run1["id"]])["pairs"][0]["n"] == 20
    assert tool(svc, "gates_correlation", runs=[run1["id"], run2["id"]])["pairs"][0]["n"] == 40
    assert tool(svc, "gates_correlation")["pairs"][0]["n"] == 40
    with pytest.raises(GaltonError):
        tool(svc, "gates_correlation", runs=["r_nope"])


def test_judge_abstentions_and_pending_answers_stay_out_of_the_pair(svc):
    m = add_gguf(svc, "alfa-q4")
    seed(svc, m, [(F, F)] * 15 + [(P, P)] * 15 + [(X, F)] * 10)
    out = tool(svc, "gates_correlation")
    single = {g["gate"]: g for g in out["gates"]}
    assert single["regex"]["abstained"] == 10 and single["contains"]["abstained"] == 0
    assert out["pairs"][0]["n"] == 30


def test_stale_digest_and_discarded_runs_do_not_count(svc):
    m = add_gguf(svc, "alfa-q4")
    suite, run = seed(svc, m, [(F, F)] * 10 + [(P, P)] * 10)
    seed(svc, m, [(F, P)] * 10, suite=suite, digest="older-file")
    assert tool(svc, "gates_correlation")["pairs"][0]["n"] == 20, "results of an older file are left out"
    assert tool(svc, "gates_correlation", include_stale=True)["pairs"][0]["n"] == 30
    tool(svc, "run_discard", run=run["id"], reason="test of the filter", confirm=True)
    empty = tool(svc, "gates_correlation")
    assert empty["answers"] == 0 and empty["warnings"]


def test_a_single_checker_case_counts_for_its_gate_only(svc):
    m = add_gguf(svc, "alfa-q4")
    seed(svc, m, [(F, F)] * 3, spec={"type": "exact", "value": "x"})
    out = tool(svc, "gates_correlation")
    assert out["pairs"] == [] and [g["gate"] for g in out["gates"]] == ["exact"], "a plain checker is one gate and makes no pair"
    assert out["gates"][0]["n"] == 3 and out["gates"][0]["fails"] == 3


def test_the_route_returns_the_same_report(client):
    m = add_gguf(client.svc, "alfa-q4")
    seed(client.svc, m, [(F, F)] * 12 + [(P, P)] * 30 + [(F, P)] * 8)
    r = client.get("/api/gates/correlation", params={"min_samples": 10, "resamples": 200})
    assert r.status_code == 200
    body = r.json()
    assert body["pairs"][0]["n"] == 50 and body["thresholds"]["min_samples"] == 10
    ui = client.post("/api/ui/call", json={"name": "gates_correlation", "arguments": {"min_samples": 10, "resamples": 200}}).json()
    assert ui["pairs"][0]["counts"] == body["pairs"][0]["counts"]
    agent = client.post("/api/agent/call", json={"name": "gates_correlation", "arguments": {"min_samples": 10, "resamples": 200}}, headers=client.bearer)
    assert agent.status_code == 200
    assert client.get("/api/gates/correlation", params={"min_samples": 1}).status_code in (400, 422)
    assert client.get("/api/gates/correlation", params={"run": "r_nope"}).status_code in (400, 404)
