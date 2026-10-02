"""Continuing an interrupted run: only the cases that were not measured are asked again, the earlier results keep counting, and the answers the
interruption left waiting for the judge are graded."""

from __future__ import annotations

import pytest

from galton_hoard.errors import GaltonError
from galton_hoard.fakes import FakeWorld, reference_responder
from galton_hoard.messages import CodedText, recognise, text
from galton_hoard.runner import LABEL_MAX, normalise_settings
from conftest import tool
from helpers import add_gguf, build_services, references, run_inline

JUDGE_REPLY = '{"score": 9, "reasons": "bien"}'
CASES = 12          # the cases of the quick suite


def results(svc, run_id, **kw):
    return svc.store.results(run_id=run_id, **kw)


def calls_to_cases(svc) -> int:
    return len([c for c in svc.fake_world.calls if "evaluador" not in str(c.messages[0].get("content", ""))])


def interrupt(svc, run, *, keep: int, state: str = "failed") -> dict:
    """Make a finished run look like one that was interrupted after ``keep`` results: the later results never existed."""
    rows = results(svc, run["id"], with_output=False)
    for row in rows[keep:]:
        svc.db.execute("DELETE FROM results WHERE id = ?", (row["id"],))
    svc.store.update_run(run["id"], state=state, error=text("run_stopped") if state == "failed" else "")
    for rc in svc.store.run_contestants(run["id"]):
        svc.store.upsert_run_contestant(run["id"], rc["contestant_id"], state=state, done=keep, error=text("rc_interrupted") if state == "failed" else "")
    return svc.store.run(run["id"])


def asked_case_ids(svc, run_id) -> set[str]:
    return {r["case_id"] for r in results(svc, run_id, with_output=False)}


@pytest.fixture
def started(svc):
    """A model that answers well, and a failed run of the quick suite that got through 5 of the 12 cases."""
    model = add_gguf(svc, "alfa-q4")
    first = run_inline(svc, ["rapida"], [model["id"]])
    assert first["state"] == "done" and len(results(svc, first["id"])) == CASES
    return {"model": model, "run": interrupt(svc, first, keep=5)}


def test_resuming_a_failed_run_asks_only_the_missing_cases(svc, started):
    model, first = started["model"], started["run"]
    svc.fake_world.calls.clear()
    out = svc.resume_run(first["id"], source="test")
    second = svc.store.run(out["run"]["id"])
    assert second["state"] == "done" and second["continues"] == first["id"] and out["continues"] == first["id"]
    assert second["suites"] == first["suites"] and second["contestants"] == first["contestants"] and second["settings"] == first["settings"]
    assert calls_to_cases(svc) == CASES - 5, "the five measured cases are not asked again"
    rc = svc.store.run_contestant(second["id"], model["id"])
    assert rc["total"] == CASES - 5 and rc["done"] == CASES - 5 and rc["state"] == "done"
    note = rc["warnings"][0]
    assert isinstance(note, CodedText) and note.key == "note_resumed" and note.params == {"n": 5} and note == "5 cases already measured in the earlier run are not asked again"
    assert not asked_case_ids(svc, first["id"]) & asked_case_ids(svc, second["id"]) and len(asked_case_ids(svc, first["id"]) | asked_case_ids(svc, second["id"])) == CASES
    assert out["run"]["progress"]["total"] == CASES - 5 and out["run"]["continues"] == first["id"] and out["run"]["continues_label"] == first["label"]


def test_the_results_of_both_runs_count_in_the_ranking(svc, started):
    first = started["run"]
    second = svc.resume_run(first["id"], source="test")["run"]["id"]
    rows = svc.store.scoring_rows(contestant_ids=[started["model"]["id"]])
    assert len(rows) == CASES and {r["run_id"] for r in rows} == {first["id"], second}
    assert tool(svc, "leaderboard", suite="rapida")["rows"][0]["n"] == CASES


def test_a_cancelled_run_can_be_continued_too(tmp_path):
    svc = build_services(tmp_path, world=FakeWorld())
    seen = {"n": 0, "run": None}
    inner = reference_responder(references())

    def respond(req):
        seen["n"] += 1
        if seen["n"] == 4:
            svc.runner.cancel(seen["run"])
        return inner(req)

    m = add_gguf(svc, "cancelado", responder=respond)
    created = svc.runner.create(suites=["rapida"], contestants=[m["id"]], source="test")
    seen["run"] = created["id"]
    assert svc.runner.execute(created["id"])["state"] == "cancelled"
    kept = len(results(svc, created["id"]))
    assert 0 < kept < CASES
    svc.fake_world.calls.clear()
    second = svc.store.run(svc.resume_run(created["id"], source="test")["run"]["id"])
    assert second["state"] == "done" and calls_to_cases(svc) == CASES - kept
    assert svc.store.run_contestant(second["id"], m["id"])["total"] == CASES - kept
    svc.db.close()


def test_errored_skipped_and_unavailable_results_are_asked_again(svc, started):
    first = started["run"]
    rows = results(svc, first["id"], with_output=False)
    assert len(rows) == 5
    svc.store.update_result(rows[0]["id"], error="timeout: no answer within 30 s", passed=False)
    svc.store.update_result(rows[1]["id"], skipped=True, error="skipped: context (9000 tokens needed, 2048 available)")
    svc.store.update_result(rows[2]["id"], unavailable=True)
    svc.fake_world.calls.clear()
    second = svc.store.run(svc.resume_run(first["id"], source="test")["run"]["id"])
    assert calls_to_cases(svc) == CASES - 2, "the three unusable results are asked again; the two good ones are not"
    assert {r["case_id"] for r in results(svc, second["id"], with_output=False)} >= {rows[0]["case_id"], rows[1]["case_id"], rows[2]["case_id"]}
    assert not {rows[3]["case_id"], rows[4]["case_id"]} & asked_case_ids(svc, second["id"])
    assert svc.store.run_contestant(second["id"], started["model"]["id"])["warnings"][0].params == {"n": 2}


def test_results_waiting_for_the_judge_count_as_measured(svc, started):
    first = started["run"]
    rows = results(svc, first["id"], with_output=False)
    svc.store.update_result(rows[0]["id"], judge_pending=True, passed=False)
    svc.fake_world.calls.clear()
    svc.resume_run(first["id"], source="test")
    assert calls_to_cases(svc) == CASES - 5


def test_a_changed_model_is_asked_everything_again(svc, started):
    first, model = started["run"], started["model"]
    svc.store.update_contestant(model["id"], digest="d-alfa-q4-v2")
    svc.fake_world.calls.clear()
    second = svc.store.run(svc.resume_run(first["id"], source="test")["run"]["id"])
    rc = svc.store.run_contestant(second["id"], model["id"])
    assert calls_to_cases(svc) == CASES and rc["total"] == CASES and rc["warnings"] == []
    assert {r["digest"] for r in results(svc, second["id"])} == {"d-alfa-q4-v2"}
    rows = svc.store.scoring_rows(contestant_ids=[model["id"]])
    assert len(rows) == CASES + 5, "the old digest's five results are in the store, but only the new digest is ranked"
    assert tool(svc, "leaderboard", suite="rapida")["rows"][0]["n"] == CASES


def test_a_chain_of_two_continuations_asks_only_what_is_left(svc, started):
    first, model = started["run"], started["model"]
    second = svc.store.run(svc.resume_run(first["id"], source="test")["run"]["id"])
    assert second["state"] == "done"
    second = interrupt(svc, second, keep=3)                   # the continuation was interrupted as well: 5 + 3 cases are measured
    svc.fake_world.calls.clear()
    third = svc.store.run(svc.resume_run(second["id"], source="test")["run"]["id"])
    assert third["continues"] == second["id"] and third["state"] == "done"
    assert calls_to_cases(svc) == CASES - 8
    assert svc.store.run_contestant(third["id"], model["id"])["warnings"][0].params == {"n": 8}
    every = asked_case_ids(svc, first["id"]) | asked_case_ids(svc, second["id"]) | asked_case_ids(svc, third["id"])
    assert len(every) == CASES and len(svc.store.scoring_rows(contestant_ids=[model["id"]])) == CASES
    assert [r["id"] for r in svc.runner.continued_chain(third["continues"])] == [second["id"], first["id"]]


def test_the_label_gets_one_suffix_however_many_times_a_run_is_continued(svc, started):
    first = started["run"]
    second = svc.store.run(svc.resume_run(first["id"], source="test")["run"]["id"])
    assert second["label"].key == "label_continued" and str(second["label"]) == f"{first['label']} (continued)"
    assert second["label"].params == {"label": str(first["label"])}
    third = svc.store.run(svc.resume_run(interrupt(svc, second, keep=2)["id"], source="test")["run"]["id"])
    assert str(third["label"]) == f"{first['label']} (continued)"
    assert recognise(str(third["label"]), prefix="label_").key == "label_continued"


def test_a_long_label_stays_within_the_limit(svc, started):
    first = started["run"]
    svc.store.update_run(first["id"], label="x" * 100)
    second = svc.store.run(svc.resume_run(first["id"], source="test")["run"]["id"])
    assert len(second["label"]) == LABEL_MAX and second["label"].endswith("x… (continued)") and second["label"].key == "label_continued"
    third = svc.store.run(svc.resume_run(interrupt(svc, second, keep=1)["id"], source="test")["run"]["id"])
    assert third["label"] == second["label"], "an already shortened label is not shortened again"


def test_a_loop_in_the_links_ends_the_walk(svc, started):
    first = started["run"]
    second = svc.store.run(svc.resume_run(first["id"], source="test")["run"]["id"])
    svc.store.update_run(first["id"], continues=second["id"])           # a -> b -> a: cannot happen through the tools, may be in a damaged file
    assert [r["id"] for r in svc.runner.continued_chain(second["id"])] == [second["id"], first["id"]]
    assert [r["id"] for r in svc.runner.continued_chain("r_gone")] == []
    planned, _notes = svc.runner.plan_cases(started["model"], [svc.store.suite("s_rapida")], normalise_settings(None), second["id"])
    assert planned == [], "everything is measured in the two runs"


def test_nothing_left_is_refused_and_no_run_is_made(svc, started):
    first = started["run"]
    second = svc.store.run(svc.resume_run(first["id"], source="test")["run"]["id"])
    both = svc.store.update_run(second["id"], state="failed", error=text("run_stopped"))           # the continuation did finish its cases, then was marked failed
    before = len(svc.store.runs(limit=100))
    with pytest.raises(GaltonError) as raised:
        svc.resume_run(both["id"], source="test")
    assert raised.value.key == "run_nothing_left" and raised.value.code == "invalid" and raised.value.params == {"id": both["id"]}
    assert raised.value.hint == "Start a new run if you want to measure them again." and len(svc.store.runs(limit=100)) == before


def test_only_a_failed_or_cancelled_run_that_was_not_discarded_can_be_continued(svc, started):
    first, model = started["run"], started["model"]
    done = run_inline(svc, ["rapida"], [model["id"]])
    with pytest.raises(GaltonError) as raised:
        svc.resume_run(done["id"], source="test")
    assert raised.value.key == "run_not_resumable" and raised.value.params == {"id": done["id"], "state": "done"} and "failed or was cancelled" in raised.value.hint
    queued = svc.runner.create(suites=["rapida"], contestants=[model["id"]], source="test")
    with pytest.raises(GaltonError) as raised:
        svc.runner.resume(queued["id"])
    assert raised.value.key == "run_not_resumable" and raised.value.params["state"] == "queued"
    svc.store.update_run(first["id"], discarded=True, discard_reason="mal")
    with pytest.raises(GaltonError) as raised:
        svc.resume_run(first["id"], source="test")
    assert raised.value.key == "run_discarded_not_resumable" and "run_restore" in raised.value.hint
    with pytest.raises(GaltonError) as raised:
        svc.resume_run("r_nope", source="test")
    assert raised.value.key == "no_run"


def test_the_results_of_a_discarded_run_in_the_chain_do_not_count_as_measured(svc, started):
    first, model = started["run"], started["model"]
    second = svc.store.run(svc.resume_run(first["id"], source="test")["run"]["id"])
    interrupted = interrupt(svc, second, keep=2)
    svc.store.update_run(first["id"], discarded=True, discard_reason="mal")          # its five results are in no statistic: those cases must be measured again
    svc.fake_world.calls.clear()
    third = svc.store.run(svc.resume_run(interrupted["id"], source="test")["run"]["id"])
    assert calls_to_cases(svc) == CASES - 2 and third["state"] == "done"
    assert len(svc.store.scoring_rows(contestant_ids=[model["id"]])) == CASES


def test_the_models_are_checked_as_for_any_run(svc, started):
    first, model = started["run"], started["model"]
    before = len(svc.store.runs(limit=100))
    svc.store.update_contestant(model["id"], enabled=False)
    with pytest.raises(GaltonError) as raised:
        svc.resume_run(first["id"], source="test")
    assert raised.value.key == "model_disabled"
    svc.store.update_contestant(model["id"], enabled=True, missing=True)
    with pytest.raises(GaltonError) as raised:
        svc.runner.resume(first["id"])
    assert raised.value.key == "model_gone" and len(svc.store.runs(limit=100)) == before


def test_a_model_with_nothing_left_is_not_loaded(svc):
    done, other = add_gguf(svc, "completo-q4"), add_gguf(svc, "parcial-q4", seed=1)
    run = run_inline(svc, ["rapida"], [done["id"], other["id"]])
    rows = [r for r in results(svc, run["id"], with_output=False) if r["contestant_id"] == other["id"]]
    for row in rows[4:]:
        svc.db.execute("DELETE FROM results WHERE id = ?", (row["id"],))
    first = svc.store.update_run(run["id"], state="failed", error=text("run_stopped"))
    svc.fake_world.started.clear()
    svc.fake_world.calls.clear()
    second = svc.store.run(svc.resume_run(first["id"], source="test")["run"]["id"])
    assert second["state"] == "done" and [s.name for s in svc.fake_world.started] == ["parcial-q4"], "the model with nothing left was not started"
    complete = svc.store.run_contestant(second["id"], done["id"])
    assert complete["state"] == "done" and complete["total"] == 0 and complete["warnings"][0].params == {"n": CASES}
    assert svc.store.run_contestant(second["id"], other["id"])["total"] == CASES - 4 and calls_to_cases(svc) == CASES - 4


def test_repeats_are_measured_one_by_one(svc):
    model = add_gguf(svc, "alfa-q4")
    run = run_inline(svc, ["rapida"], [model["id"]], repeats=2)
    assert len(results(svc, run["id"])) == 2 * CASES
    first = interrupt(svc, run, keep=CASES + 3)                   # both repeats of the first nine cases, and the first repeat of the next three
    second = svc.store.run(svc.resume_run(first["id"], source="test")["run"]["id"])
    assert svc.store.run_contestant(second["id"], model["id"])["total"] == CASES - 3 and second["settings"]["repeats"] == 2
    assert len(svc.store.scoring_rows(contestant_ids=[model["id"]])) == 2 * CASES


def test_a_continuation_grades_the_answers_the_interruption_left_for_the_judge(tmp_path):
    from galton_hoard.suites import normalise_case
    svc = build_services(tmp_path, world=FakeWorld())
    subject = add_gguf(svc, "sujeto-q4", responder=lambda req: "Una explicación breve. Y otra frase.")
    suite = svc.store.create_suite(name="Con juez", description="", category="custom", builtin=False, max_tokens=200, position=100)
    for i in range(4):
        case = normalise_case({"title": f"Explica el tema {i}", "prompt": f"Explica el tema número {i} en dos frases.", "checker": {"type": "judge", "rubric": "Es correcta y clara."}})
        svc.store.create_case(suite_id=suite["id"], position=i, source="user", **case)
    first = run_inline(svc, [suite["id"]], [subject["id"]])
    assert len(svc.store.pending_judge(first["id"])) == 4, "no judge was chosen: every answer waits"
    first = interrupt(svc, first, keep=3)
    judge = add_gguf(svc, "juez-q4", responder=lambda req: JUDGE_REPLY)
    svc.settings.set_many({"judge.contestant": judge["id"]})
    second = svc.store.run(svc.resume_run(first["id"], source="test")["run"]["id"])
    assert second["state"] == "done" and len(results(svc, second["id"])) == 1
    assert svc.store.pending_judge(first["id"]) == [] and svc.store.pending_judge(second["id"]) == []
    graded = results(svc, first["id"])
    assert len(graded) == 3 and all(r["score"] == 0.9 and r["passed"] and not r["judge_pending"] for r in graded)
    assert len(svc.store.scoring_rows(contestant_ids=[subject["id"]])) == 4
    svc.db.close()


def test_cancelling_the_continuation_stops_the_grading_of_the_earlier_runs(tmp_path):
    from galton_hoard.suites import normalise_case
    svc = build_services(tmp_path, world=FakeWorld())
    subject = add_gguf(svc, "sujeto-q4", responder=lambda req: "Una explicación breve. Y otra frase.")
    suite = svc.store.create_suite(name="Con juez", description="", category="custom", builtin=False, max_tokens=200, position=100)
    for i in range(3):
        case = normalise_case({"title": f"Explica el tema {i}", "prompt": f"Explica el tema número {i} en dos frases.", "checker": {"type": "judge", "rubric": "Es correcta y clara."}})
        svc.store.create_case(suite_id=suite["id"], position=i, source="user", **case)
    first = interrupt(svc, run_inline(svc, [suite["id"]], [subject["id"]]), keep=2)
    judge = add_gguf(svc, "juez-q4", responder=lambda req: JUDGE_REPLY)
    svc.settings.set_many({"judge.contestant": judge["id"]})
    queued = svc.runner.resume(first["id"], source="test")
    svc.runner.cancel(queued["id"])
    assert svc.runner.execute(queued["id"])["state"] == "cancelled" and len(svc.store.pending_judge(first["id"])) == 2, "a cancelled run grades nothing"
    assert svc.runner.judge_pending(first["id"], cancel_as=queued["id"])["graded"] == 0 and len(svc.store.pending_judge(first["id"])) == 2
    assert svc.runner.judge_pending(first["id"])["graded"] == 2 and svc.store.pending_judge(first["id"]) == []
    svc.db.close()


def test_the_run_card_and_the_migration_carry_the_link(svc, started):
    first = started["run"]
    assert svc.db.version() >= 6 and "continues" in [r["name"] for r in svc.db.query("PRAGMA table_info(runs)")]
    assert first["continues"] == "" and svc.run_card(first)["continues"] == "" and svc.run_card(first)["continues_label"] == ""
    second = svc.store.run(svc.resume_run(first["id"], source="test")["run"]["id"])
    card = svc.run_card(second)
    assert card["continues"] == first["id"] and card["continues_label"] == first["label"]
    svc.store.delete_run(first["id"])
    assert svc.run_card(svc.store.run(second["id"]))["continues_label"] == "", "a deleted earlier run leaves the link without a label"
    assert svc.runner.continued_chain(second["continues"]) == []


def test_the_state_machine_of_a_continued_run_is_the_normal_one(svc, started):
    out = svc.resume_run(started["run"]["id"], source="test")
    run = svc.store.run(out["run"]["id"])
    assert run["source"] == "test" and run["started_ts"] is not None and run["finished_ts"] is not None and run["summary"]["contestants"]
    assert svc.store.run(started["run"]["id"])["state"] == "failed", "the earlier run is left as it was"
