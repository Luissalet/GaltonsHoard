"""Harness bench: tasks, the four ways of calling tools, the metrics of a cell, None for infrastructure, the conclusion, the store and the scheduler path."""

from __future__ import annotations

import sys

import pytest

from helpers import add_gguf
from galton_hoard.backends import Completion
from galton_hoard.errors import GaltonError
from galton_hoard.fakes import agent_responder
from galton_hoard.harnessbench import (
    BASES, DEFAULT_HARNESSES, EXAMPLE_LINE, KINDS, TASKS, TASKS_BY_ID, TOOL_SPECS, World, build_matrix, cell_stats, conclude, parse_external, parse_harness_id,
    run_external, run_trial, select_tasks,
)
from galton_hoard.harnessbench.external import ExternalSpec
from galton_hoard.harnessbench.harnesses import json_objects, trim_history
from galton_hoard.harnessbench.runner import normalise_matrix_settings
from galton_hoard.harnessbench.tasks import new_world
from galton_hoard.settings import EXTERNAL_EXAMPLE

VARIANTS = [f"{b}{s}" for b in BASES for s in ("", ".long", ".trim", ".long.trim")]


def _chat(responder, api="openai"):
    """The chat callable ``run_trial`` expects, over a fake responder."""
    from galton_hoard.backends import ChatRequest
    from galton_hoard.fakes import completion_from

    def chat(messages, tools):
        req = ChatRequest(messages=messages, tools=tools or [])
        return completion_from(responder(req), req, tps=40.0, ttft_ms=100.0)
    return chat


def _trials(responder, harness, tasks=None, **kw):
    h = parse_harness_id(harness)
    return [run_trial(t, h, _chat(responder), **kw).as_dict() for t in (tasks or TASKS)]


def _models(svc, **specs):
    return [add_gguf(svc, name, responder=r) for name, r in specs.items()]


# ---------------------------------------------------------------- the tasks and their world
def test_there_are_enough_tasks_with_every_kind_and_unique_ids():
    ids = [t["id"] for t in TASKS]
    assert 20 <= len(ids) <= 30 and len(set(ids)) == len(ids)
    assert {t["kind"] for t in TASKS} == set(KINDS) == {"calc", "convert", "docs", "files", "chain", "notool", "recovery"}
    assert all(not t["oracle"] for t in TASKS if t["kind"] == "notool")
    assert max(len(t["oracle"]) for t in TASKS if t["kind"] == "chain") >= 3
    assert all(set(t["tools"]) <= set(TOOL_SPECS) for t in TASKS)
    assert [t["id"] for t in select_tasks(kinds=["notool"])] == [t["id"] for t in TASKS if t["kind"] == "notool"]
    assert [t["id"] for t in select_tasks(only=[ids[0]])] == [ids[0]]


@pytest.mark.parametrize("harness", VARIANTS)
def test_the_oracle_passes_every_task_in_every_harness_without_extras(harness):
    rows = _trials(agent_responder(), harness)
    bad = [(r["task_id"], r["outcome"], r["reason"]) for r in rows if r["outcome"] != "pass"]
    assert not bad, bad
    assert all(r["extra_calls"] == 0 and r["format_failures"] == 0 for r in rows)


def test_a_wrong_final_answer_is_a_bad_answer_not_a_format_failure():
    answered = [t for t in TASKS if t["state"] is None]            # the tasks that write a file are judged on the file, not on the words
    assert 0 < len(answered) < len(TASKS)
    rows = _trials(agent_responder(accuracy=0.0), "native_tools", answered)
    assert {r["outcome"] for r in rows} == {"fail_answer"}
    assert {r["reason"] for r in rows} == {"wrong_answer"} and all(r["format_failures"] == 0 for r in rows)


def test_the_calculator_refuses_code_and_the_world_validates_arguments():
    w = World.build(TASKS_BY_ID[next(t["id"] for t in TASKS if t["kind"] == "files")]["world"])
    assert w.run("calculator", {"expression": "2 + 3 * 4"}).text.strip().startswith("14")
    assert not w.run("calculator", {"expression": "__import__('os').system('x')"}).ok
    assert not w.run("calculator", {"expression": "2 ^ 3"}).ok
    clean, why = World.check_args(TOOL_SPECS["calculator"], {})
    assert clean is None and "expression" in why
    clean, why = World.check_args(TOOL_SPECS["calculator"], {"expression": ["3"]})
    assert clean is None and why
    assert not w.run("read_file", {"path": "/nowhere.txt"}).ok


def test_the_recovery_tool_fails_first_and_the_tasks_are_independent():
    task = next(t for t in TASKS if t["kind"] == "recovery" and "lookup_record" in t["tools"])
    a, b = new_world(task), new_world(task)
    name, args = next((n, x) for n, x in task["oracle"] if n == "lookup_record")
    first = a.run(name, args)
    assert not first.ok
    assert a.run(name, args).ok
    assert not b.run(name, args).ok, "a second world starts fresh"


# ---------------------------------------------------------------- harnesses and format failures
def test_harness_ids_and_variants_round_trip():
    assert [parse_harness_id(h).id for h in DEFAULT_HARNESSES] == DEFAULT_HARNESSES
    h = parse_harness_id("json_text.long.trim")
    assert (h.base, h.prompt, h.compact) == ("json_text", "long", "trim") and h.id == "json_text.long.trim"
    with pytest.raises(GaltonError) as e:
        parse_harness_id("telepathy")
    assert e.value.key == "hb_unknown_harness"
    specs = [TOOL_SPECS["calculator"]]
    assert len(parse_harness_id("xml_text.long").system_prompt(specs)) > len(parse_harness_id("xml_text").system_prompt(specs))
    assert parse_harness_id("native_tools").tools_payload(specs) and not parse_harness_id("json_text").tools_payload(specs)


def test_json_objects_finds_blocks_in_noise():
    assert json_objects('thinking... {"tool": "a", "args": {"x": 1}} done') == [{"tool": "a", "args": {"x": 1}}]
    assert json_objects("no json here") == []


def test_a_native_only_model_fails_format_under_text_harnesses():
    r = agent_responder(speaks=("native",))
    native = _trials(r, "native_tools")
    assert {x["outcome"] for x in native} == {"pass"}
    tool_tasks = [t for t in TASKS if t["oracle"]]
    for h in ("json_text", "xml_text", "react_text"):
        rows = _trials(r, h, tool_tasks)
        assert {x["outcome"] for x in rows} == {"fail_format"}, h
        assert {x["reason"] for x in rows} == {"format_limit"}
        assert all(x["format_failures"] == 3 and x["detail"] for x in rows)
    notool = [t for t in TASKS if not t["oracle"]]
    assert {x["outcome"] for x in _trials(r, "json_text", notool)} == {"pass"}, "a task without tools needs no format at all"


def test_a_model_that_writes_tool_calls_as_text_fails_format_under_native_tools():
    r = agent_responder(speaks=("xml",))
    tool_tasks = [t for t in TASKS if t["oracle"]]
    assert {x["outcome"] for x in _trials(r, "xml_text", tool_tasks)} == {"pass"}
    rows = _trials(r, "native_tools", tool_tasks[:4])
    assert {x["outcome"] for x in rows} == {"fail_format"}
    assert all("text" in (x["trace"][0]["format_error"] or "").lower() or x["trace"][0]["format_error"] for x in rows)


def test_a_server_that_refuses_the_tools_field_is_a_native_format_failure():
    task = next(t for t in TASKS if t["oracle"])

    def refusing(messages, tools):
        return Completion(text="", error="HTTP 400: this model does not support tools")
    t = run_trial(task, parse_harness_id("native_tools"), refusing)
    assert (t.outcome, t.reason) == ("fail_format", "tools_unsupported") and t.passed is False
    other = run_trial(task, parse_harness_id("json_text"), refusing)
    assert other.outcome == "none", "the same error under a text harness is the server's, not the format's"


def test_repeated_tools_count_as_extras_but_do_not_fail_the_task():
    rows = _trials(agent_responder(extra_calls=2), "native_tools")
    assert {x["outcome"] for x in rows} == {"pass"}
    assert all(x["extra_calls"] >= 2 for x in rows if x["calls"])
    stats = cell_stats(rows)
    assert stats["extra_calls_mean"] >= 1.5 and stats["extra_rate"] > 0.8 and stats["pass_at_1"] == 1.0
    assert sum(x["extra_repeat"] + x["extra_after_solved"] for x in rows) == sum(x["extra_calls"] for x in rows) > 0


def test_a_task_that_needs_no_tool_counts_a_tool_call_as_extra():
    notool = [t for t in TASKS if t["kind"] == "notool"]
    rows = _trials(agent_responder(extra_calls=1), "json_text", notool)
    assert all(r["outcome"] in ("pass", "fail_format", "fail_answer") for r in rows)


def test_trim_compaction_shortens_old_observations_only():
    entries = [{"kind": "system", "msg": {"role": "system", "content": "s"}}, {"kind": "user", "msg": {"role": "user", "content": "task"}}]
    for i in range(5):
        entries.append({"kind": "assistant", "msg": {"role": "assistant", "content": f"call {i}"}})
        entries.append({"kind": "obs", "short": {"role": "user", "content": "[old result]"}, "msg": {"role": "user", "content": f"result {i} " + "x" * 400}})
    assert trim_history(entries, 0) == [e["msg"] for e in entries]
    trimmed = trim_history(entries, 2)
    assert len(trimmed) == len(entries)
    contents = [m["content"] for m in trimmed]
    assert contents[0] == "s" and contents[1] == "task"
    assert len(contents[-1]) > 400 and len(contents[-3]) > 400, "the last two observations are intact"
    assert contents[3] == "[old result]", "an old observation is cut"


def test_turn_cap_is_the_same_everywhere_and_ends_a_looping_model():
    def loop(messages, tools):
        return Completion(text="", tool_calls=[{"name": "calculator", "arguments": {"expression": "1+1"}}], completion_tokens=5, prompt_tokens=10)
    task = next(t for t in TASKS if t["kind"] == "calc")
    t = run_trial(task, parse_harness_id("native_tools"), loop, max_turns=4)
    assert t.turns == 4 and t.outcome == "fail_answer" and t.reason == "turn_limit"


def test_a_truncated_reply_is_a_failure_of_the_answer_and_infra_is_none():
    task = next(t for t in TASKS if t["kind"] == "calc")
    cut = run_trial(task, parse_harness_id("json_text"), lambda m, t: Completion(text="", finish_reason="length", completion_tokens=1024))
    assert cut.outcome == "fail_answer" and cut.reason == "truncated"
    down = run_trial(task, parse_harness_id("json_text"), lambda m, t: Completion(text="", error="timeout after 60 s"))
    assert down.outcome == "none" and down.passed is None and down.detail or down.reason


# ---------------------------------------------------------------- cell stats, None, pass@k and the conclusion
def _row(task, outcome, **kw):
    base = {"task_id": task, "outcome": outcome, "turns": 2, "calls": 1, "format_failures": 0, "extra_calls": 0, "tokens_in": 100, "tokens_out": 20, "latency_ms": 50.0,
            "detail": "", "reason": ""}
    return base | kw


def test_none_is_left_out_of_every_denominator_and_reported_apart():
    rows = [_row("a", "pass"), _row("b", "pass"), _row("c", "fail_answer"), _row("d", "none", reason="server", detail="timeout"), _row("e", "none", reason="server", detail="timeout")]
    s = cell_stats(rows)
    assert s["valid"] == 3 and s["none_n"] == 2 and s["pass_at_1"] == pytest.approx(0.6667, abs=1e-3)
    assert s["none_reasons"] == {"timeout": 2}
    only_none = cell_stats([_row("a", "none", detail="net")])
    assert only_none["pass_at_1"] is None and only_none["valid"] == 0 and only_none["none_n"] == 1 and only_none["ci"] is None


def test_format_failures_are_separate_from_bad_answers():
    s = cell_stats([_row("a", "pass"), _row("b", "fail_format", format_failures=3), _row("c", "fail_answer"), _row("d", "fail_answer")])
    assert s["format_fail_rate"] == 0.25 and s["answer_fail_rate"] == 0.5 and s["pass_at_1"] == 0.25
    assert s["counts"] == {"pass": 1, "fail_answer": 2, "fail_format": 1, "none": 0}


def test_pass_at_k_and_pass_hat_k_per_task():
    rows = [_row("a", "pass"), _row("a", "fail_answer"), _row("a", "pass"), _row("b", "fail_answer"), _row("b", "fail_answer"), _row("b", "fail_answer")]
    s = cell_stats(rows, k=3)
    assert s["pass_at_1"] == pytest.approx(2 / 6, abs=1e-3)
    assert s["pass_at_k"] == pytest.approx(0.5) and s["pass_hat_k"] == pytest.approx(0.0)
    assert "pass_at_k" not in cell_stats(rows, k=1) or cell_stats(rows, k=1).get("pass_at_k") is None


def _matrix_from(table):
    """table: {model: {harness: pass count out of 10}} -> (matrix, ids, harnesses)."""
    rows = []
    for m, per in table.items():
        for h, ok in per.items():
            for i in range(10):
                rows.append(_row(f"t{i}", "pass" if i < ok else "fail_answer") | {"contestant_id": m, "harness": h})
    ids, hs_ = list(table), list(next(iter(table.values())))
    return build_matrix(rows, ids, hs_, 1), ids, hs_


def test_the_conclusion_says_whether_harness_or_model_matters_more():
    matrix, ids, hs_ = _matrix_from({"m1": {"native_tools": 9, "json_text": 2}, "m2": {"native_tools": 8, "json_text": 3}})
    c = conclude(matrix, {i: i.upper() for i in ids}, hs_)
    assert c["verdict"] == "harness_larger" and c["harness_spread_points"] > c["model_spread_points"]
    assert c["best_harness_by_model"] == {"M1": "native_tools", "M2": "native_tools"}
    assert "larger" in c["summary"]["en"] and "mayor" in c["summary"]["es"] and "M1" in c["summary"]["en"]
    assert c["per_model"]["m1"]["verdict"] == "harness_sensitive"

    matrix, ids, hs_ = _matrix_from({"m1": {"native_tools": 9, "json_text": 9}, "m2": {"native_tools": 3, "json_text": 3}})
    c = conclude(matrix, {i: i for i in ids}, hs_)
    assert c["verdict"] == "model_larger" and "smaller" in c["summary"]["en"] and "menor" in c["summary"]["es"]
    assert c["per_model"]["m1"]["verdict"] == "stable" and c["per_model"]["m2"]["verdict"] == "weak_everywhere"

    matrix, ids, hs_ = _matrix_from({"m1": {"native_tools": 9}})
    assert conclude(matrix, {"m1": "m1"}, hs_)["verdict"] == "not_comparable"


def test_the_conclusion_flags_weak_evidence_and_counts_none():
    rows = [_row("a", "pass") | {"contestant_id": "m", "harness": "h1"}, _row("a", "none", detail="net") | {"contestant_id": "m", "harness": "h2"}]
    c = conclude(build_matrix(rows, ["m"], ["h1", "h2"], 1), {"m": "m"}, ["h1", "h2"])
    assert c["weak"] and c["none_total"] == 1 and "left out" in c["summary"]["en"] and "sin resultado" in c["summary"]["es"]


# ---------------------------------------------------------------- the whole path through the services
HARNESSES = ["native_tools", "json_text", "xml_text", "react_text"]


def _start(svc, models, **kw):
    out = svc.start_matrix(models=[m["id"] for m in models], **kw)
    return out["matrix"]["id"]


def test_a_matrix_runs_through_the_scheduler_path_and_is_stored(svc):
    native, writer = _models(svc, nativo=agent_responder(speaks=("native",)), escritor=agent_responder(speaks=("xml", "native"), accuracy=0.9))
    mid = _start(svc, [native, writer], harnesses=HARNESSES)
    m = svc.matrix_store.matrix(mid)
    assert m["state"] == "done" and m["done"] == m["total"] == 2 * 4 * len(TASKS)
    res = svc.matrices.result(mid)
    cn, cw = res["cells"][native["id"]], res["cells"][writer["id"]]
    assert cn["native_tools"]["pass_at_1"] == 1.0
    assert cn["json_text"]["pass_at_1"] < 0.2 and cn["json_text"]["format_fail_rate"] > 0.7
    assert cw["xml_text"]["pass_at_1"] is not None and cw["xml_text"]["pass_at_1"] > 0.7
    assert cw["json_text"]["format_fail_rate"] > 0.7
    assert res["conclusion"]["verdict"] == "harness_larger" and res["conclusion"]["best_harness_by_model"]["nativo"] == "native_tools"
    assert "| nativo |" in res["markdown"] and res["trials"] == m["total"]
    assert svc.matrix_store.latest()["id"] == mid and m["summary"]["verdict"] == "harness_larger"


def test_the_same_caps_apply_in_every_cell(svc):
    caps = {}

    def spy(req):
        caps.setdefault((req.max_tokens, req.temperature), 0)
        caps[(req.max_tokens, req.temperature)] += 1
        return agent_responder()(req)
    (a,) = _models(svc, espia=spy)
    _start(svc, [a], harnesses=["native_tools", "react_text"], kinds=["calc"], settings={"max_tokens": 300, "max_turns": 5})
    assert set(caps) == {(300, 0.0)}


def test_infrastructure_errors_become_none_not_zero(svc):
    def broken(req):
        return {"text": "", "error": "connection reset"}
    (a, b) = _models(svc, roto=broken, bueno=agent_responder())
    mid = _start(svc, [a, b], harnesses=["native_tools", "json_text"], kinds=["calc", "notool"])
    res = svc.matrices.result(mid)
    bad = res["cells"][a["id"]]["native_tools"]
    assert bad["pass_at_1"] is None and bad["valid"] == 0 and bad["none_n"] == bad["trials"] > 0
    assert res["cells"][b["id"]]["native_tools"]["pass_at_1"] == 1.0
    assert res["by_model"][a["id"]] is None
    assert res["conclusion"]["none_total"] >= bad["none_n"] and res["conclusion"]["best_harness_by_model"].get("roto") is None
    assert svc.matrix_store.matrix(mid)["state"] == "done"


def test_k_above_one_runs_samples_and_reports_pass_at_k(svc):
    (a,) = _models(svc, solo=agent_responder(accuracy=0.5, seed=3))
    mid = _start(svc, [a], harnesses=["native_tools"], kinds=["calc"], k=3)
    m = svc.matrix_store.matrix(mid)
    assert m["total"] == 3 * len(select_tasks(kinds=["calc"]))
    cell = svc.matrices.result(mid)["cells"][a["id"]]["native_tools"]
    assert cell["trials"] == m["total"] and "pass_at_k" in cell and "pass_hat_k" in cell
    assert svc.matrix_store.matrix(mid)["settings"]["temperature"] > 0


def test_dry_run_plans_without_creating_anything(svc):
    (a,) = _models(svc, plan=agent_responder())
    plan = svc.matrices.plan(models=[a["id"]], harnesses=["native_tools", "json_text.trim"], kinds=["calc"], k=2)
    assert plan["trials"] == 2 * 2 * len(select_tasks(kinds=["calc"])) and plan["same_for_every_cell"]["max_turns"]
    assert svc.matrix_store.matrices() == []


def test_requests_that_cannot_run_are_refused_with_a_catalogue_message(svc):
    (a,) = _models(svc, uno=agent_responder())
    for kw, key in (({"harnesses": ["nope"]}, "hb_unknown_harness"), ({"tasks": ["no-such-task"]}, "hb_no_task"), ({"harnesses": ["cli:x"]}, "hb_external_off")):
        with pytest.raises(GaltonError) as e:
            svc.start_matrix(models=[a["id"]], **kw)
        assert e.value.key == key
    with pytest.raises(GaltonError):
        svc.start_matrix(models=[])


def test_normalise_matrix_settings_bounds_and_idempotence():
    s = normalise_matrix_settings(None, 1)
    assert s["max_turns"] == 8 and s["max_format_errors"] == 3 and s["temperature"] == 0.0 and s["max_tokens"] == 1024
    assert normalise_matrix_settings(s, 1) == s
    assert normalise_matrix_settings({}, 4)["temperature"] == 0.7
    with pytest.raises(GaltonError):
        normalise_matrix_settings({"max_turns": 500}, 1)


def test_cancel_before_running_leaves_a_cancelled_matrix(svc):
    (a,) = _models(svc, uno=agent_responder())
    m = svc.matrices.create(models=[a["id"]], kinds=["calc"], settings=normalise_matrix_settings(None, 1))
    svc.matrices.cancel(m["id"])
    done = svc.matrices.execute(m["id"])
    assert done["state"] == "cancelled" and svc.matrix_store.count_trials(m["id"]) == 0


def test_cell_detail_shows_traces_with_the_format_error(svc):
    (a,) = _models(svc, nativo=agent_responder(speaks=("native",)))
    mid = _start(svc, [a], harnesses=["json_text"], kinds=["calc"])
    detail = svc.matrices.cell(mid, "nativo", "json_text")
    assert detail["stats"]["format_fail_rate"] == 1.0 and len(detail["tasks"]) == len(select_tasks(kinds=["calc"])) and "trials" not in detail
    one = svc.matrices.cell(mid, "nativo", "json_text", detail["tasks"][0]["task"])
    assert one["trials"][0]["trace"] and any(step["format_error"] or step["rejected"] for step in one["trials"][0]["trace"])
    with pytest.raises(GaltonError) as e:
        svc.matrices.cell(mid, "nadie", "json_text")
    assert e.value.key == "hb_no_model_in_matrix"
    with pytest.raises(GaltonError) as e:
        svc.matrices.cell(mid, "nativo", "xml_text")
    assert e.value.key == "hb_unknown_harness"


def test_a_changed_model_marks_the_matrix_stale_and_removal_cleans_up(svc):
    (a,) = _models(svc, uno=agent_responder())
    mid = _start(svc, [a], harnesses=["native_tools"], kinds=["calc"])
    assert svc.matrices.result(mid)["models"][0]["stale"] is False
    svc.db.execute("UPDATE contestants SET digest = 'another' WHERE id = ?", [a["id"]])
    assert svc.matrices.result(mid)["models"][0]["stale"] is True
    svc.remove_model(a["id"])
    assert svc.matrix_store.count_trials(mid) == 0


def test_restart_abandons_running_matrices_and_requeues_queued_ones(svc):
    (a,) = _models(svc, uno=agent_responder())
    rs = normalise_matrix_settings(None, 1)
    running = svc.matrices.create(models=[a["id"]], kinds=["calc"], settings=rs)
    svc.matrix_store.update_matrix(running["id"], state="running")
    svc._abandon_stale_matrices()
    assert svc.matrix_store.matrix(running["id"])["state"] == "failed" and svc.matrix_store.matrix(running["id"])["error"]
    assert not svc.matrix_busy()
    queued = svc.matrices.create(models=[a["id"]], kinds=["calc"], settings=rs)
    assert svc.matrix_busy()
    svc._requeue_queued_matrices()
    assert svc.matrix_store.matrix(queued["id"])["state"] in ("queued", "done")


def test_the_watch_waits_while_a_matrix_is_active(svc):
    (a,) = _models(svc, uno=agent_responder())
    assert svc.matrix_busy() is False
    svc.matrices.create(models=[a["id"]], kinds=["calc"], settings=normalise_matrix_settings(None, 1))
    assert svc.matrix_busy() is True


# ---------------------------------------------------------------- command-line harnesses
def test_the_example_line_ships_disabled_and_matches_the_setting_default(svc):
    assert EXAMPLE_LINE == EXTERNAL_EXAMPLE
    assert svc.settings.get("harnessbench.external") == [EXAMPLE_LINE] and svc.settings.get("harnessbench.allow_external") is False
    specs = parse_external([EXAMPLE_LINE])
    assert len(specs) == 1 and specs[0].enabled is False and specs[0].id == "cli:example" and "{base_url}" in specs[0].argv
    assert svc.matrices.catalog()["external"] == {"allowed": False, "configured": [specs[0].info()], "how": svc.matrices.catalog()["external"]["how"]}
    assert parse_external(["broken line", "x =", "= y", "ok = echo hi"])[0].argv == ("echo", "hi")


def test_a_disabled_or_unknown_external_harness_cannot_be_chosen(svc):
    (a,) = _models(svc, uno=agent_responder())
    svc.settings.set_many({"harnessbench.allow_external": True})
    for ref, key in (("cli:example", "hb_external_disabled"), ("cli:nothing", "hb_external_unknown")):
        with pytest.raises(GaltonError) as e:
            svc.start_matrix(models=[a["id"]], harnesses=[ref])
        assert e.value.key == key


def _python_agent(code: str) -> ExternalSpec:
    return ExternalSpec("fake", (sys.executable, "-c", code, "{workdir}", "{base_url}", "{model}"))


def test_an_external_command_runs_in_a_scratch_folder_and_is_verified():
    task = TASKS_BY_ID[next(t["id"] for t in TASKS if t["kind"] == "files" and t["external"])]
    answer = task["final"]
    t = run_external(_python_agent(f"import sys; print('thinking'); print({answer!r})"), next(x for x in TASKS if x["kind"] == "calc" and x["external"]), base_url="http://h:1", model="m", timeout_s=30)
    assert t.outcome in ("pass", "fail_answer") and t.turns is None and t.format_failures is None and t.extra_calls is None and t.tokens_out is None
    calc = next(x for x in TASKS if x["kind"] == "calc" and x["external"])
    ok = run_external(_python_agent(f"print({calc['final']!r})"), calc, base_url="http://h:1", model="m", timeout_s=30)
    assert ok.outcome == "pass" and ok.passed is True and ok.trace[0]["exit"] == 0
    wrong = run_external(_python_agent("print('I do not know')"), calc, base_url="http://h:1", model="m", timeout_s=30)
    assert wrong.outcome == "fail_answer" and wrong.passed is False
    crash = run_external(_python_agent("import sys; sys.exit(3)"), calc, base_url="http://h:1", model="m", timeout_s=30)
    assert crash.outcome == "none" and crash.passed is None and crash.reason == "command_failed"
    missing = run_external(ExternalSpec("gone", ("/no/such/binary-xyz",)), calc, base_url="http://h:1", model="m", timeout_s=30)
    assert missing.outcome == "none" and missing.reason == "cannot_start"
    slow = run_external(_python_agent("import time; time.sleep(30)"), calc, base_url="http://h:1", model="m", timeout_s=1)
    assert slow.outcome == "fail_answer" and slow.reason == "timeout"


def test_an_external_agent_can_write_files_that_the_verifier_reads():
    task = next(t for t in TASKS if t["kind"] == "files" and t["external"] and t["state"] is not None)
    path, content = next(((n, a) for n, a in task["oracle"] if n == "write_file"), (None, None))
    assert path and content
    p, text = content["path"].lstrip("/"), content["content"]
    code = ("import sys, pathlib; w = pathlib.Path(sys.argv[1]); t = w / " + repr(p) + "; t.parent.mkdir(parents=True, exist_ok=True); t.write_text(" + repr(text) +
            "); print(" + repr(task["final"]) + ")")
    t = run_external(_python_agent(code), task, base_url="http://h:1", model="m", timeout_s=30)
    assert t.outcome == "pass", (t.outcome, t.detail)


def test_a_matrix_with_an_external_harness_runs_and_marks_unseen_metrics_as_unknown(svc):
    (a,) = _models(svc, uno=agent_responder())
    svc.settings.set_many({"harnessbench.allow_external": True, "harnessbench.external": [f"fake = {sys.executable} -c \"print('x')\" {{workdir}}"]})
    mid = _start(svc, [a], harnesses=["cli:fake"], kinds=["calc"])
    cell = svc.matrices.result(mid)["cells"][a["id"]]["cli:fake"]
    assert cell["trials"] > 0 and cell["turns_mean"] is None and cell["format_error_rate"] is None and cell["extra_rate"] is None


# ---------------------------------------------------------------- persistence and the demo
def test_the_migration_creates_the_tables(svc):
    names = {r["name"] for r in svc.db.query("SELECT name FROM sqlite_master WHERE type='table'")}
    assert {"hb_matrices", "hb_trials"} <= names


def test_the_demo_world_carries_models_with_different_dialects(tmp_path, clock):
    from helpers import build_services
    from galton_hoard.fakes import DEMO_DIALECTS, demo_seed
    s = build_services(tmp_path, clock=clock)
    try:
        demo_seed(s)
        models = [c["id"] for c in s.store.contestants()][:3]
        assert len(DEMO_DIALECTS) >= 3
        mid = _start(s, [{"id": i} for i in models], harnesses=["native_tools", "json_text"], kinds=["calc"])
        assert s.matrix_store.matrix(mid)["state"] == "done"
        assert s.matrices.result(mid)["conclusion"]["verdict"] != "not_comparable"
    finally:
        s.db.close()


# ---------------------------------------------------------------- the standalone script against a real HTTP endpoint
def _serve(responder):
    """A tiny chat-completions endpoint (streaming) over a responder, on a free local port."""
    import json
    import threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    from galton_hoard.backends import ChatRequest
    from galton_hoard.fakes import completion_from

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers.get("content-length") or 0)))
            req = ChatRequest(messages=body["messages"], tools=body.get("tools") or [])
            comp = completion_from(responder(req), req, tps=40.0, ttft_ms=1.0)
            delta = {"content": comp.text} if not comp.tool_calls else {"tool_calls": [
                {"index": i, "function": {"name": c["name"], "arguments": json.dumps(c["arguments"])}} for i, c in enumerate(comp.tool_calls)]}
            chunks = [{"choices": [{"delta": delta}]}, {"choices": [{"delta": {}, "finish_reason": "tool_calls" if comp.tool_calls else "stop"}]},
                      {"choices": [], "usage": {"prompt_tokens": comp.prompt_tokens, "completion_tokens": comp.completion_tokens}}]
            self.send_response(200)
            self.send_header("content-type", "text/event-stream")
            self.end_headers()
            for chunk in chunks:
                self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
            self.wfile.write(b"data: [DONE]\n\n")

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, f"http://127.0.0.1:{server.server_address[1]}"


def test_the_standalone_script_runs_a_small_matrix_against_an_http_endpoint(tmp_path):
    import json
    import subprocess
    from pathlib import Path
    root = Path(__file__).resolve().parent.parent
    plan = subprocess.run([sys.executable, str(root / "scripts" / "harnessbench_eval.py"), "--dry-run", "--kinds", "calc"], capture_output=True, text=True, cwd=root)
    assert plan.returncode == 0 and "harnesses: native_tools, json_text, xml_text, react_text" in plan.stdout
    server, url = _serve(agent_responder(speaks=("native", "xml")))
    try:
        out = tmp_path / "report.json"
        done = subprocess.run([sys.executable, str(root / "scripts" / "harnessbench_eval.py"), "--url", url, "--model", "fake-8b", "--harnesses", "native_tools,json_text,xml_text",
                               "--kinds", "calc,notool", "--out", str(out)], capture_output=True, text=True, cwd=root, timeout=300)
        assert done.returncode == 0, done.stderr
    finally:
        server.shutdown()
    report = json.loads(out.read_text(encoding="utf-8"))
    cells = report["cells"]["fake-8b"]
    assert cells["native_tools"]["pass_at_1"] == 1.0 and cells["xml_text"]["pass_at_1"] == 1.0
    assert cells["json_text"]["format_fail_rate"] == pytest.approx(0.8, abs=0.21) and cells["json_text"]["pass_at_1"] < 0.5
    assert report["conclusion"]["best_harness_by_model"]["fake-8b"] in ("native_tools", "xml_text") and report["trials"] and "| fake-8b |" in done.stderr


def test_the_standalone_script_reports_a_dead_server_as_none(tmp_path):
    import json
    import subprocess
    from pathlib import Path
    root = Path(__file__).resolve().parent.parent
    out = tmp_path / "dead.json"
    done = subprocess.run([sys.executable, str(root / "scripts" / "harnessbench_eval.py"), "--url", "http://127.0.0.1:9", "--model", "x", "--harnesses", "native_tools",
                           "--kinds", "calc", "--timeout", "2", "--out", str(out)], capture_output=True, text=True, cwd=root, timeout=120)
    assert done.returncode == 0, done.stderr
    cell = json.loads(out.read_text(encoding="utf-8"))["cells"]["x"]["native_tools"]
    assert cell["pass_at_1"] is None and cell["none_n"] >= 1 and cell["valid"] == 0


def test_an_empty_reply_is_a_format_failure_with_a_retry_not_a_wrong_answer():
    """A server that ends the turn on a special token returns nothing visible: the model tried to act."""
    task = next(t for t in TASKS if t["kind"] == "calc")
    empty = run_trial(task, parse_harness_id("xml_text"),
                      lambda m, t: Completion(text="", finish_reason="stop", reasoning="I should call the calculator."))
    assert empty.outcome == "fail_format" and empty.reason == "format_limit"
    assert empty.format_failures == 3 and empty.trace[0]["format_error"] == "empty reply"
    assert "calculator" in empty.trace[0]["reasoning"]
    oracle = agent_responder(accuracy=1.0)
    replies = iter([None])

    def once_empty(messages, tools):
        if next(replies, "go") is None:
            return Completion(text="", finish_reason="stop")
        return _chat(oracle)(messages, tools)
    recovered = run_trial(task, parse_harness_id("xml_text"), once_empty)
    assert recovered.outcome == "pass" and recovered.format_failures == 1
