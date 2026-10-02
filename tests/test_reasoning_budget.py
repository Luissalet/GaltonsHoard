"""Models that think before they answer: the room their thinking gets in ``max_tokens``, the effort ``off`` switches, detection, truncated results and
the warnings about them (leaderboard, routes, recommendation, run page)."""

from __future__ import annotations

import json
import sqlite3

import httpx
import pytest

from galton_hoard.backends import ChatRequest, Completion, OllamaBackend, OpenAIBackend
from galton_hoard.db import MIGRATIONS, Database
from galton_hoard.errors import GaltonError
from galton_hoard.fakes import reference_responder
from galton_hoard.servers import probe_openai, template_reasons
from galton_hoard.stats import TRUNCATED_WARN_SHARE, truncation
from helpers import add_gguf, json_response, mock_client_factory, references, run_inline
from conftest import tool

EXTRA = 8192


def thinker(share_cut: float = 0.0, reasoning: str = "pienso un momento"):
    """A model that thinks and answers from the reference table; a fixed share of its answers is cut while it is still thinking."""
    base = reference_responder(references())
    counter = {"n": 0}

    def respond(req):
        counter["n"] += 1
        if share_cut and (counter["n"] % round(1 / share_cut)) == 0:
            return {"text": "", "reasoning": reasoning * 40, "finish_reason": "length"}
        reply = base(req)
        return {**reply, "reasoning": reasoning} if isinstance(reply, dict) else {"text": reply, "reasoning": reasoning}
    return respond


def mark_reasoning(svc, contestant, how="props"):
    return svc.store.update_contestant(contestant["id"], meta={**contestant["meta"], "reasons": True, "reasons_how": how})


def sent(svc):
    return [call.max_tokens for call in svc.fake_world.calls]


def baseline(svc, **settings):
    """The answer budgets of every case of the quick suite for a model that does not reason, sorted."""
    baseline.count += 1
    plain = add_gguf(svc, f"base-{baseline.count}")
    before = len(svc.fake_world.calls)
    run_inline(svc, ["rapida"], [plain["id"]], **settings)
    out = sorted(call.max_tokens for call in svc.fake_world.calls[before:])
    del svc.fake_world.calls[before:]
    return out


baseline.count = 0


def shifted(svc, extra, **settings):
    return sorted(m + extra for m in baseline(svc, **settings))


# ---- what the backends send ---------------------------------------------------------------------------------------------------------------

def capture(backend_factory, **request):
    seen = {}

    def handler(req):
        seen["body"] = json.loads(req.content)
        if req.url.path.endswith("/chat/completions"):
            body = "".join(f"data: {json.dumps(c)}\n\n" for c in [{"choices": [{"delta": {"content": "ok"}, "finish_reason": "stop"}]}]) + "data: [DONE]\n\n"
            return httpx.Response(200, content=body.encode(), headers={"content-type": "text/event-stream"})
        return httpx.Response(200, content=(json.dumps({"message": {"content": "ok"}}) + "\n" + json.dumps({"done": True}) + "\n").encode())
    backend_factory(httpx.MockTransport(handler)).chat(ChatRequest(messages=[{"role": "user", "content": "hola"}], **request))
    return seen["body"]


def openai_body(**request):
    return capture(lambda transport: OpenAIBackend("http://x:1", "m", transport=transport), **request)


def ollama_body(**request):
    return capture(lambda transport: OllamaBackend("http://x:11434", "qwen3:8b", transport=transport), **request)


def test_effort_off_switches_thinking_off_on_llama_server():
    body = openai_body(effort="off", max_tokens=300)
    assert body["chat_template_kwargs"] == {"enable_thinking": False} and body["max_tokens"] == 300
    assert not any(key in body for key in ("thinking_budget_tokens", "reasoning_budget", "reasoning_effort"))


def test_effort_off_switches_thinking_off_on_ollama():
    body = ollama_body(effort="off", max_tokens=300)
    assert body["think"] is False and body["options"]["num_predict"] == 300


def test_no_effort_leaves_the_server_default_alone():
    assert "chat_template_kwargs" not in openai_body(max_tokens=300) and "think" not in ollama_body(max_tokens=300)


def test_an_effort_widens_max_tokens_unless_the_runner_already_made_room():
    plain = openai_body(effort="high", max_tokens=300)
    assert plain["max_tokens"] == 300 + 8192 and plain["chat_template_kwargs"]["enable_thinking"] is True
    ready = openai_body(effort="high", max_tokens=300 + EXTRA, reasoning_tokens=EXTRA)
    assert ready["max_tokens"] == 300 + EXTRA
    assert ollama_body(effort="high", max_tokens=300)["options"]["num_predict"] == 300 + 8192
    assert ollama_body(effort="high", max_tokens=300 + EXTRA, reasoning_tokens=EXTRA)["options"]["num_predict"] == 300 + EXTRA


def test_truncated_means_cut_with_nothing_to_show():
    assert Completion(text="", reasoning="pienso", finish_reason="length").truncated
    assert not Completion(text="algo", finish_reason="length").truncated
    assert not Completion(text="", finish_reason="stop").truncated
    assert not Completion(text="", finish_reason="length", error="HTTP 500").truncated
    assert not Completion(text="", finish_reason="length", tool_calls=[{"name": "f", "arguments": {}}]).truncated


# ---- does the model reason? -----------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("template,expected", [("{% if enable_thinking %}x{% endif %}", True), ("<|im_start|>assistant\n<think>\n", True), ("{{ messages }}", False),
                                                ("", None), (None, None), (7, None)])
def test_template_reasons(template, expected):
    assert template_reasons(template) is expected


def props_handler(template):
    def handler(req):
        if req.url.path == "/props":
            return json_response({"chat_template": template, "default_generation_settings": {"n_ctx": 4096}})
        if req.url.path == "/v1/models":
            return json_response({"data": [{"id": "m"}]})
        return httpx.Response(404)
    return handler


def test_probe_openai_reports_reasoning_from_props():
    for template, expected in (("{% if enable_thinking %}", True), ("plain", False)):
        with httpx.Client(transport=httpx.MockTransport(props_handler(template))) as client:
            assert probe_openai(client, "http://x:1")["reasoning"] is expected
    with httpx.Client(transport=httpx.MockTransport(lambda req: httpx.Response(404))) as client:
        assert probe_openai(client, "http://x:1").get("reasoning") is None


def real_gguf(svc, name="real", **meta):
    """A contestant that is probed like a real one (no demo flag)."""
    m = add_gguf(svc, name)
    svc.store.update_contestant(m["id"], meta={**{k: v for k, v in m["meta"].items() if k != "demo"}, **meta})
    return svc.store.contestant(m["id"])


def test_detect_reasoning_on_llama_server_reads_the_chat_template(svc):
    c = real_gguf(svc)
    svc.runner.client_factory = mock_client_factory(props_handler("... enable_thinking ..."))
    assert svc.runner.detect_reasoning(c, url="http://x:1", api="openai") == (True, "props")
    svc.runner.client_factory = mock_client_factory(props_handler("no thinking here"))
    assert svc.runner.detect_reasoning(c, url="http://x:1", api="openai") == (False, "props")


def test_detect_reasoning_on_ollama_reads_the_capabilities(svc):
    c = real_gguf(svc)

    def show(caps):
        return lambda req: json_response({"capabilities": caps}) if req.url.path == "/api/show" else httpx.Response(404)
    with httpx.Client(transport=httpx.MockTransport(show(["completion", "thinking"]))) as client:
        assert svc.runner.detect_reasoning(c, url="http://x", api="ollama", model="qwen3:8b", client=client) == (True, "show")
    with httpx.Client(transport=httpx.MockTransport(show(["completion", "tools"]))) as client:
        assert svc.runner.detect_reasoning(c, url="http://x", api="ollama", model="qwen3:8b", client=client) == (False, "show")


def test_detect_reasoning_falls_back_to_what_is_stored(svc):
    unknown = real_gguf(svc, "unknown")
    stored = real_gguf(svc, "stored", reasons=True, reasons_how="answer")
    svc.runner.client_factory = mock_client_factory(lambda req: (_ for _ in ()).throw(httpx.ConnectError("down")))
    assert svc.runner.detect_reasoning(unknown, url="http://x:1", api="openai") == (None, "")
    assert svc.runner.detect_reasoning(stored, url="http://x:1", api="openai") == (True, "answer")
    svc.runner.client_factory = mock_client_factory(lambda req: httpx.Response(404))
    assert svc.runner.detect_reasoning(stored, url="http://x:1", api="openai") == (True, "answer")


def test_demo_contestants_are_never_probed(svc):
    m = add_gguf(svc, "demo")
    svc.runner.client_factory = mock_client_factory(lambda req: (_ for _ in ()).throw(AssertionError("a demo model must not be probed")))
    assert svc.runner.detect_reasoning(m, url="http://x:1", api="openai") == (None, "")


# ---- the request the runner builds ----------------------------------------------------------------------------------------------------------

def test_the_setting_exists_with_a_default_and_zero_switches_it_off(svc):
    assert svc.settings.get("runner.reasoning_tokens") == EXTRA
    svc.settings.set_many({"runner.reasoning_tokens": 0})
    assert svc.settings.get("runner.reasoning_tokens") == 0
    with pytest.raises(GaltonError):
        svc.settings.set_many({"runner.reasoning_tokens": -1})
    described = {s["key"]: s for s in svc.settings.describe()}
    assert described["runner.reasoning_tokens"]["default"] == EXTRA


def test_the_setting_can_be_changed_through_the_settings_tool(svc):
    tool(svc, "settings_set", values={"runner.reasoning_tokens": 2048})
    assert svc.settings.get("runner.reasoning_tokens") == 2048


def test_a_reasoning_model_gets_the_answer_budget_plus_the_reasoning_tokens(svc):
    expected = shifted(svc, EXTRA)
    m = mark_reasoning(svc, add_gguf(svc, "piensa"))
    svc.fake_world.register("piensa", thinker())
    run_inline(svc, ["rapida"], [m["id"]])
    assert sorted(sent(svc)) == expected and min(expected) > EXTRA
    assert all(call.reasoning_tokens == EXTRA for call in svc.fake_world.calls)


def test_the_run_budget_is_the_answer_part_and_the_reasoning_tokens_come_on_top(svc):
    expected = shifted(svc, EXTRA, max_tokens=300)
    m = mark_reasoning(svc, add_gguf(svc, "piensa"))
    run_inline(svc, ["rapida"], [m["id"]], max_tokens=300)
    assert sorted(sent(svc)) == expected


def test_the_reasoning_tokens_setting_is_used_and_zero_disables_it(svc):
    m = mark_reasoning(svc, add_gguf(svc, "piensa"))
    svc.settings.set_many({"runner.reasoning_tokens": 2048})
    expected = shifted(svc, 2048, max_tokens=300)
    run_inline(svc, ["rapida"], [m["id"]], max_tokens=300)
    assert sorted(sent(svc)) == expected
    svc.fake_world.calls.clear()
    svc.settings.set_many({"runner.reasoning_tokens": 0})
    expected = shifted(svc, 0, max_tokens=300)
    run_inline(svc, ["rapida"], [m["id"]], max_tokens=300)
    assert sorted(sent(svc)) == expected and not any(call.reasoning_tokens for call in svc.fake_world.calls)


def test_a_bigger_effort_asks_for_a_bigger_allowance(svc):
    expected = shifted(svc, 16384, max_tokens=300, effort="max")
    m = mark_reasoning(svc, add_gguf(svc, "piensa"))
    run_inline(svc, ["rapida"], [m["id"]], max_tokens=300, effort="max")
    assert sorted(sent(svc)) == expected


def test_effort_off_gets_no_reasoning_room(svc):
    expected = shifted(svc, 0, max_tokens=300, effort="off")
    m = mark_reasoning(svc, add_gguf(svc, "piensa"))
    run_inline(svc, ["rapida"], [m["id"]], max_tokens=300, effort="off")
    assert sorted(sent(svc)) == expected and all(call.effort == "off" and call.reasoning_tokens == 0 for call in svc.fake_world.calls)


def test_a_model_that_does_not_reason_keeps_its_budget(svc):
    m = add_gguf(svc, "directo")
    run_inline(svc, ["rapida"], [m["id"]], max_tokens=300)
    assert not any(call.reasoning_tokens for call in svc.fake_world.calls) and max(sent(svc)) < EXTRA


def test_the_timeout_grows_with_the_thinking_room(svc):
    thinking = mark_reasoning(svc, add_gguf(svc, "piensa"))
    plain = add_gguf(svc, "directo")
    run_inline(svc, ["rapida"], [thinking["id"], plain["id"]], max_tokens=300, timeout_s=60)
    by = {}
    for call in svc.fake_world.calls:
        by.setdefault(call.reasoning_tokens, []).append(call.timeout_s)
    assert min(by[EXTRA]) - min(by[0]) == pytest.approx(EXTRA * 0.1)


def test_the_run_warns_that_the_budget_was_widened(svc):
    m = mark_reasoning(svc, add_gguf(svc, "piensa"))
    run = run_inline(svc, ["rapida"], [m["id"]])
    shown = json.dumps(svc.store.run_contestant(run["id"], m["id"])["warnings"], ensure_ascii=False)
    assert str(EXTRA) in shown and "reasons" in shown


# ---- a model that was not known to reason --------------------------------------------------------------------------------------------------

def test_an_unknown_model_cut_while_thinking_is_asked_again_with_room_and_remembered(svc):
    base = reference_responder(references())

    def respond(req):
        if req.reasoning_tokens == 0:
            return {"text": "", "reasoning": "pienso " * 50, "finish_reason": "length"}
        return {"text": base(req) if isinstance(base(req), str) else base(req)["text"], "reasoning": "pienso", "finish_reason": "stop"}
    m = add_gguf(svc, "sorpresa", responder=respond)
    run = run_inline(svc, ["rapida"], [m["id"]], max_tokens=300)
    rows = svc.store.results(run_id=run["id"])
    assert rows and not any(r["truncated"] for r in rows) and all(r["detail"]["max_tokens"] > EXTRA for r in rows)
    assert str(EXTRA) in json.dumps(rows[0]["detail"]["notes"])
    assert svc.store.contestant(m["id"])["meta"]["reasons"] is True
    # the next run knows it and does not need the second try
    svc.fake_world.calls.clear()
    run_inline(svc, ["rapida"], [m["id"]], max_tokens=300)
    assert all(t >= 300 + EXTRA for t in sent(svc)) and len(sent(svc)) == 12


def test_a_model_that_never_reasons_is_not_asked_twice(svc):
    m = add_gguf(svc, "mudo", responder=lambda req: {"text": "", "finish_reason": "length"})
    run_inline(svc, ["rapida"], [m["id"]], max_tokens=300)
    assert max(sent(svc)) < EXTRA and len(sent(svc)) == 12


def test_effort_off_is_not_asked_twice_either(svc):
    m = add_gguf(svc, "sigue", responder=lambda req: {"text": "", "reasoning": "pienso", "finish_reason": "length"})
    run_inline(svc, ["rapida"], [m["id"]], max_tokens=300, effort="off")
    assert max(sent(svc)) < EXTRA and len(sent(svc)) == 12


# ---- truncated results ----------------------------------------------------------------------------------------------------------------------

def test_a_result_cut_while_thinking_is_flagged_truncated(svc):
    m = mark_reasoning(svc, add_gguf(svc, "corta"))
    svc.fake_world.register("corta", thinker(share_cut=0.25))
    run = run_inline(svc, ["rapida"], [m["id"]])
    rows = svc.store.results(run_id=run["id"])
    cut = [r for r in rows if r["truncated"]]
    assert len(cut) == 3 and all(not r["passed"] and r["detail"].get("truncated") and r["detail"]["finish_reason"] == "length" for r in cut)
    assert not any(r["truncated"] for r in rows if r["passed"])
    assert svc.store.count_results(run_id=run["id"])["truncated"] == 3


def test_an_empty_answer_that_ended_normally_is_not_truncated(svc):
    m = add_gguf(svc, "calla", responder=lambda req: {"text": "", "finish_reason": "stop"})
    rows = svc.store.results(run_id=run_inline(svc, ["rapida"], [m["id"]])["id"])
    assert rows and all(r["detail"].get("empty_answer") and not r["truncated"] for r in rows)


def test_the_run_page_counts_truncated_results_per_contestant(svc):
    m = mark_reasoning(svc, add_gguf(svc, "corta"))
    svc.fake_world.register("corta", thinker(share_cut=0.25))
    run = run_inline(svc, ["rapida"], [m["id"]])
    card = svc.run_card(run)
    assert card["contestants"][0]["truncated"] == 3
    assert run["summary"]["contestants"][m["id"]]["truncated"] == 3
    cards = [svc.result_card(r, {}, {}) for r in svc.store.results(run_id=run["id"])]
    assert sum(1 for c in cards if c["truncated"]) == 3


# ---- the share that triggers a warning --------------------------------------------------------------------------------------------------------

def test_truncation_counts_per_key_and_warns_above_five_percent():
    rows = [{"category": "a", "truncated": 1}] + [{"category": "a", "truncated": 0}] * 19 + [{"category": "b", "truncated": 1}] + [{"category": "b", "truncated": 0}] * 9
    out = truncation(rows)
    assert TRUNCATED_WARN_SHARE == 0.05
    assert out["a"] == {"results": 20, "truncated": 1, "share": 0.05, "warn": False}       # exactly 5 % is not "more than 5 %"
    assert out["b"]["warn"] is True and out["b"]["truncated"] == 1 and out["b"]["results"] == 10


def test_the_leaderboard_warns_about_a_model_cut_too_often(svc):
    cut = mark_reasoning(svc, add_gguf(svc, "corta"))
    fine = mark_reasoning(svc, add_gguf(svc, "entera", seed=1))
    svc.fake_world.register("corta", thinker(share_cut=0.25))
    svc.fake_world.register("entera", thinker())
    run_inline(svc, ["rapida"], [cut["id"], fine["id"]])
    rows = {r["name"]: r for r in svc.board.leaderboard()["rows"]}
    assert rows["corta"]["truncated"] == 3 and rows["entera"]["truncated"] == 0
    assert rows["entera"]["warnings"] == [] and len(rows["corta"]["warnings"]) >= 1
    assert "3 of 12" in json.dumps(rows["corta"]["warnings"], ensure_ascii=False) and "too small" in str(rows["corta"]["warnings"][0])
    category = next(iter(rows["corta"]["by_category"]))
    assert rows["corta"]["by_category"][category]["truncated"] == 3 and rows["entera"]["by_category"][category]["truncated"] == 0


def test_the_routes_and_the_recommendation_carry_the_warning(svc):
    cut = mark_reasoning(svc, add_gguf(svc, "corta"))
    svc.fake_world.register("corta", thinker(share_cut=0.25))
    run_inline(svc, ["rapida", "razonamiento", "matematicas", "instrucciones"], [cut["id"]])
    view = svc.routes.get()
    warned = {cat: d["warnings"] for cat, d in view["detail"].items() if d.get("warnings")}
    assert warned, view["detail"].keys()
    entry = next(iter(warned.values()))[0]
    assert entry["name"] == "corta" and entry["truncated"] >= 1 and entry["results"] >= entry["truncated"] and entry["why"]
    assert any(item.get("truncated") for d in view["detail"].values() for item in d.get("ranked", []))


def test_a_clean_run_has_no_warnings(svc, measured):
    assert all(r["warnings"] == [] and r["truncated"] == 0 for r in svc.board.leaderboard()["rows"])
    view = svc.routes.get()
    assert not any(d.get("warnings") for d in view["detail"].values())


# ---- the stored schema ----------------------------------------------------------------------------------------------------------------------

def test_migration_two_adds_the_column_and_flags_old_cut_results(tmp_path):
    path = tmp_path / "old.db"
    raw = sqlite3.connect(path)
    raw.executescript(MIGRATIONS[0])
    raw.execute("CREATE TABLE schema_version (version INTEGER NOT NULL)")
    raw.execute("INSERT INTO schema_version(version) VALUES (1)")
    cols = [r[1] for r in raw.execute("PRAGMA table_info(results)")]
    assert "truncated" not in cols
    base = {"run_id": "r", "contestant_id": "c", "suite_id": "s", "case_id": "k", "category": "reasoning", "repeat": 0, "score": 0, "passed": 0, "ts": 1.0}

    def insert(detail):
        raw.execute("INSERT INTO results(%s) VALUES (%s)" % (", ".join([*base, "detail"]), ", ".join("?" * (len(base) + 1))), [*base.values(), detail])
    raw.execute("PRAGMA foreign_keys=OFF")
    insert(json.dumps({"empty_answer": True, "finish_reason": "length"}))
    insert(json.dumps({"empty_answer": True, "finish_reason": "stop"}))
    insert(json.dumps({"finish_reason": "length"}))
    raw.commit()
    raw.close()
    db = Database(path)
    try:
        assert db.version() == len(MIGRATIONS)
        assert [r["truncated"] for r in db.query("SELECT truncated FROM results ORDER BY id")] == [1, 0, 0]
    finally:
        db.conn.close()
