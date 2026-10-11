"""Simulated users: cases, scoring, the conversation loop and the HTTP targets, all with fake simulators and fake servers."""
from __future__ import annotations

import json

import httpx
import pytest

from galton_hoard.simusers import (FaustusTarget, OpenAISimulator, OpenAITarget, asked_question, load_cases, persona_prompt, run_case,
                                   run_family, summarise)
from galton_hoard.simusers.cases import PERSONAS, Case


def test_starter_cases_are_well_formed_and_references_pass():
    cases = load_cases()
    assert 6 <= len(cases) <= 12
    assert len({c.id for c in cases}) == len(cases)
    assert {c.persona for c in cases} == set(PERSONAS)
    assert any(c.ambiguous for c in cases) and any(not c.ambiguous for c in cases)
    for c in cases:
        assert c.check(c.reference)["passed"], c.id
        assert not c.check("No lo se.")["passed"], c.id
        assert persona_prompt(c).count(c.hidden_goal) == 1
        assert c.max_turns >= 2


def test_only_filters_cases():
    assert [c.id for c in load_cases(only=["hurry-speed"])] == ["hurry-speed"]


def test_asked_question():
    assert asked_question("¿Cuánto cobras?") and asked_question("how much?")
    assert not asked_question("Te sobran 800 euros.")


def scripted(*replies):
    it = iter(replies)
    return lambda history: next(it)


def sim_says(*messages):
    it = iter(messages)
    seen = []

    def sim(system, history):
        seen.append((system, history))
        return next(it)
    sim.seen = seen
    return sim


def test_ambiguous_case_with_clarification_scores_full():
    case = load_cases(only=["vague-savings"])[0]
    target = scripted("¿Cuánto cobras al mes y qué gastos fijos tienes?", "Te sobran 800 euros al mes.")
    sim = sim_says("Cobro 2400, alquiler 900 y 700 de otras cosas", "[FIN]")
    res = run_case(case, target, sim)
    assert res.goal_reached and res.asked_clarifying and res.turns == 2 and res.ended == "simulator_done"
    assert res.clarify_score == 1.0 and res.score > 0.9
    system, history = sim.seen[0]
    assert case.hidden_goal in system
    assert history[0] == {"role": "assistant", "content": case.opening}      # the persona sees itself as the assistant
    assert history[1]["role"] == "user"


def test_ambiguous_case_guessed_without_asking_fails_the_goal():
    case = load_cases(only=["vague-savings"])[0]
    res = run_case(case, scripted("Depende, ahorra el 20 %.", "Ahorra lo que puedas."), sim_says("[FIN]"))
    assert not res.goal_reached and not res.asked_clarifying
    assert res.clarify_score == 0.0 and res.score == 0.0


def test_clear_case_answered_at_once_and_needless_question_costs_half():
    case = load_cases(only=["hurry-speed"])[0]
    good = run_case(case, scripted("20"), sim_says("[FIN]"))
    assert good.goal_reached and good.turns == 1 and good.clarify_score == 1.0 and good.score == 1.0
    chatty = run_case(case, scripted("¿Quieres más decimales?", "20"), sim_says("No.", "[FIN]"))
    assert chatty.goal_reached and chatty.clarify_score == 0.5 and chatty.score < good.score
    assert summarise([good, chatty])["overall"]["needless_questions"] == 1


def test_self_correction_is_judged_on_the_final_answer():
    case = load_cases(only=["selfcorrect-sum"])[0]
    stale = run_case(case, scripted("195", "Perdón: sigue siendo 195."), sim_says("Me equivoqué, era 54 y no 45", "[FIN]"))
    assert not stale.goal_reached
    fixed = run_case(case, scripted("195", "Con 54 la suma es 204."), sim_says("Me equivoqué, era 54 y no 45", "[FIN]"))
    assert fixed.goal_reached and fixed.turns == 2


def test_turn_limit_and_turn_score():
    case = Case(id="x", persona="vague", opening="hola", hidden_goal="g", checker={"type": "contains", "all": ["ok"]}, reference="ok", max_turns=3)
    res = run_case(case, scripted("a", "b", "c", "d"), sim_says("uno", "dos", "tres"))
    assert res.ended == "max_turns" and res.turns == 3 and not res.goal_reached and res.turn_score == 0.0
    slow = run_case(case, scripted("a", "b", "ok"), sim_says("uno", "dos", "tres"))
    assert slow.goal_reached and slow.turn_score == 0.0 and slow.turns == 3
    quick = run_case(case, scripted("a", "ok"), sim_says("uno", "[FIN]"))
    assert quick.turn_score == 0.5


def test_errors_are_scored_not_raised():
    case = load_cases(only=["hurry-speed"])[0]

    def broken(history):
        raise RuntimeError("down")
    res = run_case(case, broken, sim_says())
    assert res.ended == "target_error" and res.error == "down" and not res.goal_reached and res.score == 0.0

    def bad_sim(system, history):
        raise RuntimeError("sim down")
    res2 = run_case(case, scripted("no sé"), bad_sim)
    assert res2.ended == "simulator_error" and "sim down" in res2.error


def test_run_family_uses_a_fresh_target_per_case_and_summarises():
    made = []

    def factory():
        made.append(1)
        return scripted("20" if len(made) == 1 else "45 dias")

    cases = load_cases(only=["hurry-speed", "hurry-days-between"])
    got = []
    results = run_family(cases, factory, sim_says("[FIN]", "[FIN]"), on_result=got.append)
    assert len(made) == 2 and len(got) == 2 and all(r.goal_reached for r in results)
    s = summarise(results)
    assert s["overall"]["goal_rate"] == 1.0 and s["by_persona"]["hurry"]["cases"] == 2
    assert s["overall"]["clarify_rate_ambiguous"] is None


def sse(*events):
    return "".join(f"data: {json.dumps(e)}\n\n" for e in events) + "data: [DONE]\n\n"


def fake_faustus(auth_disabled=True, answers=("Te sobran 800 euros.",)):
    calls = {"forms": [], "login": 0, "n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path == "/api/auth/status":
            return httpx.Response(200, json={"auth_disabled": auth_disabled})
        if path == "/api/auth/login":
            calls["login"] += 1
            return httpx.Response(200, json={"ok": True})
        if path == "/api/auth/settings":
            return httpx.Response(200, json={"default_model": "m1"})
        if path == "/api/session":
            return httpx.Response(200, json={"id": "s1"})
        if path == "/api/chat_stream":
            calls["forms"].append(request.content.decode())
            text = answers[min(calls["n"], len(answers) - 1)]
            calls["n"] += 1
            return httpx.Response(200, text=sse({"delta": "pensando", "thinking": True}, {"delta": text[:5]}, {"delta": text[5:]}))
        return httpx.Response(404)
    return handler, calls


def test_faustus_target_no_auth_streams_text_and_keeps_one_session():
    handler, calls = fake_faustus()
    t = FaustusTarget(transport=httpx.MockTransport(handler))
    assert t([{"role": "user", "content": "hola"}]) == "Te sobran 800 euros."
    t([{"role": "user", "content": "hola"}, {"role": "assistant", "content": "x"}, {"role": "user", "content": "segunda"}])
    assert calls["login"] == 0 and len(calls["forms"]) == 2
    assert "session=s1" in calls["forms"][1] and "segunda" in calls["forms"][1] and "model=m1" in calls["forms"][1]


def test_faustus_target_logs_in_when_auth_is_on_and_explains_missing_credentials():
    handler, calls = fake_faustus(auth_disabled=False)
    FaustusTarget(username="u", password="p", transport=httpx.MockTransport(handler))([{"role": "user", "content": "hola"}])
    assert calls["login"] == 1
    with pytest.raises(RuntimeError, match="sign-in"):
        FaustusTarget(transport=httpx.MockTransport(handler))([{"role": "user", "content": "hola"}])


def test_faustus_target_reports_unreachable_and_error_events():
    def down(request):
        raise httpx.ConnectError("refused")
    with pytest.raises(RuntimeError, match="cannot reach Faustus"):
        FaustusTarget(transport=httpx.MockTransport(down))([{"role": "user", "content": "x"}])

    def failing(request):
        if request.url.path == "/api/chat_stream":
            return httpx.Response(200, text=sse({"type": "error", "error": "no model"}))
        return fake_faustus()[0](request)
    with pytest.raises(RuntimeError, match="no model"):
        FaustusTarget(transport=httpx.MockTransport(failing))([{"role": "user", "content": "x"}])


def chat_sse(text):
    return httpx.Response(200, text=sse({"choices": [{"delta": {"content": text}, "finish_reason": "stop"}]}))


def test_openai_target_and_simulator_send_the_history():
    bodies = []

    def handler(request):
        bodies.append(json.loads(request.content))
        return chat_sse("hola de vuelta")
    tgt = OpenAITarget("http://x", "m", system="sé breve", transport=httpx.MockTransport(handler))
    assert tgt([{"role": "user", "content": "hola"}]) == "hola de vuelta"
    assert bodies[0]["messages"][0] == {"role": "system", "content": "sé breve"} and bodies[0]["messages"][1]["content"] == "hola"
    sim = OpenAISimulator("http://x", "m", transport=httpx.MockTransport(handler))
    assert sim("persona", [{"role": "assistant", "content": "a"}]) == "hola de vuelta"
    assert bodies[1]["messages"][0]["content"] == "persona"

    def server_error(request):
        return httpx.Response(500, text="boom")
    with pytest.raises(RuntimeError, match="500"):
        OpenAITarget("http://x", "m", transport=httpx.MockTransport(server_error))([{"role": "user", "content": "x"}])
