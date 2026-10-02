"""The runner with the fake world: progress, cancel, errors, skips, settings that reach the request, GPU and server cleanup, judge hand-over."""

from __future__ import annotations

import pytest

from galton_hoard.errors import GaltonError
from galton_hoard.fakes import FakeWorld, reference_responder
from galton_hoard.runner import Runner, normalise_settings
from helpers import Clock, add_gguf, build_services, json_response, mock_client_factory, references, run_inline


def results(svc, run_id, **kw):
    return svc.store.results(run_id=run_id, **kw)


def test_a_perfect_model_passes_every_quick_case(svc):
    m = add_gguf(svc, "perfecto")
    run = run_inline(svc, ["rapida"], [m["id"]])
    rows = results(svc, run["id"])
    assert run["state"] == "done" and len(rows) == 12 and all(r["passed"] for r in rows)
    rc = svc.store.run_contestant(run["id"], m["id"])
    assert rc["state"] == "done" and rc["done"] == rc["total"] == 12 and rc["runs_on"].startswith("llama.cpp (own)") and rc["vram_mb"] > 0 and rc["load_ms"] == 1234.0
    assert run["summary"]["contestants"][m["id"]]["mean_score"] == 1.0


def test_results_keep_timings_and_the_digest(svc):
    m = add_gguf(svc, "tiempos", tps=50.0)
    row = results(svc, run_inline(svc, ["rapida"], [m["id"]])["id"])[0]
    assert row["decode_tps"] == 50.0 and row["ttft_ms"] == 150.0 and row["digest"] == "d-tiempos" and row["latency_ms"] > 150 and row["completion_tokens"] >= 1


def test_wrong_answers_are_scored_zero_not_errors(svc):
    m = add_gguf(svc, "torpe", accuracy=0.0)
    rows = results(svc, run_inline(svc, ["rapida"], [m["id"]])["id"])
    assert not any(r["passed"] for r in rows) and not any(r["error"] for r in rows)


def test_repeats_multiply_the_results(svc):
    m = add_gguf(svc, "repite")
    run = run_inline(svc, ["rapida"], [m["id"]], repeats=2)
    assert len(results(svc, run["id"])) == 24 and {r["repeat"] for r in results(svc, run["id"])} == {0, 1}


def test_backend_errors_are_stored_as_errors_and_excluded_from_scoring(svc):
    svc.fake_world.register("roto", lambda req: {"error": "HTTP 500 from the server"})
    m = svc.store.create_contestant(key="gguf:/t/roto.gguf", kind="gguf", name="roto", path="/t/roto.gguf", provider="llamacpp", aliases=["roto"], digest="d", size_bytes=3 * 1024 ** 3,
                                    context=8192, meta={"demo": True})
    run = run_inline(svc, ["rapida"], [m["id"]])
    rows = results(svc, run["id"])
    assert run["state"] == "done" and all(r["error"] == "HTTP 500 from the server" and not r["passed"] for r in rows)
    assert svc.store.scoring_rows(contestant_ids=[m["id"]]) == []


def test_empty_answers_are_flagged(svc):
    svc.fake_world.register("mudo", lambda req: "")
    m = svc.store.create_contestant(key="gguf:/t/mudo.gguf", kind="gguf", name="mudo", path="/t/mudo.gguf", provider="llamacpp", aliases=["mudo"], digest="d", size_bytes=2 * 1024 ** 3,
                                    context=8192, meta={"demo": True})
    rows = results(svc, run_inline(svc, ["rapida"], [m["id"]])["id"])
    assert all(r["detail"].get("empty_answer") for r in rows)


def test_settings_reach_the_request(svc):
    m = add_gguf(svc, "ajustes")
    run_inline(svc, ["rapida"], [m["id"]], temperature=0.7, effort="high", max_tokens=77, seed=5, timeout_s=33)
    req = svc.fake_world.calls[0]
    assert req.temperature == 0.7 and req.effort == "high" and req.max_tokens == 77 and req.seed == 5 and req.timeout_s >= 33


def test_the_suite_default_applies_when_the_run_sets_no_max_tokens(svc):
    m = add_gguf(svc, "defecto")
    run_inline(svc, ["rapida"], [m["id"]])
    assert svc.fake_world.calls[0].max_tokens == svc.store.find_suite("rapida")["max_tokens"]


def test_tool_cases_send_the_tools(svc):
    m = add_gguf(svc, "herramientas")
    run_inline(svc, ["herramientas"], [m["id"]])
    assert any(call.tools for call in svc.fake_world.calls)


def test_vision_cases_are_left_out_for_models_without_eyes(svc):
    blind, sighted = add_gguf(svc, "ciego"), add_gguf(svc, "vidente", vision=True)
    run = run_inline(svc, ["vision"], [blind["id"], sighted["id"]])
    assert len(results(svc, run["id"], contestant_id=blind["id"])) == 0
    assert len(results(svc, run["id"], contestant_id=sighted["id"])) == 20
    assert any("no vision" in w for w in svc.store.run_contestant(run["id"], blind["id"])["warnings"])
    assert svc.fake_world.calls[0].images


def test_cases_that_do_not_fit_the_context_are_skipped_with_a_reason(svc):
    m = add_gguf(svc, "corto")
    run = run_inline(svc, ["contexto-largo"], [m["id"]], context=4096)
    rows = results(svc, run["id"])
    skipped = [r for r in rows if r["skipped"]]
    assert skipped and all("context" in r["error"] for r in skipped) and len(rows) == 20
    assert svc.store.scoring_rows(contestant_ids=[m["id"]]) != [] or len(skipped) == 20


def test_gpu_lease_and_server_are_released_after_the_run(svc):
    m = add_gguf(svc, "limpio")
    run_inline(svc, ["rapida"], [m["id"]])
    fake = svc.fake_gpus
    assert fake.leases and len(fake.released) == len(fake.leases)
    assert svc.fake_world.stopped == ["limpio"] and len(svc.fake_world.started) == 1


def test_only_allowed_gpus_are_used(svc):
    m = add_gguf(svc, "permitido", size_gb=3.0)
    run = run_inline(svc, ["rapida"], [m["id"]])
    assert set(svc.store.run_contestant(run["id"], m["id"])["gpus"]) <= {2, 3}
    assert {lease.gpu for lease in svc.fake_gpus.leases} <= {2, 3}


def test_a_big_model_is_split_and_says_so(svc):
    m = add_gguf(svc, "enorme", size_gb=22.0)
    rc = svc.store.run_contestant(run_inline(svc, ["rapida"], [m["id"]])["id"], m["id"])
    assert sorted(rc["gpus"]) == [2, 3] and any("split" in w for w in rc["warnings"])


def test_a_model_that_cannot_fit_fails_alone_with_the_reason(svc):
    huge, fine = add_gguf(svc, "inmenso", size_gb=80.0), add_gguf(svc, "bueno")
    run = run_inline(svc, ["rapida"], [huge["id"], fine["id"]])
    bad = svc.store.run_contestant(run["id"], huge["id"])
    assert bad["state"] == "failed" and "hold" in bad["error"] and run["state"] == "done"
    assert svc.store.run_contestant(run["id"], fine["id"])["state"] == "done"


def test_when_every_contestant_fails_the_run_fails(svc):
    huge = add_gguf(svc, "inmenso", size_gb=80.0)
    created = svc.runner.create(suites=["rapida"], contestants=[huge["id"]], source="test")
    svc.runner.execute(created["id"])
    run = svc.store.run(created["id"])
    assert run["state"] == "failed" and run["error"].key == "run_all_failed"
    reason = svc.store.run_contestant(created["id"], huge["id"])["error"]
    assert reason.key == "gpu_capacity" and "GB" in reason


def test_starting_a_run_that_cannot_fit_anywhere_is_refused_early(svc):
    huge = add_gguf(svc, "inmenso", size_gb=80.0)
    with pytest.raises(GaltonError) as exc:
        svc.start_run(suites=["rapida"], contestants=[huge["id"]], source="test")
    assert "GB" in exc.value.message


def test_a_launcher_failure_is_reported_and_the_lease_is_returned(svc):
    m = add_gguf(svc, "no-arranca")
    svc.fake_world.fail_start["no-arranca"] = "llama-server exited with code 1"
    run = run_inline(svc, ["rapida"], [m["id"]])
    assert "exited with code 1" in svc.store.run_contestant(run["id"], m["id"])["error"] and len(svc.fake_gpus.released) == len(svc.fake_gpus.leases)


def test_no_gpu_right_now_fails_fast_when_the_run_does_not_wait(svc):
    svc.fake_gpus.refuse = {2, 3}
    m = add_gguf(svc, "espera")
    run = run_inline(svc, ["rapida"], [m["id"]], wait_s=0)
    assert svc.store.run_contestant(run["id"], m["id"])["state"] == "failed"


def test_remote_servers_are_not_called_without_opt_in(svc):
    c = svc.store.create_contestant(key="server:remote", kind="server", name="nube", url="https://api.example.com", api="openai", model="m", provider="remote", remote=True,
                                    remote_ok=False, aliases=["nube"], digest="x")
    run = run_inline(svc, ["rapida"], [c["id"]])
    assert "remote" in svc.store.run_contestant(run["id"], c["id"])["error"] and svc.fake_world.calls == []


def test_cancel_in_the_middle_keeps_partial_results(tmp_path):
    world = FakeWorld()
    svc = build_services(tmp_path, world=world)
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
    final = svc.runner.execute(created["id"])
    rows = svc.store.results(run_id=created["id"])
    assert final["state"] == "cancelled" and 0 < len(rows) < 12
    assert svc.store.run_contestant(created["id"], m["id"])["state"] == "cancelled" and len(svc.fake_gpus.released) == len(svc.fake_gpus.leases)
    svc.db.close()


def test_cancelling_a_queued_run_never_starts_it(svc):
    m = add_gguf(svc, "pendiente")
    created = svc.runner.create(suites=["rapida"], contestants=[m["id"]], source="test")
    assert svc.runner.cancel(created["id"])["state"] == "cancelled"
    assert svc.runner.execute(created["id"])["state"] == "cancelled" and svc.fake_world.calls == []
    assert svc.runner.cancel(created["id"])["cancelled"] is False


def test_create_validates_the_selection(svc):
    m = add_gguf(svc, "valido")
    with pytest.raises(GaltonError, match="Choose at least one model"):
        svc.runner.create(suites=["rapida"], contestants=[])
    with pytest.raises(GaltonError):
        svc.runner.create(suites=["no-existe"], contestants=[m["id"]])
    svc.store.update_contestant(m["id"], enabled=False)
    with pytest.raises(GaltonError, match="disabled"):
        svc.runner.create(suites=["rapida"], contestants=[m["id"]])
    svc.store.update_contestant(m["id"], enabled=True, missing=True)
    with pytest.raises(GaltonError, match="no longer installed"):
        svc.runner.create(suites=["rapida"], contestants=[m["id"]])


@pytest.mark.parametrize("raw,fragment", [({"temperature": 5}, "temperature"), ({"repeats": 0}, "repeats"), ({"bogus": 1}, "Unknown run settings"), ({"effort": "extreme"}, "effort"),
                                          ({"max_tokens": "many"}, "number")])
def test_settings_are_validated(raw, fragment):
    with pytest.raises(GaltonError, match=fragment):
        normalise_settings(raw)


def test_settings_defaults():
    s = normalise_settings(None)
    assert s["temperature"] == 0.0 and s["repeats"] == 1 and s["effort"] is None and s["seed"] is None


# ------------------------------------------------------------------ judge
JUDGE_SUITE_CASE = {"title": "Explica la fotosíntesis", "prompt": "Explica la fotosíntesis en dos frases.", "checker": {"type": "judge", "rubric": "Es correcta y clara."}}


def make_judge_suite(svc):
    suite = svc.store.create_suite(name="Con juez", description="", category="custom", builtin=False, max_tokens=200, position=100)
    from galton_hoard.suites import normalise_case
    case = normalise_case(JUDGE_SUITE_CASE)
    svc.store.create_case(suite_id=suite["id"], position=0, source="user", **case)
    return suite


def test_judge_cases_wait_when_there_is_no_judge_and_resolve_later(svc):
    subject = add_gguf(svc, "sujeto", responder=lambda req: "Las plantas convierten luz en azúcar. Liberan oxígeno.")
    suite = make_judge_suite(svc)
    run = run_inline(svc, [suite["id"]], [subject["id"]])
    row = results(svc, run["id"])[0]
    assert row["judge_pending"] and not row["passed"]
    assert svc.store.scoring_rows(contestant_ids=[subject["id"]]) == []
    assert svc.runner.judge_pending()["reason"].startswith("no judge model")
    judge = add_gguf(svc, "juez", responder=lambda req: '{"score": 9, "reasons": "correcto"}')
    svc.settings.set_many({"judge.contestant": judge["id"]})
    outcome = svc.runner.judge_pending()
    graded = results(svc, run["id"])[0]
    assert outcome["graded"] == 1 and not graded["judge_pending"] and graded["passed"] and graded["score"] == 0.9 and not graded["self_judged"]


def test_a_model_judging_itself_is_flagged_and_counts_less(svc):
    me = add_gguf(svc, "yo", responder=lambda req: '{"score": 10, "reasons": "perfecto"}')
    svc.settings.set_many({"judge.contestant": me["id"]})
    suite = make_judge_suite(svc)
    row = results(svc, run_inline(svc, [suite["id"]], [me["id"]])["id"])[0]
    assert row["self_judged"] and row["confidence"] == 0.5 and not row["judge_pending"]


def test_judge_answers_are_cached_by_content(svc):
    judge = add_gguf(svc, "juez", responder=lambda req: '{"score": 8, "reasons": "bien"}')
    svc.settings.set_many({"judge.contestant": judge["id"]})
    subject = add_gguf(svc, "sujeto", responder=lambda req: "Una respuesta fija.")
    suite = make_judge_suite(svc)
    run_inline(svc, [suite["id"]], [subject["id"]])
    asked = len([c for c in svc.fake_world.calls if "rubric" in str(c.messages).lower() or "puntu" in str(c.messages).lower()])
    run_inline(svc, [suite["id"]], [subject["id"]])
    asked_again = len([c for c in svc.fake_world.calls if "rubric" in str(c.messages).lower() or "puntu" in str(c.messages).lower()])
    assert asked_again == asked


# ------------------------------------------------------------------ the judge shares the server being measured
JUDGE_REPLY = '{"score": 9, "reasons": "bien"}'


def is_judge_call(req) -> bool:
    return "evaluador" in str(req.messages[0].get("content", "")) if req.messages else False


def judge_suite(svc, n=4):
    from galton_hoard.suites import normalise_case
    suite = svc.store.create_suite(name="Con juez", description="", category="custom", builtin=False, max_tokens=200, position=100)
    for i in range(n):
        case = normalise_case({"title": f"Explica el tema {i}", "prompt": f"Explica el tema número {i} en dos frases.", "checker": {"type": "judge", "rubric": "Es correcta y clara."}})
        svc.store.create_case(suite_id=suite["id"], position=i, source="user", **case)
    return suite


def kinds(world):
    return ["judge" if is_judge_call(c) else "case" for c in world.calls]


def test_a_model_judging_itself_is_graded_after_its_cases_not_between_them(tmp_path):
    """The judge is the same llama-server: its requests must not run beside the questions whose latency and first-token time are recorded."""
    world = FakeWorld(delay_s=0.03)
    svc = build_services(tmp_path, world=world)
    me = add_gguf(svc, "yo", responder=lambda req: JUDGE_REPLY if is_judge_call(req) else "Una explicación breve. Y otra frase.")
    svc.settings.set_many({"judge.contestant": me["id"]})
    suite = judge_suite(svc, 4)
    run = run_inline(svc, [suite["id"]], [me["id"]])
    assert run["state"] == "done"
    assert kinds(world) == ["case"] * 4 + ["judge"] * 4, "no grading request between two questions"
    rows = results(svc, run["id"])
    assert len(rows) == 4 and all(r["self_judged"] and r["confidence"] == 0.5 and not r["judge_pending"] and r["score"] == 0.9 and r["passed"] for r in rows)
    assert run["summary"]["pending_judge"] == 0
    svc.db.close()


def test_the_judge_runs_after_each_contestants_cases_and_before_the_next_model(tmp_path):
    world = FakeWorld(delay_s=0.02)
    svc = build_services(tmp_path, world=world)
    me = add_gguf(svc, "yo", responder=lambda req: JUDGE_REPLY if is_judge_call(req) else "Mi respuesta.")
    other = add_gguf(svc, "otro", responder=lambda req: "La del otro.")
    svc.settings.set_many({"judge.contestant": me["id"]})
    suite = judge_suite(svc, 3)
    run = run_inline(svc, [suite["id"]], [me["id"], other["id"]])
    assert run["state"] == "done"
    mine = [r for r in results(svc, run["id"]) if r["contestant_id"] == me["id"]]
    theirs = [r for r in results(svc, run["id"]) if r["contestant_id"] == other["id"]]
    assert len(mine) == len(theirs) == 3 and all(r["self_judged"] and not r["judge_pending"] for r in mine)
    # the other model is judged by "yo", whose server is not the one being measured: it waits for the end of the run
    assert all(not r["judge_pending"] and not r["self_judged"] and r["score"] == 0.9 for r in theirs)
    assert kinds(world)[:6] == ["case"] * 3 + ["judge"] * 3, "the first model's grades come before the second model's questions"
    svc.db.close()


def server_contestant(svc, name, model, url):
    return svc.store.create_contestant(key=f"server:{name}", kind="server", name=name, url=url, api="openai", model=model, provider="openai_compat", aliases=[name, model],
                                       digest=f"d-{name}", source="manual", context=8192)


def llama_server_world(models):
    def handler(req):
        path = req.url.path
        if path == "/v1/models":
            return json_response({"data": [{"id": m} for m in models]})
        if path == "/props":
            return json_response({"default_generation_settings": {"n_ctx": 8192}})
        if path == "/slots":
            return json_response([{"is_processing": False}])
        return json_response({}, 404)
    return mock_client_factory(handler)


@pytest.mark.parametrize("judge_url,inline", [("http://127.0.0.1:8080", False), ("http://localhost:8080/", False), ("http://127.0.0.1:8081", True)],
                         ids=["same-url", "same-host-other-spelling", "another-server"])
def test_a_judge_on_the_measured_server_waits_and_one_on_another_server_does_not(tmp_path, judge_url, inline):
    world = FakeWorld(delay_s=0.03)
    clock = Clock()
    svc = build_services(tmp_path, clock=clock, world=world, client_factory=llama_server_world(["modelo", "juez"]))
    world.register("modelo", lambda req: (clock.advance(25), "Una explicación breve.")[1])      # each answer is older than the server check, so it is stored at once
    world.register("juez", lambda req: JUDGE_REPLY)
    measured = server_contestant(svc, "medido", "modelo", "http://127.0.0.1:8080")
    judge = server_contestant(svc, "juez", "juez", judge_url)
    svc.settings.set_many({"judge.contestant": judge["id"]})
    suite = judge_suite(svc, 4)
    run = run_inline(svc, [suite["id"]], [measured["id"]])
    assert run["state"] == "done", run
    rows = results(svc, run["id"])
    assert len(rows) == 4 and all(not r["judge_pending"] and r["score"] == 0.9 and not r["self_judged"] for r in rows)
    order = kinds(world)
    if not inline:
        assert order == ["case"] * 4 + ["judge"] * 4
    else:
        assert order.count("judge") == 4 and order.index("judge") < len(order) - 4, "a separate server may grade while the next question is asked"
    svc.db.close()


@pytest.mark.parametrize("a,b,same", [("http://127.0.0.1:8080", "http://localhost:8080/v1", True), ("127.0.0.1:8080", "http://127.0.0.1:8080", True),
                                      ("http://[::1]:8080", "http://127.0.0.1:8080", True), ("http://127.0.0.1:8080", "http://127.0.0.1:8081", False),
                                      ("http://192.168.1.5:8080", "http://127.0.0.1:8080", False), ("https://api.example.com", "https://api.example.com/v1", True),
                                      ("http://api.example.com", "https://api.example.com", False)])
def test_two_addresses_are_the_same_server_when_host_and_port_agree(a, b, same):
    from galton_hoard.runner import Session, _endpoint
    assert (_endpoint(a) == _endpoint(b)) is same
    judge = {"id": "j", "kind": "server", "url": a}
    contestant = {"id": "c", "kind": "server", "url": b}
    assert Runner.judge_shares_server(judge, contestant, Session(backend=None, runs_on="x", url=b)) is same


def test_the_judge_is_this_server_when_it_is_the_same_contestant_whatever_the_url():
    from galton_hoard.runner import Session
    me = {"id": "c", "kind": "gguf", "url": ""}
    assert Runner.judge_shares_server(me, me, Session(backend=None, runs_on="x", url="http://127.0.0.1:8091")) is True
    gguf_judge = {"id": "j", "kind": "gguf", "url": ""}
    assert Runner.judge_shares_server(gguf_judge, me, Session(backend=None, runs_on="x", url="http://127.0.0.1:8091")) is False


# ------------------------------------------------------------------ try a single case
def test_try_case_runs_one_case_and_stores_nothing(svc):
    m = add_gguf(svc, "prueba")
    case = svc.store.cases(svc.store.find_suite("rapida")["id"])[0]
    out = svc.runner.try_case(case, m["id"])
    assert out["verdict"]["passed"] and out["output"] and svc.store.count_results.__name__ and svc.store.results() == []


def test_try_case_refuses_vision_for_a_blind_model(svc):
    blind = add_gguf(svc, "ciego")
    case = svc.store.cases(svc.store.find_suite("vision")["id"])[0]
    with pytest.raises(GaltonError, match="no vision"):
        svc.runner.try_case(case, blind["id"])
