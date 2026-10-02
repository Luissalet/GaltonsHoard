"""Messages the UI shows: stable keys with parameters (the English sentence stays for assistants), stored sentences read back with their keys,
notices with parameters and without repeats, and servers that answer /v1/models but cannot chat."""

from __future__ import annotations

import json
import re
from pathlib import Path

import httpx
import pytest

from galton_hoard import messages
from galton_hoard.discovery import Discovery, NOT_CHAT_RECHECK_S
from galton_hoard.errors import GaltonError
from galton_hoard.messages import CodedText, ERRORS, NOTICES, TEXTS, fields_of
from helpers import add_gguf, json_response, mock_client_factory

ROOT = Path(__file__).resolve().parent.parent
SOURCES = [p for p in (ROOT / "galton_hoard").rglob("*.py") if "hoard_link" not in p.parts and p.name != "messages.py"]


def sample(names) -> dict[str, str]:
    return {name: f"v{i}" for i, name in enumerate(sorted(names), 1)}


# ---- the catalogue ---------------------------------------------------------------------------------------------------------------------------

def test_every_key_the_code_uses_is_in_the_catalogue():
    errors, texts, kinds = set(), set(), set()
    for path in SOURCES:
        source = path.read_text(encoding="utf-8")
        errors |= set(re.findall(r'GaltonError\(\s*"[a-z_]+",\s*"([a-z_]+)"\s*[,)]', source))
        texts |= set(re.findall(r'\b(?:text|coded|message)\("([a-z_]+)"', source))
        kinds |= set(re.findall(r'add_notice\(kind="([a-z_]+)"', source))
    assert errors and texts and kinds
    assert errors <= set(ERRORS), errors - set(ERRORS)
    assert texts <= set(TEXTS), texts - set(TEXTS)
    assert kinds <= set(NOTICES), kinds - set(NOTICES)


def test_every_entry_formats_without_leaving_a_placeholder():
    literal = "{kind: 'gguf'"                       # the one hint that shows a spec object
    for key, (message, hint) in ERRORS.items():
        err = GaltonError("invalid", key, **sample(fields_of(message) | fields_of(hint)))
        assert "{" not in err.message and "{" not in err.hint.replace(literal, ""), key
        assert err.key == key and set(err.params) == fields_of(message) | fields_of(hint)
    for key in TEXTS:
        assert "{" not in str(messages.text(key, **sample(fields_of(TEXTS[key])))).replace("{role, content}", ""), key
    for kind, (title, body) in NOTICES.items():
        assert not any("{" in part for part in messages.notice_text(kind, sample(fields_of(title) | fields_of(body)))), kind


def test_keys_are_lowercase_words_and_unique_across_the_catalogues():
    for key in (*ERRORS, *TEXTS, *NOTICES):
        assert messages.KEY.match(key), key
    assert not set(ERRORS) & set(TEXTS), "a client key would be ambiguous"


def test_an_unknown_error_key_is_a_programming_error_and_free_text_is_still_allowed():
    with pytest.raises(KeyError):
        GaltonError("invalid", "no_such_message_key")
    free = GaltonError("unavailable", "The system said no.", "Try later.")
    assert free.key == "" and free.message == "The system said no." and "key" not in free.to_dict()


def test_an_error_keeps_english_text_key_and_parameters():
    err = GaltonError("not_found", "no_suite", ref="x")
    assert str(err) == "No suite 'x'." and err.hint == "List suites with suites_list." and err.status == 404
    assert err.to_dict() == {"error": "No suite 'x'.", "code": "not_found", "hint": "List suites with suites_list.", "key": "no_suite", "params": {"ref": "x"}}
    coded = err.coded()
    assert isinstance(coded, CodedText) and coded.key == "no_suite" and coded == "No suite 'x'." and coded.item()["params"] == {"ref": "x"}


def test_lists_become_one_text_and_extra_details_travel_with_the_error():
    err = GaltonError("confirm_required", "gpu_reserved", gpus=[1, 0], reserved=[0, 1])
    assert err.message == "GPU 1, 0 is reserved for the owner of this computer." and err.details["reserved"] == [0, 1]
    assert err.to_dict()["reserved"] == [0, 1]


# ---- sentences read back from storage --------------------------------------------------------------------------------------------------------

def test_every_stored_sentence_is_recognised_with_its_key_and_parameters():
    for key, template in TEXTS.items():
        if key in messages.GENERIC:
            continue
        params = sample(fields_of(template))
        found = messages.recognise(str(messages.text(key, **params)))
        assert isinstance(found, CodedText), key
        assert TEXTS[found.key] == template, (key, found.key)             # two keys with the same words are interchangeable
        assert {k: str(v) for k, v in found.params.items()} == params, key
    for key, (template, _hint) in ERRORS.items():
        found = messages.recognise(GaltonError("invalid", key, **sample(fields_of(template))).message)
        assert isinstance(found, CodedText) and ERRORS[found.key][0] == template, key


def test_recognising_leaves_free_text_alone_and_the_longer_template_wins():
    assert messages.recognise("Connection refused by the peer") == "Connection refused by the peer"
    assert not isinstance(messages.recognise("Connection refused by the peer"), CodedText)
    assert messages.recognise("ollama http://x:11434 (loaded by Galton)").key == "runs_on_ollama_loaded"
    assert messages.recognise("ollama http://x:11434").key == "runs_on_ollama"
    assert messages.recognise("timeout: no answer within 30 s").params == {"seconds": 30}
    assert messages.recognise("timeout: read timed out").key == "chat_timeout_detail"
    assert messages.recognise(42) == 42 and messages.recognise("") == ""


def test_rows_written_before_the_keys_existed_are_read_with_keys(svc):
    c = add_gguf(svc, "alfa-q4")
    run = svc.runner.create(suites=["rapida"], contestants=[c["id"]], source="test")
    svc.store.upsert_run_contestant(run["id"], c["id"], runs_on="llama-server http://127.0.0.1:8082", spill="12 % of the model is in system RAM, not on the GPU (speeds are lower than a full GPU load)",
                                    warnings=["1 vision case(s) were not run: alfa-q4 has no vision", "something only the server says"], error="Ollama at http://x does not answer.")
    svc.store.update_run(run["id"], error="Galton was stopped while this run was in progress.", label="watch: alfa-q4")
    rc = svc.store.run_contestant(run["id"], c["id"])
    assert rc["runs_on"].key == "runs_on_server" and rc["runs_on"].params == {"url": "http://127.0.0.1:8082"}
    assert rc["spill"].key == "warn_spill" and rc["spill"].params == {"pct": 12}
    assert rc["warnings"][0].key == "note_vision_dropped" and rc["warnings"][0].params == {"n": 1, "name": "alfa-q4"}
    assert not isinstance(rc["warnings"][1], CodedText) and rc["warnings"][1] == "something only the server says"
    assert rc["error"].key == "ollama_no_answer" and messages.hint_of(rc["error"]) == "Start Ollama, or refresh the models."
    stored = svc.store.run(run["id"])
    assert stored["error"].key == "run_stopped" and stored["label"].key == "label_watch"
    card = svc.run_card(stored)["contestants"][0]
    assert card["hint"] == "Start Ollama, or refresh the models." and card["error"] == "Ollama at http://x does not answer."


def test_a_failed_run_stores_english_text_that_reads_back_with_its_key(svc):
    huge = add_gguf(svc, "inmenso", size_gb=80.0)
    created = svc.runner.create(suites=["rapida"], contestants=[huge["id"]], source="test")
    svc.runner.execute(created["id"])
    rc = svc.store.run_contestant(created["id"], huge["id"])
    assert rc["error"].key == "gpu_capacity" and "GB" in rc["error"] and not rc["error"].endswith(ERRORS["gpu_capacity"][1])
    assert svc.run_card(svc.store.run(created["id"]))["contestants"][0]["hint"].startswith("Use a smaller quantisation")
    raw = svc.db.one("SELECT error FROM run_contestants WHERE run_id = ?", (created["id"],))["error"]
    assert type(raw) is str and raw == str(rc["error"]), "storage keeps plain text"


def test_session_texts_have_keys(svc):
    c = add_gguf(svc, "alfa-q4")
    with svc.runner.session(c, {"context": None, "wait_s": 0}, lambda: False) as session:
        assert isinstance(session.runs_on, CodedText) and session.runs_on.key == "runs_on_own"
    server = svc.store.create_contestant(key="server:t", kind="server", name="srv", url="http://127.0.0.1:9", api="openai", model="m", provider="llamacpp", enabled=True)
    with pytest.raises(GaltonError) as exc:
        with svc.runner.session(server, {"wait_s": 0}, lambda: False):
            pass
    assert exc.value.key == "server_no_answer" and exc.value.params == {"url": "http://127.0.0.1:9"}


# ---- notices ---------------------------------------------------------------------------------------------------------------------------------

def test_a_notice_keeps_its_kind_and_parameters_and_an_english_text(svc):
    nid = svc.store.add_notice(kind="regression", severity="high", params={"name": "alfa", "n": 12, "diff": "0.31", "p": 0.01}, data={"run": "r1"})
    note = svc.store.notices()[0]
    assert nid == note["id"] and note["kind"] == "regression"
    assert note["title"] == "alfa got worse" and note["body"] == "On 12 shared cases it scores 0.31 lower than before the model changed (p = 0.01)."
    assert note["params"] == {"name": "alfa", "n": 12, "diff": "0.31", "p": 0.01} and note["data"]["run"] == "r1"


def test_a_notice_stored_before_parameters_existed_gets_them_from_its_text(svc):
    svc.db.execute("INSERT INTO notices(ts, kind, severity, title, body, data) VALUES (?, ?, ?, ?, ?, ?)",
                   (1.0, "regression", "high", "Mi modelo got worse", "On 9 shared cases it scores 0.25 lower than before the model changed (p = 0.0312).", "{}"))
    svc.db.execute("INSERT INTO notices(ts, kind, severity, title, body, data) VALUES (?, ?, ?, ?, ?, ?)", (2.0, "new_model", "low", "New model: qwen3:8b", "Measure it to see where it fits.", "{}"))
    svc.db.execute("INSERT INTO notices(ts, kind, severity, title, body, data) VALUES (?, ?, ?, ?, ?, ?)", (3.0, "run_failed", "medium", "Run failed: a, b", "one\ntwo", "{}"))
    by = {n["kind"]: n["params"] for n in svc.store.notices()}
    assert by["regression"] == {"name": "Mi modelo", "n": 9, "diff": 0.25, "p": 0.0312}
    assert by["new_model"] == {"name": "qwen3:8b"} and by["run_failed"] == {"label": "a, b", "error": "one\ntwo"}


def test_notices_do_not_repeat_for_the_same_model_and_digest(svc):
    c = add_gguf(svc, "alfa-q4")
    first = svc.store.add_notice(kind="new_model", params={"name": "alfa"}, data={"digest": "d1"}, contestant_id=c["id"], once=True)
    assert first
    assert svc.store.add_notice(kind="new_model", params={"name": "alfa"}, data={"digest": "d1"}, contestant_id=c["id"], once=True) is None
    assert svc.store.add_notice(kind="new_model", params={"name": "alfa again"}, data={"digest": "d1"}, contestant_id=c["id"], dedupe="other", once=True) is None
    assert svc.store.add_notice(kind="changed_model", params={"name": "alfa"}, data={"digest": "d1"}, contestant_id=c["id"], once=True), "another kind is another notice"
    assert svc.store.add_notice(kind="new_model", params={"name": "alfa"}, data={"digest": "d2"}, contestant_id=c["id"], once=True), "another file is news again"
    assert len(svc.store.notices(kind="new_model")) == 2


def test_the_dashboard_notices_carry_kind_and_parameters_for_the_ui(client):
    client.svc.store.add_notice(kind="new_model", params={"name": "alfa"})
    notice = client.get("/api/dashboard").json()["notices"][0]
    assert notice["kind"] == "new_model" and notice["params"] == {"name": "alfa"} and notice["title"] == "New model: alfa"


# ---- what the UI and the assistants receive --------------------------------------------------------------------------------------------------

def ui(client, name, /, **arguments):
    return client.post("/api/ui/call", json={"name": name, "arguments": arguments})


def agent(client, name, /, **arguments):
    return client.post("/api/agent/call", json={"name": name, "arguments": arguments}, headers=client.bearer)


def test_the_ui_receives_keys_and_parameters_and_assistants_receive_plain_sentences(client):
    huge = add_gguf(client.svc, "inmenso", size_gb=80.0)
    ui_plan = ui(client, "run_plan", suites=["rapida"], contestants=[huge["id"]]).json()
    problem = ui_plan["problems"][0]
    assert problem["problem"]["key"] == "gpu_capacity" and problem["problem"]["params"]["need_gb"] and problem["key"] == "gpu_capacity"
    where = ui_plan["contestants"][0]["where"]
    assert where["key"] == "gpu_capacity" and where["text"].startswith("The model needs about")
    agent_plan = agent(client, "run_plan", suites=["rapida"], contestants=[huge["id"]]).json()
    result = agent_plan.get("result", agent_plan)
    assert isinstance(result["contestants"][0]["where"], str) and result["contestants"][0]["where"].startswith("The model needs about")
    assert isinstance(result["problems"][0]["problem"], str)


def test_ui_errors_carry_the_key_and_the_problem_list(client):
    r = ui(client, "case_add", suite="rapida", title="x", prompt="hola")
    body = r.json()
    assert r.status_code == 403 and body["key"] == "suite_builtin" and body["params"]["name"] and body["code"] == "builtin_readonly"
    made = ui(client, "suite_create", name="mia", category="custom", cases=[]).json()
    bad = ui(client, "case_add", suite=made["suite"]["id"], title="x", prompt="")
    assert bad.status_code == 400 and bad.json()["key"] == "case_invalid"
    assert bad.json()["problem_items"][0]["key"] == "prob_prompt_empty" and "prompt: empty" in bad.json()["error"]
    again = agent(client, "case_add", suite=made["suite"]["id"], title="x", prompt="")
    assert again.status_code == 400 and again.json()["problem_items"] == ["prompt: empty"], "assistants get plain sentences"


def test_run_events_and_cards_reach_the_ui_with_keys(client):
    c = add_gguf(client.svc, "alfa-q4")
    run = client.svc.runner.create(suites=["rapida"], contestants=[c["id"]], source="test")
    client.svc.store.upsert_run_contestant(run["id"], c["id"], state="failed", runs_on=str(messages.text("runs_on_server", url="http://x")), error="Ollama at http://x does not answer.")
    card = client.get(f"/api/runs/{run['id']}/events").json()["run"]["contestants"][0]
    assert card["runs_on"] == {"key": "runs_on_server", "params": {"url": "http://x"}, "text": "llama-server http://x"}
    assert card["error"]["key"] == "ollama_no_answer" and card["hint"] == "Start Ollama, or refresh the models."


def test_the_model_remove_and_case_notes_are_keyed(client):
    c = add_gguf(client.svc, "alfa-q4")
    client.svc.store.update_contestant(c["id"], source="ollama")
    out = ui(client, "model_remove", model=c["id"], confirm=True).json()
    assert out["note"]["key"] == "model_removed_note"
    made = ui(client, "suite_create", name="mia", category="custom", cases=[]).json()
    added = ui(client, "case_add", suite=made["suite"]["id"], title="t", prompt="hola").json()
    assert added["notes"][0]["key"] == "case_no_checker"


# ---- servers that answer /v1/models but cannot chat ------------------------------------------------------------------------------------------

def stub_server(*, chat_status=404, props=None, model=lambda: "stub-model", calls=None, port=8083):
    def handler(req: httpx.Request) -> httpx.Response:
        if req.url.port != port:
            raise httpx.ConnectError("refused")
        if req.url.path == "/v1/models":
            return json_response({"data": [{"id": model()}]})
        if req.url.path == "/props":
            return json_response(props if props is not None else {"model_alias": model()})
        if req.url.path == "/slots":
            return json_response([{"is_processing": False}])
        if req.url.path == "/v1/chat/completions":
            if calls is not None:
                calls.append(json.loads(req.content))
            if chat_status == 200:
                return json_response({"choices": [{"message": {"content": "hi"}}]})
            return httpx.Response(chat_status, content="{}")
        raise httpx.ConnectError("refused")
    return handler


def discovery(svc, handler):
    return Discovery(svc.store, svc.settings, client_factory=mock_client_factory(handler), clock=svc.clock, offline=False)


def stubs(svc):
    return [c for c in svc.store.contestants(kind="server", include_missing=True) if c["source"] == "llamacpp"]


@pytest.mark.parametrize("status", [404, 500])
def test_a_server_that_cannot_chat_is_marked_after_one_failed_probe_and_never_announced(svc, status):
    calls: list = []
    d = discovery(svc, stub_server(chat_status=status, calls=calls))
    for _ in range(4):
        d.refresh()
    (stub,) = stubs(svc)
    assert stub["meta"]["chat"]["ok"] is False and stub["meta"]["chat"]["status"] == status and stub["enabled"] is False
    assert len(calls) == 1 and calls[0]["max_tokens"] == 1, "one probe, not one per refresh"
    assert svc.store.notices(kind="new_model") == []
    card = svc.contestant_card(stub)
    assert card["not_chat"] is True and card["not_chat_reason"].key == "not_chat_failed" and card["not_chat_reason"].params == {"status": status}
    assert svc.attention()["never_measured"] == [], "a stub is not something to measure"


def test_a_server_without_a_chat_template_is_not_a_chat_model_without_any_probe(svc):
    calls: list = []
    discovery(svc, stub_server(props={"model_alias": "stub-model", "chat_template": ""}, calls=calls)).refresh()
    (stub,) = stubs(svc)
    assert stub["meta"]["chat"] == {"ok": False, "reason": "no_template", "checked_ts": svc.clock(), "model": "stub-model"} and calls == []
    assert svc.contestant_card(stub)["not_chat_reason"].key == "not_chat_no_template" and svc.store.notices(kind="new_model") == []


def test_a_stub_whose_model_id_changes_stays_one_entry_and_one_silence(svc):
    names = iter(["stub-a", "stub-b", "stub-c", "stub-d"])
    state = {"name": next(names)}
    d = discovery(svc, stub_server(model=lambda: state["name"], props={}))
    for _ in range(3):
        d.refresh()
        state["name"] = next(names)
    assert len(stubs(svc)) == 1 and svc.store.notices(kind="new_model") == []


def test_a_server_that_chats_is_announced_once(svc):
    calls: list = []
    d = discovery(svc, stub_server(chat_status=200, calls=calls))
    d.refresh()
    d.refresh()
    (server,) = stubs(svc)
    assert server["meta"]["chat"]["ok"] is True and server["enabled"] is True and len(calls) == 1
    assert [n["params"] for n in svc.store.notices(kind="new_model")] == [{"name": "stub-model"}]
    assert svc.contestant_card(server)["not_chat"] is False and svc.contestant_card(server)["not_chat_reason"] is None


def test_a_loading_or_unreachable_server_is_no_verdict(svc):
    d = discovery(svc, stub_server(chat_status=503))
    d.refresh()
    (server,) = stubs(svc)
    assert "chat" not in server["meta"] and server["enabled"] is True
    assert len(svc.store.notices(kind="new_model")) == 1


def test_a_no_is_asked_again_after_a_while_and_a_server_that_now_chats_is_usable(svc, clock):
    state = {"status": 404}
    calls: list = []

    def handler(req):
        return stub_server(chat_status=state["status"], calls=calls)(req)

    d = discovery(svc, handler)
    d.refresh()
    assert stubs(svc)[0]["meta"]["chat"]["ok"] is False
    clock.advance(NOT_CHAT_RECHECK_S + 1)
    state["status"] = 200
    d.refresh()
    (server,) = stubs(svc)
    assert server["meta"]["chat"]["ok"] is True and len(calls) == 2
    assert server["enabled"] is False, "switching it back on is the user's call"


def test_offline_discovery_never_probes_chat(svc):
    calls: list = []
    d = Discovery(svc.store, svc.settings, client_factory=mock_client_factory(stub_server(calls=calls)), clock=svc.clock, offline=True)
    d.refresh()
    assert calls == []
