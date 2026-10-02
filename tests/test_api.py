"""HTTP surface: health, dashboard, the UI and agent bridges, run events, vision images, PWA, guard."""

from __future__ import annotations

import time

import pytest

from helpers import add_gguf


def ui(client, name, /, **arguments):
    return client.post("/api/ui/call", json={"name": name, "arguments": arguments})


def agent(client, name, /, headers=True, **arguments):
    return client.post("/api/agent/call", json={"name": name, "arguments": arguments}, headers=client.bearer if headers else {})


def wait_done(client, run_id, seconds=20):
    end = time.time() + seconds
    while time.time() < end:
        events = client.get(f"/api/runs/{run_id}/events").json()
        if events["finished"]:
            return events
        time.sleep(0.05)
    pytest.fail("run did not finish")


def test_health_and_status(client):
    r = client.get("/api/health")
    assert r.status_code == 200 and r.json()["service"] == "galton-hoard"
    body = r.json()
    assert body["counts"]["suites"] == 13 and body["scheduler"] is True and body["hoard_link"]["app"] == "galton"
    s = client.get("/api/status").json()
    assert s["counts"]["cases"] >= 220 and "gpus" in s and "scheduler" in s and s["routes_path"].endswith("routes.json")


def test_dashboard_has_every_block(client):
    add_gguf(client.svc, "alfa-q4")
    d = client.get("/api/dashboard").json()
    for key in ("counts", "routes", "routes_diff", "running", "queued", "recent_runs", "notices", "attention", "gpus", "scheduler", "watch"):
        assert key in d, key
    assert {r["category"] for r in d["routes"]} >= {"general", "code", "vision"} and d["attention"]["never_measured"][0]["name"] == "alfa-q4"
    assert d["gpus"]["allowed"] == [2, 3]
    assert client.post("/api/dashboard/visit").json() == {"marked_seen": 0}


def test_ui_call_runs_the_same_handlers_uncapped(client):
    r = ui(client, "suites_list")
    assert r.status_code == 200 and r.json()["count"] == 13
    assert ui(client, "nope").json() == {"error": "Unknown tool: nope"}
    assert client.get("/api/ui/tools").json()["tools"].count("run_start") == 1


def test_errors_keep_their_code_and_status(client):
    r = ui(client, "model_get", model="no-existe")
    assert r.status_code == 404 and r.json()["code"] == "not_found" and r.json()["hint"]
    r = ui(client, "model_remove", model="x")
    assert r.status_code in (400, 404)
    r = ui(client, "run_start", suites=[], contestants=["x"])
    assert r.status_code == 400 and "suites" in r.json()["error"]
    r = ui(client, "settings_set", values={"gpus.allowed": [0]})
    assert r.status_code == 400 and r.json()["code"] == "confirm_required"


def test_agent_bridge_needs_the_token(client):
    assert agent(client, "galton_overview", headers=False).status_code == 401
    bad = client.post("/api/agent/call", json={"name": "galton_overview"}, headers={"Authorization": "Bearer nope"})
    assert bad.status_code == 401
    ok = agent(client, "galton_overview")
    assert ok.status_code == 200 and "counts" in ok.json()
    assert agent(client, "does_not_exist").status_code == 404
    assert agent(client, "model_get", model="zzz").status_code == 404
    listing = client.get("/api/agent/tools").json()
    assert len(listing["tools"]) >= 38 and "Galton" in listing["instructions"]


def test_agent_results_are_capped_ui_results_are_not(client):
    add_gguf(client.svc, "alfa-q4")
    cases = [{"title": f"Caso {i} " + "x" * 80, "prompt": f"pregunta {i} " + "y" * 200, "expected": "z"} for i in range(150)]
    ui(client, "suite_create", name="Larga", category="custom", cases=cases)
    sid = client.svc.store.find_suite("Larga")["id"]
    capped = agent(client, "suite_get", suite=sid, limit=150).json()
    full = ui(client, "suite_get", suite=sid, limit=150).json()
    assert len(full["cases"]) == 150 and "truncated" not in full
    assert "truncated" in capped and len(capped["cases"]) < 150


def test_run_through_the_lane_with_polling_events(client):
    add_gguf(client.svc, "alfa-q4")
    started = ui(client, "run_start", suites=["razonamiento"], contestants=["alfa-q4"]).json()
    rid = started["run"]["id"]
    events = wait_done(client, rid)
    assert events["run"]["state"] == "done" and len(events["results"]) == 20 and events["last_id"] > 0
    assert events["results"][0]["title"] and "output" not in events["results"][0]
    after = client.get(f"/api/runs/{rid}/events", params={"after": events["last_id"]}).json()
    assert after["results"] == [] and after["last_id"] == events["last_id"]
    assert client.get("/api/runs/r_nope/events").status_code == 404


def test_cancel_a_queued_run_through_the_ui(client):
    add_gguf(client.svc, "alfa-q4")
    client.svc.fake_world.delay_s = 0.2
    first = ui(client, "run_start", suites=["razonamiento"], contestants=["alfa-q4"]).json()["run"]["id"]
    second = ui(client, "run_start", suites=["matematicas"], contestants=["alfa-q4"]).json()["run"]["id"]
    out = ui(client, "run_cancel", run=second).json()
    assert out["cancelled"] is True
    ui(client, "run_cancel", run=first)
    assert wait_done(client, second)["run"]["state"] == "cancelled"
    assert wait_done(client, first)["run"]["state"] == "cancelled"


def test_vision_images_are_served_and_deterministic(client):
    case = client.svc.store.cases("s_vision")[0]
    r = client.get(f"/api/vision/{case['id']}/1.png")
    assert r.status_code == 200 and r.headers["content-type"] == "image/png" and r.content.startswith(b"\x89PNG")
    assert client.get(f"/api/vision/{case['id']}/1.png").content == r.content
    assert client.get(f"/api/vision/{case['id']}/2.png").status_code == 404
    assert client.get(f"/api/vision/{case['id']}/1.png", headers={"sec-fetch-site": "same-site"}).status_code == 403
    assert client.get("/api/vision/k_nope/1.png").status_code == 404


def test_pwa_manifest_and_service_worker(client):
    m = client.get("/manifest.webmanifest")
    assert m.status_code == 200 and m.json()["short_name"] == "Galton" and m.json()["icons"]
    sw = client.get("/sw.js")
    assert sw.status_code == 200 and "galton-hoard-assets" in sw.text and sw.headers["service-worker-allowed"] == "/"


def test_spa_fallback_and_unknown_api(client):
    assert client.get("/api/nothing").status_code == 404
    r = client.get("/")
    assert r.status_code in (200, 503)
    assert client.get("/%2e%2e/%2e%2e/etc/passwd").status_code in (200, 404, 503)


def test_guard_blocks_other_hosts_and_cross_site_posts(client):
    assert client.get("/api/health", headers={"host": "evil.example"}).status_code == 403
    r = client.post("/api/ui/call", json={"name": "suites_list"}, headers={"origin": "https://evil.example"})
    assert r.status_code == 403
    r = client.post("/api/ui/call", json={"name": "suites_list"}, headers={"sec-fetch-site": "cross-site", "sec-fetch-mode": "cors"})
    assert r.status_code == 403
