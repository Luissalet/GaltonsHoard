"""Request guard: host allow-list, Origin rule and Fetch Metadata rules (the rules are the shared ones; this checks the app wires them)."""

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from helpers import build_services

from galton_hoard.hoard_link.guard import check_request, install_guard, parse_allowed_hosts
from galton_hoard.main import create_app

NAV = {"sec-fetch-site": "cross-site", "sec-fetch-mode": "navigate", "sec-fetch-dest": "document"}
CORS = {"sec-fetch-site": "cross-site", "sec-fetch-mode": "cors", "sec-fetch-dest": "empty"}
IFRAME = {"sec-fetch-site": "cross-site", "sec-fetch-mode": "navigate", "sec-fetch-dest": "iframe"}


def test_wiring_uses_the_shared_guard_rules():
    """The rules themselves are tested in Hoard Link; this checks the app feeds them its allowed hosts and keeps the family messages."""
    allowed = parse_allowed_hosts("pc.example, *.ts.net,, pc2.example:8443")
    assert allowed == ("pc.example", "*.ts.net", "pc2.example:8443")  # name:port entries are pinned to that port
    local = {"host": "localhost:5201"}
    assert check_request("GET", local, 5201) is None  # curl / MCP bridge: no Sec-Fetch headers
    assert check_request("GET", {**local, **NAV}, 5201) is None  # top-level navigation from another site
    assert check_request("GET", {**local, **CORS}, 5201)  # cross-site fetch
    assert check_request("GET", {**local, **IFRAME}, 5201)
    assert check_request("POST", {**local, **NAV}, 5201)  # form post from another site
    assert check_request("GET", {"host": "evil.example"}, 5201) == (403, "Only local access is allowed.")
    headers = lambda origin: {"host": "my-pc.ts.net", "origin": origin}  # noqa: E731
    assert check_request("GET", headers("https://my-pc.ts.net:8443"), 5201, allowed) is None
    assert check_request("GET", headers("http://localhost:5173"), 5201, allowed) is None  # vite dev
    assert check_request("GET", headers("https://evil.example"), 5201, allowed)


def test_middleware_navigation_reaches_root_but_not_embeds_or_fetches():
    app = FastAPI()
    install_guard(app, port_getter=lambda: 5201, allowed_hosts=parse_allowed_hosts("*.ts.net"), allowed_env="")

    @app.get("/")
    def home():
        return {"ok": True}

    with TestClient(app, base_url="http://127.0.0.1") as c:
        assert c.get("/", headers=NAV).status_code == 200
        assert c.get("/", headers={**NAV, "host": "my-pc.ts.net"}).status_code == 200
        assert c.get("/", headers=IFRAME).status_code == 403
        assert c.get("/", headers=CORS).status_code == 403
        assert c.get("/", headers={**NAV, "host": "other.example"}).status_code == 403


@pytest.fixture
def guarded(tmp_path):
    services = build_services(tmp_path, allowed_hosts=parse_allowed_hosts("pc.example, *.ts.net"))
    with TestClient(create_app(services.config, services=services), base_url="http://127.0.0.1") as c:
        yield c
    services.db.close()


def test_app_host_origin_and_cross_site_rules(guarded):
    get = lambda **headers: guarded.get("/api/health", headers=headers).status_code  # noqa: E731
    # Host rule: local, exact, wildcard; unknown rejected; case and port ignored.
    assert get() == 200
    assert get(host="pc.example") == 200
    assert get(host="My-PC.ts.net:8443") == 200
    assert get(host="evil.example") == 403
    assert get(host="ts.net") == 403
    # Origin rule: allowed host with any scheme/port; anything else 403.
    assert get(origin="https://my-pc.ts.net:8443") == 200
    assert get(origin="http://localhost:5173") == 200
    assert get(origin="https://evil.example") == 403
    # Fetch Metadata: navigation ok, cross-site fetch / iframe / form post rejected.
    assert get(**NAV) == 200
    assert get(**CORS) == 403
    assert get(**IFRAME) == 403
    post = lambda **headers: guarded.post("/api/agent/call", json={}, headers=headers).status_code  # noqa: E731
    assert post(**NAV) == 403
    assert post(**CORS, origin="https://evil.example") == 403
    assert post(**{"sec-fetch-site": "same-origin", "sec-fetch-mode": "cors"}) != 403  # reaches the route


def test_app_without_allowed_hosts_is_local_only(client):
    assert client.get("/api/health", headers={"host": "my-pc.ts.net"}).status_code == 403
    assert client.get("/api/health", headers={"host": "[::1]:5201"}).status_code == 200


def test_allowed_hosts_from_the_environment_variable(tmp_path, monkeypatch):
    from galton_hoard.config import Config

    monkeypatch.setenv("GALTON_ALLOWED_HOSTS", "box.example")
    monkeypatch.setenv("GALTON_DATA_DIR", str(tmp_path))
    config = Config.from_env()
    assert config.allowed_hosts == ("box.example",)
    services = build_services(tmp_path / "svc", allowed_hosts=config.allowed_hosts)
    with TestClient(create_app(services.config, services=services), base_url="http://127.0.0.1") as c:
        assert c.get("/api/health", headers={"host": "box.example"}).status_code == 200
        assert c.get("/api/health", headers={"host": "other.example"}).status_code == 403
    services.db.close()


def test_error_envelopes_are_json_with_a_code(client):
    missing = client.get("/api/nope")
    assert missing.status_code == 404 and missing.json()["code"] == "not_found"
    bad = client.post("/api/ui/call", json={})
    assert bad.status_code == 400 and bad.json()["code"] == "invalid_arguments" and bad.json()["issues"]
    domain = client.post("/api/ui/call", json={"name": "model_get", "arguments": {"model": "nada"}})
    assert domain.status_code == 404 and domain.json()["code"] == "not_found" and domain.json()["hint"]
