"""Request guard: host allow-list, Origin rule and Fetch Metadata rules (the module is shared with the sibling apps)."""

from fastapi import FastAPI
from fastapi.testclient import TestClient

from galton_hoard.guard import check_request, host_of, install_guard, is_allowed_host, parse_allowed_hosts

NAV = {"sec-fetch-site": "cross-site", "sec-fetch-mode": "navigate", "sec-fetch-dest": "document"}
CORS = {"sec-fetch-site": "cross-site", "sec-fetch-mode": "cors", "sec-fetch-dest": "empty"}
IFRAME = {"sec-fetch-site": "cross-site", "sec-fetch-mode": "navigate", "sec-fetch-dest": "iframe"}


def test_host_of_strips_scheme_path_port_and_case():
    assert host_of("LocalHost:5201") == "localhost"
    assert host_of("https://My-PC.ts.net:8443/x") == "my-pc.ts.net"
    assert host_of("[::1]:5201") == "[::1]"
    assert host_of("") == "" and host_of(None) == ""


def test_parse_allowed_hosts():
    assert parse_allowed_hosts(" pc.example , *.TS.net,, pc2.example:8443") == ("pc.example", "*.ts.net", "pc2.example")
    assert parse_allowed_hosts(None) == () and parse_allowed_hosts("*.") == ()


def test_is_allowed_host_exact_wildcard_unknown():
    allowed = parse_allowed_hosts("pc.example,*.ts.net")
    for host in ("localhost", "127.0.0.1", "[::1]", "pc.example", "my-pc.ts.net", "a.b.ts.net"):
        assert is_allowed_host(host, allowed), host
    for host in ("ts.net", "evil.example", "pc.example.evil", "", None):
        assert not is_allowed_host(host, allowed), host


def test_check_request_fetch_metadata_rules():
    local = {"host": "localhost:5201"}
    assert check_request("GET", local) is None
    assert check_request("POST", {"host": "127.0.0.1:5201"}) is None
    assert check_request("GET", {**local, **NAV}) is None
    assert check_request("GET", {**local, **CORS})
    assert check_request("GET", {**local, **IFRAME})
    assert check_request("POST", {**local, **NAV})
    assert check_request("GET", {"host": "evil.example"})


def test_middleware_blocks_cross_site_embeds_and_fetches():
    app = FastAPI()
    install_guard(app, parse_allowed_hosts("*.ts.net"))

    @app.get("/x")
    def x():
        return {"ok": True}

    c = TestClient(app, base_url="http://127.0.0.1")
    assert c.get("/x").status_code == 200
    assert c.get("/x", headers=CORS).status_code == 403
    assert c.get("/x", headers=IFRAME).status_code == 403
    assert c.get("/x", headers={"host": "box.ts.net"}).status_code == 200
    assert c.get("/x", headers={"host": "box.example"}).status_code == 403
