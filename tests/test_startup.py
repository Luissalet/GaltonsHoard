"""Start-up of `python -m galton_hoard` (the shared launcher): no console under `pythonw.exe`, a taken port, a second copy.

The logging, the exception hook and the port search are tested in Hoard Link; these tests check that this app wires them up."""

from __future__ import annotations

import json
import logging
import socket
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest
import uvicorn

from galton_hoard import __main__ as entry
from galton_hoard.hoard_link import net


@pytest.fixture(autouse=True)
def restore_logging(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["galton_hoard"])                 # the launcher reads the command line: pytest's arguments are not ours
    root = logging.getLogger()
    handlers, level, hook = list(root.handlers), root.level, sys.excepthook
    yield
    for handler in list(root.handlers):
        if handler not in handlers:
            handler.close()
    root.handlers[:] = handlers
    root.setLevel(level)
    sys.excepthook = hook


def free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def test_main_starts_without_stdout_and_stderr(tmp_path, monkeypatch):
    """The whole entry point with both streams missing: it logs, builds the app and hands over to uvicorn with no log config."""
    monkeypatch.setenv("GALTON_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("GALTON_PORT", str(free_port()))
    monkeypatch.delenv("PORT_STRICT", raising=False)
    monkeypatch.setattr(sys, "stdout", None)
    monkeypatch.setattr(sys, "stderr", None)
    seen = {}
    monkeypatch.setattr("galton_hoard.main.create_app", lambda config=None, services=None: object())
    monkeypatch.setattr(uvicorn, "run", lambda app, **kwargs: seen.update(kwargs))
    assert entry.main() == 0
    assert seen["log_config"] is None and seen["host"] == "127.0.0.1"
    for handler in logging.getLogger().handlers:
        handler.flush()
    assert "listening on" in (tmp_path / "data" / "logs" / "galton-hoard.log").read_text(encoding="utf-8")


def test_a_taken_port_with_port_strict_is_an_error_and_touches_nothing(tmp_path, monkeypatch, capsys):
    port = free_port()
    monkeypatch.setenv("GALTON_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("GALTON_PORT", str(port))
    monkeypatch.setenv("PORT_STRICT", "1")
    with socket.socket() as holder:
        holder.bind(("127.0.0.1", port))
        holder.listen(1)
        assert entry.main() == 1
    assert "taken by another program" in capsys.readouterr().out
    assert not (tmp_path / "data").exists()


def test_a_second_instance_exits_cleanly_and_touches_nothing(tmp_path, monkeypatch, capsys):
    class Health(BaseHTTPRequestHandler):
        def do_GET(self):
            body = json.dumps({"service": "galton-hoard", "version": "x"}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    server = HTTPServer(("127.0.0.1", 0), Health)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        monkeypatch.setenv("GALTON_DATA_DIR", str(tmp_path / "data"))
        monkeypatch.setenv("GALTON_PORT", str(server.server_address[1]))
        monkeypatch.setenv("PORT_STRICT", "1")
        assert net.already_running("galton-hoard", server.server_address[1])
        assert entry.main() == 0
    finally:
        server.shutdown()
    assert "already running" in capsys.readouterr().out
    assert not (tmp_path / "data").exists()                     # the running copy's token, database and log are left alone
