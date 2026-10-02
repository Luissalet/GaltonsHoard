"""Start-up without a console (`pythonw.exe`): no stream to print to, and a log file that survives the run."""

from __future__ import annotations

import logging
import sys

import pytest

from galton_hoard import __main__ as entry
from galton_hoard.config import Config
from galton_hoard.startup import LOG_BACKUPS, LOG_MAX_BYTES, install_excepthook, say, setup_logging


@pytest.fixture(autouse=True)
def restore_logging():
    root = logging.getLogger()
    handlers, level, hook = list(root.handlers), root.level, sys.excepthook
    yield
    for handler in list(root.handlers):
        if handler not in handlers:
            handler.close()
    root.handlers[:] = handlers
    root.setLevel(level)
    sys.excepthook = hook


def test_say_writes_when_there_is_a_console(capsys):
    say("hola")
    assert capsys.readouterr().out == "hola\n"


def test_say_is_silent_without_a_console(monkeypatch):
    monkeypatch.setattr(sys, "stdout", None)
    monkeypatch.setattr(sys, "stderr", None)
    say("nobody listens")                                   # must not raise


def test_say_survives_a_broken_stream(monkeypatch):
    class Broken:
        def write(self, _):
            raise OSError("closed")

        def flush(self):
            raise OSError("closed")

    monkeypatch.setattr(sys, "stdout", Broken())
    say("still fine")


def test_logging_goes_to_a_rotating_file_even_without_a_console(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "stdout", None)
    monkeypatch.setattr(sys, "stderr", None)
    path = setup_logging(tmp_path / "logs", "galton")
    assert path == tmp_path / "logs" / "galton.log"
    assert not path.exists()                                # opened on the first message only
    logging.getLogger("galton").info("hello from pythonw")
    for handler in logging.getLogger().handlers:
        handler.flush()
    assert "hello from pythonw" in path.read_text(encoding="utf-8")
    rotating = [h for h in logging.getLogger().handlers if getattr(h, "_hoard_app_log", False)]
    assert len(rotating) == 1 and rotating[0].maxBytes == LOG_MAX_BYTES and rotating[0].backupCount == LOG_BACKUPS
    assert not [h for h in logging.getLogger().handlers if type(h) is logging.StreamHandler]


def test_logging_with_a_console_also_writes_to_stderr(tmp_path):
    setup_logging(tmp_path / "logs", "galton")
    assert any(type(h) is logging.StreamHandler for h in logging.getLogger().handlers)


def test_the_log_rotates(tmp_path, monkeypatch):
    monkeypatch.setattr("galton_hoard.startup.LOG_MAX_BYTES", 400)
    path = setup_logging(tmp_path / "logs", "galton")
    for i in range(40):
        logging.getLogger("galton").info("line %03d %s", i, "x" * 40)
    assert (tmp_path / "logs" / "galton.log.1").exists() and path.stat().st_size < 1000


def test_an_unwritable_log_folder_does_not_stop_the_app(tmp_path):
    blocker = tmp_path / "logs"
    blocker.write_text("a file where the folder should be", encoding="utf-8")
    assert setup_logging(blocker, "galton") is None


def test_uncaught_exceptions_are_logged(tmp_path, caplog):
    install_excepthook(logging.getLogger("galton"))
    with caplog.at_level(logging.CRITICAL, logger="galton"):
        try:
            raise RuntimeError("boom")
        except RuntimeError:
            sys.excepthook(*sys.exc_info())
    assert "Uncaught exception" in caplog.text and "boom" in caplog.text


def test_main_starts_without_stdout_and_stderr(tmp_path, monkeypatch):
    """The whole entry point with both streams missing: it logs, builds the app and hands over to uvicorn with no log config."""
    monkeypatch.setenv("GALTON_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setattr(sys, "stdout", None)
    monkeypatch.setattr(sys, "stderr", None)
    seen = {}
    monkeypatch.setattr(entry, "create_app", lambda config: object())
    monkeypatch.setattr(entry.uvicorn, "run", lambda app, **kwargs: seen.update(kwargs))
    entry.main()
    assert seen["log_config"] is None and seen["host"] == "127.0.0.1"
    for handler in logging.getLogger().handlers:
        handler.flush()
    assert "listening on" in (tmp_path / "data" / "logs" / "galton.log").read_text(encoding="utf-8")


def test_a_taken_port_is_reported_in_the_log_without_a_console(tmp_path, monkeypatch):
    monkeypatch.setenv("GALTON_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("PORT_STRICT", "1")
    monkeypatch.setattr(sys, "stdout", None)
    monkeypatch.setattr(entry, "can_listen", lambda port: False)
    monkeypatch.setattr(entry, "_already_running", lambda port: False)
    with pytest.raises(SystemExit) as stop:
        entry.main()
    assert stop.value.code == 1
    for handler in logging.getLogger().handlers:
        handler.flush()
    assert "taken by another program" in (tmp_path / "data" / "logs" / "galton.log").read_text(encoding="utf-8")


def test_a_second_instance_exits_cleanly(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("GALTON_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("PORT_STRICT", "1")
    monkeypatch.setattr(entry, "can_listen", lambda port: False)
    monkeypatch.setattr(entry, "_already_running", lambda port: True)
    with pytest.raises(SystemExit) as stop:
        entry.main()
    assert stop.value.code == 0 and "already running" in capsys.readouterr().out
    assert Config.from_env().data_dir == tmp_path / "data"
