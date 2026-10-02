"""`python -m galton_hoard` — run the app with uvicorn on 127.0.0.1."""

from __future__ import annotations

import logging

import uvicorn

from .config import Config
from .main import create_app
from .port import can_listen, find_available_port
from .startup import install_excepthook, say, setup_logging


def _already_running(port: int) -> bool:
    try:
        import httpx

        response = httpx.get(f"http://127.0.0.1:{port}/api/health", timeout=3, trust_env=False)
        return response.status_code == 200 and response.json().get("service") == "galton-hoard"
    except Exception:  # noqa: BLE001
        return False


def main() -> None:
    config = Config.from_env()
    log = logging.getLogger("galton")
    setup_logging(config.logs_dir, "galton")
    install_excepthook(log)
    if config.port_strict and not can_listen(config.port):
        # Decide before building the app (which opens the database): a second instance must never
        # touch anything under the feet of the one that is serving. The log file is opened lazily and only appended to.
        if _already_running(config.port):
            note = f"Galton's Hoard is already running on http://127.0.0.1:{config.port}"
            log.info(note)
            say(note)
            raise SystemExit(0)
        note = f"Port {config.port} is taken by another program (PORT_STRICT=1)."
        log.error(note)
        say(note)
        raise SystemExit(1)
    port = config.port if config.port_strict else find_available_port(config.port)
    config.port = port
    app = create_app(config)
    note = f"Galton's Hoard listening on http://127.0.0.1:{port}"
    log.info(note)
    say(note)
    # log_config=None: uvicorn's default formatter asks the console whether it is a terminal, which fails when there is none.
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning", log_config=None)


if __name__ == "__main__":
    main()
