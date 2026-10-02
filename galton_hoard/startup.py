"""Safe start-up plumbing for `python -m galton_hoard`.

Under `pythonw.exe` there is no console: `sys.stdout` and `sys.stderr` are `None`, so a bare `print` raises and the app would
die before serving. Messages are therefore written only when the stream exists, and everything is also logged to
`data/logs/galton.log` (rotated) so a run without a console still leaves a trace.
"""

from __future__ import annotations

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path
from types import TracebackType
from typing import Optional

LOG_MAX_BYTES = 1_000_000
LOG_BACKUPS = 3
LOG_FORMAT = "%(asctime)s %(name)s %(levelname)s %(message)s"
_FILE_HANDLER_FLAG = "_hoard_app_log"


def say(message: str) -> None:
    """Print a start-up message to the console when there is one; never raises."""
    stream = sys.stdout
    if stream is None:
        return
    try:
        stream.write(message + "\n")
        stream.flush()
    except Exception:  # noqa: BLE001 - a closed or broken console must not stop the app
        pass


def setup_logging(logs_dir: Path, name: str = "galton", level: int = logging.INFO) -> Optional[Path]:
    """Log to the console (only if there is one) and to `<logs_dir>/<name>.log` with rotation. Returns the log path, or None when
    the folder cannot be written. The file is opened on the first message, so a second instance that exits early creates nothing."""
    handlers: list[logging.Handler] = []
    if sys.stderr is not None:
        handlers.append(logging.StreamHandler(sys.stderr))
    path: Optional[Path] = None
    try:
        logs_dir.mkdir(parents=True, exist_ok=True)
        path = logs_dir / f"{name}.log"
        file_handler = RotatingFileHandler(path, maxBytes=LOG_MAX_BYTES, backupCount=LOG_BACKUPS, encoding="utf-8", delay=True)
        setattr(file_handler, _FILE_HANDLER_FLAG, True)
        handlers.append(file_handler)
    except OSError:
        path = None
    if not handlers:
        handlers.append(logging.NullHandler())
    for handler in handlers:
        handler.setFormatter(logging.Formatter(LOG_FORMAT))
    logging.basicConfig(level=level, handlers=handlers, force=True)
    logging.getLogger("httpx").setLevel(logging.WARNING)  # one line per request would drown the log
    return path


def install_excepthook(logger: logging.Logger) -> None:
    """Without a console an uncaught exception vanishes; send it to the log."""
    previous = sys.excepthook

    def hook(kind: type[BaseException], value: BaseException, tb: Optional[TracebackType]) -> None:
        logger.critical("Uncaught exception", exc_info=(kind, value, tb))
        try:
            previous(kind, value, tb)
        except Exception:  # noqa: BLE001
            pass

    sys.excepthook = hook
