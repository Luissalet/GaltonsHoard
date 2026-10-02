"""Process-level configuration read from the environment (never from the DB)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from .guard import parse_allowed_hosts

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PORT = 5201


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


def _int(raw: str, default: int, low: int, high: int) -> int:
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return default
    return value if low <= value <= high else default


def _float(raw: str, default: float, low: float, high: float) -> float:
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return default
    return value if low <= value <= high else default


def load_dotenv(path: Path) -> dict[str, str]:
    """Minimal .env reader (KEY=VALUE, # comments, optional quotes). Never overrides the real environment."""
    values: dict[str, str] = {}
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return values
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        if key:
            values[key] = value
    return values


def default_routes_file() -> Path:
    """Where the routing table is published, the place Hoard Link reads: ``HOARD_ROUTES_FILE``, else ``routes.json`` in ``HOARD_HOME``,
    else ``~/.hoard/routes.json``."""
    override = _env("HOARD_ROUTES_FILE")
    if override:
        return Path(override).expanduser()
    home = _env("HOARD_HOME")
    return (Path(home).expanduser() if home else Path.home() / ".hoard") / "routes.json"


@dataclass
class Config:
    """Everything the process needs before the database exists."""

    data_dir: Path = field(default_factory=lambda: REPO_ROOT / "data")
    port: int = DEFAULT_PORT
    port_strict: bool = False
    allowed_hosts: tuple[str, ...] = ()
    data_dir_configured: bool = False
    http_timeout_s: float = 10.0
    offline: bool = False  # never touch the network or the GPUs (tests): discovery and the watch answer "offline"
    fake: bool = False  # demo mode: fake GPUs, a fake launcher and models that answer from a table (GALTON_FAKE=1); never touches hardware
    scheduler: bool = True  # background refresh, regression watch and queued runs; tests and the MCP-only mode switch it off
    routes_file: Path | None = None  # None: resolved at publish time from HOARD_ROUTES_FILE / ~/.hoard/routes.json
    secrets: dict[str, str] = field(default_factory=dict)  # from .env (+ environment); never written back

    @property
    def db_path(self) -> Path:
        return self.data_dir / "galton.db"

    @property
    def token_path(self) -> Path:
        return self.data_dir / "mcp-token"

    @property
    def url_path(self) -> Path:
        return self.data_dir / "url"

    @property
    def logs_dir(self) -> Path:
        return self.data_dir / "logs"

    @property
    def images_dir(self) -> Path:
        """Images attached to user cases, content-addressed."""
        return self.data_dir / "images"

    @property
    def cache_dir(self) -> Path:
        return self.data_dir / "cache"

    @property
    def servers_file(self) -> Path:
        """Processes this app started (llama-server children), so a crash does not leave them running."""
        return self.data_dir / "servers.json"

    def routes_path(self) -> Path:
        return self.routes_file or default_routes_file()

    def secret(self, name: str) -> str:
        """Environment first, then .env. ``name`` without the GALTON_ prefix."""
        key = f"GALTON_{name}"
        return (os.environ.get(key) or self.secrets.get(key) or "").strip()

    @classmethod
    def from_env(cls) -> "Config":
        raw_dir = _env("GALTON_DATA_DIR")
        port = _int(_env("GALTON_PORT") or _env("PORT") or str(DEFAULT_PORT), DEFAULT_PORT, 1, 65535)
        return cls(
            data_dir=Path(raw_dir).expanduser() if raw_dir else REPO_ROOT / "data",
            port=port,
            port_strict=_env("PORT_STRICT") == "1",
            allowed_hosts=parse_allowed_hosts(_env("GALTON_ALLOWED_HOSTS")),
            data_dir_configured=bool(raw_dir),
            http_timeout_s=_float(_env("GALTON_HTTP_TIMEOUT_S"), 10.0, 1.0, 120.0),
            offline=_env("GALTON_OFFLINE") == "1",
            fake=_env("GALTON_FAKE") == "1",
            scheduler=_env("GALTON_SCHEDULER", "1") != "0",
            secrets=load_dotenv(REPO_ROOT / ".env"),
        )
