"""Process-level configuration read from the environment (never from the DB)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from .hoard_link.appconfig import AppPaths, env_flag, env_float, env_int, env_str, load_dotenv as _read_dotenv
from .hoard_link.guard import parse_allowed_hosts

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PORT = 5201


def load_dotenv(path: Path) -> dict[str, str]:
    """The pairs of a ``.env`` file (the shared reader). Unlike the shared reader's default they are **not** left in ``os.environ``: these are secrets
    for this process, and a llama-server child that inherited the environment would get them too. A variable the real environment already has stays."""
    before = set(os.environ)
    values = _read_dotenv(path)
    for key in values:
        if key not in before:
            os.environ.pop(key, None)
    return values


def default_routes_file() -> Path:
    """Where the routing table is published, the place Hoard Link reads: ``HOARD_ROUTES_FILE``, else ``routes.json`` in ``HOARD_HOME``,
    else ``~/.hoard/routes.json``."""
    override = env_str("HOARD_ROUTES_FILE")
    if override:
        return Path(override).expanduser()
    home = env_str("HOARD_HOME")
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
    def paths(self) -> AppPaths:
        """The shared data-folder layout (``galton.db``, ``mcp-token``, ``url``, ``logs/``)."""
        return AppPaths("galton", REPO_ROOT, self.data_dir, self.data_dir_configured)

    @property
    def db_path(self) -> Path:
        return self.paths.db_path

    @property
    def token_path(self) -> Path:
        return self.paths.token_path

    @property
    def url_path(self) -> Path:
        return self.paths.url_path

    @property
    def logs_dir(self) -> Path:
        return self.paths.logs_dir

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
        raw_dir = env_str("GALTON_DATA_DIR") or ""
        port = env_int("GALTON_PORT", "PORT", default=DEFAULT_PORT)
        if not 1 <= port <= 65535:
            port = DEFAULT_PORT
        timeout = env_float("GALTON_HTTP_TIMEOUT_S", default=10.0)
        if not 1.0 <= timeout <= 120.0:
            timeout = 10.0
        return cls(
            data_dir=Path(raw_dir).expanduser() if raw_dir else REPO_ROOT / "data",
            port=port,
            port_strict=env_flag("PORT_STRICT"),
            allowed_hosts=parse_allowed_hosts(env_str("GALTON_ALLOWED_HOSTS")),
            data_dir_configured=bool(raw_dir),
            http_timeout_s=timeout,
            offline=env_flag("GALTON_OFFLINE"),
            fake=env_flag("GALTON_FAKE"),
            scheduler=env_flag("GALTON_SCHEDULER", True),
            secrets=load_dotenv(REPO_ROOT / ".env"),
        )
