"""`python -m galton_hoard` — run the app with uvicorn on 127.0.0.1 (the shared Hoard Link launcher)."""

from __future__ import annotations

from .hoard_link.service import run_main


def main() -> int:
    return run_main(service="galton-hoard", package="galton_hoard", default_port=5201, app_factory="galton_hoard.main:create_app",
                    data_dir_env="GALTON_DATA_DIR", port_env="GALTON_PORT", open_browser_default=False, title="Galton's Hoard")


if __name__ == "__main__":
    raise SystemExit(main())
