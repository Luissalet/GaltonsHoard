"""FastAPI application factory: request guard, API routers, static SPA (guard, error envelope, PWA and SPA come from Hoard Link)."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from . import SERVICE, __version__
from .api import ROUTERS
from .config import Config
from .errors import GaltonError
from .hoard_link import family
from .hoard_link.agentkit import format_issues, issues_of
from .hoard_link.guard import install_guard
from .hoard_link.service import health_router, install_error_handlers, install_pwa, install_spa
from .messages import wire
from .services import Services

STATIC_DIR = Path(__file__).resolve().parent / "static"


def create_app(config: Config | None = None, services: Services | None = None) -> FastAPI:
    """``services`` lets tests inject a pre-built instance (fake transport, fake link)."""
    config = config or (services.config if services else Config.from_env())

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        svc = services or Services(config)
        app.state.services = svc
        svc.start()
        logging.getLogger("galton").info("Galton's Hoard %s - data in %s", __version__, config.data_dir)
        try:
            yield
        finally:
            svc.stop()

    app = FastAPI(title="Galton's Hoard", version=__version__, lifespan=lifespan, docs_url=None, redoc_url=None)
    app.state.config = config
    family.configure("galton", str(config.data_dir), token_file=str(config.token_path))

    install_guard(app, port_getter=lambda: config.port, allowed_env="GALTON_ALLOWED_HOSTS", allowed_hosts=config.allowed_hosts)
    install_error_handlers(app)  # one {"error", "code"} envelope: HTTP errors, validation, GaltonError (an AppError), 500

    @app.exception_handler(GaltonError)
    async def galton_error(request: Request, exc: GaltonError):
        body = exc.to_dict()
        if request.url.path.startswith(("/api/ui/", "/api/runs/")):  # the bundled UI gets the parts of a message it translates; assistants get plain text
            body = wire(body)
        return JSONResponse(body, status_code=exc.status)

    @app.exception_handler(ValidationError)
    async def model_error(_: Request, exc: ValidationError):  # a pydantic model that failed inside a handler (not a request body)
        return JSONResponse({"error": format_issues(exc), "code": "invalid_arguments", "issues": issues_of(exc)}, status_code=400)

    @app.exception_handler(ValueError)
    async def value_error(_: Request, exc: ValueError):
        return JSONResponse({"error": str(exc), "code": "invalid"}, status_code=400)

    app.include_router(health_router(SERVICE, __version__, extra=lambda: _health(app, config)))
    for router in ROUTERS:
        app.include_router(router)

    install_pwa(app, name="Galton's Hoard", short_name="Galton", theme="#16140d", background="#16140d", cache="galton-hoard-assets", lang="es",
                static_dir=STATIC_DIR, version=__version__)
    install_spa(app, STATIC_DIR)  # last: everything that is not an API route or a real file is the single page app
    return app


def _health(app: FastAPI, config: Config) -> dict:
    # Cheap on purpose: the launcher, the hub and the MCP bridge poll this.
    svc = getattr(app.state, "services", None)
    return {"dataDirConfigured": config.data_dir_configured, "offline": config.offline, "demo": config.fake,
            "counts": svc.counts() if svc else {}, "scheduler": svc.scheduler.status()["running"] if svc else False}
