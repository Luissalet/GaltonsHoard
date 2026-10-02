"""API routers."""

from .agent import router as agent_router
from .health import router as health_router
from .pwa import router as pwa_router
from .runs import router as runs_router
from .ui import router as ui_router

ROUTERS = [health_router, ui_router, agent_router, runs_router, pwa_router]
