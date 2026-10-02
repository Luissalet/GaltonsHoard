"""Shared helpers for the API routers: the services object and a thin bridge to the agent tools.

The UI calls the same handlers as the assistant (through ``call_tool``), so the two can never drift apart.
"""

from __future__ import annotations

from typing import Any

from fastapi import Request

from ..agent_tools import call_tool, uncapped
from ..services import Services


def services(request: Request) -> Services:
    return request.app.state.services


def tool(request: Request, name: str, arguments: dict[str, Any] | None = None) -> Any:
    with uncapped():
        return call_tool(services(request), name, arguments, caller="ui")
