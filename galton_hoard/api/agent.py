"""/api/agent/* — the bridge used by mcp_server.py (Bearer token from <DATA_DIR>/mcp-token)."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, Request

from ..agent_tools import AGENT_INSTRUCTIONS, call_tool, tool_catalog
from ..hoard_link.agentkit import make_agent_router
from .deps import services


async def _remember_caller(request: Request) -> None:
    """The ``caller`` of ``POST /api/agent/call`` (which app asked) labels the runs it starts. The shared router uses it only for the audit event and does
    not hand it to the tool, so it is read here, from the (cached) body, before the tool runs."""
    caller = ""
    if request.method == "POST":
        try:
            data = await request.json()
            caller = str(data.get("caller") or "")[:80] if isinstance(data, dict) else ""
        except Exception:  # noqa: BLE001 - a bad body is reported by the router itself
            caller = ""
    request.state.caller = caller


def _call(name: str, arguments: dict[str, Any], request: Request) -> Any:
    return call_tool(services(request), name, arguments, caller=getattr(request.state, "caller", ""))


# GaltonError is an AppError: the router answers it with its own status, code, hint and key
router = APIRouter(dependencies=[Depends(_remember_caller)])
router.include_router(make_agent_router(
    tools_fn=tool_catalog,
    call_fn=_call,
    token_fn=lambda request: services(request).token,
    instructions=AGENT_INSTRUCTIONS,
    app_name="galton",
))
