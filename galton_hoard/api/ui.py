"""Routes for the bundled web UI.

The dashboard has its own endpoint (it is what the app shows on entry); everything else goes through
``POST /api/ui/call`` with ``{name, arguments}``, which runs the same tool handlers the assistant uses, uncapped.
The request guard (same-origin JSON only) protects these routes; the Bearer token is for the MCP bridge.
"""

from __future__ import annotations

import time
from typing import Any

from fastapi import APIRouter, Request
from pydantic import BaseModel, Field

from ..agent_tools import TOOLS_BY_NAME, tool_catalog
from ..hoard_link import family
from ..messages import wire
from .deps import services, tool

router = APIRouter(prefix="/api")


class CallBody(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)
    arguments: dict[str, Any] | None = None


@router.get("/dashboard")
def dashboard(request: Request):
    return wire(services(request).dashboard())


@router.post("/dashboard/visit")
def visit(request: Request):
    return services(request).mark_visit()


@router.get("/ui/tools")
def ui_tools():
    return {"tools": [t["name"] for t in tool_catalog()]}


@router.post("/ui/call")
def ui_call(request: Request, body: CallBody):
    if body.name not in TOOLS_BY_NAME:
        return {"error": f"Unknown tool: {body.name}"}
    t0 = time.monotonic()
    ok, error = False, ""
    try:
        result = tool(request, body.name, body.arguments)
        ok = True
        return wire(result)
    except Exception as exc:  # noqa: BLE001 — re-raised: the app's exception handlers shape the response
        error = str(exc)[:200]
        raise
    finally:
        family.record_call(body.name, ok, int((time.monotonic() - t0) * 1000), caller="ui", error=error)
