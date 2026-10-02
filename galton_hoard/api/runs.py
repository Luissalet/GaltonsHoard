"""Run progress for the UI (polling JSON) and the generated images of vision cases."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import Response

from .. import suites as suite_lib
from ..errors import GaltonError
from ..messages import wire
from .deps import services

router = APIRouter()
SAFE_HEADERS = {"X-Content-Type-Options": "nosniff", "Cache-Control": "private, max-age=3600", "Cross-Origin-Resource-Policy": "same-origin"}


@router.get("/api/runs/{run_id}/events")
def events(request: Request, run_id: str, after: int = 0, limit: int = 300):
    """The run card plus the results written after result id ``after``. The page polls this every second or two while the run is live."""
    return wire(services(request).run_events(run_id, after=max(0, after), limit=max(1, min(limit, 1000))))


@router.get("/api/vision/{case_id}/{number}.png")
def vision_image(request: Request, case_id: str, number: int):
    """Image ``number`` (1-based) of a case: a stored image or the one a vision case generates at run time (same seed, same picture)."""
    site = request.headers.get("sec-fetch-site")
    if site not in (None, "same-origin", "none"):
        raise HTTPException(403, "Images are only served to the app itself.")
    svc = services(request)
    case = svc.store.case(case_id)
    images = suite_lib.materialize_case(case, svc.config.images_dir)["images"]
    if number < 1 or number > len(images):
        raise GaltonError("not_found", "case_images", n=len(images))
    data = images[number - 1]
    kind = "image/png" if data.startswith(b"\x89PNG") else "image/jpeg" if data.startswith(b"\xff\xd8") else "image/webp" if data.startswith(b"RIFF") else "image/gif"
    return Response(data, media_type=kind, headers=SAFE_HEADERS)
