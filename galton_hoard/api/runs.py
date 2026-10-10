"""Run progress for the UI (polling JSON) and the generated images of vision cases."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import Response

from .. import suites as suite_lib
from ..errors import GaltonError
from ..hoard_link.docs.sniff import sniff
from ..messages import wire
from .deps import services, tool

router = APIRouter()
SAFE_HEADERS = {"X-Content-Type-Options": "nosniff", "Cache-Control": "private, max-age=3600", "Cross-Origin-Resource-Policy": "same-origin"}


@router.get("/api/runs/{run_id}/events")
def events(request: Request, run_id: str, after: int = 0, limit: int = 300):
    """The run card plus the results written after result id ``after``. The page polls this every second or two while the run is live."""
    return wire(services(request).run_events(run_id, after=max(0, after), limit=max(1, min(limit, 1000))))


@router.get("/api/gates/correlation")
def gates_correlation(request: Request, run: list[str] = Query(default_factory=list), category: str = "", suite: str = "", include_stale: bool = False,
                      min_samples: int = Query(20, ge=2, le=1000), strong_phi: float = Query(0.7, gt=0, le=1), resamples: int = Query(1000, ge=100, le=5000)):
    """Pairwise failure correlation of the gates (checks inside the cases) over the stored results: the same report as the ``gates_correlation`` tool.
    ``run`` may be repeated to analyse several runs together."""
    return wire(tool(request, "gates_correlation", {"runs": run, "category": category, "suite": suite, "include_stale": include_stale, "min_samples": min_samples,
                                                    "strong_phi": strong_phi, "resamples": resamples}))


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
    found = sniff("", data)
    return Response(data, media_type=found.mime if found.kind == "image" else "image/png", headers=SAFE_HEADERS)
