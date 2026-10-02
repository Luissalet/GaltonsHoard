"""Builders shared by the tests: a Services object wired to fakes, models that answer from the reference answers, and a mock HTTP world."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Optional

import httpx

from galton_hoard.config import Config
from galton_hoard.fakes import FakeGpus, FakeWorld, reference_responder
from galton_hoard.services import Services
from galton_hoard.suites import load_definitions

_REFS: dict[str, str] = {}


def references() -> dict[str, str]:
    """Prompt text (first 200 characters) -> reference answer, for every static builtin case."""
    if not _REFS:
        for suite in load_definitions():
            for case in suite["cases"]:
                text = (case.get("prompt") or {}).get("text") or ""
                if text and case.get("reference"):
                    _REFS[text[:200]] = case["reference"]
    return _REFS


class Clock:
    """An injected clock: tests move time by hand."""

    def __init__(self, start: float = 1_760_000_000.0):
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> float:
        self.now += seconds
        return self.now


def build_services(tmp_path: Path, *, clock: Optional[Callable[[], float]] = None, world: Optional[FakeWorld] = None, fake_gpus: Optional[FakeGpus] = None,
                   client_factory: Optional[Callable[[], httpx.Client]] = None, family_call: Optional[Callable[..., dict]] = None, sleep: Optional[Callable[[float], None]] = None, **config_kw: Any) -> Services:
    config = Config(data_dir=tmp_path / "data", offline=config_kw.pop("offline", True), scheduler=config_kw.pop("scheduler", False),
                    routes_file=config_kw.pop("routes_file", tmp_path / "routes.json"), **config_kw)
    return Services(config, clock_fn=clock or Clock(), world=world or FakeWorld(), fake_gpus=fake_gpus or FakeGpus(), client_factory=client_factory,
                    family_call=family_call, sleep=sleep or (lambda _s: None))


def add_gguf(svc: Services, name: str, *, accuracy: float = 1.0, tps: float = 40.0, vision: bool = False, size_gb: float = 5.0, seed: int = 0, digest: str = "",
             responder: Optional[Callable] = None, enabled: bool = True) -> dict[str, Any]:
    """A GGUF contestant (no file on disk) whose answers come from the reference table with the given accuracy."""
    assert svc.fake_world is not None
    svc.fake_world.register(name, responder or reference_responder(references(), accuracy=accuracy, seed=seed), tps=tps)
    return svc.store.create_contestant(key=f"gguf:/test/{name}.gguf", kind="gguf", name=name, path=f"/test/{name}.gguf", provider="llamacpp", aliases=[name, f"{name}.gguf"],
                                       family="test", params_b=8.0, quant="Q4_K_M", context=32768, vision=vision, digest=digest or f"d-{name}", source="manual",
                                       size_bytes=int(size_gb * 1024 ** 3), enabled=enabled,
                                       meta={"demo": True, "gguf": {"block_count": 32, "head_count_kv": 8, "head_dim": 128, "context_length": 32768}})


def run_inline(svc: Services, suites: list[str], contestants: list[Any], **settings: Any) -> dict[str, Any]:
    """Start a run and, since the lanes are not running in tests, get it back finished."""
    out = svc.start_run(suites=suites, contestants=contestants, settings=settings or None, source="test")
    return svc.store.run(out["run"]["id"])


def mock_client_factory(handler: Callable[[httpx.Request], httpx.Response]) -> Callable[[], httpx.Client]:
    return lambda: httpx.Client(transport=httpx.MockTransport(handler), timeout=5.0)


def json_response(data: Any, status: int = 200) -> httpx.Response:
    return httpx.Response(status, content=json.dumps(data), headers={"content-type": "application/json"})
