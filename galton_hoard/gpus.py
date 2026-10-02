"""GPU access: which GPUs Galton may use, what each one holds, and the leases it takes before loading a model.

Rules (they do not bend):

* only GPUs in the ``gpus.allowed`` setting are ever used (default 2 and 3); the others are the owner's;
* a lease is taken per GPU on a concrete index, trying the allowed GPUs by free memory, before anything is loaded;
* the index granted is checked against the allowed list also when the lease client falls back to ``local`` (no hub);
* a model that fits on no single allowed GPU is split across several, one lease each.
"""

from __future__ import annotations

import logging
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from .errors import GaltonError
from .hoard_link import GpuMemory, LeaseError, LeaseTimeout, gpu_free_mb, lease as hub_lease
from .hoard_link import _hubclient

log = logging.getLogger("galton.gpus")
OWNER = "galton"
NAMES_TTL_S = 60.0


def smi_names(timeout_s: float = 3.0) -> dict[int, str]:
    """GPU index -> product name via nvidia-smi (empty without it)."""
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if sys.platform == "win32" else 0
    try:
        proc = subprocess.run(["nvidia-smi", "--query-gpu=index,name", "--format=csv,noheader,nounits"], capture_output=True, text=True, encoding="utf-8",
                              errors="replace", stdin=subprocess.DEVNULL, timeout=timeout_s, creationflags=flags)
    except (OSError, subprocess.SubprocessError):
        return {}
    out: dict[int, str] = {}
    for line in (proc.stdout or "").splitlines():
        index, _, name = line.partition(",")
        if index.strip().isdigit():
            out[int(index)] = name.strip()
    return out


def hub_status() -> Optional[dict[str, Any]]:
    """The hub's view (``GET /api/lease``): per-GPU availability after its own promises, leases and queue. ``None`` when the hub is down."""
    base = _hubclient.hub_url()
    status, body = _hubclient.fetch(base + "/api/lease", timeout=1.5)
    return body if status == 200 and isinstance(body, dict) and body.get("ok") else None


@dataclass
class GpuGrant:
    """GPU memory held for one model load. ``release`` gives it back; it is safe to call twice."""

    gpus: list[int]
    shares_mb: dict[int, int]
    via: str
    leases: list[Any] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    _released: bool = False

    @property
    def tensor_split(self) -> Optional[list[float]]:
        if len(self.gpus) < 2:
            return None
        total = sum(self.shares_mb[g] for g in self.gpus) or 1
        return [round(self.shares_mb[g] / total, 4) for g in self.gpus]

    @property
    def cuda_visible_devices(self) -> str:
        return ",".join(str(g) for g in self.gpus)

    def release(self) -> None:
        if self._released:
            return
        self._released = True
        for item in self.leases:
            try:
                item.release()
            except Exception:  # noqa: BLE001 — giving memory back must never raise
                log.warning("could not release a GPU lease", exc_info=True)


class GpuManager:
    def __init__(self, settings: Any, *, probe: Optional[Callable[[], list[GpuMemory]]] = None, hub: Optional[Callable[[], Optional[dict[str, Any]]]] = None,
                 lease_factory: Optional[Callable[..., Any]] = None, names: Optional[Callable[[], dict[int, str]]] = None, sleep: Callable[[float], None] = time.sleep,
                 clock: Callable[[], float] = time.monotonic):
        self.settings = settings
        self._probe = probe or gpu_free_mb
        self._hub = hub or hub_status
        self._lease = lease_factory or hub_lease
        self._names = names or smi_names
        self._sleep, self._clock = sleep, clock
        self._names_cache: tuple[float, dict[int, str]] = (0.0, {})
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ settings
    def allowed(self) -> list[int]:
        return [int(i) for i in self.settings.get("gpus.allowed")]

    def reserved(self) -> list[int]:
        return [int(i) for i in self.settings.get("gpus.reserved")]

    # ------------------------------------------------------------------ state
    def _names_now(self) -> dict[int, str]:
        with self._lock:
            ts, cached = self._names_cache
            if time.monotonic() - ts < NAMES_TTL_S and cached:
                return cached
        names = self._names()
        with self._lock:
            self._names_cache = (time.monotonic(), names)
        return names

    def state(self) -> dict[str, Any]:
        """Per GPU: memory, whether it is allowed or reserved for the owner, and who holds leases. Honest about what could not be read."""
        gpus = {g.index: g for g in self._probe()}
        hub = self._hub()
        hub_gpus = {g["index"]: g for g in (hub or {}).get("gpus", [])}
        leases = (hub or {}).get("leases", [])
        names = self._names_now() if gpus else {}
        allowed, reserved = set(self.allowed()), set(self.reserved())
        rows = []
        for index in sorted(set(gpus) | allowed | reserved):
            g = gpus.get(index)
            h = hub_gpus.get(index, {})
            rows.append({
                "index": index, "name": names.get(index, ""), "present": g is not None,
                "total_mb": g.total_mb if g else None, "used_mb": g.used_mb if g else None, "free_mb": g.free_mb if g else None,
                "available_mb": h.get("available_mb"), "allowed": index in allowed, "reserved": index in reserved,
                "leases": [{"owner": l.get("owner"), "purpose": l.get("purpose"), "vram_mb": l.get("vram_mb")} for l in leases if l.get("gpu") == index],
            })
        return {"gpus": rows, "allowed": sorted(allowed), "reserved": sorted(reserved), "hub": hub is not None, "inventory": bool(gpus),
                "queue": len((hub or {}).get("queue", []))}

    def used_mb(self, indices: list[int]) -> int:
        by = {g.index: g.used_mb for g in self._probe()}
        return sum(by.get(i, 0) for i in indices)

    # ------------------------------------------------------------------ planning
    def _free(self) -> dict[int, tuple[int, int]]:
        """Allowed, present GPUs: index -> (free MB the machine reports, MB the hub would grant). The smaller of both is what counts."""
        allowed = set(self.allowed())
        hub = {g["index"]: g for g in (self._hub() or {}).get("gpus", [])}
        out: dict[int, tuple[int, int]] = {}
        for g in self._probe():
            if g.index in allowed:
                avail = hub.get(g.index, {}).get("available_mb")
                out[g.index] = (g.free_mb, g.free_mb if avail is None else int(avail))
        return out

    def capacity_mb(self) -> int:
        allowed = set(self.allowed())
        return sum(g.total_mb for g in self._probe() if g.index in allowed)

    def plan(self, need_mb: int) -> Optional[dict[int, int]]:
        """``{gpu: share_mb}`` that fits right now on allowed GPUs, one GPU if possible. ``None`` when nothing fits at the moment."""
        free = {i: min(a, b) for i, (a, b) in self._free().items()}
        if not free:
            return None
        ordered = sorted(free, key=lambda i: free[i], reverse=True)
        for index in ordered:
            if free[index] >= need_mb:
                return {index: need_mb}
        total = 0
        chosen: list[int] = []
        for index in ordered:
            if free[index] <= 0:
                continue
            chosen.append(index)
            total += free[index]
            if total >= need_mb:
                break
        if total < need_mb:
            return None
        shares = {i: max(1, int(need_mb * free[i] / total)) for i in chosen}
        shares[chosen[0]] += need_mb - sum(shares.values())
        return shares

    def check_possible(self, need_mb: int) -> None:
        """Raise a clear error when the allowed GPUs could never hold ``need_mb``, however long we wait."""
        allowed = self.allowed()
        probed = {g.index: g for g in self._probe()}
        present = [i for i in allowed if i in probed]
        if not probed:
            raise GaltonError("no_gpu", "no_nvidia")
        if not present:
            raise GaltonError("no_gpu", "allowed_missing", allowed=allowed)
        capacity = self.capacity_mb()
        if need_mb > capacity:
            raise GaltonError("no_gpu", "gpu_capacity", need_gb=f"{need_mb / 1024:.1f}", capacity_gb=f"{capacity / 1024:.1f}", present=present,
                              need_mb=need_mb, capacity_mb=capacity, allowed=present)

    # ------------------------------------------------------------------ leases
    def _take(self, shares: dict[int, int], purpose: str, timeout_s: float) -> Optional[GpuGrant]:
        allowed = set(self.allowed())
        taken: list[Any] = []
        granted_shares: dict[int, int] = {}
        via = "hub"
        warnings: list[str] = []

        def undo() -> None:
            for item in taken:
                try:
                    item.release()
                except Exception:  # noqa: BLE001
                    pass

        for index, share in shares.items():
            item = self._lease(vram_mb=share, purpose=purpose, owner=OWNER, gpu=index, timeout_s=timeout_s)
            try:
                item.acquire()
            except LeaseTimeout:
                undo()
                return None
            except LeaseError as exc:
                undo()
                raise GaltonError("no_gpu", "lease_refused", reason=str(exc)) from exc
            taken.append(item)
            granted = item.gpu if item.gpu is not None else index
            if granted not in allowed:
                undo()
                raise GaltonError("gpu_not_allowed", "lease_not_allowed", granted=granted, allowed=sorted(allowed))
            # the child sees the GPU the lease holds memory on, which is not always the one asked for
            granted_shares[int(granted)] = granted_shares.get(int(granted), 0) + share
            if getattr(item, "via", "hub") == "local":
                via = "local"
                if getattr(item, "warning", None):
                    warnings.append(item.warning)
        return GpuGrant(gpus=list(granted_shares), shares_mb=granted_shares, via=via, leases=taken, warnings=warnings)

    def acquire(self, need_mb: int, purpose: str, *, wait_s: float = 0.0, cancel: Callable[[], bool] = lambda: False,
                on_wait: Optional[Callable[[str], None]] = None) -> GpuGrant:
        """Hold ``need_mb`` on allowed GPU(s). Waits up to ``wait_s`` (0: one attempt, never queue). Raises ``no_gpu`` when it cannot be done."""
        self.check_possible(need_mb)
        deadline = self._clock() + max(0.0, wait_s)
        announced = False
        while True:
            plan = self.plan(need_mb)
            if plan is not None:
                remaining = max(1.0, deadline - self._clock())
                grant = self._take(plan, purpose, timeout_s=min(10.0, remaining) if wait_s > 0 else 1.0)
                if grant is not None:
                    return grant
            if cancel():
                raise GaltonError("busy", "cancelled_waiting_gpu")
            if self._clock() >= deadline:
                free = {i: min(a, b) for i, (a, b) in self._free().items()}
                raise GaltonError("no_gpu", "no_gpu_free_waited" if wait_s > 0 else "no_gpu_free", need_gb=f"{need_mb / 1024:.1f}", need_mb=need_mb, free_mb=free)
            if on_wait and not announced:
                announced = True
                on_wait(f"waiting for {need_mb / 1024:.1f} GB on GPUs {self.allowed()}")
            self._sleep(2.0)

    def try_acquire(self, need_mb: int, purpose: str) -> Optional[GpuGrant]:
        """For background work that must never queue: a grant now, or ``None``."""
        try:
            return self.acquire(need_mb, purpose, wait_s=0)
        except GaltonError:
            return None
