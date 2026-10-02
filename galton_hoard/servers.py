"""Galton's own llama-server: start it on a free port (8091-8099) for one GGUF, wait until it answers, and kill its whole process tree afterwards.

Ports 8080-8090 are left alone: other apps of the family scan them for shared servers. Children are recorded in ``servers.json`` so a
crash of this app does not leave a model loaded on a GPU forever (``reap_orphans`` kills what the last run left behind).
"""

from __future__ import annotations

import json
import logging
import os
import shlex
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

import httpx

from .errors import GaltonError
from .gpus import GpuGrant
from .port import can_listen
from .procs import kill_pid_tree, kill_tree, popen_kwargs, process_name
from .util import slugify

log = logging.getLogger("galton.servers")
PORT_RANGE = range(8091, 8100)


@dataclass
class LaunchSpec:
    model_path: str
    name: str                                   # contestant name (log file, alias)
    context: int = 8192
    mmproj_path: Optional[str] = None
    grant: Optional[GpuGrant] = None
    cpu: bool = False                           # run on the CPU only: no layers on a GPU, no GPU visible to the child
    threads: Optional[int] = None               # CPU threads (``-t``) when ``cpu``
    extra_args: list[str] = field(default_factory=list)


@dataclass
class ServerHandle:
    url: str
    port: int
    pid: int
    log_path: str
    load_ms: float
    alias: str
    proc: Any = None
    _stop: Optional[Callable[[], None]] = None

    def alive(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def stop(self) -> None:
        if self._stop:
            self._stop()


def which_on_path(name: str) -> Optional[str]:
    """A program on the PATH; the one place the launcher asks (tests replace it so the computer running them does not matter)."""
    return shutil.which(name)


class Launcher:
    """Starts and stops llama-server children. Everything process-related is injectable so tests use a fake."""

    def __init__(self, settings: Any, logs_dir: Path, registry: Path, *, popen: Callable[..., Any] = subprocess.Popen, which: Optional[Callable[[str], Optional[str]]] = None,
                 client_factory: Callable[[], httpx.Client] = lambda: httpx.Client(trust_env=False, timeout=3.0), port_free: Callable[[int], bool] = can_listen,
                 sleep: Callable[[float], None] = time.sleep, clock: Callable[[], float] = time.monotonic):
        self.settings = settings
        self.logs_dir = logs_dir
        self.registry = registry
        self._popen, self._client_factory, self._port_free = popen, client_factory, port_free
        self._which = which if which is not None else (lambda name: which_on_path(name))
        self._sleep, self._clock = sleep, clock

    # ------------------------------------------------------------------ binary and ports
    def binary(self) -> str:
        configured = str(self.settings.get("llama.server_path") or "").strip()
        if configured and Path(configured).is_file():
            return configured
        found = self._which("llama-server") or self._which("llama-server.exe")
        if found:
            return found
        raise GaltonError("unavailable", "no_llama_server", where=configured or "the llama.server_path setting")

    def pick_port(self) -> int:
        used = {int(e["port"]) for e in self._read_registry()}
        for port in PORT_RANGE:
            if port not in used and self._port_free(port):
                return port
        raise GaltonError("busy", "ports_busy")

    # ------------------------------------------------------------------ registry of children
    def _read_registry(self) -> list[dict[str, Any]]:
        try:
            data = json.loads(self.registry.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return []
        return [e for e in data if isinstance(e, dict) and "pid" in e and "port" in e] if isinstance(data, list) else []

    def _write_registry(self, entries: list[dict[str, Any]]) -> None:
        try:
            self.registry.parent.mkdir(parents=True, exist_ok=True)
            self.registry.write_text(json.dumps(entries), encoding="utf-8")
        except OSError:
            log.warning("could not write %s", self.registry)

    def reap_orphans(self) -> list[int]:
        """Kill llama-server processes recorded by a previous run that no longer has an owner. Returns the pids it killed."""
        killed = []
        for entry in self._read_registry():
            pid = int(entry["pid"])
            if "llama" in process_name(pid):
                kill_pid_tree(pid)
                killed.append(pid)
        self._write_registry([])
        return killed

    # ------------------------------------------------------------------ start / stop
    def command(self, spec: LaunchSpec, port: int, alias: str) -> list[str]:
        cmd = [self.binary(), "-m", spec.model_path, "-c", str(spec.context), "-ngl", "0" if spec.cpu else "99", "-fa", "on", "--jinja", "-np", "1", "--host", "127.0.0.1",
               "--port", str(port), "--metrics", "--alias", alias]
        if spec.mmproj_path:
            cmd += ["--mmproj", spec.mmproj_path]
        if spec.cpu and spec.threads:
            cmd += ["-t", str(int(spec.threads))]
        split = spec.grant.tensor_split if spec.grant and not spec.cpu else None
        if split:
            cmd += ["--tensor-split", ",".join(f"{x:g}" for x in split)]
        cmd += shlex.split(str(self.settings.get("llama.extra_args") or ""), posix=os.name != "nt") + list(spec.extra_args)
        return cmd

    def start(self, spec: LaunchSpec, *, cancel: Callable[[], bool] = lambda: False, timeout_s: Optional[float] = None) -> ServerHandle:
        timeout_s = float(timeout_s or self.settings.get("llama.load_timeout_s"))
        port = self.pick_port()
        alias = slugify(spec.name)[:60]
        cmd = self.command(spec, port, alias)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        log_path = self.logs_dir / f"llama-{slugify(spec.name)[:60]}.log"
        env = dict(os.environ)
        # CUDA numbers GPUs fastest-first by default, nvidia-smi (and the hub, and the allowed list) by PCI bus: without this the
        # child could land on one of the owner's GPUs. No grant means no GPU at all, never "every GPU".
        env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        env["CUDA_VISIBLE_DEVICES"] = spec.grant.cuda_visible_devices if spec.grant and not spec.cpu else ""
        started = self._clock()
        with log_path.open("ab") as logfile:
            logfile.write(f"\n=== {time.strftime('%Y-%m-%d %H:%M:%S')} {' '.join(shlex.quote(c) for c in cmd)}\n".encode("utf-8", "replace"))
            logfile.flush()
            try:
                proc = self._popen(cmd, stdout=logfile, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, env=env, **popen_kwargs())
            except OSError as exc:
                raise GaltonError("unavailable", "server_start_failed", detail=str(exc)) from exc
        entries = self._read_registry() + [{"pid": proc.pid, "port": port, "name": spec.name, "started": time.time()}]
        self._write_registry(entries)

        def stop() -> None:
            kill_tree(proc)
            self._write_registry([e for e in self._read_registry() if int(e["pid"]) != proc.pid])

        url = f"http://127.0.0.1:{port}"
        try:
            self._wait_ready(proc, url, log_path, started, timeout_s, cancel)
        except BaseException:
            stop()
            raise
        return ServerHandle(url=url, port=port, pid=proc.pid, log_path=str(log_path), load_ms=round((self._clock() - started) * 1000, 1), alias=alias, proc=proc, _stop=stop)

    def _wait_ready(self, proc: Any, url: str, log_path: Path, started: float, timeout_s: float, cancel: Callable[[], bool]) -> None:
        with self._client_factory() as client:
            while True:
                if cancel():
                    raise GaltonError("busy", "cancelled_loading")
                code = proc.poll()
                if code is not None:
                    raise GaltonError("unavailable", "server_exited", exit_code=code, log_tail=self.log_tail(log_path), log=str(log_path))
                try:
                    response = client.get(url + "/health")
                    if response.status_code == 200 and (response.json() or {}).get("status", "ok") == "ok":
                        return
                except (httpx.HTTPError, ValueError):
                    pass
                if self._clock() - started > timeout_s:
                    raise GaltonError("unavailable", "server_not_ready", seconds=f"{timeout_s:.0f}", log_tail=self.log_tail(log_path), log=str(log_path))
                self._sleep(1.0)

    @staticmethod
    def log_tail(path: Path, lines: int = 6) -> str:
        try:
            tail = path.read_bytes()[-4000:].decode("utf-8", "replace").strip().splitlines()[-lines:]
        except OSError:
            return "(no log)"
        return " | ".join(t.strip() for t in tail if t.strip())[:700]


# ------------------------------------------------------------------------------------------------- probing running servers
def template_reasons(template: Any) -> Optional[bool]:
    """Does a chat template make the model think before it answers? True when it mentions ``enable_thinking`` or ``<think>``, False when there is a
    template that does not, None when the server did not give one."""
    if not isinstance(template, str) or not template.strip():
        return None
    return "enable_thinking" in template or "<think>" in template


def probe_openai(client: httpx.Client, url: str) -> dict[str, Any]:
    """What a running chat-completions server (llama-server and similar) says about itself: models, props (context, vision), busy slots. Missing endpoints are skipped."""
    out: dict[str, Any] = {"url": url.rstrip("/"), "models": [], "busy": None}
    base = url.rstrip("/")
    try:
        r = client.get(base + "/v1/models")
        if r.status_code == 200:
            out["models"] = [m.get("id") for m in (r.json().get("data") or []) if isinstance(m, dict) and m.get("id")]
            out["raw_models"] = r.json().get("data") or []
        else:
            return {**out, "up": False}
    except (httpx.HTTPError, ValueError):
        return {**out, "up": False}
    out["up"] = True
    try:
        r = client.get(base + "/props")
        if r.status_code == 200 and isinstance(r.json(), dict):
            props = r.json()
            settings = props.get("default_generation_settings") or {}
            out["context"] = settings.get("n_ctx") or (settings.get("params") or {}).get("n_ctx")
            out["alias"] = props.get("model_alias") or props.get("model_path")
            out["model_path"] = props.get("model_path")
            modalities = props.get("modalities") or {}
            out["vision"] = bool(modalities.get("vision"))
            if "chat_template" in props:          # unknown (None) when the server does not say, so an old build is not taken for a model without one
                out["has_template"] = bool(props.get("chat_template"))
                out["reasoning"] = template_reasons(props.get("chat_template"))
    except (httpx.HTTPError, ValueError):
        pass
    out["busy"] = slots_busy(client, base)
    return out


def slots_busy(client: httpx.Client, url: str) -> Optional[bool]:
    """Is any slot of a llama-server processing a request right now? ``None`` when the server does not say (no ``/slots``, or it does not answer)."""
    try:
        r = client.get(url.rstrip("/") + "/slots")
        if r.status_code == 200 and isinstance(r.json(), list):
            return any(bool(s.get("is_processing")) for s in r.json() if isinstance(s, dict))
    except (httpx.HTTPError, ValueError):
        pass
    return None
