"""Stand-ins for the parts that need real hardware, used by the tests and by the demo mode (``GALTON_FAKE=1``).

Nothing here is used in a normal run. A ``FakeWorld`` answers prompts from a table or a function, starts "servers" that are only a name,
and ``FakeGpus`` reports a small inventory and hands out leases without a hub. It lets the whole pipeline (runs, checks, statistics,
routes) be exercised without a model or a GPU, and it is labelled as a demo in the UI so its numbers are never mistaken for measurements.
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Union

from .backends import Backend, Cancelled, ChatRequest, Completion
from .gpus import GpuGrant
from .hoard_link import GpuMemory
from .servers import LaunchSpec, ServerHandle
from .util import slugify

Reply = Union[str, dict[str, Any], Completion]
Responder = Callable[[ChatRequest], Reply]


def completion_from(reply: Reply, req: ChatRequest, *, tps: float, ttft_ms: float) -> Completion:
    if isinstance(reply, Completion):
        return reply
    data = {"text": reply} if isinstance(reply, str) else dict(reply)
    text = data.get("text", "")
    tokens = data.get("completion_tokens") or max(1, len(text) // 4)
    c = Completion(text=text, reasoning=data.get("reasoning", ""), tool_calls=data.get("tool_calls", []), finish_reason=data.get("finish_reason", "stop"),
                   error=data.get("error", ""), served_model=data.get("served_model", ""), prompt_tokens=data.get("prompt_tokens") or sum(len(str(m.get("content", ""))) for m in req.messages) // 4,
                   completion_tokens=tokens)
    c.ttft_ms = float(data.get("ttft_ms", ttft_ms))
    c.decode_tps = float(data.get("decode_tps", tps))
    c.prompt_tps = float(data.get("prompt_tps", tps * 20))
    c.latency_ms = round(c.ttft_ms + tokens / c.decode_tps * 1000, 1)
    return c


class FakeBackend(Backend):
    def __init__(self, responder: Responder, *, tps: float = 40.0, ttft_ms: float = 150.0, delay_s: float = 0.0, calls: Optional[list[ChatRequest]] = None):
        self.responder, self.tps, self.ttft_ms, self.delay_s = responder, tps, ttft_ms, delay_s
        self.calls = calls if calls is not None else []

    def chat(self, req: ChatRequest, cancel: Callable[[], bool] = lambda: False) -> Completion:
        self.calls.append(req)
        if self.delay_s:
            end = time.monotonic() + self.delay_s
            while time.monotonic() < end:
                if cancel():
                    raise Cancelled()
                time.sleep(0.01)
        if cancel():
            raise Cancelled()
        reply = self.responder(req)
        return completion_from(reply, req, tps=self.tps, ttft_ms=self.ttft_ms)


def user_text(req: ChatRequest) -> str:
    return next((str(m.get("content", "")) for m in reversed(req.messages) if m.get("role") == "user"), "")


def table_responder(table: dict[str, Reply], default: Reply = "") -> Responder:
    """Answer with the first table value whose key appears in the user message."""
    def respond(req: ChatRequest) -> Reply:
        text = user_text(req)
        for key, value in table.items():
            if key in text:
                return value
        return default
    return respond


def reference_responder(references: dict[str, str], *, accuracy: float = 1.0, seed: int = 0, wrong: str = "No lo sé.") -> Responder:
    """A model that knows the reference answers: perfect at ``accuracy`` 1, wrong on a fixed random share of prompts otherwise (reproducible)."""
    def respond(req: ChatRequest) -> Reply:
        text = user_text(req)
        key = next((k for k in references if k and k in text), None)
        if key is None:
            return wrong
        if accuracy < 1.0 and random.Random(f"{seed}:{key}").random() > accuracy:
            return wrong
        return references[key]
    return respond


# ------------------------------------------------------------------------------------------------- agents for the harness bench
def harness_task(req: ChatRequest) -> Optional[dict[str, Any]]:
    """The harness-bench task a request belongs to (its first user message is the task prompt), or ``None``."""
    from .harnessbench.tasks import TASKS
    first = next((str(m.get("content", "")) for m in req.messages if m.get("role") == "user"), "")
    return next((t for t in TASKS if t["prompt"] == first), None)


def harness_dialect(req: ChatRequest) -> str:
    """Which way of calling tools the harness behind a request asks for: native, xml, react or json."""
    system = next((str(m.get("content", "")) for m in req.messages if m.get("role") == "system"), "")
    if req.tools:
        return "native"
    if "<tool_call>" in system:
        return "xml"
    if "Action Input" in system:
        return "react"
    return "json"


def _speak(dialect: str, name: str, args: dict[str, Any]) -> Reply:
    import json
    if dialect == "native":
        return {"text": "", "tool_calls": [{"name": name, "arguments": args}]}
    if dialect == "xml":
        return "<tool_call>\n" + json.dumps({"name": name, "arguments": args}) + "\n</tool_call>"
    if dialect == "react":
        return f"Thought: I need a tool.\nAction: {name}\nAction Input: {json.dumps(args)}"
    return json.dumps({"tool": name, "args": args})


def _fumble(dialect: str, name: str, args: dict[str, Any]) -> Reply:
    """The same intention written in a way the harness of that dialect cannot read."""
    import json
    if dialect == "json":
        return json.dumps({"name": name, "arguments": args})                      # the keys of another convention
    if dialect == "xml":
        return f"<tool_call>{name}(" + ", ".join(f"{k}={v!r}" for k, v in args.items()) + ")</tool_call>"
    if dialect == "react":
        return f"Thought: I need a tool.\nAction: {name}\nAction Input: " + ", ".join(f"{k}={v!r}" for k, v in args.items())
    return {"text": "<tool_call>" + json.dumps({"name": name, "arguments": args}) + "</tool_call>"}     # native: the call written as text


def _finish(dialect: str, answer: str) -> Reply:
    return f"Thought: I now know the final answer\nFinal Answer: {answer}" if dialect == "react" else answer


def agent_responder(*, speaks: tuple[str, ...] = ("native", "json", "xml", "react"), accuracy: float = 1.0, seed: int = 0, extra_calls: int = 0,
                    wrong: str = "I could not find it.") -> Responder:
    """A scripted agent for the harness bench. It follows the task's oracle path, speaking the dialects in ``speaks`` correctly and fumbling the
    others (the harness then reports a format failure). ``accuracy`` below 1 makes it give a wrong final answer on a fixed share of tasks;
    ``extra_calls`` makes it repeat the last tool call that many times before answering (a task that needs no tool gets one calculator call)."""
    def respond(req: ChatRequest) -> Reply:
        task = harness_task(req)
        dialect = harness_dialect(req)
        if task is None:
            return ""
        steps = [(n, a) for n, a in task["oracle"]]
        extra = [steps[-1] if steps else ("calculator", {"expression": "1 + 1"})] * extra_calls
        script = steps + extra
        done = sum(1 for m in req.messages if m.get("role") == "assistant") if dialect in speaks else 0      # what it fumbles, it fumbles again
        if done < len(script):
            name, args = script[done]
            return _speak(dialect, name, args) if dialect in speaks else _fumble(dialect, name, args)
        if accuracy < 1.0 and random.Random(f"{seed}:{task['id']}:{dialect}").random() > accuracy:
            return _finish(dialect, wrong)
        return _finish(dialect, task["final"])
    return respond


def with_agent(base: Responder, agent: Responder) -> Responder:
    """Answer harness-bench requests with ``agent`` and everything else with ``base``."""
    return lambda req: agent(req) if harness_task(req) is not None else base(req)


@dataclass
class FakeWorld:
    """Responders by model name, a launcher and a backend factory that agree with each other."""

    responders: dict[str, Responder] = field(default_factory=dict)
    default: Optional[Responder] = None
    tps: dict[str, float] = field(default_factory=dict)
    delay_s: float = 0.0
    started: list[LaunchSpec] = field(default_factory=list)
    stopped: list[str] = field(default_factory=list)
    calls: list[ChatRequest] = field(default_factory=list)
    fail_start: dict[str, str] = field(default_factory=dict)
    _port: int = 8090

    def register(self, name: str, responder: Responder, tps: float = 40.0) -> None:
        for key in {name, slugify(name)}:
            self.responders[key] = responder
            self.tps[key] = tps

    def backend_factory(self, api: str, url: str, model: str, **_: Any) -> FakeBackend:
        key = model if model in self.responders else slugify(model)
        responder = self.responders.get(key) or self.default or (lambda req: "")
        return FakeBackend(responder, tps=self.tps.get(key, 40.0), delay_s=self.delay_s, calls=self.calls)

    # the Launcher interface -------------------------------------------------
    def start(self, spec: LaunchSpec, *, cancel: Callable[[], bool] = lambda: False, timeout_s: Optional[float] = None) -> ServerHandle:
        from .errors import GaltonError
        if spec.name in self.fail_start:
            raise GaltonError("unavailable", self.fail_start[spec.name], "Fake launcher configured to fail.")
        self.started.append(spec)
        self._port += 1
        alias = slugify(spec.name)
        name = spec.name
        return ServerHandle(url=f"http://fake.invalid:{self._port}", port=self._port, pid=90_000 + self._port, log_path="", load_ms=1234.0, alias=alias,
                            proc=None, _stop=lambda: self.stopped.append(name))

    def reap_orphans(self) -> list[int]:
        return []


class FakeGpus:
    """A small inventory plus a lease factory that never talks to a hub. Mimics the hub's behaviour of granting the requested index."""

    def __init__(self, gpus: Optional[list[tuple[int, int, int]]] = None):
        self.gpus = [GpuMemory(i, total, used) for i, total, used in (gpus or [(0, 12_282, 600), (1, 16_311, 700), (2, 16_311, 900), (3, 16_311, 800)])]
        self.leases: list["FakeLease"] = []
        self.released: list["FakeLease"] = []
        self.refuse: set[int] = set()
        self.override_gpu: Optional[int] = None

    def probe(self) -> list[GpuMemory]:
        held: dict[int, int] = {}
        for lease in self.leases:
            if lease not in self.released and lease.gpu is not None:
                held[lease.gpu] = held.get(lease.gpu, 0) + lease.vram_mb
        return [GpuMemory(g.index, g.total_mb, g.used_mb + held.get(g.index, 0)) for g in self.gpus]

    def hub(self) -> None:
        return None

    def names(self) -> dict[int, str]:
        return {0: "NVIDIA GeForce RTX 4070 Ti", 1: "NVIDIA GeForce RTX 5060 Ti", 2: "NVIDIA GeForce RTX 5060 Ti", 3: "NVIDIA GeForce RTX 5060 Ti"}

    def lease_factory(self, vram_mb: int, purpose: str = "", owner: str = "", gpu: Any = None, timeout_s: Optional[float] = None, **_: Any) -> "FakeLease":
        item = FakeLease(self, vram_mb, purpose, owner, gpu)
        self.leases.append(item)
        return item


class FakeLease:
    def __init__(self, parent: FakeGpus, vram_mb: int, purpose: str, owner: str, gpu: Any):
        self.parent, self.vram_mb, self.purpose, self.owner, self.requested = parent, vram_mb, purpose, owner, gpu
        self.gpu: Optional[int] = None
        self.via = "local"
        self.warning: Optional[str] = None

    def acquire(self) -> "FakeLease":
        from .hoard_link import LeaseTimeout
        index = self.parent.override_gpu if self.parent.override_gpu is not None else self.requested
        if index in self.parent.refuse:
            raise LeaseTimeout("queued", 1)
        self.gpu = index
        return self

    def release(self) -> None:
        self.parent.released.append(self)


def grant_for(gpus: list[int], shares_mb: dict[int, int]) -> GpuGrant:
    return GpuGrant(gpus=gpus, shares_mb=shares_mb, via="fake")


# ------------------------------------------------------------------------------------------------- demo mode
DEMO_MODELS = [
    # name, params, quant, accuracy, tok/s, vision, size GB
    ("demo-grande-27b-q4", 27.0, "Q4_K_M", 0.88, 34.0, True, 17.1),
    ("demo-medio-8b-q5", 8.2, "Q5_K_M", 0.72, 78.0, False, 5.9),
    ("demo-pequeno-3b-q8", 3.1, "Q8_0", 0.46, 142.0, False, 3.4),
]


#: the tool-calling conventions each invented model writes correctly (the harness bench shows what happens under the others)
DEMO_DIALECTS = {"demo-grande-27b-q4": ("native", "json", "xml", "react"), "demo-medio-8b-q5": ("native", "xml", "react"), "demo-pequeno-3b-q8": ("json", "react")}


def demo_seed(services: Any) -> dict[str, Any]:
    """Create three fake GGUF contestants and wire the fake world to answer with the reference answers at different accuracies."""
    from .suites import load_definitions
    refs: dict[str, str] = {}
    for suite in load_definitions():
        for case in suite["cases"]:
            text = (case.get("prompt") or {}).get("text") or ""
            if text and case.get("reference"):
                refs[text[:200]] = case["reference"]
    world: FakeWorld = services.fake_world
    for i, (name, params, quant, accuracy, tps, vision, size) in enumerate(DEMO_MODELS):
        world.register(name, with_agent(reference_responder(refs, accuracy=accuracy, seed=i),
                                        agent_responder(speaks=DEMO_DIALECTS[name], accuracy=min(1.0, accuracy + 0.08), seed=i)), tps=tps)
        key = f"gguf:/demo/{name}.gguf"
        if services.store.contestant_by_key(key) is None:
            services.store.create_contestant(key=key, kind="gguf", name=name, path=f"/demo/{name}.gguf", provider="llamacpp", aliases=[name, f"{name}.gguf"], family="demo",
                                             params_b=params, quant=quant, context=32768, vision=vision, digest=f"demo:{name}", source="demo", size_bytes=int(size * 1024 ** 3),
                                             meta={"demo": True, "gguf": {"block_count": 48, "head_count_kv": 8, "head_dim": 128, "context_length": 32768}})
    return {"models": [m[0] for m in DEMO_MODELS]}
