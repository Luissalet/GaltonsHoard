"""Talking to a model server: chat-completions compatible (llama-server and similar) and Ollama, streaming, with timings.

``Backend.chat`` never raises for a server-side problem: it returns a ``Completion`` whose ``error`` says what happened, so a run records
the failure on that case and goes on. It raises ``Cancelled`` only when the run was cancelled.
"""

from __future__ import annotations

import base64
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator, Optional

import httpx

from .checkers.textutil import split_reasoning
from .errors import GaltonError
from .messages import text as coded
from .hoard_link import reasoning

log = logging.getLogger("galton.backends")


class Cancelled(Exception):
    """The run was cancelled while a request was in flight."""


@dataclass
class ChatRequest:
    messages: list[dict[str, Any]]
    images: list[bytes] = field(default_factory=list)
    tools: list[dict[str, Any]] = field(default_factory=list)
    temperature: float = 0.0
    top_p: Optional[float] = None
    max_tokens: int = 512
    seed: Optional[int] = None
    effort: Optional[str] = None       # off | low | medium | high | max | None (server default)
    timeout_s: float = 120.0
    context: Optional[int] = None      # Ollama: num_ctx
    reasoning_tokens: int = 0          # tokens for thinking that ``max_tokens`` already includes: the backend must not widen it again


@dataclass
class Completion:
    text: str = ""
    reasoning: str = ""
    tool_calls: list[dict[str, Any]] = field(default_factory=list)   # [{"name", "arguments": dict}]
    finish_reason: str = ""
    latency_ms: float = 0.0
    ttft_ms: Optional[float] = None
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    decode_tps: Optional[float] = None
    prompt_tps: Optional[float] = None
    load_ms: Optional[float] = None
    error: str = ""
    notes: list[str] = field(default_factory=list)
    max_tokens_sent: Optional[int] = None      # the output cap of the request that produced this answer
    served_model: str = ""                     # the ``model`` field the reply carried (empty when it carried none): it must not change within a session

    @property
    def truncated(self) -> bool:
        """The token budget ran out before any visible answer: the model was cut while it was still thinking (or writing nothing)."""
        return self.finish_reason == "length" and not self.text.strip() and not self.tool_calls and not self.error


def tools_prompt(tools: list[dict[str, Any]]) -> str:
    """For a server that rejects the ``tools`` field: describe them in the prompt and ask for a JSON call."""
    lines = ["Tienes estas herramientas. Para usar una, responde SOLO con un objeto JSON {\"name\": ..., \"arguments\": {...}} (o una lista de objetos si necesitas varias). Si no necesitas ninguna, responde con normalidad.", ""]
    for tool in tools:
        fn = tool.get("function", {})
        lines.append(f"- {fn.get('name')}: {fn.get('description', '')} Parámetros: {json.dumps(fn.get('parameters', {}), ensure_ascii=False)}")
    return "\n".join(lines)


def _with_tools_in_prompt(messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = [dict(m) for m in messages]
    note = tools_prompt(tools)
    if out and out[0].get("role") == "system":
        out[0]["content"] = f"{out[0]['content']}\n\n{note}"
    else:
        out.insert(0, {"role": "system", "content": note})
    return out


def _finish(c: Completion, started: float, first: Optional[float]) -> Completion:
    c.latency_ms = round((time.monotonic() - started) * 1000, 1)
    if first is not None:
        c.ttft_ms = round((first - started) * 1000, 1)
    visible, thought = split_reasoning(c.text)
    if thought:
        c.text, c.reasoning = visible, (c.reasoning + "\n\n" + thought).strip() if c.reasoning else thought
    if c.decode_tps is None and c.completion_tokens and c.ttft_ms is not None:
        span = (c.latency_ms - c.ttft_ms) / 1000
        if span > 0 and c.completion_tokens > 1:
            c.decode_tps = round((c.completion_tokens - 1) / span, 2)
    return c


def _tool_calls(raw: Any) -> list[dict[str, Any]]:
    out = []
    for item in raw or []:
        fn = item.get("function") if isinstance(item, dict) else None
        if not isinstance(fn, dict) or not fn.get("name"):
            continue
        args = fn.get("arguments", {})
        if isinstance(args, str):
            try:
                args = json.loads(args) if args.strip() else {}
            except ValueError:
                args = {"_unparsed": args}
        out.append({"name": str(fn["name"]), "arguments": args if isinstance(args, dict) else {"_value": args}})
    return out


def b64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


class Backend:
    """Interface: ``chat`` plus the identity of what is answering."""

    def chat(self, req: ChatRequest, cancel: Callable[[], bool] = lambda: False) -> Completion:  # pragma: no cover - interface
        raise NotImplementedError

    def close(self) -> None:
        return None


class _HttpBackend(Backend):
    def __init__(self, url: str, model: str, *, transport: Optional[httpx.BaseTransport] = None, headers: Optional[dict[str, str]] = None):
        self.url = url.rstrip("/")
        self.model = model
        self.client = httpx.Client(transport=transport, headers=headers or {}, trust_env=False)

    def close(self) -> None:
        self.client.close()

    def _timeout(self, req: ChatRequest) -> httpx.Timeout:
        return httpx.Timeout(connect=10.0, read=max(5.0, reasoning.timeout_for(req.timeout_s, reasoning.normalize(req.effort))), write=30.0, pool=10.0)

    def _lines(self, payload: dict[str, Any], path: str, req: ChatRequest, started: float, cancel: Callable[[], bool]) -> Iterator[str]:
        deadline = started + reasoning.timeout_for(req.timeout_s, reasoning.normalize(req.effort))
        with self.client.stream("POST", self.url + path, json=payload, timeout=self._timeout(req)) as response:
            if response.status_code >= 400:
                body = response.read().decode("utf-8", "replace")
                raise _HttpFailure(response.status_code, body)
            for line in response.iter_lines():
                if cancel():
                    raise Cancelled()
                if time.monotonic() > deadline:
                    raise TimeoutError(f"no answer within {req.timeout_s:.0f} s")
                if line:
                    yield line


class _HttpFailure(Exception):
    def __init__(self, status: int, body: str):
        super().__init__(f"HTTP {status}: {body[:300]}")
        self.status, self.body = status, body


class OpenAIBackend(_HttpBackend):
    """``POST /v1/chat/completions`` with ``stream: true``; llama-server's final ``timings`` give the exact speeds."""

    def _payload(self, req: ChatRequest, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> dict[str, Any]:
        msgs = [dict(m) for m in messages]
        if req.images:
            for m in reversed(msgs):
                if m["role"] == "user":
                    m["content"] = [{"type": "text", "text": m["content"]}] + [{"type": "image_url", "image_url": {"url": "data:image/png;base64," + b64(i)}} for i in req.images]
                    break
        payload: dict[str, Any] = {"model": self.model, "messages": msgs, "stream": True, "stream_options": {"include_usage": True},
                                   "temperature": req.temperature, "max_tokens": req.max_tokens}
        if req.top_p is not None:
            payload["top_p"] = req.top_p
        if req.seed is not None:
            payload["seed"] = req.seed
        if tools:
            payload["tools"] = tools
        reasoning.apply_openai(payload, reasoning.normalize(req.effort))
        if req.reasoning_tokens:                    # the runner already added the thinking allowance to ``max_tokens``
            payload["max_tokens"] = req.max_tokens
        return payload

    def chat(self, req: ChatRequest, cancel: Callable[[], bool] = lambda: False) -> Completion:
        messages, tools, notes = req.messages, req.tools, []
        for attempt in range(3):
            started = time.monotonic()
            payload = self._payload(req, messages, tools)
            try:
                return self._once(payload, req, started, cancel, notes)
            except _HttpFailure as exc:
                if reasoning.looks_like_reasoning_error(exc.status, exc.body) and reasoning.strip(payload) and attempt == 0:
                    req = ChatRequest(**{**req.__dict__, "effort": None})
                    notes.append(coded("note_reasoning_retry"))
                    continue
                if tools and exc.status in (400, 422, 501) and "tool" in exc.body.lower():
                    messages, tools = _with_tools_in_prompt(messages, tools), []
                    notes.append(coded("note_tools_in_prompt_server"))
                    continue
                return Completion(error=str(exc), latency_ms=round((time.monotonic() - started) * 1000, 1), notes=notes)
            except Cancelled:
                raise
            except TimeoutError as exc:
                return Completion(error=coded("chat_timeout_detail", detail=str(exc)), latency_ms=round((time.monotonic() - started) * 1000, 1), notes=notes)
            except httpx.TimeoutException:
                return Completion(error=coded("chat_timeout", seconds=f"{req.timeout_s:.0f}"), latency_ms=round((time.monotonic() - started) * 1000, 1), notes=notes)
            except httpx.HTTPError as exc:
                return Completion(error=coded("chat_connection", detail=f"{type(exc).__name__}: {exc}"), latency_ms=round((time.monotonic() - started) * 1000, 1), notes=notes)
        return Completion(error=coded("chat_rejected"), notes=notes)

    def _once(self, payload: dict[str, Any], req: ChatRequest, started: float, cancel: Callable[[], bool], notes: list[str]) -> Completion:
        c = Completion(notes=list(notes))
        first: Optional[float] = None
        calls: dict[int, dict[str, Any]] = {}
        text, thought = [], []
        for line in self._lines(payload, "/v1/chat/completions", req, started, cancel):
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            try:
                chunk = json.loads(data)
            except ValueError:
                continue
            if chunk.get("error"):
                raise _HttpFailure(500, json.dumps(chunk["error"]))
            if chunk.get("model"):
                c.served_model = str(chunk["model"])
            if usage := chunk.get("usage"):
                c.prompt_tokens = usage.get("prompt_tokens", c.prompt_tokens)
                c.completion_tokens = usage.get("completion_tokens", c.completion_tokens)
            if timings := chunk.get("timings"):
                c.prompt_tokens = timings.get("prompt_n", c.prompt_tokens)
                c.completion_tokens = timings.get("predicted_n", c.completion_tokens)
                c.decode_tps = round(timings["predicted_per_second"], 2) if timings.get("predicted_per_second") else c.decode_tps
                c.prompt_tps = round(timings["prompt_per_second"], 2) if timings.get("prompt_per_second") else c.prompt_tps
            for choice in chunk.get("choices") or []:
                delta = choice.get("delta") or {}
                piece = delta.get("content")
                rpiece = delta.get("reasoning_content") or delta.get("reasoning")
                if (piece or rpiece or delta.get("tool_calls")) and first is None:
                    first = time.monotonic()
                if piece:
                    text.append(piece)
                if rpiece:
                    thought.append(rpiece)
                for tc in delta.get("tool_calls") or []:
                    slot = calls.setdefault(int(tc.get("index", len(calls))), {"function": {"name": "", "arguments": ""}})
                    fn = tc.get("function") or {}
                    if fn.get("name"):
                        slot["function"]["name"] += fn["name"]
                    if fn.get("arguments"):
                        slot["function"]["arguments"] += fn["arguments"] if isinstance(fn["arguments"], str) else json.dumps(fn["arguments"])
                if choice.get("finish_reason"):
                    c.finish_reason = choice["finish_reason"]
        c.text, c.reasoning = "".join(text), "".join(thought)
        c.tool_calls = _tool_calls([calls[k] for k in sorted(calls)])
        return _finish(c, started, first)


class OllamaBackend(_HttpBackend):
    """``POST /api/chat`` streaming NDJSON; the last line carries the durations in nanoseconds."""

    def _payload(self, req: ChatRequest, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> dict[str, Any]:
        msgs = [dict(m) for m in messages]
        if req.images:
            for m in reversed(msgs):
                if m["role"] == "user":
                    m["images"] = [b64(i) for i in req.images]
                    break
        options: dict[str, Any] = {"temperature": req.temperature, "num_predict": req.max_tokens}
        if req.top_p is not None:
            options["top_p"] = req.top_p
        if req.seed is not None:
            options["seed"] = req.seed
        if req.context:
            options["num_ctx"] = req.context
        payload: dict[str, Any] = {"model": self.model, "messages": msgs, "stream": True, "options": options}
        if tools:
            payload["tools"] = tools
        reasoning.apply_ollama(payload, reasoning.normalize(req.effort))
        if req.reasoning_tokens:
            options["num_predict"] = req.max_tokens
        return payload

    def chat(self, req: ChatRequest, cancel: Callable[[], bool] = lambda: False) -> Completion:
        messages, tools, notes = req.messages, req.tools, []
        for attempt in range(3):
            started = time.monotonic()
            payload = self._payload(req, messages, tools)
            try:
                return self._once(payload, req, started, cancel, notes)
            except _HttpFailure as exc:
                if "think" in exc.body.lower() and "think" in payload and attempt == 0:
                    req = ChatRequest(**{**req.__dict__, "effort": None})
                    notes.append(coded("note_think_retry"))
                    continue
                if tools and exc.status in (400, 422, 501) and "tool" in exc.body.lower():
                    messages, tools = _with_tools_in_prompt(messages, tools), []
                    notes.append(coded("note_tools_in_prompt_model"))
                    continue
                return Completion(error=str(exc), latency_ms=round((time.monotonic() - started) * 1000, 1), notes=notes)
            except Cancelled:
                raise
            except TimeoutError as exc:
                return Completion(error=coded("chat_timeout_detail", detail=str(exc)), latency_ms=round((time.monotonic() - started) * 1000, 1), notes=notes)
            except httpx.TimeoutException:
                return Completion(error=coded("chat_timeout", seconds=f"{req.timeout_s:.0f}"), latency_ms=round((time.monotonic() - started) * 1000, 1), notes=notes)
            except httpx.HTTPError as exc:
                return Completion(error=coded("chat_connection", detail=f"{type(exc).__name__}: {exc}"), latency_ms=round((time.monotonic() - started) * 1000, 1), notes=notes)
        return Completion(error=coded("chat_rejected"), notes=notes)

    def _once(self, payload: dict[str, Any], req: ChatRequest, started: float, cancel: Callable[[], bool], notes: list[str]) -> Completion:
        c = Completion(notes=list(notes))
        first: Optional[float] = None
        text, thought, calls = [], [], []
        for line in self._lines(payload, "/api/chat", req, started, cancel):
            try:
                chunk = json.loads(line)
            except ValueError:
                continue
            if chunk.get("error"):
                raise _HttpFailure(500, str(chunk["error"]))
            if chunk.get("model"):
                c.served_model = str(chunk["model"])
            message = chunk.get("message") or {}
            if (message.get("content") or message.get("thinking") or message.get("tool_calls")) and first is None:
                first = time.monotonic()
            if message.get("content"):
                text.append(message["content"])
            if message.get("thinking"):
                thought.append(message["thinking"])
            calls.extend(message.get("tool_calls") or [])
            if chunk.get("done"):
                c.finish_reason = chunk.get("done_reason", "")
                c.prompt_tokens, c.completion_tokens = chunk.get("prompt_eval_count"), chunk.get("eval_count")
                if chunk.get("eval_count") and chunk.get("eval_duration"):
                    c.decode_tps = round(chunk["eval_count"] / (chunk["eval_duration"] / 1e9), 2)
                if chunk.get("prompt_eval_count") and chunk.get("prompt_eval_duration"):
                    c.prompt_tps = round(chunk["prompt_eval_count"] / (chunk["prompt_eval_duration"] / 1e9), 2)
                if chunk.get("load_duration"):
                    c.load_ms = round(chunk["load_duration"] / 1e6, 1)
        c.text, c.reasoning = "".join(text), "".join(thought)
        c.tool_calls = _tool_calls(calls)
        return _finish(c, started, first)


def make_backend(api: str, url: str, model: str, *, transport: Optional[httpx.BaseTransport] = None, headers: Optional[dict[str, str]] = None) -> Backend:
    if api == "ollama":
        return OllamaBackend(url, model, transport=transport, headers=headers)
    if api == "openai":
        return OpenAIBackend(url, model, transport=transport, headers=headers)
    raise GaltonError("invalid", "unknown_api", api=api)
