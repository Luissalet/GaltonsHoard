"""The real ends of a conversation: the system under test and the user simulator.

* ``FaustusTarget``: one chat of a running Faustus server, driven through the same HTTP API the app uses (``/api/session`` and
  ``/api/chat_stream``). The server keeps the history, so each turn sends only the new user message.
* ``OpenAITarget``: any chat-completions compatible chat endpoint; the whole history is sent on every turn.
* ``OpenAISimulator``: the persona model, on any endpoint ``backends.OpenAIBackend`` supports.

Failures raise ``RuntimeError`` with a readable sentence; ``session.run_case`` turns that into a scored, explained case.
"""

from __future__ import annotations

import json
from typing import Any, Optional

import httpx

from ..backends import ChatRequest, OpenAIBackend


class FaustusTarget:
    """Callable target: ``target(history) -> str``. Create one per case (a case is one fresh chat)."""

    def __init__(self, base_url: str = "http://127.0.0.1:7000", *, model: str = "", endpoint_url: str = "", username: str = "",
                 password: str = "", token: str = "", timeout_s: float = 600.0, transport: Optional[httpx.BaseTransport] = None):
        self.base = base_url.rstrip("/")
        self.model, self.endpoint_url = model, endpoint_url
        self.username, self.password, self.token = username, password, token
        self.session = ""
        headers = {"Authorization": f"Bearer {token}"} if token else {}
        self.http = httpx.Client(base_url=self.base, headers=headers, timeout=httpx.Timeout(timeout_s, connect=10.0), transport=transport, trust_env=False)
        self._ready = False

    def close(self) -> None:
        self.http.close()

    # -- setup -------------------------------------------------------------------------------------------------------------
    def _json(self, resp: httpx.Response) -> Any:
        if resp.status_code >= 400:
            raise RuntimeError(f"Faustus answered HTTP {resp.status_code} on {resp.request.url.path}: {resp.text[:200]}")
        try:
            return resp.json()
        except ValueError:
            return {}

    def _prepare(self) -> None:
        try:
            if not self.token:
                status = self._json(self.http.get("/api/auth/status"))
                if not status.get("auth_disabled"):
                    if not self.username:
                        raise RuntimeError("Faustus needs a sign-in: pass a username and password or an API token")
                    self._json(self.http.post("/api/auth/login", json={"username": self.username, "password": self.password}))
            model, endpoint = self.model, self.endpoint_url
            if not model:
                settings = self._json(self.http.get("/api/auth/settings"))
                model = str(settings.get("default_model") or "")
            form = {"name": "galton simulated user", "skip_validation": "true"}
            if model:
                form["model"] = model
            if endpoint:
                form["endpoint_url"] = endpoint
            made = self._json(self.http.post("/api/session", data=form))
            self.session = str(made.get("id") or made.get("session_id") or "")
            if not self.session:
                raise RuntimeError(f"Faustus did not open a chat: {made}")
            self.model = model
            self._ready = True
        except httpx.HTTPError as exc:
            raise RuntimeError(f"cannot reach Faustus at {self.base}: {type(exc).__name__}: {exc}") from exc

    # -- one turn ----------------------------------------------------------------------------------------------------------
    def __call__(self, history: list[dict[str, str]]) -> str:
        if not self._ready:
            self._prepare()
        message = next((m["content"] for m in reversed(history) if m["role"] == "user"), "")
        form = {"session": self.session, "mode": "agent", "message": message}
        if self.model:
            form["model"] = self.model
        text, error = "", ""
        try:
            with self.http.stream("POST", "/api/chat_stream", data=form) as resp:
                if resp.status_code >= 400:
                    raise RuntimeError(f"Faustus answered HTTP {resp.status_code}: {resp.read().decode('utf-8', 'replace')[:200]}")
                for line in resp.iter_lines():
                    line = line.strip()
                    if not line.startswith("data:"):
                        continue
                    payload = line[5:].strip()
                    if payload == "[DONE]":
                        break
                    try:
                        ev = json.loads(payload)
                    except ValueError:
                        continue
                    if not isinstance(ev, dict):
                        continue
                    kind = ev.get("type")
                    if isinstance(ev.get("delta"), str) and not ev.get("thinking") and not kind:
                        text += ev["delta"]
                    elif kind == "response_replace" and isinstance(ev.get("text"), str):
                        text = ev["text"]
                    elif kind == "ask_user":
                        error = "Faustus stopped on an approval card; a simulated user cannot approve tools"
                    elif kind == "error" or ("error" in ev and not kind):
                        error = str(ev.get("error") or ev.get("message") or ev)[:300]
        except httpx.HTTPError as exc:
            raise RuntimeError(f"Faustus stream failed: {type(exc).__name__}: {exc}") from exc
        if error and not text.strip():
            raise RuntimeError(error)
        return text.strip()


class OpenAITarget:
    """Callable target over an chat-completions compatible endpoint (``base_url`` without ``/v1``)."""

    def __init__(self, base_url: str, model: str, *, system: str = "", max_tokens: int = 800, timeout_s: float = 300.0,
                 transport: Optional[httpx.BaseTransport] = None):
        self.backend = OpenAIBackend(base_url, model, transport=transport)
        self.system, self.max_tokens, self.timeout_s = system, max_tokens, timeout_s

    def __call__(self, history: list[dict[str, str]]) -> str:
        messages = ([{"role": "system", "content": self.system}] if self.system else []) + history
        done = self.backend.chat(ChatRequest(messages=messages, temperature=0.2, max_tokens=self.max_tokens, timeout_s=self.timeout_s))
        if done.error:
            raise RuntimeError(done.error)
        return done.text.strip()


class OpenAISimulator:
    """The persona model: ``simulator(system, history) -> str`` with a little temperature so personas do not sound alike."""

    def __init__(self, base_url: str, model: str, *, max_tokens: int = 200, timeout_s: float = 120.0, temperature: float = 0.6,
                 seed: Optional[int] = 7, transport: Optional[httpx.BaseTransport] = None):
        self.backend = OpenAIBackend(base_url, model, transport=transport)
        self.max_tokens, self.timeout_s, self.temperature, self.seed = max_tokens, timeout_s, temperature, seed

    def __call__(self, system: str, history: list[dict[str, str]]) -> str:
        messages = [{"role": "system", "content": system}, *history]
        done = self.backend.chat(ChatRequest(messages=messages, temperature=self.temperature, max_tokens=self.max_tokens,
                                             timeout_s=self.timeout_s, seed=self.seed, effort="off"))
        if done.error:
            raise RuntimeError(done.error)
        return done.text.strip()
