"""Which model a shared server serves, and whether it is still the one a contestant was registered for.

A server that other people use can be restarted with another model on the same address. Its requests keep working, so without a look at what
it serves a run would measure the new model and store the answers under the old contestant and its digest. The identity of a server
contestant is the address plus what the server says about itself:

* a chat-completions server (llama-server and similar): the model ids of ``/v1/models``, the alias and the model file of ``/props``;
* Ollama: the tag and the digest of ``/api/tags``.

Before a contestant starts, and again before the cases (at most every ``CHECK_TTL_S`` seconds), the runner asks again and compares. The answer
of every request may also carry a ``model`` field: it must stay what the first answer of the session said. Nothing here changes a server; it
only reads.
"""

from __future__ import annotations

import posixpath
from dataclasses import dataclass, field
from typing import Any, Callable, Optional
from urllib.parse import urlsplit

import httpx

from .errors import GaltonError

#: seconds a successful look at a server is trusted (a run asks the server again after this long, before the next case)
CHECK_TTL_S = 20.0
#: contestants of these sources have a model name somebody typed or an app gave: what the server serves the first time is taken as theirs
PIN_ON_FIRST_USE = ("spec", "faustus")
LOOPBACK = ("localhost", "127.0.0.1", "::1", "0.0.0.0", "")


def endpoint(url: str) -> str:
    """``host:port`` of a server address, with every spelling of this computer as one host (``localhost``, ``127.0.0.1``, ``[::1]``, ``0.0.0.0``)."""
    parts = urlsplit(url if "//" in url else "//" + url)
    host = (parts.hostname or "").lower()
    if host in LOOPBACK:
        host = "localhost"
    default = 443 if parts.scheme == "https" else 80
    return f"{host}:{parts.port or default}"


def _cf(value: Any) -> str:
    return str(value or "").strip().casefold()


def _path(value: Any) -> str:
    return str(value or "").strip().replace("\\", "/").casefold()


def _base(value: Any) -> str:
    return posixpath.basename(str(value or "").replace("\\", "/"))


def _tag_variants(name: Any) -> set[str]:
    name = _cf(name)
    return {name, name[: -len(":latest")]} if name.endswith(":latest") else ({name, name + ":latest"} if name and ":" not in name else {name})


@dataclass(frozen=True)
class Served:
    """What a server says it serves right now."""

    url: str
    api: str                                                 # openai | ollama
    models: tuple[str, ...] = ()                             # ids of /v1/models, or the tags of /api/tags
    alias: str = ""
    path: str = ""                                           # the model file llama-server loaded
    digests: dict[str, str] = field(default_factory=dict)    # Ollama: tag -> digest

    @property
    def label(self) -> str:
        """A short name for what is served: the alias, else the first model id, else the file name."""
        return self.alias or (self.models[0] if self.models else "") or _base(self.path) or "?"

    def names(self) -> set[str]:
        return {n for n in (*(_cf(m) for m in self.models), _cf(self.alias), _path(self.path), _cf(_base(self.path))) if n}

    def card(self) -> dict[str, Any]:
        """What is kept on a contestant as its pinned identity."""
        return {k: v for k, v in {"model": self.models[0] if self.models else "", "alias": self.alias, "path": self.path}.items() if v}


# ------------------------------------------------------------------------------------------------- looking
def served_from_probe(probe: dict[str, Any], url: str) -> Served:
    """The ``Served`` of what ``servers.probe_openai`` already found out (no new request)."""
    models = tuple(str(m) for m in probe.get("models") or [])
    path = str(probe.get("model_path") or "")
    alias = str(probe.get("alias") or "")
    return Served(url=url.rstrip("/"), api="openai", models=models, alias="" if alias == path else alias, path=path)


def look_openai(client: httpx.Client, url: str) -> Optional[Served]:
    """``/v1/models`` and ``/props`` of a chat-completions server; ``None`` when it does not answer (it may be restarting)."""
    base = url.rstrip("/")
    try:
        response = client.get(base + "/v1/models")
        if response.status_code != 200:
            return None
        models = tuple(str(m["id"]) for m in (response.json().get("data") or []) if isinstance(m, dict) and m.get("id"))
    except (httpx.HTTPError, ValueError, AttributeError):
        return None
    alias = path = ""
    try:
        response = client.get(base + "/props")
        if response.status_code == 200 and isinstance(response.json(), dict):
            props = response.json()
            alias, path = str(props.get("model_alias") or ""), str(props.get("model_path") or "")
    except (httpx.HTTPError, ValueError):
        pass
    return Served(url=base, api="openai", models=models, alias="" if alias == path else alias, path=path)


def look_ollama(client: httpx.Client, url: str) -> Optional[Served]:
    """``/api/tags`` of an Ollama server: the tags and their digests; ``None`` when it does not answer."""
    try:
        response = client.get(url.rstrip("/") + "/api/tags")
        if response.status_code != 200:
            return None
        items = [m for m in (response.json().get("models") or []) if isinstance(m, dict) and (m.get("name") or m.get("model"))]
    except (httpx.HTTPError, ValueError, AttributeError):
        return None
    digests = {str(m.get("name") or m.get("model")): str(m.get("digest") or "") for m in items}
    return Served(url=url.rstrip("/"), api="ollama", models=tuple(digests), digests=digests)


def look(client_factory: Callable[[], httpx.Client], api: str, url: str) -> Optional[Served]:
    try:
        with client_factory() as client:
            return look_ollama(client, url) if api == "ollama" else look_openai(client, url)
    except httpx.HTTPError:
        return None


# ------------------------------------------------------------------------------------------------- comparing
def expected(c: dict[str, Any]) -> dict[str, Any]:
    """The identity a server contestant stands for: the one pinned on it, else what its fields say."""
    pin = (c.get("meta") or {}).get("identity")
    if isinstance(pin, dict) and pin:
        return pin
    return {"model": c.get("model") or "", "path": c.get("path") or "", "digest": c.get("digest") or "", "alias": ""}


def needs_pin(c: dict[str, Any]) -> bool:
    return c.get("source") in PIN_ON_FIRST_USE and not (c.get("meta") or {}).get("identity")


def judge(c: dict[str, Any], served: Served) -> tuple[str, str]:
    """``(verdict, label)``: ``same`` when the server serves what ``c`` stands for, ``different`` when it serves something else (``label`` says what),
    ``unknown`` when the server tells too little to say."""
    want = expected(c)
    if served.api == "ollama":
        tags = _tag_variants(c.get("model")) | _tag_variants(c.get("ollama_ref")) | _tag_variants(c.get("name"))
        tag = next((m for m in served.models if _cf(m) in tags), None)
        got, digest = (served.digests.get(tag, "") if tag else ""), str(want.get("digest") or "")
        if not tag or not got or not digest:
            return "unknown", served.label
        return ("same", tag) if got == digest else ("different", f"{tag} ({got[:19]})")
    names, have_path = served.names(), bool(served.path)
    wanted = [w for w in (want.get("model"), want.get("alias")) if w]
    wants_path = bool(want.get("path"))
    if not names and not have_path:
        return "unknown", served.label
    if wants_path and have_path and _path(want["path"]) != _path(served.path):
        return "different", served.label
    if wanted and names and not any(_cf(w) in names or _cf(_base(w)) in names for w in wanted):
        return "different", served.label
    if not wanted and not (wants_path and have_path):
        return "unknown", served.label
    return "same", served.label


def pin(store: Any, c: dict[str, Any], served: Served) -> dict[str, Any]:
    """Keep what the server serves now as the identity of ``c`` (first use of a contestant whose model name was given by hand)."""
    if served.api == "ollama":
        tags = _tag_variants(c.get("model")) | _tag_variants(c.get("ollama_ref")) | _tag_variants(c.get("name"))
        tag = next((m for m in served.models if _cf(m) in tags), None)
        card = {"model": tag or "", "digest": served.digests.get(tag or "", "")} if tag else {}
    else:
        card = served.card()
    if not card:
        return c
    return store.update_contestant(c["id"], meta={**c["meta"], "identity": card})


def changed(url: str, label: str, expected_label: str) -> GaltonError:
    return GaltonError("conflict", "server_changed", url=url, model=label, expected=expected_label)


# ------------------------------------------------------------------------------------------------- the "not served now" mark
def mark_not_served(store: Any, c: dict[str, Any], label: str, url: str, now: float) -> bool:
    """Say on a server contestant that its address now serves another model (``label``). The entry, its history and its results stay; it can
    not be measured until the server serves it again. Returns whether the mark is new."""
    try:
        fresh = store.contestant(c["id"])
    except GaltonError:
        return False
    flag = (fresh["meta"] or {}).get("not_served") or {}
    if flag.get("model") == label and flag.get("url") == url:
        return False
    store.update_contestant(fresh["id"], meta={**fresh["meta"], "not_served": {"since": now, "model": label, "url": url}})
    store.add_notice(kind="not_served", severity="medium", params={"name": fresh["name"], "url": url, "model": label}, data={"contestant": fresh["id"], "model": label},
                     contestant_id=fresh["id"], dedupe=f"not_served:{fresh['id']}:{label}")
    return True


def clear_not_served(store: Any, c: dict[str, Any]) -> bool:
    """The server serves this contestant again: remove the mark. Returns whether there was one."""
    try:
        fresh = store.contestant(c["id"])
    except GaltonError:
        return False
    if "not_served" not in (fresh["meta"] or {}):
        return False
    store.update_contestant(fresh["id"], meta={k: v for k, v in fresh["meta"].items() if k != "not_served"})
    return True


# ------------------------------------------------------------------------------------------------- the guard of a session
class IdentityGuard:
    """Keeps a run honest about the server it measures. ``check`` looks again at most every ``ttl_s`` seconds; ``answer`` compares the ``model``
    field of each reply with the one of the first reply. Either raises ``server_changed``."""

    def __init__(self, *, url: str, expected_label: str, look: Callable[[], Optional[Served]], verdict: Callable[[Served], tuple[str, str]],
                 clock: Callable[[], float], on_change: Callable[[str], None] = lambda label: None, ttl_s: float = CHECK_TTL_S):
        self.url, self.expected_label, self._look, self._verdict, self.clock, self.on_change, self.ttl_s = url, expected_label, look, verdict, clock, on_change, ttl_s
        self.verified_at: float = clock()
        self.baseline = ""
        self.looks = 0

    def _different(self, label: str) -> GaltonError:
        self.on_change(label)
        return changed(self.url, label, self.expected_label)

    def check(self, *, force: bool = False) -> str:
        """``cached`` (looked at recently), ``fresh`` (looked at now and it is what it was) or ``unknown`` (the server did not answer)."""
        if not force and self.clock() - self.verified_at < self.ttl_s:
            return "cached"
        served = self._look()
        self.looks += 1
        if served is None:
            return "unknown"
        verdict, label = self._verdict(served)
        if verdict == "different":
            raise self._different(label)
        self.verified_at = self.clock()
        return "fresh"

    def answer(self, served_model: str) -> None:
        """The ``model`` field a reply carried (empty when it carried none) must stay what the first reply said."""
        model = (served_model or "").strip()
        if not model:
            return
        if not self.baseline:
            self.baseline = model
        elif _cf(model) != _cf(self.baseline):
            raise self._different(model)
