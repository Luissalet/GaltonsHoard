"""Sharing a server politely: wait while somebody else is using it, and only go on once it has been quiet for a while.

Galton measures models that other people use too (a llama-server the family shares, an Ollama with loaded models). A measurement must never
sit between two questions of a chat, so a run

* waits before it starts while the server is busy (``wait_until_idle``), and
* looks again before every question, and pauses the same way when the server turned out to be busy with a request that is not its own.

"Quiet" means the server has not been seen busy for ``runner.idle_grace_s`` seconds: a person who just got an answer is usually about to
type the next message. A server that was idle from the start is not made to wait for that grace. The wait ends after ``runner.wait_idle_max_s``
seconds (0: never) or the run's own ``wait_s``.

How "busy" is known depends on the server:

* llama-server lists its slots on ``/slots``; one that is processing while we have no request in flight is somebody else's.
* Ollama reports no running requests, but ``/api/ps`` gives each loaded model an ``expires_at`` that moves whenever the model answers. The
  guard keeps the last picture it saw (refreshed after each question of ours); a different picture means somebody else used the server.
* A server that says nothing (no ``/slots``) is taken as idle: nothing can be known, so nothing is waited for.
"""

from __future__ import annotations

from typing import Callable, Optional

import httpx

from .errors import GaltonError
from .servers import slots_busy

#: seconds between two looks at a busy server
POLL_S = 3.0
#: a llama-server that looks busy between two of our questions is looked at once more this much later (the slot of our own last answer can take a moment to free)
SETTLE_S = 1.0


def wait_until_idle(busy: Callable[[], Optional[bool]], *, grace_s: float, max_s: Optional[float], cancel: Callable[[], bool], clock: Callable[[], float],
                    sleep: Callable[[float], None], on_wait: Callable[[float], None] = lambda waited: None, url: str = "", busy_now: bool = False,
                    poll_s: float = POLL_S) -> float:
    """Block while ``busy()`` is true and for ``grace_s`` seconds after it was last true; return the seconds waited (0 when the server was idle).

    ``busy_now`` says the caller has just seen it busy (a guard whose reading changes when it is read). ``max_s`` is the longest wait: ``None``
    waits for ever, ``0`` does not wait at all. Reaching it raises ``server_busy``. A cancelled run returns at once."""
    start = clock()
    if not busy_now and not busy():
        return 0.0
    last_busy = clock()
    while True:
        waited = clock() - start
        if cancel():
            return waited
        if max_s is not None and waited >= max_s:
            raise GaltonError("busy", "server_busy", url=url, limit=f"{max_s:.0f}")
        on_wait(waited)
        sleep(poll_s)
        now = clock()
        if busy():
            last_busy = now
        elif now - last_busy >= grace_s:
            return now - start


class LlamaGuard:
    """Is a llama-server busy with a request of somebody else? (Used between our questions, when none of ours is in flight.)"""

    settle_s = SETTLE_S

    def __init__(self, client_factory: Callable[[], httpx.Client], url: str):
        self.client_factory, self.url = client_factory, url

    def others_busy(self) -> Optional[bool]:
        try:
            with self.client_factory() as client:
                return slots_busy(client, self.url)
        except httpx.HTTPError:
            return None

    def mark_own(self) -> None:
        """Nothing to remember: a slot is busy or it is not."""


class OllamaGuard:
    """Has an Ollama server been used by somebody else since the last time we looked? (Its ``/api/ps`` shows when each model answered last.)"""

    settle_s = 0.0

    def __init__(self, client_factory: Callable[[], httpx.Client], url: str):
        self.client_factory, self.url = client_factory, url
        self.seen = self._picture()

    def _picture(self) -> Optional[frozenset]:
        try:
            with self.client_factory() as client:
                response = client.get(self.url.rstrip("/") + "/api/ps")
                response.raise_for_status()
                return frozenset((str(m.get("name") or m.get("model")), str(m.get("expires_at") or "")) for m in response.json().get("models") or [] if isinstance(m, dict))
        except (httpx.HTTPError, ValueError, AttributeError):
            return None

    def mark_own(self) -> None:
        """Our own question moved a model's ``expires_at``: that is the picture to compare with next time."""
        picture = self._picture()
        if picture is not None:
            self.seen = picture

    def others_busy(self) -> Optional[bool]:
        picture = self._picture()
        if picture is None:
            return None
        changed = self.seen is not None and picture != self.seen
        self.seen = picture
        return changed
