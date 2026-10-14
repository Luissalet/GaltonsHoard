"""Matrices and trials in Galton's database (tables ``hb_matrices`` and ``hb_trials``, migration 8)."""

from __future__ import annotations

import json
import time
from typing import Any, Callable, Iterable, Optional

from ..db import Database
from ..errors import GaltonError
from ..messages import recognise
from ..util import new_id

MATRIX_JSON = ("models", "harnesses", "tasks", "settings", "progress", "summary")
MATRIX_FIELDS = ("label", "state", "models", "harnesses", "tasks", "k", "settings", "progress", "total", "done", "source", "caller", "error", "cancel", "summary",
                 "started_ts", "finished_ts")
ACTIVE = ("queued", "waiting_gpu", "waiting_server", "running")
TERMINAL = ("done", "failed", "cancelled")
_DEFAULTS = {"models": [], "harnesses": [], "tasks": [], "settings": {}, "progress": {}, "summary": {}}
TRIAL_FIELDS = ("task_id", "harness", "sample", "outcome", "passed", "reason", "detail", "turns", "calls", "format_failures", "extra_calls", "extra_repeat",
                "extra_after_solved", "tokens_in", "tokens_out", "latency_ms", "final")
LIGHT = "id, matrix_id, contestant_id, digest, harness, task_id, sample, outcome, passed, reason, detail, turns, calls, format_failures, extra_calls, extra_repeat, extra_after_solved, tokens_in, tokens_out, latency_ms, ts"


def _enc(value: Any) -> Any:
    if isinstance(value, bool):
        return int(value)
    return value if isinstance(value, (str, int, float)) or value is None else json.dumps(value, ensure_ascii=False, default=str)


def _loads(value: Any, default: Any) -> Any:
    if not isinstance(value, str):
        return default if value is None else value
    try:
        return json.loads(value)
    except ValueError:
        return default


def _matrix(row: Any) -> Optional[dict[str, Any]]:
    if row is None:
        return None
    data = dict(row)
    for key in MATRIX_JSON:
        data[key] = _loads(data.get(key), _DEFAULTS[key])
    data["cancel"] = bool(data.get("cancel"))
    data["error"] = recognise(data.get("error"))
    data["label"] = recognise(data.get("label"), prefix="label_")
    for item in (data.get("progress") or {}).values():
        if isinstance(item, dict):
            item["error"] = recognise(item.get("error"))
            item["warnings"] = [recognise(w) for w in item.get("warnings") or []]
            item["runs_on"] = recognise(item.get("runs_on"))
    return data


def _trial(row: Any, *, trace: bool = False) -> dict[str, Any]:
    data = dict(row)
    data["passed"] = None if data.get("passed") is None else bool(data["passed"])
    data["detail"] = recognise(data.get("detail"))
    if trace:
        data["trace"] = _loads(data.get("trace"), [])
    return data


class MatrixStore:
    def __init__(self, db: Database, clock: Callable[[], float] = time.time):
        self.db, self.clock = db, clock

    # ------------------------------------------------------------------ matrices
    def create_matrix(self, **fields: Any) -> dict[str, Any]:
        mid = new_id("m")
        data = {k: _enc(v) for k, v in fields.items() if k in MATRIX_FIELDS}
        cols = ["id", "created_ts", *data]
        self.db.execute(f"INSERT INTO hb_matrices({', '.join(cols)}) VALUES ({', '.join('?' for _ in cols)})", [mid, self.clock(), *data.values()])
        return self.matrix(mid)

    def update_matrix(self, mid: str, **fields: Any) -> dict[str, Any]:
        data = {k: _enc(v) for k, v in fields.items() if k in MATRIX_FIELDS}
        if data:
            self.db.execute(f"UPDATE hb_matrices SET {', '.join(f'{k} = ?' for k in data)} WHERE id = ?", [*data.values(), mid])
        return self.matrix(mid)

    def find_matrix(self, ref: str) -> Optional[dict[str, Any]]:
        ref = (ref or "").strip()
        return _matrix(self.db.one("SELECT * FROM hb_matrices WHERE id = ?", (ref,)))

    def matrix(self, ref: str) -> dict[str, Any]:
        row = self.find_matrix(ref)
        if row is None:
            raise GaltonError("not_found", "hb_no_matrix", id=ref)
        return row

    def latest(self) -> Optional[dict[str, Any]]:
        """The running matrix, else the most recent one."""
        rows = self.matrices(limit=20)
        return next((m for m in rows if m["state"] in ACTIVE), rows[0] if rows else None)

    def matrices(self, *, states: Optional[Iterable[str]] = None, limit: int = 30) -> list[dict[str, Any]]:
        sql, params = "SELECT * FROM hb_matrices WHERE 1=1", []
        if states:
            states = list(states)
            sql += f" AND state IN ({', '.join('?' for _ in states)})"
            params.extend(states)
        sql += " ORDER BY created_ts DESC, id DESC LIMIT ?"
        params.append(max(1, min(int(limit), 500)))
        return [_matrix(r) for r in self.db.query(sql, params)]

    def delete_matrix(self, mid: str) -> None:
        self.db.execute("DELETE FROM hb_trials WHERE matrix_id = ?", (mid,))
        self.db.execute("DELETE FROM hb_matrices WHERE id = ?", (mid,))

    # ------------------------------------------------------------------ trials
    def add_trial(self, matrix_id: str, contestant_id: str, digest: str, trial: dict[str, Any]) -> int:
        data = {k: _enc(trial.get(k)) for k in TRIAL_FIELDS}
        data["trace"] = json.dumps(trial.get("trace") or [], ensure_ascii=False, default=str)
        data["final"] = str(trial.get("final") or "")[:4000]
        cols = ["matrix_id", "contestant_id", "digest", "ts", *data]
        cur = self.db.execute(f"INSERT INTO hb_trials({', '.join(cols)}) VALUES ({', '.join('?' for _ in cols)})", [matrix_id, contestant_id, digest, self.clock(), *data.values()])
        return int(cur.lastrowid)

    def trials(self, matrix_id: str, *, contestant_id: str = "", harness: str = "", task_id: str = "", outcome: str = "", trace: bool = False,
               limit: int = 100_000, offset: int = 0) -> list[dict[str, Any]]:
        sql, params = f"SELECT {'*' if trace else LIGHT} FROM hb_trials WHERE matrix_id = ?", [matrix_id]
        for column, value in (("contestant_id", contestant_id), ("harness", harness), ("task_id", task_id), ("outcome", outcome)):
            if value:
                sql += f" AND {column} = ?"
                params.append(value)
        sql += " ORDER BY id LIMIT ? OFFSET ?"
        params.extend([max(1, int(limit)), max(0, int(offset))])
        return [_trial(r, trace=trace) for r in self.db.query(sql, params)]

    def count_trials(self, matrix_id: str) -> int:
        return int(self.db.scalar("SELECT COUNT(*) FROM hb_trials WHERE matrix_id = ?", (matrix_id,), 0))

    def digests(self, matrix_id: str) -> dict[str, str]:
        """The digest each model had when its trials in this matrix were measured."""
        return {r["contestant_id"]: r["digest"] for r in self.db.query("SELECT contestant_id, digest FROM hb_trials WHERE matrix_id = ? GROUP BY contestant_id", (matrix_id,))}

    def delete_for_contestant(self, contestant_id: str) -> int:
        n = int(self.db.scalar("SELECT COUNT(*) FROM hb_trials WHERE contestant_id = ?", (contestant_id,), 0))
        self.db.execute("DELETE FROM hb_trials WHERE contestant_id = ?", (contestant_id,))
        return n
