"""Agent-bench campaigns and trials in Galton's database (tables ``ab_campaigns`` and ``ab_trials``, migration 9).

A trial row is a copy of what the runner wrote in the trial's folder; it is upserted by ``(campaign_id, trial_name)``, so reading the job folders
again never duplicates a row and a trial that was still running the last time is completed in place.
"""

from __future__ import annotations

import json
import time
from typing import Any, Callable, Iterable, Optional

from ..db import Database
from ..errors import GaltonError
from ..util import new_id

CAMPAIGN_JSON = ("benchmarks", "harnesses", "harness_lines", "splits", "summary")
CAMPAIGN_FIELDS = ("label", "name", "state", "benchmarks", "harnesses", "harness_lines", "model", "base_url", "k", "splits", "slots", "per_cell",
                   "retries", "launch", "source", "caller", "error", "cancel", "summary", "started_ts", "finished_ts")
ACTIVE = ("queued", "running")
TERMINAL = ("done", "failed", "cancelled")
TRIAL_FIELDS = ("benchmark", "harness", "task", "split", "job", "trial_name", "outcome", "passed", "reward", "reason", "error_type", "error",
                "tokens_in", "tokens_out", "tokens_cache", "agent_s", "setup_s", "total_s", "agent_version", "trial_dir", "started_at", "finished_at")
_DEFAULTS: dict[str, Any] = {"benchmarks": [], "harnesses": [], "harness_lines": [], "splits": ["dev", "test"], "summary": {}}


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


def _campaign(row: Any) -> Optional[dict[str, Any]]:
    if row is None:
        return None
    data = dict(row)
    for key in CAMPAIGN_JSON:
        data[key] = _loads(data.get(key), _DEFAULTS[key])
    data["cancel"] = bool(data.get("cancel"))
    data["launch"] = bool(data.get("launch"))
    return data


def _trial(row: Any) -> dict[str, Any]:
    data = dict(row)
    data["passed"] = None if data.get("passed") is None else bool(data["passed"])
    return data


class AgentBenchStore:
    def __init__(self, db: Database, clock: Callable[[], float] = time.time):
        self.db, self.clock = db, clock

    # -------------------------------------------------------------- campaigns
    def create(self, **fields: Any) -> dict[str, Any]:
        cid = new_id("ab")
        data = {k: _enc(v) for k, v in fields.items() if k in CAMPAIGN_FIELDS}
        data.setdefault("name", cid)
        cols = ["id", "created_ts", *data]
        self.db.execute(f"INSERT INTO ab_campaigns({', '.join(cols)}) VALUES ({', '.join('?' for _ in cols)})", [cid, self.clock(), *data.values()])
        return self.campaign(cid)

    def update(self, cid: str, **fields: Any) -> dict[str, Any]:
        data = {k: _enc(v) for k, v in fields.items() if k in CAMPAIGN_FIELDS}
        if data:
            self.db.execute(f"UPDATE ab_campaigns SET {', '.join(f'{k} = ?' for k in data)} WHERE id = ?", [*data.values(), cid])
        return self.campaign(cid)

    def find(self, ref: str) -> Optional[dict[str, Any]]:
        ref = (ref or "").strip()
        return _campaign(self.db.one("SELECT * FROM ab_campaigns WHERE id = ? OR name = ? ORDER BY created_ts DESC LIMIT 1", (ref, ref)))

    def campaign(self, ref: str) -> dict[str, Any]:
        row = self.find(ref)
        if row is None:
            raise GaltonError("not_found", "ab_no_campaign", id=ref)
        return row

    def campaigns(self, *, states: Optional[Iterable[str]] = None, limit: int = 30) -> list[dict[str, Any]]:
        sql, params = "SELECT * FROM ab_campaigns WHERE 1=1", []
        if states:
            states = list(states)
            sql += f" AND state IN ({', '.join('?' for _ in states)})"
            params.extend(states)
        sql += " ORDER BY created_ts DESC, id DESC LIMIT ?"
        params.append(max(1, min(int(limit), 500)))
        return [_campaign(r) for r in self.db.query(sql, params)]

    def latest(self) -> Optional[dict[str, Any]]:
        rows = self.campaigns(limit=20)
        return next((c for c in rows if c["state"] in ACTIVE), rows[0] if rows else None)

    def delete(self, cid: str) -> None:
        self.db.execute("DELETE FROM ab_trials WHERE campaign_id = ?", (cid,))
        self.db.execute("DELETE FROM ab_campaigns WHERE id = ?", (cid,))

    # -------------------------------------------------------------- trials
    def upsert_trials(self, cid: str, rows: Iterable[dict[str, Any]]) -> int:
        n = 0
        now = self.clock()
        for row in rows:
            if not row.get("trial_name"):
                continue
            data = {k: _enc(row.get(k)) for k in TRIAL_FIELDS}
            data["error"] = str(row.get("error") or "")[:2000]
            cols = ["campaign_id", "ts", *data]
            updates = ", ".join(f"{k} = excluded.{k}" for k in data if k != "trial_name")
            self.db.execute(
                f"INSERT INTO ab_trials({', '.join(cols)}) VALUES ({', '.join('?' for _ in cols)}) "
                f"ON CONFLICT(campaign_id, trial_name) DO UPDATE SET {updates}, ts = excluded.ts",
                [cid, now, *data.values()])
            n += 1
        return n

    def trials(self, cid: str, *, benchmark: str = "", harness: str = "", task: str = "", split: str = "", outcome: str = "",
               limit: int = 100_000, offset: int = 0) -> list[dict[str, Any]]:
        sql, params = "SELECT * FROM ab_trials WHERE campaign_id = ?", [cid]
        for column, value in (("benchmark", benchmark), ("harness", harness), ("task", task), ("split", split), ("outcome", outcome)):
            if value:
                sql += f" AND {column} = ?"
                params.append(value)
        sql += " ORDER BY benchmark, harness, task, id LIMIT ? OFFSET ?"
        params.extend([max(1, int(limit)), max(0, int(offset))])
        return [_trial(r) for r in self.db.query(sql, params)]

    def trial(self, tid: int) -> dict[str, Any]:
        row = self.db.one("SELECT * FROM ab_trials WHERE id = ?", (int(tid),))
        if row is None:
            raise GaltonError("not_found", "ab_no_trial", id=str(tid))
        return _trial(row)

    def count(self, cid: str) -> dict[str, int]:
        rows = self.db.query("SELECT outcome, COUNT(*) AS n FROM ab_trials WHERE campaign_id = ? GROUP BY outcome", (cid,))
        return {r["outcome"]: int(r["n"]) for r in rows}
