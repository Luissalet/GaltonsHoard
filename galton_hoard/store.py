"""Typed access to the database: contestants, suites and cases, runs and results, judge cache, notices, route publications,
arena pairs and votes, activity."""

from __future__ import annotations

import json
import time
from typing import Any, Callable, Iterable, Optional

from . import messages
from .db import Database
from .errors import GaltonError
from .util import clean_aliases, fold, new_id

CONTESTANT_JSON = ("aliases", "meta")
CONTESTANT_FIELDS = ("key", "kind", "name", "url", "api", "model", "provider", "aliases", "family", "params_b", "quant", "context", "vision",
                     "digest", "remote", "remote_ok", "enabled", "path", "mmproj", "ollama_ref", "source", "size_bytes", "meta", "missing",
                     "adhoc", "last_seen_ts")
CONTESTANT_BOOLS = ("vision", "remote", "remote_ok", "enabled", "missing", "adhoc")
SUITE_FIELDS = ("name", "description", "category", "builtin", "version", "max_tokens", "notes", "position", "content_hash")
CASE_JSON = ("prompt", "images", "tools", "checker", "tags")
CASE_FIELDS = ("suite_id", "position", "title", "prompt", "images", "tools", "checker", "weight", "tags", "source", "notes", "reference",
               "max_tokens", "min_context")
RUN_JSON = ("suites", "contestants", "settings", "summary")
RUN_FIELDS = ("label", "state", "suites", "contestants", "settings", "source", "caller", "started_ts", "finished_ts", "error", "cancel", "summary", "discarded",
              "discard_reason", "discarded_ts")
RC_JSON = ("gpus", "warnings")
RC_FIELDS = ("state", "error", "load_ms", "vram_mb", "vram_method", "gpus", "context", "warnings", "total", "done", "digest", "runs_on", "spill",
             "started_ts", "finished_ts", "wait_since", "waited_s", "device")
RESULT_JSON = ("tool_calls", "detail")
RESULT_FIELDS = ("cpu", "truncated", "run_id", "contestant_id", "suite_id", "case_id", "category", "repeat", "output", "reasoning", "tool_calls", "score", "passed",
                 "detail", "latency_ms", "ttft_ms", "prompt_tokens", "completion_tokens", "decode_tps", "prompt_tps", "error", "skipped",
                 "unavailable", "self_judged", "judge_pending", "confidence", "weight", "digest")
OUTPUT_CAP = 20_000

ACTIVE_RUN_STATES = ("queued", "waiting_gpu", "waiting_server", "running")
#: SQL for "this result belongs to a run that was not discarded": every statistic, ranking, route, comparison and regression check reads results through it
LIVE = "run_id NOT IN (SELECT id FROM runs WHERE discarded = 1)"


def _loads(value: Any, default: Any) -> Any:
    if not isinstance(value, str):
        return value if value is not None else default
    try:
        return json.loads(value or json.dumps(default))
    except ValueError:
        return default


def _enc(value: Any) -> Any:
    if isinstance(value, bool):
        return int(value)
    return value if isinstance(value, (str, int, float)) or value is None else json.dumps(value, ensure_ascii=False, default=str)


def _row(row: Any, json_keys: Iterable[str], bool_keys: Iterable[str] = (), defaults: Optional[dict[str, Any]] = None) -> Optional[dict[str, Any]]:
    if row is None:
        return None
    data = dict(row)
    for key in json_keys:
        data[key] = _loads(data.get(key), (defaults or {}).get(key, {}))
    for key in bool_keys:
        data[key] = bool(data.get(key))
    return data


_C_DEFAULTS = {"aliases": [], "meta": {}}
_K_DEFAULTS = {"prompt": {}, "images": [], "tools": [], "checker": {}, "tags": []}
_R_DEFAULTS = {"suites": [], "contestants": [], "settings": {}, "summary": {}}
_RC_DEFAULTS = {"gpus": [], "warnings": []}
_RES_DEFAULTS = {"tool_calls": [], "detail": {}}


def _contestant(row: Any) -> Optional[dict[str, Any]]:
    return _row(row, CONTESTANT_JSON, CONTESTANT_BOOLS, _C_DEFAULTS)


def _case(row: Any) -> Optional[dict[str, Any]]:
    return _row(row, CASE_JSON, (), _K_DEFAULTS)


# Runs keep their sentences as plain English text; reading them back recognises the ones of the catalogue (``messages``), so the UI can show them in
# its own language. Anything else (a server's own error text) stays a plain string.
def _coded(data: Optional[dict[str, Any]], *fields: str) -> Optional[dict[str, Any]]:
    if data is None:
        return None
    for name in fields:
        value = data.get(name)
        data[name] = messages.recognise_all(value) if isinstance(value, list) else messages.recognise(value)
    return data


def _run(row: Any) -> Optional[dict[str, Any]]:
    data = _coded(_row(row, RUN_JSON, ("cancel", "discarded"), _R_DEFAULTS), "error")
    if data is not None:
        data["label"] = messages.recognise(data.get("label"), prefix="label_")  # labels are free text; only our own label sentences are coded
    return data


def _rc(row: Any) -> Optional[dict[str, Any]]:
    return _coded(_row(row, RC_JSON, (), _RC_DEFAULTS), "error", "warnings", "runs_on", "spill")


def _result(row: Any) -> Optional[dict[str, Any]]:
    data = _coded(_row(row, RESULT_JSON, ("passed", "skipped", "unavailable", "self_judged", "judge_pending", "truncated", "cpu"), _RES_DEFAULTS), "error")
    if data is not None and isinstance(data.get("detail"), dict) and isinstance(data["detail"].get("notes"), list):
        data["detail"]["notes"] = messages.recognise_all(data["detail"]["notes"])
    return data


class Store:
    def __init__(self, db: Database, clock: Callable[[], float] = time.time):
        self.db = db
        self.clock = clock

    # ------------------------------------------------------------------ contestants
    def create_contestant(self, **fields: Any) -> dict[str, Any]:
        now = self.clock()
        cid = fields.pop("id", None) or new_id("c", now)
        data = {k: _enc(v) for k, v in fields.items() if k in CONTESTANT_FIELDS}
        cols = ["id", "created_ts", "updated_ts", *data.keys()]
        self.db.execute(f"INSERT INTO contestants({', '.join(cols)}) VALUES ({', '.join('?' for _ in cols)})", [cid, now, now, *data.values()])
        return self.contestant(cid)

    def update_contestant(self, cid: str, **fields: Any) -> dict[str, Any]:
        data = {k: _enc(v) for k, v in fields.items() if k in CONTESTANT_FIELDS}
        if data:
            sets = ", ".join(f"{k} = ?" for k in data)
            self.db.execute(f"UPDATE contestants SET {sets}, updated_ts = ? WHERE id = ?", [*data.values(), self.clock(), cid])
        return self.contestant(cid)

    def contestant(self, cid: str) -> dict[str, Any]:
        row = _contestant(self.db.one("SELECT * FROM contestants WHERE id = ?", ((cid or "").strip(),)))
        if row is None:
            raise GaltonError("not_found", "no_model", ref=cid)
        return row

    def find_contestant(self, ref: str) -> Optional[dict[str, Any]]:
        """By id, exact key, exact name/alias (accent- and case-insensitive), then a unique name prefix."""
        ref = (ref or "").strip()
        if not ref:
            return None
        row = self.db.one("SELECT * FROM contestants WHERE id = ? OR key = ?", (ref, ref))
        if row is not None:
            return _contestant(row)
        needle = fold(ref)
        rows = [_contestant(r) for r in self.db.query("SELECT * FROM contestants ORDER BY enabled DESC, missing, created_ts")]
        for c in rows:
            if fold(c["name"]) == needle or needle in {fold(a) for a in c["aliases"]}:
                return c
        starts = [c for c in rows if fold(c["name"]).startswith(needle)]
        return starts[0] if len(starts) == 1 else None

    def contestant_by_key(self, key: str) -> Optional[dict[str, Any]]:
        return _contestant(self.db.one("SELECT * FROM contestants WHERE key = ?", (key,)))

    def contestants(self, *, kind: str = "", enabled: Optional[bool] = None, text: str = "", include_missing: bool = True,
                    include_adhoc: bool = True) -> list[dict[str, Any]]:
        sql, params = "SELECT * FROM contestants WHERE 1=1", []
        if kind:
            sql += " AND kind = ?"
            params.append(kind)
        if enabled is not None:
            sql += " AND enabled = ?"
            params.append(int(enabled))
        if not include_missing:
            sql += " AND missing = 0"
        if not include_adhoc:
            sql += " AND adhoc = 0"
        sql += " ORDER BY name COLLATE NOCASE, created_ts"
        rows = [_contestant(r) for r in self.db.query(sql, params)]
        if text:
            needle = fold(text)
            rows = [c for c in rows if needle in fold(" ".join([c["name"], c["model"], c["family"], c["quant"], c["provider"], c["kind"], *c["aliases"]]))]
        return rows

    def clean_aliases(self) -> int:
        """Drop from every contestant the aliases that no server or app can report (file paths, blob digests) and repeated names, which earlier
        versions stored. Returns how many contestants changed."""
        changed = 0
        for c in self.contestants():
            cleaned = clean_aliases(c["aliases"])
            if cleaned != c["aliases"]:
                self.db.execute("UPDATE contestants SET aliases = ? WHERE id = ?", (json.dumps(cleaned, ensure_ascii=False), c["id"]))
                changed += 1
        return changed

    def delete_contestant(self, cid: str) -> None:
        self.db.execute("DELETE FROM contestants WHERE id = ?", (cid,))

    # ------------------------------------------------------------------ suites and cases
    def create_suite(self, **fields: Any) -> dict[str, Any]:
        now = self.clock()
        sid = fields.pop("id", None) or new_id("s", now)
        data = {k: _enc(v) for k, v in fields.items() if k in SUITE_FIELDS}
        cols = ["id", "created_ts", "updated_ts", *data.keys()]
        self.db.execute(f"INSERT INTO suites({', '.join(cols)}) VALUES ({', '.join('?' for _ in cols)})", [sid, now, now, *data.values()])
        return self.suite(sid)

    def update_suite(self, sid: str, **fields: Any) -> dict[str, Any]:
        data = {k: _enc(v) for k, v in fields.items() if k in SUITE_FIELDS}
        if data:
            sets = ", ".join(f"{k} = ?" for k in data)
            self.db.execute(f"UPDATE suites SET {sets}, updated_ts = ? WHERE id = ?", [*data.values(), self.clock(), sid])
        return self.suite(sid)

    def suite(self, sid: str) -> dict[str, Any]:
        row = self.find_suite(sid)
        if row is None:
            raise GaltonError("not_found", "no_suite", ref=sid)
        return row

    def find_suite(self, ref: str) -> Optional[dict[str, Any]]:
        """By id, then by name (accent- and case-insensitive)."""
        ref = (ref or "").strip()
        row = self.db.one("SELECT * FROM suites WHERE id = ?", (ref,))
        if row is None and ref and not ref.startswith("s_"):
            row = self.db.one("SELECT * FROM suites WHERE id = ?", ("s_" + ref,))
        if row is None and ref:
            needle = fold(ref)
            for r in self.db.query("SELECT * FROM suites"):
                if fold(r["name"]) == needle:
                    row = r
                    break
        if row is None:
            return None
        data = dict(row)
        data["builtin"] = bool(data["builtin"])
        return data

    def suites(self) -> list[dict[str, Any]]:
        out = []
        for r in self.db.query("SELECT s.*, (SELECT COUNT(*) FROM cases c WHERE c.suite_id = s.id) AS cases FROM suites s ORDER BY s.builtin DESC, s.position, s.name"):
            d = dict(r)
            d["builtin"] = bool(d["builtin"])
            out.append(d)
        return out

    def delete_suite(self, sid: str) -> None:
        self.db.execute("DELETE FROM suites WHERE id = ?", (sid,))

    def create_case(self, **fields: Any) -> dict[str, Any]:
        now = self.clock()
        kid = fields.pop("id", None) or new_id("k", now)
        data = {k: _enc(v) for k, v in fields.items() if k in CASE_FIELDS}
        cols = ["id", "created_ts", "updated_ts", *data.keys()]
        self.db.execute(f"INSERT INTO cases({', '.join(cols)}) VALUES ({', '.join('?' for _ in cols)})", [kid, now, now, *data.values()])
        return self.case(kid)

    def update_case(self, kid: str, **fields: Any) -> dict[str, Any]:
        data = {k: _enc(v) for k, v in fields.items() if k in CASE_FIELDS}
        if data:
            sets = ", ".join(f"{k} = ?" for k in data)
            self.db.execute(f"UPDATE cases SET {sets}, updated_ts = ? WHERE id = ?", [*data.values(), self.clock(), kid])
        return self.case(kid)

    def case(self, kid: str) -> dict[str, Any]:
        row = _case(self.db.one("SELECT * FROM cases WHERE id = ?", ((kid or "").strip(),)))
        if row is None:
            raise GaltonError("not_found", "no_case", id=kid)
        return row

    def find_case(self, kid: str) -> Optional[dict[str, Any]]:
        return _case(self.db.one("SELECT * FROM cases WHERE id = ?", ((kid or "").strip(),)))

    def cases(self, suite_id: str, *, limit: int = 100_000, offset: int = 0) -> list[dict[str, Any]]:
        rows = self.db.query("SELECT * FROM cases WHERE suite_id = ? ORDER BY position, created_ts, id LIMIT ? OFFSET ?", (suite_id, limit, offset))
        return [_case(r) for r in rows]

    def count_cases(self, suite_id: str) -> int:
        row = self.db.one("SELECT COUNT(*) FROM cases WHERE suite_id = ?", (suite_id,))
        return int(row[0]) if row else 0

    def next_case_position(self, suite_id: str) -> int:
        row = self.db.one("SELECT COALESCE(MAX(position), -1) + 1 FROM cases WHERE suite_id = ?", (suite_id,))
        return int(row[0]) if row else 0

    def delete_case(self, kid: str) -> None:
        self.db.execute("DELETE FROM cases WHERE id = ?", (kid,))

    def case_suite_map(self) -> dict[str, dict[str, Any]]:
        """case id -> {suite_id, category} for every case (results are grouped by category through it)."""
        rows = self.db.query("SELECT c.id AS case_id, c.suite_id AS suite_id, s.category AS category FROM cases c JOIN suites s ON s.id = c.suite_id")
        return {r["case_id"]: {"suite_id": r["suite_id"], "category": r["category"]} for r in rows}

    # ------------------------------------------------------------------ runs
    def create_run(self, **fields: Any) -> dict[str, Any]:
        now = self.clock()
        rid = fields.pop("id", None) or new_id("r", now)
        data = {k: _enc(v) for k, v in fields.items() if k in RUN_FIELDS}
        cols = ["id", "created_ts", *data.keys()]
        self.db.execute(f"INSERT INTO runs({', '.join(cols)}) VALUES ({', '.join('?' for _ in cols)})", [rid, now, *data.values()])
        return self.run(rid)

    def update_run(self, rid: str, **fields: Any) -> dict[str, Any]:
        data = {k: _enc(v) for k, v in fields.items() if k in RUN_FIELDS}
        if data:
            sets = ", ".join(f"{k} = ?" for k in data)
            self.db.execute(f"UPDATE runs SET {sets} WHERE id = ?", [*data.values(), rid])
        return self.run(rid)

    def run(self, rid: str) -> dict[str, Any]:
        row = _run(self.db.one("SELECT * FROM runs WHERE id = ?", ((rid or "").strip(),)))
        if row is None:
            raise GaltonError("not_found", "no_run", id=rid)
        return row

    def find_run(self, rid: str) -> Optional[dict[str, Any]]:
        return _run(self.db.one("SELECT * FROM runs WHERE id = ?", ((rid or "").strip(),)))

    def runs(self, *, states: Optional[Iterable[str]] = None, limit: int = 50, source: str = "") -> list[dict[str, Any]]:
        sql, params = "SELECT * FROM runs WHERE 1=1", []
        if states:
            states = list(states)
            sql += f" AND state IN ({', '.join('?' for _ in states)})"
            params.extend(states)
        if source:
            sql += " AND source = ?"
            params.append(source)
        sql += " ORDER BY created_ts DESC, id DESC LIMIT ?"
        params.append(max(1, min(int(limit), 1000)))
        return [_run(r) for r in self.db.query(sql, params)]

    def delete_run(self, rid: str) -> None:
        self.db.execute("DELETE FROM run_contestants WHERE run_id = ?", (rid,))
        self.db.execute("DELETE FROM results WHERE run_id = ?", (rid,))
        self.db.execute("DELETE FROM runs WHERE id = ?", (rid,))

    def upsert_run_contestant(self, run_id: str, contestant_id: str, **fields: Any) -> dict[str, Any]:
        data = {k: _enc(v) for k, v in fields.items() if k in RC_FIELDS}
        exists = self.db.one("SELECT 1 FROM run_contestants WHERE run_id = ? AND contestant_id = ?", (run_id, contestant_id))
        if exists is None:
            cols = ["run_id", "contestant_id", *data.keys()]
            self.db.execute(f"INSERT INTO run_contestants({', '.join(cols)}) VALUES ({', '.join('?' for _ in cols)})", [run_id, contestant_id, *data.values()])
        elif data:
            sets = ", ".join(f"{k} = ?" for k in data)
            self.db.execute(f"UPDATE run_contestants SET {sets} WHERE run_id = ? AND contestant_id = ?", [*data.values(), run_id, contestant_id])
        return self.run_contestant(run_id, contestant_id)

    def run_contestant(self, run_id: str, contestant_id: str) -> dict[str, Any]:
        row = _rc(self.db.one("SELECT * FROM run_contestants WHERE run_id = ? AND contestant_id = ?", (run_id, contestant_id)))
        if row is None:
            raise GaltonError("not_found", "no_run_contestant", contestant=contestant_id, run=run_id)
        return row

    def run_contestants(self, run_id: str) -> list[dict[str, Any]]:
        return [_rc(r) for r in self.db.query("SELECT * FROM run_contestants WHERE run_id = ? ORDER BY rowid", (run_id,))]

    def measurements(self, contestant_id: str, *, limit: int = 20) -> list[dict[str, Any]]:
        rows = self.db.query("SELECT rc.*, r.finished_ts AS run_finished FROM run_contestants rc JOIN runs r ON r.id = rc.run_id "
                             "WHERE rc.contestant_id = ? AND rc.state IN ('done', 'failed') AND r.discarded = 0 ORDER BY COALESCE(rc.finished_ts, 0) DESC LIMIT ?",
                             (contestant_id, limit))
        return [_rc(r) for r in rows]

    # ------------------------------------------------------------------ results
    def add_result(self, **fields: Any) -> int:
        data = {k: _enc(v) for k, v in fields.items() if k in RESULT_FIELDS}
        for key in ("output", "reasoning"):
            if key in data and isinstance(data[key], str) and len(data[key]) > OUTPUT_CAP:
                data[key] = data[key][:OUTPUT_CAP]
        cols = ["ts", *data.keys()]
        cur = self.db.execute(f"INSERT INTO results({', '.join(cols)}) VALUES ({', '.join('?' for _ in cols)})", [self.clock(), *data.values()])
        return int(cur.lastrowid)

    def update_result(self, result_id: int, **fields: Any) -> None:
        data = {k: _enc(v) for k, v in fields.items() if k in RESULT_FIELDS}
        if data:
            sets = ", ".join(f"{k} = ?" for k in data)
            self.db.execute(f"UPDATE results SET {sets} WHERE id = ?", [*data.values(), result_id])

    def result(self, result_id: int) -> Optional[dict[str, Any]]:
        return _result(self.db.one("SELECT * FROM results WHERE id = ?", (result_id,)))

    def results(self, *, run_id: str = "", contestant_id: str = "", suite_id: str = "", case_id: str = "", only_failed: bool = False,
                only_passed: bool = False, after_id: int = 0, limit: int = 200, offset: int = 0, with_output: bool = True,
                include_skipped: bool = True, live_only: bool = False) -> list[dict[str, Any]]:
        """Result rows with their text. ``live_only`` leaves out the results of discarded runs (a run's own page still shows them)."""
        cols = "*" if with_output else ("id, run_id, contestant_id, suite_id, case_id, category, repeat, score, passed, detail, latency_ms, ttft_ms, "
                                        "prompt_tokens, completion_tokens, decode_tps, prompt_tps, error, skipped, unavailable, self_judged, "
                                        "judge_pending, truncated, cpu, confidence, weight, digest, ts, '' AS output, '' AS reasoning, '[]' AS tool_calls")
        sql, params = f"SELECT {cols} FROM results WHERE 1=1", []
        for column, value in (("run_id", run_id), ("contestant_id", contestant_id), ("suite_id", suite_id), ("case_id", case_id)):
            if value:
                sql += f" AND {column} = ?"
                params.append(value)
        if only_failed:
            sql += " AND passed = 0 AND skipped = 0 AND unavailable = 0 AND judge_pending = 0"
        if only_passed:
            sql += " AND passed = 1"
        if not include_skipped:
            sql += " AND skipped = 0"
        if live_only:
            sql += f" AND {LIVE}"
        if after_id:
            sql += " AND id > ?"
            params.append(after_id)
        sql += " ORDER BY id LIMIT ? OFFSET ?"
        params.extend([max(1, min(int(limit), 100_000)), max(0, int(offset))])
        return [_result(r) for r in self.db.query(sql, params)]

    def count_results(self, run_id: str, contestant_id: str = "") -> dict[str, int]:
        sql = ("SELECT COUNT(*) AS n, COALESCE(SUM(passed), 0) AS passed, COALESCE(SUM(skipped), 0) AS skipped, COALESCE(SUM(unavailable), 0) AS unavailable, "
               "COALESCE(SUM(CASE WHEN error != '' AND skipped = 0 THEN 1 ELSE 0 END), 0) AS errors, COALESCE(SUM(truncated), 0) AS truncated FROM results WHERE run_id = ?")
        params: list[Any] = [run_id]
        if contestant_id:
            sql += " AND contestant_id = ?"
            params.append(contestant_id)
        row = self.db.one(sql, params)
        return {k: int(row[k] or 0) for k in ("n", "passed", "skipped", "unavailable", "errors", "truncated")} if row else {}

    def pending_judge(self, run_id: str = "") -> list[dict[str, Any]]:
        """Results waiting for the judge (of one run, or of all)."""
        if run_id:
            return [_result(r) for r in self.db.query("SELECT * FROM results WHERE run_id = ? AND judge_pending = 1 ORDER BY id", (run_id,))]
        return [_result(r) for r in self.db.query(f"SELECT * FROM results WHERE judge_pending = 1 AND {LIVE} ORDER BY id")]

    def scoring_rows(self, *, contestant_ids: Optional[Iterable[str]] = None, suite_ids: Optional[Iterable[str]] = None,
                     case_ids: Optional[Iterable[str]] = None, since_ts: float = 0.0, run_ids: Optional[Iterable[str]] = None) -> list[dict[str, Any]]:
        """Slim result rows (no text) for statistics. Skipped, unavailable and pending-judge rows are left out, and so are the results of discarded runs."""
        sql = ("SELECT id, run_id, contestant_id, suite_id, case_id, category, repeat, score, passed, latency_ms, ttft_ms, prompt_tokens, completion_tokens, "
               "decode_tps, prompt_tps, confidence, weight, digest, ts, self_judged, truncated, cpu FROM results "
               f"WHERE skipped = 0 AND unavailable = 0 AND judge_pending = 0 AND error = '' AND {LIVE}")
        params: list[Any] = []
        for column, values in (("contestant_id", contestant_ids), ("suite_id", suite_ids), ("case_id", case_ids), ("run_id", run_ids)):
            if values is not None:
                values = list(values)
                if not values:
                    return []
                sql += f" AND {column} IN ({', '.join('?' for _ in values)})"
                params.extend(values)
        if since_ts:
            sql += " AND ts >= ?"
            params.append(since_ts)
        sql += " ORDER BY id"
        return [dict(r) for r in self.db.query(sql, params)]

    def speed_rows(self, contestant_id: str, *, since_ts: float = 0.0, digest: str = "") -> list[dict[str, Any]]:
        sql = ("SELECT decode_tps, prompt_tps, ttft_ms, latency_ms, completion_tokens, cpu FROM results WHERE contestant_id = ? AND skipped = 0 "
               f"AND unavailable = 0 AND error = '' AND decode_tps IS NOT NULL AND {LIVE}")
        params: list[Any] = [contestant_id]
        if digest:
            sql += " AND digest = ?"
            params.append(digest)
        if since_ts:
            sql += " AND ts >= ?"
            params.append(since_ts)
        return [dict(r) for r in self.db.query(sql, params)]

    def last_measured(self) -> dict[str, dict[str, Any]]:
        """Per contestant: when it was last measured, with which digest, and how many checked results it has."""
        rows = self.db.query("SELECT contestant_id, MAX(ts) AS last_ts, COUNT(*) AS n FROM results WHERE skipped = 0 AND unavailable = 0 "
                             f"AND error = '' AND {LIVE} GROUP BY contestant_id")
        out = {r["contestant_id"]: {"last_ts": r["last_ts"], "n": int(r["n"]), "digests": []} for r in rows}
        for r in self.db.query("SELECT contestant_id, digest, MAX(ts) AS last_ts FROM results WHERE skipped = 0 AND unavailable = 0 AND error = '' "
                               f"AND {LIVE} GROUP BY contestant_id, digest ORDER BY last_ts"):
            if r["contestant_id"] in out:
                out[r["contestant_id"]]["digests"].append({"digest": r["digest"], "last_ts": r["last_ts"]})
        return out

    # ------------------------------------------------------------------ judge cache and Ollama info
    def judge_cached(self, key: str) -> Optional[dict[str, Any]]:
        row = self.db.one("SELECT result FROM judge_cache WHERE hash = ?", (key,))
        return _loads(row["result"], None) if row else None

    def judge_store(self, key: str, result: dict[str, Any]) -> None:
        self.db.execute("INSERT INTO judge_cache(hash, result, ts) VALUES (?, ?, ?) ON CONFLICT(hash) DO UPDATE SET result = excluded.result, ts = excluded.ts",
                        (key, json.dumps(result, ensure_ascii=False), self.clock()))

    def ollama_info(self, digest: str) -> Optional[dict[str, Any]]:
        row = self.db.one("SELECT info FROM ollama_info WHERE digest = ?", (digest,))
        return _loads(row["info"], None) if row else None

    def store_ollama_info(self, digest: str, info: dict[str, Any]) -> None:
        self.db.execute("INSERT INTO ollama_info(digest, info, ts) VALUES (?, ?, ?) ON CONFLICT(digest) DO UPDATE SET info = excluded.info, ts = excluded.ts",
                        (digest, json.dumps(info, ensure_ascii=False), self.clock()))

    # ------------------------------------------------------------------ notices
    def add_notice(self, *, kind: str, params: Optional[dict[str, Any]] = None, severity: str = "medium", data: Optional[dict[str, Any]] = None,
                   contestant_id: Optional[str] = None, dedupe: str = "", once: bool = False) -> Optional[int]:
        """Store a notice of ``kind`` (``messages.NOTICES``) with its ``params``: the English title and body are made from them, and the UI formats the
        same kind and params in its own language. Returns the notice id, or None when it would repeat one: the same ``dedupe`` key, or, with ``once``,
        the same kind for the same model and the same digest (``data["digest"]``)."""
        if dedupe and self.db.one("SELECT 1 FROM notices WHERE dedupe = ? LIMIT 1", (dedupe,)):
            return None
        data = dict(data or {})
        if once and contestant_id and self.has_notice(kind, contestant_id, str(data.get("digest") or "")):
            return None
        params = {k: messages.scalar(v) for k, v in (params or {}).items()}
        title, body = messages.notice_text(kind, params)
        data["params"] = params
        cur = self.db.execute("INSERT INTO notices(ts, kind, severity, title, body, data, contestant_id, dedupe) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                              (self.clock(), kind, severity, title, body, json.dumps(data, ensure_ascii=False, default=str), contestant_id, dedupe))
        return int(cur.lastrowid)

    def has_notice(self, kind: str, contestant_id: str, digest: str = "") -> bool:
        """Is there already a notice of ``kind`` for this model (and, when given, this digest)?"""
        for row in self.db.query("SELECT data FROM notices WHERE kind = ? AND contestant_id = ?", (kind, contestant_id)):
            if not digest or str(_loads(row["data"], {}).get("digest") or "") in ("", digest):
                return True
        return False

    def notices(self, *, unseen: bool = False, kind: str = "", limit: int = 50) -> list[dict[str, Any]]:
        sql, params = "SELECT * FROM notices WHERE 1=1", []
        if unseen:
            sql += " AND seen = 0"
        if kind:
            sql += " AND kind = ?"
            params.append(kind)
        sql += " ORDER BY ts DESC, id DESC LIMIT ?"
        params.append(max(1, min(limit, 500)))
        out = []
        for r in self.db.query(sql, params):
            d = dict(r)
            d["data"] = _loads(d.get("data"), {})
            d["params"] = d["data"].get("params") or messages.notice_params(d["kind"], d["title"], d["body"])
            d["seen"] = bool(d["seen"])
            out.append(d)
        return out

    def delete_run_notices(self, run_id: str, kinds: Iterable[str]) -> int:
        """Remove the notices of these kinds that a run raised (a regression found by a run that is then discarded is not true any more)."""
        removed = 0
        for row in self.db.query("SELECT id, data FROM notices WHERE kind IN (%s)" % ", ".join("?" for _ in list(kinds)), list(kinds)):
            if _loads(row["data"], {}).get("run") == run_id:
                self.db.execute("DELETE FROM notices WHERE id = ?", (row["id"],))
                removed += 1
        return removed

    def mark_notices_seen(self, before: Optional[float] = None) -> int:
        return self.db.execute("UPDATE notices SET seen = 1 WHERE seen = 0 AND ts <= ?", (before if before is not None else self.clock() + 1,)).rowcount

    # ------------------------------------------------------------------ route publications
    def add_publication(self, *, path: str, doc: dict[str, Any], diff: dict[str, Any], note: str = "") -> int:
        cur = self.db.execute("INSERT INTO route_publications(ts, path, doc, diff, note) VALUES (?, ?, ?, ?, ?)",
                              (self.clock(), path, json.dumps(doc, ensure_ascii=False), json.dumps(diff, ensure_ascii=False), note))
        return int(cur.lastrowid)

    def publications(self, limit: int = 20) -> list[dict[str, Any]]:
        out = []
        for r in self.db.query("SELECT * FROM route_publications ORDER BY id DESC LIMIT ?", (limit,)):
            d = dict(r)
            d["doc"] = _loads(d["doc"], {})
            d["diff"] = _loads(d["diff"], {})
            out.append(d)
        return out

    # ------------------------------------------------------------------ arena
    def add_arena_pair(self, **fields: Any) -> str:
        pid = new_id("a", self.clock())
        self.db.execute("INSERT INTO arena_pairs(id, case_id, category, a_result, b_result, a_contestant, b_contestant, created_ts) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                        (pid, fields["case_id"], fields.get("category", ""), fields["a_result"], fields["b_result"], fields["a_contestant"],
                         fields["b_contestant"], self.clock()))
        return pid

    def arena_pair(self, pid: str) -> Optional[dict[str, Any]]:
        row = self.db.one("SELECT * FROM arena_pairs WHERE id = ?", (pid,))
        return dict(row) if row else None

    def mark_pair_voted(self, pid: str) -> None:
        self.db.execute("UPDATE arena_pairs SET voted = 1 WHERE id = ?", (pid,))

    def add_vote(self, *, pair_id: str, case_id: str, category: str, a: str, b: str, vote: str) -> int:
        cur = self.db.execute("INSERT INTO arena_votes(pair_id, case_id, category, a, b, vote, ts) VALUES (?, ?, ?, ?, ?, ?, ?)",
                              (pair_id, case_id, category, a, b, vote, self.clock()))
        return int(cur.lastrowid)

    def votes(self, category: str = "") -> list[dict[str, Any]]:
        """The votes that count: a vote on a pair with an answer of a discarded run is left out."""
        sql = ("SELECT v.* FROM arena_votes v LEFT JOIN arena_pairs p ON p.id = v.pair_id WHERE NOT EXISTS "
               "(SELECT 1 FROM results r JOIN runs u ON u.id = r.run_id WHERE u.discarded = 1 AND r.id IN (p.a_result, p.b_result))")
        params: list[Any] = []
        if category:
            sql += " AND v.category = ?"
            params.append(category)
        return [dict(r) for r in self.db.query(sql + " ORDER BY v.id", params)]

    def voted_pairs(self) -> set[tuple[str, str, str]]:
        """(case, a, b) combinations that already have a vote, in either order."""
        out: set[tuple[str, str, str]] = set()
        for r in self.db.query("SELECT case_id, a, b FROM arena_votes"):
            out.add((r["case_id"], r["a"], r["b"]))
            out.add((r["case_id"], r["b"], r["a"]))
        return out

    # ------------------------------------------------------------------ activity
    def add_activity(self, kind: str, ref: str, ok: bool, duration_ms: int, detail: str) -> None:
        self.db.execute("INSERT INTO activity(ts, kind, ref, ok, duration_ms, detail) VALUES (?, ?, ?, ?, ?, ?)",
                        (self.clock(), kind, ref, int(ok), duration_ms, detail[:500]))
        self.db.execute("DELETE FROM activity WHERE id IN (SELECT id FROM activity ORDER BY id DESC LIMIT -1 OFFSET 2000)")

    def activity(self, kind: str = "", limit: int = 50) -> list[dict[str, Any]]:
        if kind:
            rows = self.db.query("SELECT * FROM activity WHERE kind = ? ORDER BY id DESC LIMIT ?", (kind, limit))
        else:
            rows = self.db.query("SELECT * FROM activity ORDER BY id DESC LIMIT ?", (limit,))
        return [dict(r) for r in rows]

    # ------------------------------------------------------------------ counts
    def counts(self) -> dict[str, int]:
        one = lambda sql, p=(): int((self.db.one(sql, p) or [0])[0] or 0)  # noqa: E731
        return {
            "models": one("SELECT COUNT(*) FROM contestants WHERE missing = 0 AND adhoc = 0"),
            "models_enabled": one("SELECT COUNT(*) FROM contestants WHERE missing = 0 AND enabled = 1"),
            "suites": one("SELECT COUNT(*) FROM suites"),
            "cases": one("SELECT COUNT(*) FROM cases"),
            "runs": one("SELECT COUNT(*) FROM runs"),
            "runs_active": one("SELECT COUNT(*) FROM runs WHERE state IN ('queued', 'waiting_gpu', 'waiting_server', 'running')"),
            "results": one(f"SELECT COUNT(*) FROM results WHERE skipped = 0 AND unavailable = 0 AND error = '' AND {LIVE}"),
            "runs_discarded": one("SELECT COUNT(*) FROM runs WHERE discarded = 1"),
            "votes": one("SELECT COUNT(*) FROM arena_votes"),
            "unseen_notices": one("SELECT COUNT(*) FROM notices WHERE seen = 0"),
        }
