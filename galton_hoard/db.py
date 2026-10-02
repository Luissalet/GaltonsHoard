"""SQLite connection (WAL) and ordered schema migrations."""

from __future__ import annotations

import sqlite3
import threading
from pathlib import Path

MIGRATIONS: list[str] = [
    # 1: settings, contestants, suites and cases, runs and results, judge cache, notices, route publications, arena, activity
    """
    CREATE TABLE settings (key TEXT PRIMARY KEY, value TEXT NOT NULL);
    CREATE TABLE contestants (
      id TEXT PRIMARY KEY,
      key TEXT NOT NULL UNIQUE,
      kind TEXT NOT NULL,
      name TEXT NOT NULL DEFAULT '',
      url TEXT NOT NULL DEFAULT '',
      api TEXT NOT NULL DEFAULT '',
      model TEXT NOT NULL DEFAULT '',
      provider TEXT NOT NULL DEFAULT '',
      aliases TEXT NOT NULL DEFAULT '[]',
      family TEXT NOT NULL DEFAULT '',
      params_b REAL,
      quant TEXT NOT NULL DEFAULT '',
      context INTEGER,
      vision INTEGER NOT NULL DEFAULT 0,
      digest TEXT NOT NULL DEFAULT '',
      remote INTEGER NOT NULL DEFAULT 0,
      remote_ok INTEGER NOT NULL DEFAULT 0,
      enabled INTEGER NOT NULL DEFAULT 1,
      path TEXT NOT NULL DEFAULT '',
      mmproj TEXT NOT NULL DEFAULT '',
      ollama_ref TEXT NOT NULL DEFAULT '',
      source TEXT NOT NULL DEFAULT 'manual',
      size_bytes INTEGER,
      meta TEXT NOT NULL DEFAULT '{}',
      missing INTEGER NOT NULL DEFAULT 0,
      adhoc INTEGER NOT NULL DEFAULT 0,
      last_seen_ts REAL,
      created_ts REAL NOT NULL,
      updated_ts REAL NOT NULL
    );
    CREATE TABLE suites (
      id TEXT PRIMARY KEY,
      name TEXT NOT NULL,
      description TEXT NOT NULL DEFAULT '',
      category TEXT NOT NULL DEFAULT 'custom',
      builtin INTEGER NOT NULL DEFAULT 0,
      version INTEGER NOT NULL DEFAULT 1,
      max_tokens INTEGER NOT NULL DEFAULT 512,
      notes TEXT NOT NULL DEFAULT '',
      position INTEGER NOT NULL DEFAULT 0,
      content_hash TEXT NOT NULL DEFAULT '',
      created_ts REAL NOT NULL,
      updated_ts REAL NOT NULL
    );
    CREATE TABLE cases (
      id TEXT PRIMARY KEY,
      suite_id TEXT NOT NULL REFERENCES suites(id) ON DELETE CASCADE,
      position INTEGER NOT NULL DEFAULT 0,
      title TEXT NOT NULL DEFAULT '',
      prompt TEXT NOT NULL DEFAULT '{}',
      images TEXT NOT NULL DEFAULT '[]',
      tools TEXT NOT NULL DEFAULT '[]',
      checker TEXT NOT NULL DEFAULT '{}',
      weight REAL NOT NULL DEFAULT 1.0,
      tags TEXT NOT NULL DEFAULT '[]',
      source TEXT NOT NULL DEFAULT 'user',
      notes TEXT NOT NULL DEFAULT '',
      reference TEXT NOT NULL DEFAULT '',
      max_tokens INTEGER,
      min_context INTEGER NOT NULL DEFAULT 0,
      created_ts REAL NOT NULL,
      updated_ts REAL NOT NULL
    );
    CREATE INDEX cases_suite ON cases(suite_id, position);
    CREATE TABLE runs (
      id TEXT PRIMARY KEY,
      label TEXT NOT NULL DEFAULT '',
      state TEXT NOT NULL DEFAULT 'queued',
      suites TEXT NOT NULL DEFAULT '[]',
      contestants TEXT NOT NULL DEFAULT '[]',
      settings TEXT NOT NULL DEFAULT '{}',
      source TEXT NOT NULL DEFAULT 'ui',
      caller TEXT NOT NULL DEFAULT '',
      created_ts REAL NOT NULL,
      started_ts REAL,
      finished_ts REAL,
      error TEXT NOT NULL DEFAULT '',
      cancel INTEGER NOT NULL DEFAULT 0,
      summary TEXT NOT NULL DEFAULT '{}'
    );
    CREATE INDEX runs_created ON runs(created_ts);
    CREATE TABLE run_contestants (
      run_id TEXT NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
      contestant_id TEXT NOT NULL,
      state TEXT NOT NULL DEFAULT 'queued',
      error TEXT NOT NULL DEFAULT '',
      load_ms INTEGER,
      vram_mb INTEGER,
      vram_method TEXT NOT NULL DEFAULT '',
      gpus TEXT NOT NULL DEFAULT '[]',
      context INTEGER,
      warnings TEXT NOT NULL DEFAULT '[]',
      total INTEGER NOT NULL DEFAULT 0,
      done INTEGER NOT NULL DEFAULT 0,
      digest TEXT NOT NULL DEFAULT '',
      runs_on TEXT NOT NULL DEFAULT '',
      spill TEXT NOT NULL DEFAULT '',
      started_ts REAL,
      finished_ts REAL,
      PRIMARY KEY (run_id, contestant_id)
    );
    CREATE TABLE results (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      run_id TEXT NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
      contestant_id TEXT NOT NULL,
      suite_id TEXT NOT NULL,
      case_id TEXT NOT NULL,
      category TEXT NOT NULL DEFAULT '',
      repeat INTEGER NOT NULL DEFAULT 0,
      output TEXT NOT NULL DEFAULT '',
      reasoning TEXT NOT NULL DEFAULT '',
      tool_calls TEXT NOT NULL DEFAULT '[]',
      score REAL NOT NULL DEFAULT 0,
      passed INTEGER NOT NULL DEFAULT 0,
      detail TEXT NOT NULL DEFAULT '{}',
      latency_ms REAL,
      ttft_ms REAL,
      prompt_tokens INTEGER,
      completion_tokens INTEGER,
      decode_tps REAL,
      prompt_tps REAL,
      error TEXT NOT NULL DEFAULT '',
      skipped INTEGER NOT NULL DEFAULT 0,
      unavailable INTEGER NOT NULL DEFAULT 0,
      self_judged INTEGER NOT NULL DEFAULT 0,
      judge_pending INTEGER NOT NULL DEFAULT 0,
      confidence REAL NOT NULL DEFAULT 1.0,
      weight REAL NOT NULL DEFAULT 1.0,
      digest TEXT NOT NULL DEFAULT '',
      ts REAL NOT NULL
    );
    CREATE INDEX results_run ON results(run_id, contestant_id);
    CREATE INDEX results_contestant ON results(contestant_id, suite_id, case_id);
    CREATE INDEX results_case ON results(case_id);
    CREATE TABLE judge_cache (
      hash TEXT PRIMARY KEY,
      result TEXT NOT NULL,
      ts REAL NOT NULL
    );
    CREATE TABLE ollama_info (
      digest TEXT PRIMARY KEY,
      info TEXT NOT NULL,
      ts REAL NOT NULL
    );
    CREATE TABLE notices (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      ts REAL NOT NULL,
      kind TEXT NOT NULL,
      severity TEXT NOT NULL DEFAULT 'medium',
      title TEXT NOT NULL DEFAULT '',
      body TEXT NOT NULL DEFAULT '',
      data TEXT NOT NULL DEFAULT '{}',
      contestant_id TEXT,
      dedupe TEXT NOT NULL DEFAULT '',
      seen INTEGER NOT NULL DEFAULT 0
    );
    CREATE INDEX notices_ts ON notices(ts);
    CREATE INDEX notices_dedupe ON notices(dedupe);
    CREATE TABLE route_publications (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      ts REAL NOT NULL,
      path TEXT NOT NULL DEFAULT '',
      doc TEXT NOT NULL DEFAULT '{}',
      diff TEXT NOT NULL DEFAULT '{}',
      note TEXT NOT NULL DEFAULT ''
    );
    CREATE TABLE arena_pairs (
      id TEXT PRIMARY KEY,
      case_id TEXT NOT NULL,
      category TEXT NOT NULL DEFAULT '',
      a_result INTEGER NOT NULL,
      b_result INTEGER NOT NULL,
      a_contestant TEXT NOT NULL,
      b_contestant TEXT NOT NULL,
      created_ts REAL NOT NULL,
      voted INTEGER NOT NULL DEFAULT 0
    );
    CREATE TABLE arena_votes (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      pair_id TEXT NOT NULL,
      case_id TEXT NOT NULL,
      category TEXT NOT NULL DEFAULT '',
      a TEXT NOT NULL,
      b TEXT NOT NULL,
      vote TEXT NOT NULL,
      ts REAL NOT NULL
    );
    CREATE INDEX arena_votes_cat ON arena_votes(category);
    CREATE TABLE activity (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      ts REAL NOT NULL,
      kind TEXT NOT NULL,
      ref TEXT NOT NULL DEFAULT '',
      ok INTEGER NOT NULL DEFAULT 1,
      duration_ms INTEGER NOT NULL DEFAULT 0,
      detail TEXT NOT NULL DEFAULT ''
    );
    CREATE INDEX activity_ts ON activity(ts);
    """,
    # 2: a result whose visible answer is empty because the token budget ran out (finish_reason length) is flagged; earlier results are flagged from their detail
    """
    ALTER TABLE results ADD COLUMN truncated INTEGER NOT NULL DEFAULT 0;
    UPDATE results SET truncated = 1 WHERE detail LIKE '%"empty_answer": true%' AND detail LIKE '%"finish_reason": "length"%';
    """,
    # 3: a model waiting for a shared server that somebody else is using: since when it waits now (NULL: not waiting) and how long it waited in total
    """
    ALTER TABLE run_contestants ADD COLUMN wait_since REAL;
    ALTER TABLE run_contestants ADD COLUMN waited_s REAL NOT NULL DEFAULT 0;
    """,
    # 4: a run whose results are known to be wrong is discarded: it stays in the history with the reason, and no statistic reads its results
    """
    ALTER TABLE runs ADD COLUMN discarded INTEGER NOT NULL DEFAULT 0;
    ALTER TABLE runs ADD COLUMN discard_reason TEXT NOT NULL DEFAULT '';
    ALTER TABLE runs ADD COLUMN discarded_ts REAL;
    """,
    # 5: a model that ran on the CPU: the contestant's run card says where, and each result is flagged so its speed is never mixed with GPU speeds
    """
    ALTER TABLE run_contestants ADD COLUMN device TEXT NOT NULL DEFAULT '';
    ALTER TABLE results ADD COLUMN cpu INTEGER NOT NULL DEFAULT 0;
    """,
    # 6: a run that continues an interrupted one (failed or cancelled): the id of the run it continues, so that the cases already measured are not asked again
    """
    ALTER TABLE runs ADD COLUMN continues TEXT NOT NULL DEFAULT '';
    """,
]


class Database:
    """One connection shared by every thread, guarded by a re-entrant lock (one writer at a time).

    The app is the only writer; the MCP bridge never opens this file.
    """

    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.lock = threading.RLock()
        self.conn = sqlite3.connect(str(path), check_same_thread=False, isolation_level=None)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")
        self.conn.execute("PRAGMA foreign_keys=ON")
        self.migrate()

    def migrate(self) -> None:
        with self.lock:
            self.conn.execute("CREATE TABLE IF NOT EXISTS schema_version (version INTEGER NOT NULL)")
            row = self.conn.execute("SELECT MAX(version) AS v FROM schema_version").fetchone()
            current = row["v"] or 0
            for index, sql in enumerate(MIGRATIONS, start=1):
                if index <= current:
                    continue
                script = f"BEGIN;\n{sql}\nINSERT INTO schema_version(version) VALUES ({index});\nCOMMIT;"
                try:
                    self.conn.executescript(script)
                except Exception:
                    if self.conn.in_transaction:
                        self.conn.execute("ROLLBACK")
                    raise

    def version(self) -> int:
        row = self.one("SELECT MAX(version) AS v FROM schema_version")
        return int(row["v"] or 0)

    def query(self, sql: str, params: tuple | list = ()) -> list[sqlite3.Row]:
        with self.lock:
            return self.conn.execute(sql, params).fetchall()

    def one(self, sql: str, params: tuple | list = ()) -> sqlite3.Row | None:
        with self.lock:
            return self.conn.execute(sql, params).fetchone()

    def execute(self, sql: str, params: tuple | list = ()) -> sqlite3.Cursor:
        with self.lock:
            return self.conn.execute(sql, params)

    def get_setting(self, key: str, default: str | None = None) -> str | None:
        row = self.one("SELECT value FROM settings WHERE key = ?", (key,))
        return row["value"] if row else default

    def set_setting(self, key: str, value: str) -> None:
        self.execute("INSERT INTO settings(key, value) VALUES (?, ?) ON CONFLICT(key) DO UPDATE SET value = excluded.value", (key, value))

    def transaction(self):
        """`with db.transaction():` — BEGIN IMMEDIATE / COMMIT (ROLLBACK on error) under the lock."""
        return _Transaction(self)

    def close(self) -> None:
        with self.lock:
            try:
                self.conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            except sqlite3.Error:
                pass
            self.conn.close()


class _Transaction:
    def __init__(self, db: Database):
        self.db = db

    def __enter__(self):
        self.db.lock.acquire()
        self.db.conn.execute("BEGIN IMMEDIATE")
        return self.db.conn

    def __exit__(self, exc_type, exc, tb):
        try:
            if exc_type is None:
                self.db.conn.execute("COMMIT")
            else:
                self.db.conn.execute("ROLLBACK")
        finally:
            self.db.lock.release()
        return False
