"""SQLite access. No SQLite call ever runs on the asyncio event loop.

Writes go through one writer thread (a single-worker executor, i.e. a queue); reads use a small reader pool.
Each thread keeps its own connection. WAL mode lets readers run while the writer writes.
"""

from __future__ import annotations

import asyncio
import math
import sqlite3
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable, TypeVar

T = TypeVar("T")

SCHEMA_VERSION = 4

SCHEMA_V1 = """
CREATE TABLE IF NOT EXISTS sessions (
    id INTEGER PRIMARY KEY,
    colab_name TEXT NOT NULL,
    endpoint TEXT,
    gpu_requested TEXT NOT NULL,
    gpu_class TEXT,
    gpu_name TEXT, vram_mib INTEGER, ram_gb REAL, torch_cuda TEXT, driver TEXT,
    proxy_port INTEGER NOT NULL, hidden_port INTEGER NOT NULL, vm_port INTEGER NOT NULL,
    tunnel_pid INTEGER, new_pid INTEGER, new_exit INTEGER, name_reserved INTEGER NOT NULL DEFAULT 0,
    state TEXT NOT NULL, state_since REAL NOT NULL, stop_reason TEXT,
    stop_unconfirmed INTEGER NOT NULL DEFAULT 0,
    created_at REAL NOT NULL, ready_at REAL, stopped_at REAL,
    last_activity REAL, last_exec_ok REAL,
    rate_cu_h REAL NOT NULL, rate_source TEXT NOT NULL,
    assignments_before INTEGER, usage_rate_before REAL,
    idle_min INTEGER NOT NULL, deadline REAL, cu_cap REAL,
    event_offset INTEGER NOT NULL DEFAULT 0, log_offset INTEGER NOT NULL DEFAULT 0
);
CREATE UNIQUE INDEX IF NOT EXISTS sessions_live_name ON sessions(colab_name)
    WHERE state NOT IN ('stopped', 'failed') OR stop_unconfirmed = 1 OR name_reserved = 1;

CREATE TABLE IF NOT EXISTS runs (
    id INTEGER PRIMARY KEY,
    session_id INTEGER NOT NULL REFERENCES sessions(id),
    engine TEXT NOT NULL, engine_version TEXT, preset_id TEXT,
    repo TEXT, file_or_rev TEXT, mmproj TEXT, alias TEXT,
    ctx INTEGER, kv TEXT, parallel INTEGER, extra_args TEXT,
    started_at REAL NOT NULL, ready_at REAL, ended_at REAL,
    install_s REAL, download_s REAL, download_gb REAL, load_s REAL, vram_used_mib INTEGER,
    status TEXT, error TEXT, spec_json TEXT
);

CREATE TABLE IF NOT EXISTS requests (
    id INTEGER PRIMARY KEY,
    run_id INTEGER NOT NULL REFERENCES runs(id),
    ts REAL NOT NULL, method TEXT, path TEXT, status INTEGER, stream INTEGER,
    client TEXT, model_req TEXT,
    prompt_tokens INTEGER, cached_tokens INTEGER, prefill_tokens INTEGER,
    completion_tokens INTEGER, reasoning_tokens INTEGER,
    engine_prefill_ms REAL, engine_gen_ms REAL, ttft_ms REAL, total_ms REAL,
    finish_reason TEXT, tokens_source TEXT, error_short TEXT
);
CREATE INDEX IF NOT EXISTS requests_run_ts ON requests(run_id, ts);

CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY,
    session_id INTEGER, run_id INTEGER, ts REAL NOT NULL, kind TEXT NOT NULL, detail_json TEXT
);
CREATE INDEX IF NOT EXISTS events_session_ts ON events(session_id, ts);

CREATE TABLE IF NOT EXISTS settings (key TEXT PRIMARY KEY, value TEXT);

CREATE TABLE IF NOT EXISTS gpu_rates (
    gpu TEXT PRIMARY KEY, rate_cu_h REAL NOT NULL, source TEXT NOT NULL, updated_at REAL NOT NULL
);
"""


SCHEMA_V2 = """
CREATE TABLE IF NOT EXISTS saved_queries (
    id INTEGER PRIMARY KEY, name TEXT NOT NULL UNIQUE, spec_json TEXT NOT NULL, created_at REAL NOT NULL
);

CREATE VIEW IF NOT EXISTS request_stats AS
SELECT q.id, q.run_id, q.ts, q.method, q.path, q.status, q.stream, q.client, q.model_req,
       q.prompt_tokens, q.cached_tokens, q.prefill_tokens, q.completion_tokens, q.reasoning_tokens,
       q.engine_prefill_ms, q.engine_gen_ms, q.ttft_ms, q.total_ms, q.finish_reason, q.tokens_source, q.error_short,
       r.session_id, r.engine, r.alias AS model, r.kv, r.ctx, s.colab_name AS session, s.gpu_class AS gpu,
       -- speeds only for successful requests: an error must not drag the medians down
       CASE WHEN q.status >= 400 THEN NULL
            WHEN q.engine_gen_ms > 0 THEN q.completion_tokens * 1000.0 / q.engine_gen_ms
            WHEN q.stream = 1 AND q.total_ms > q.ttft_ms THEN q.completion_tokens * 1000.0 / (q.total_ms - q.ttft_ms)
       END AS gen_tps,
       CASE WHEN q.status < 400 AND q.engine_prefill_ms > 0 THEN q.prefill_tokens * 1000.0 / q.engine_prefill_ms
       END AS prefill_tps,
       CASE WHEN q.status < 400 AND q.total_ms > 0 THEN q.completion_tokens * 1000.0 / q.total_ms END AS e2e_tps,
       CASE WHEN q.prompt_tokens IS NULL THEN NULL
            WHEN q.prompt_tokens < 4096 THEN '<4K' WHEN q.prompt_tokens < 32768 THEN '4-32K'
            WHEN q.prompt_tokens < 131072 THEN '32-128K' ELSE '>128K' END AS prompt_bucket,
       date(q.ts, 'unixepoch', 'localtime') AS day,
       CASE WHEN q.client LIKE 'colabinet-%' THEN 1 ELSE 0 END AS is_playground
FROM requests q JOIN runs r ON r.id = q.run_id JOIN sessions s ON s.id = r.session_id;

-- CU per run (rate x time; run 1 starts at the session's created_at). The Metrics page splits it by local day in
-- Python (metrics.py), the same arithmetic as the daily cap; this view is for the SQL tab.
CREATE VIEW IF NOT EXISTS run_cu AS
SELECT r.id AS run_id, r.session_id, s.colab_name AS session, s.gpu_class AS gpu, r.engine, r.alias AS model, r.kv,
       r.started_at, COALESCE(r.ended_at, s.stopped_at) AS ended_at, s.rate_cu_h,
       s.rate_cu_h * (COALESCE(r.ended_at, s.stopped_at, CAST(strftime('%s', 'now') AS REAL)) - r.started_at)
           / 3600.0 AS cu
FROM runs r JOIN sessions s ON s.id = r.session_id;
"""

SCHEMA_V3 = "ALTER TABLE sessions ADD COLUMN account_id TEXT NOT NULL DEFAULT 'default';"

SCHEMA_V4 = """
ALTER TABLE sessions ADD COLUMN create_log TEXT;
ALTER TABLE sessions ADD COLUMN create_error TEXT;
UPDATE runs SET status = 'failed' WHERE status = 'pending'
    AND session_id IN (SELECT id FROM sessions WHERE state = 'failed');
"""

MIGRATIONS = {1: SCHEMA_V1, 2: SCHEMA_V2, 3: SCHEMA_V3, 4: SCHEMA_V4}



class _Percentile:
    q = 0.5

    def __init__(self) -> None:
        self.values: list[float] = []

    def step(self, value: Any) -> None:
        if value is not None:
            self.values.append(float(value))

    def finalize(self) -> float | None:
        return percentile(self.values, self.q)


class _Median(_Percentile):
    q = 0.5


class _P90(_Percentile):
    q = 0.9


class _PercentileP:
    def __init__(self) -> None:
        self.values: list[float] = []
        self.q = 0.5

    def step(self, value: Any, percentile_value: Any) -> None:
        self.q = float(percentile_value) / 100 if float(percentile_value) > 1 else float(percentile_value)
        if value is not None:
            self.values.append(float(value))

    def finalize(self) -> float | None:
        return percentile(self.values, self.q)


def percentile(values: list[float], q: float) -> float | None:
    """Linear-interpolation percentile (q in 0..1)."""
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lower, upper = math.floor(position), math.ceil(position)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def register_aggregates(conn: sqlite3.Connection) -> None:
    conn.create_aggregate("median", 1, _Median)
    conn.create_aggregate("p90", 1, _P90)
    conn.create_aggregate("percentile", 2, _PercentileP)


class Database:
    def __init__(self, path: Path, on_connect: Callable[[sqlite3.Connection], None] | None = None) -> None:
        self.path = Path(path)
        self.on_connect = on_connect
        self._local = threading.local()
        self._writer = ThreadPoolExecutor(max_workers=1, thread_name_prefix="db-writer")
        self._readers = ThreadPoolExecutor(max_workers=4, thread_name_prefix="db-reader")

    def _conn(self) -> sqlite3.Connection:
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(self.path, timeout=5)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA busy_timeout=5000")
            conn.execute("PRAGMA foreign_keys=ON")
            register_aggregates(conn)
            if self.on_connect:
                self.on_connect(conn)
            self._local.conn = conn
        return conn

    def init(self) -> None:
        """Create or migrate the schema. Called once at start-up, before the event loop is busy."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.path)
        try:
            version = conn.execute("PRAGMA user_version").fetchone()[0]
            for migration_version in range(version + 1, SCHEMA_VERSION + 1):
                conn.executescript(MIGRATIONS[migration_version])
                conn.execute(f"PRAGMA user_version = {migration_version}")
                conn.commit()
        finally:
            conn.close()

    async def write(self, fn: Callable[[sqlite3.Connection], T]) -> T:
        def run() -> T:
            conn = self._conn()
            with conn:
                return fn(conn)

        return await asyncio.get_running_loop().run_in_executor(self._writer, run)

    async def read(self, fn: Callable[[sqlite3.Connection], T]) -> T:
        return await asyncio.get_running_loop().run_in_executor(self._readers, lambda: fn(self._conn()))

    async def execute(self, sql: str, params: tuple | dict = ()) -> int:
        """Run one write statement; returns lastrowid."""
        return await self.write(lambda c: c.execute(sql, params).lastrowid)

    async def fetchall(self, sql: str, params: tuple | dict = ()) -> list[dict[str, Any]]:
        return await self.read(lambda c: [dict(r) for r in c.execute(sql, params).fetchall()])

    async def fetchone(self, sql: str, params: tuple | dict = ()) -> dict[str, Any] | None:
        rows = await self.fetchall(sql, params)
        return rows[0] if rows else None

    def close(self) -> None:
        self._writer.shutdown(wait=True)
        self._readers.shutdown(wait=True)
