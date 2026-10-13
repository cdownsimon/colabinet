"""Metrics queries (spec §8): a whitelisted query builder over the request_stats view, CU per group computed from
runs (split by local day in Python - the same arithmetic as the daily cap), a read-only SQL runner and CSV export.
"""

from __future__ import annotations

import csv
import io
import json
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from .billing import overlap
from .db import Database, register_aggregates
from .db import percentile as percentile

TIME_PRESETS = {"24h": 86400, "7d": 7 * 86400, "30d": 30 * 86400, "all": None}


GROUPS = {
    "gpu": "gpu", "model": "model", "model_kv": "model || ' · ' || COALESCE(kv, '?')", "kv": "kv",
    "engine": "engine", "session": "session", "client": "client", "path": "path",
    "prompt_bucket": "prompt_bucket", "day": "day",
}
RUN_LEVEL_GROUPS = {"gpu", "model", "model_kv", "kv", "engine", "session", "day"}


MEASURES = {
    "requests": "COUNT(*)",
    "gen_tps_p50": "median(gen_tps)",
    "gen_tps_p90": "p90(gen_tps)",
    "prefill_tps_p50": "median(prefill_tps)",
    "prefill_tps_p90": "p90(prefill_tps)",
    "ttft_p50": "median(ttft_ms)",
    "ttft_p90": "p90(ttft_ms)",
    "e2e_tps_p50": "median(e2e_tps)",
    "tokens_in": "SUM(prompt_tokens)",
    "tokens_out": "SUM(completion_tokens)",
    "cached_in": "SUM(cached_tokens)",
    "errors": "SUM(CASE WHEN status >= 400 THEN 1 ELSE 0 END)",
}
CU_MEASURES = {"cu", "out_per_cu"}
LIST_FILTERS = {"gpu": "gpu", "model": "model", "engine": "engine", "kv": "kv", "session": "session",
                "client": "client", "path": "path"}
REQUEST_LEVEL_FILTERS = {"client", "path"}
STATUS_CLASSES = {"all": None, "2xx": (200, 300), "4xx": (400, 500), "5xx": (500, 600)}
CHARTS = {"bar", "line", "table"}


class QueryError(ValueError):
    pass



@dataclass
class QuerySpec:
    time: str = "30d"
    filters: dict[str, Any] = field(default_factory=dict)
    group_by: list[str] = field(default_factory=lambda: ["gpu", "model_kv"])
    measures: list[str] = field(default_factory=lambda: ["requests", "gen_tps_p50", "ttft_p50", "tokens_out"])
    chart: str = "bar"

    @classmethod
    def parse(cls, raw: dict[str, Any]) -> QuerySpec:
        if not isinstance(raw, dict):
            raise QueryError("query must be an object")
        fields = ("time", "filters", "group_by", "measures", "chart")
        spec = cls(**{key: value for key, value in raw.items() if key in fields})
        if isinstance(spec.time, dict):
            if set(spec.time) != {"preset"}:
                raise QueryError("time must contain a preset")
            spec.time = spec.time["preset"]
        if not isinstance(spec.time, str) or spec.time not in TIME_PRESETS:
            raise QueryError(f"unknown time range {spec.time!r}")
        for name in ("group_by", "measures"):
            values = getattr(spec, name)
            if not isinstance(values, list) or not all(isinstance(value, str) for value in values):
                raise QueryError(f"{name} must be a list of strings")
            if len(set(values)) != len(values):
                raise QueryError(f"{name} must not contain duplicates")
        bad = [group for group in spec.group_by if group not in GROUPS]
        if bad:
            raise QueryError(f"unknown group-by {bad}")
        bad = [measure for measure in spec.measures if measure not in MEASURES and measure not in CU_MEASURES]
        if bad:
            raise QueryError(f"unknown measure {bad}")
        if not spec.measures:
            raise QueryError("choose at least one measure")
        if not isinstance(spec.chart, str) or spec.chart not in CHARTS:
            raise QueryError(f"unknown chart {spec.chart!r}")
        if not isinstance(spec.filters, dict):
            raise QueryError("filters must be an object")
        filters = spec.filters = spec.filters.copy()
        for key, value in filters.items():
            if key in LIST_FILTERS:
                if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
                    raise QueryError(f"filter {key} must be a list of strings")
            elif key == "status_class":
                if not isinstance(value, str) or value not in STATUS_CLASSES:
                    raise QueryError(f"unknown status class {value!r}")
            elif key == "exclude_playground":
                if not isinstance(value, bool):
                    raise QueryError("exclude_playground must be a boolean")
            else:
                raise QueryError(f"unknown filter {key!r}")
        return spec

    def to_json(self) -> str:
        return json.dumps({"time": {"preset": self.time}, "filters": self.filters, "group_by": self.group_by,
                           "measures": self.measures, "chart": self.chart})

    def cu_available(self) -> bool:
        """CU is time-based, so it only makes sense per run-level group and without request-level filters."""
        request_filtered = any(self.filters.get(key) for key in REQUEST_LEVEL_FILTERS) or \
            self.filters.get("status_class", "all") != "all"
        return set(self.group_by) <= RUN_LEVEL_GROUPS and not request_filtered

    def window(self, now: float) -> tuple[float, float]:
        span = TIME_PRESETS[self.time]
        return (now - span if span else 0.0), now


def build_sql(spec: QuerySpec, now: float) -> tuple[str, list[Any]]:
    """Parameterised SQL over request_stats built only from whitelisted identifiers."""
    start, end = spec.window(now)
    where, params = ["ts >= ?", "ts <= ?"], [start, end]
    for key, column in LIST_FILTERS.items():
        values = spec.filters.get(key) or []
        if values:
            where.append(f"{column} IN ({','.join('?' * len(values))})")
            params.extend(values)
    status = STATUS_CLASSES[spec.filters.get("status_class", "all")]
    if status:
        where.append("status >= ? AND status < ?")
        params.extend(status)
    if spec.filters.get("exclude_playground", True):
        where.append("is_playground = 0")
    groups = [f"{GROUPS[group]} AS {group}" for group in spec.group_by]
    measures = [f"{MEASURES[measure]} AS {measure}" for measure in spec.measures if measure in MEASURES]
    if "out_per_cu" in spec.measures and "tokens_out" not in spec.measures:
        measures.append(f"{MEASURES['tokens_out']} AS tokens_out")
    select = ", ".join(groups + (measures or ["COUNT(*) AS requests"]))
    sql = f"SELECT {select} FROM request_stats WHERE {' AND '.join(where)}"
    if spec.group_by:
        sql += " GROUP BY " + ", ".join(spec.group_by) + " ORDER BY " + ", ".join(spec.group_by)
    return sql, params



def run_key(run: dict[str, Any], group: str, day: str | None) -> Any:
    return {
        "gpu": run["gpu"], "model": run["model"], "model_kv": f"{run['model']} · {run['kv'] or '?'}",
        "kv": run["kv"], "engine": run["engine"], "session": run["session"], "day": day,
    }[group]


def cu_by_group(runs: list[dict[str, Any]], group_by: list[str], start: float, end: float, now: float,
                day_bounds: Callable[[float], tuple[float, float]],
                day_key: Callable[[float], str]) -> dict[tuple, float]:
    out: dict[tuple, float] = {}
    for run in runs:
        t0 = run["started_at"]
        t1 = run["ended_at"] if run["ended_at"] is not None else now
        lo, hi = max(t0, start), min(t1, end)
        if hi <= lo:
            continue
        if "day" in group_by:
            cursor = lo
            while cursor < hi:
                d0, d1 = day_bounds(cursor)
                seg = overlap(lo, hi, d0, d1)
                key = tuple(run_key(run, group, day_key(cursor)) for group in group_by)
                out[key] = out.get(key, 0.0) + run["rate_cu_h"] * seg / 3600
                cursor = d1
        else:
            key = tuple(run_key(run, group, None) for group in group_by)
            out[key] = out.get(key, 0.0) + run["rate_cu_h"] * (hi - lo) / 3600
    return out



@dataclass
class Result:
    columns: list[str]
    rows: list[list[Any]]
    cu_available: bool = True
    note: str = ""

    def as_dicts(self) -> list[dict[str, Any]]:
        return [dict(zip(self.columns, row)) for row in self.rows]


RUNS_SQL = (
    "SELECT r.id, r.started_at, COALESCE(r.ended_at, s.stopped_at) AS ended_at, s.rate_cu_h, s.gpu_class AS gpu, "
    "r.alias AS model, r.kv, r.engine, s.colab_name AS session, r.session_id "
    "FROM runs r JOIN sessions s ON s.id = r.session_id"
)


async def run_query(db: Database, spec: QuerySpec, now: float, clock) -> Result:
    sql, params = build_sql(spec, now)
    rows = await db.fetchall(sql, tuple(params))
    wants_cu = [measure for measure in spec.measures if measure in CU_MEASURES]
    columns = list(spec.group_by) + [measure for measure in spec.measures]
    note = ""
    cu: dict[tuple, float] = {}
    if wants_cu:
        if spec.cu_available():
            gpu_f, model_f = spec.filters.get("gpu") or [], spec.filters.get("model") or []
            engine_f, kv_f = spec.filters.get("engine") or [], spec.filters.get("kv") or []
            session_f = spec.filters.get("session") or []
            runs = [row for row in await db.fetchall(RUNS_SQL)
                    if (not gpu_f or row["gpu"] in gpu_f) and (not model_f or row["model"] in model_f)
                    and (not engine_f or row["engine"] in engine_f) and (not kv_f or row["kv"] in kv_f)
                    and (not session_f or row["session"] in session_f)]
            start, end = spec.window(now)
            cu = cu_by_group(runs, spec.group_by, start, end, now, clock.day_bounds, clock.day_key)
            if spec.filters.get("exclude_playground", True):
                note = "CU includes playground time (it can't be separated from other use of the session)."
        else:
            note = ("CU needs run-level grouping (GPU, model, KV, engine, session, day) and no client/path/status "
                    "filter.")
    by_key = {tuple(row[group] for group in spec.group_by): row for row in rows}
    keys = list(by_key) + [key for key in cu if key not in by_key]
    if not spec.group_by and not keys:
        keys = [()]
    out_rows = []
    for key in keys:
        row = by_key.get(key, {})
        vals = list(key)
        for measure in spec.measures:
            if measure == "cu":
                vals.append(round(cu[key], 4) if key in cu else None)
            elif measure == "out_per_cu":
                client, cursor = cu.get(key), row.get("tokens_out")
                vals.append(round(cursor / client) if client and cursor else None)
            else:
                value = row.get(measure)
                vals.append(round(value, 2) if isinstance(value, float) else value)
        out_rows.append(vals)
    return Result(columns, out_rows, cu_available=not wants_cu or spec.cu_available(), note=note)



_ALLOWED = {sqlite3.SQLITE_SELECT, sqlite3.SQLITE_READ, sqlite3.SQLITE_FUNCTION}
_ALLOWED |= {getattr(sqlite3, "SQLITE_RECURSIVE", 33)}


def run_sql(db_path: Path, text: str, limit: int = 10_000, timeout_s: float = 5.0) -> Result:
    """Run one read-only statement. A separate `mode=ro` connection, an authorizer that allows only reads (so no
    writes, ATTACH or PRAGMA get through), sqlite3's one-statement rule, a time limit and a row cap."""
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=5)
    try:
        register_aggregates(conn)
        conn.set_authorizer(lambda action, *args: sqlite3.SQLITE_OK if action in _ALLOWED else sqlite3.SQLITE_DENY)
        deadline = time.monotonic() + timeout_s
        conn.set_progress_handler(lambda: 1 if time.monotonic() > deadline else 0, 10_000)
        try:
            cur = conn.execute(text)
            rows = cur.fetchmany(limit + 1)
        except sqlite3.OperationalError as exc:
            if "interrupted" in str(exc):
                raise QueryError(f"query took longer than {timeout_s:.0f} s") from exc
            raise QueryError(str(exc)) from exc
        except (sqlite3.DatabaseError, sqlite3.ProgrammingError, sqlite3.Warning) as exc:
            raise QueryError(str(exc)) from exc
        columns = [detail[0] for detail in cur.description or []]
        note = f"showing the first {limit} rows" if len(rows) > limit else ""
        return Result(columns, [list(row) for row in rows[:limit]], note=note)
    finally:
        conn.close()


def to_csv(result: Result) -> str:
    buf = io.StringIO()
    world = csv.writer(buf)
    world.writerow(result.columns)
    world.writerows(result.rows)
    return buf.getvalue()
