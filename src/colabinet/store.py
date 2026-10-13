"""Typed-ish helpers over the database: sessions, runs, requests, events, settings, rates."""

from __future__ import annotations

import json
from typing import Any

from .config import SEED_RATES
from .db import Database
from .diagnostics import bounded_diagnostic

LIVE_STATES = (
    "creating", "gpu_check", "gpu_mismatch", "setting_up", "reconfiguring", "setup_failed",
    "tunnelling", "ready", "engine_down", "stopping",
)
FINAL_STATES = ("stopped", "failed")

SESSION_FIELDS = {
    "account_id", "create_log", "create_error",
    "colab_name", "endpoint", "gpu_requested", "gpu_class", "gpu_name", "vram_mib", "ram_gb", "torch_cuda", "driver",
    "proxy_port", "hidden_port", "vm_port", "tunnel_pid", "new_pid", "new_exit", "name_reserved", "state",
    "state_since", "stop_reason", "stop_unconfirmed", "created_at", "ready_at", "stopped_at", "last_activity",
    "last_exec_ok", "rate_cu_h", "rate_source", "assignments_before", "usage_rate_before", "idle_min", "deadline",
    "cu_cap",
    "event_offset", "log_offset",
}
RUN_FIELDS = {
    "session_id", "engine", "engine_version", "preset_id", "repo", "file_or_rev", "mmproj", "alias", "ctx", "kv",
    "parallel", "extra_args", "started_at", "ready_at", "ended_at", "install_s", "download_s", "download_gb",
    "load_s", "vram_used_mib", "status", "error", "spec_json",
}
REQUEST_FIELDS = (
    "run_id", "ts", "method", "path", "status", "stream", "client", "model_req", "prompt_tokens", "cached_tokens",
    "prefill_tokens", "completion_tokens", "reasoning_tokens", "engine_prefill_ms", "engine_gen_ms", "ttft_ms",
    "total_ms", "finish_reason", "tokens_source", "error_short",
)


def _check(fields: dict[str, Any], allowed: set[str]) -> None:
    bad = set(fields) - allowed
    if bad:
        raise KeyError(f"unknown column(s): {sorted(bad)}")


class Store:
    def __init__(self, db: Database) -> None:
        self.db = db

    # --- sessions -------------------------------------------------------------------------------------------
    async def insert_session(self, **fields: Any) -> int:
        _check(fields, SESSION_FIELDS)
        cols = ", ".join(fields)
        marks = ", ".join(f":{k}" for k in fields)
        return await self.db.execute(f"INSERT INTO sessions ({cols}) VALUES ({marks})", fields)

    async def update_session(self, sid: int, **fields: Any) -> None:
        _check(fields, SESSION_FIELDS)
        sets = ", ".join(f"{k} = :{k}" for k in fields)
        await self.db.execute(f"UPDATE sessions SET {sets} WHERE id = :_id", {**fields, "_id": sid})

    async def session(self, sid: int) -> dict[str, Any] | None:
        return await self.db.fetchone("SELECT * FROM sessions WHERE id = ?", (sid,))

    async def creation_diagnostics(self, sid: int) -> dict[str, Any] | None:
        row = await self.session(sid)
        if row is None:
            return None
        error = row["create_error"] or ""
        if row["state"] == "failed" and not error:
            event = await self.db.fetchone(
                "SELECT detail_json FROM events WHERE session_id = ? AND kind = 'finish_message' "
                "ORDER BY id DESC LIMIT 1", (sid,))
            if event and event["detail_json"]:
                error = json.loads(event["detail_json"]).get("message", "")
            error = error or "Session creation failed; no error details were retained."
        return {"error": bounded_diagnostic(error, 2000), "output": bounded_diagnostic(row["create_log"] or ""),
                "exit_code": row["new_exit"]}

    async def sessions(self, where: str = "1=1", params: tuple = ()) -> list[dict[str, Any]]:
        return await self.db.fetchall(f"SELECT * FROM sessions WHERE {where} ORDER BY id", params)

    async def live_sessions(self) -> list[dict[str, Any]]:
        marks = ",".join("?" * len(LIVE_STATES))
        return await self.sessions(f"state IN ({marks})", LIVE_STATES)

    async def name_in_use(self, name: str) -> bool:
        row = await self.db.fetchone(
            "SELECT id FROM sessions WHERE colab_name = ? AND (state NOT IN ('stopped', 'failed') "
            "OR stop_unconfirmed = 1 OR name_reserved = 1)",
            (name,),
        )
        return row is not None

    # --- runs -----------------------------------------------------------------------------------------------
    async def insert_run(self, **fields: Any) -> int:
        _check(fields, RUN_FIELDS)
        cols = ", ".join(fields)
        marks = ", ".join(f":{k}" for k in fields)
        return await self.db.execute(f"INSERT INTO runs ({cols}) VALUES ({marks})", fields)

    async def replace_run(self, sid: int, ended_at: float, **fields: Any) -> int:
        fields = {**fields, "session_id": sid}
        _check(fields, RUN_FIELDS)
        columns = ", ".join(fields)
        marks = ", ".join(f":{key}" for key in fields)

        def replace(connection):
            connection.execute("UPDATE runs SET ended_at = ? WHERE session_id = ? AND ended_at IS NULL",
                               (ended_at, sid))
            return connection.execute(f"INSERT INTO runs ({columns}) VALUES ({marks})", fields).lastrowid

        return await self.db.write(replace)

    async def update_run(self, rid: int, **fields: Any) -> None:
        _check(fields, RUN_FIELDS)
        sets = ", ".join(f"{k} = :{k}" for k in fields)
        await self.db.execute(f"UPDATE runs SET {sets} WHERE id = :_id", {**fields, "_id": rid})

    async def current_run(self, sid: int) -> dict[str, Any] | None:
        return await self.db.fetchone("SELECT * FROM runs WHERE session_id = ? ORDER BY id DESC LIMIT 1", (sid,))

    async def close_open_runs(self, sid: int, ended_at: float) -> None:
        await self.db.execute(
            "UPDATE runs SET ended_at = ? WHERE session_id = ? AND ended_at IS NULL", (ended_at, sid)
        )

    # --- requests -------------------------------------------------------------------------------------------
    async def insert_request(self, record: dict[str, Any]) -> int:
        values = {k: record.get(k) for k in REQUEST_FIELDS}
        cols = ", ".join(REQUEST_FIELDS)
        marks = ", ".join(f":{k}" for k in REQUEST_FIELDS)
        return await self.db.execute(f"INSERT INTO requests ({cols}) VALUES ({marks})", values)

    # --- events ---------------------------------------------------------------------------------------------
    async def event(self, ts: float, kind: str, session_id: int | None = None, run_id: int | None = None,
                    **detail: Any) -> None:
        await self.db.execute(
            "INSERT INTO events (session_id, run_id, ts, kind, detail_json) VALUES (?, ?, ?, ?, ?)",
            (session_id, run_id, ts, kind, json.dumps(detail, default=str) if detail else None),
        )

    async def events(self, session_id: int, after_id: int = 0, limit: int = 200) -> list[dict[str, Any]]:
        return await self.db.fetchall(
            "SELECT * FROM events WHERE session_id = ? AND id > ? ORDER BY id LIMIT ?", (session_id, after_id, limit)
        )

    # --- settings and rates ---------------------------------------------------------------------------------
    async def get_setting(self, key: str, default: str | None = None) -> str | None:
        row = await self.db.fetchone("SELECT value FROM settings WHERE key = ?", (key,))
        return row["value"] if row else default

    async def set_setting(self, key: str, value: str) -> None:
        await self.set_settings({key: value})

    async def set_settings(self, values: dict[str, str]) -> None:
        await self.db.write(lambda connection: connection.executemany(
            "INSERT INTO settings (key, value) VALUES (?, ?) ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            values.items(),
        ))

    async def rate(self, gpu_class: str) -> tuple[float, str]:
        row = await self.db.fetchone("SELECT rate_cu_h, source FROM gpu_rates WHERE gpu = ?", (gpu_class,))
        if row:
            return row["rate_cu_h"], row["source"]
        return SEED_RATES[gpu_class]

    async def set_rate(self, gpu_class: str, rate: float, source: str, ts: float) -> None:
        await self.db.execute(
            "INSERT INTO gpu_rates (gpu, rate_cu_h, source, updated_at) VALUES (?, ?, ?, ?) "
            "ON CONFLICT(gpu) DO UPDATE SET rate_cu_h = excluded.rate_cu_h, source = excluded.source, "
            "updated_at = excluded.updated_at",
            (gpu_class, rate, source, ts),
        )


    async def save_query(self, name: str, spec_json: str, ts: float) -> int:
        return await self.db.write(lambda connection: connection.execute(
            "INSERT INTO saved_queries (name, spec_json, created_at) VALUES (?, ?, ?) "
            "ON CONFLICT(name) DO UPDATE SET spec_json = excluded.spec_json RETURNING id",
            (name, spec_json, ts)).fetchone()["id"])

    async def saved_queries(self) -> list[dict[str, Any]]:
        return await self.db.fetchall("SELECT * FROM saved_queries ORDER BY name")

    async def delete_query(self, qid: int) -> None:
        await self.db.execute("DELETE FROM saved_queries WHERE id = ?", (qid,))
