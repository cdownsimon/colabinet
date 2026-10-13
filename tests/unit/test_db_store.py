import asyncio
import sqlite3

import pytest


def _session(**over):
    base = dict(
        colab_name="llm", gpu_requested="L4", proxy_port=3080, hidden_port=23000, vm_port=8090, state="creating",
        state_since=1.0, created_at=1.0, rate_cu_h=1.54, rate_source="measured", idle_min=30,
    )
    base.update(over)
    return base


async def test_insert_and_update_session(store):
    sid = await store.insert_session(**_session())
    await store.update_session(sid, state="ready", endpoint="ep-1")
    row = await store.session(sid)
    assert row["state"] == "ready" and row["endpoint"] == "ep-1" and row["name_reserved"] == 0


async def test_unknown_column_rejected(store):
    with pytest.raises(KeyError):
        await store.insert_session(**_session(), bogus=1)


async def test_live_name_is_unique_but_reusable_after_finish(store):
    sid = await store.insert_session(**_session())
    with pytest.raises(sqlite3.IntegrityError):
        await store.insert_session(**_session())
    await store.update_session(sid, state="stopped")
    assert not await store.name_in_use("llm")
    await store.insert_session(**_session())  # reuse allowed now


async def test_unconfirmed_stop_and_reserved_name_keep_the_name(store):
    sid = await store.insert_session(**_session())
    await store.update_session(sid, state="stopped", stop_unconfirmed=1)
    assert await store.name_in_use("llm")
    await store.update_session(sid, stop_unconfirmed=0, name_reserved=1)
    assert await store.name_in_use("llm")
    with pytest.raises(sqlite3.IntegrityError):
        await store.insert_session(**_session())


async def test_concurrent_writes_are_serialised(store):
    sid = await store.insert_session(**_session())
    await asyncio.gather(*(store.event(float(i), "tick", session_id=sid, n=i) for i in range(50)))
    assert len(await store.events(sid)) == 50


async def test_runs_close_and_rates(store):
    sid = await store.insert_session(**_session())
    rid = await store.insert_run(session_id=sid, engine="llamacpp", started_at=1.0)
    await store.close_open_runs(sid, 9.0)
    assert (await store.current_run(sid))["id"] == rid
    assert (await store.current_run(sid))["ended_at"] == 9.0
    assert await store.rate("L4") == (1.54, "measured")
    await store.set_rate("L4", 1.6, "learned", 5.0)
    assert await store.rate("L4") == (1.6, "learned")


async def test_settings_roundtrip(store):
    assert await store.get_setting("x", "d") == "d"
    await store.set_setting("x", "1")
    await store.set_setting("x", "2")
    assert await store.get_setting("x") == "2"


async def test_schema_v2_views_and_migration_from_v1(tmp_path):
    import sqlite3 as sq

    from colabinet.db import SCHEMA_V1, Database
    from colabinet.store import Store

    path = tmp_path / "old.db"
    conn = sq.connect(path)
    conn.executescript(SCHEMA_V1)
    conn.execute("PRAGMA user_version = 1")
    conn.commit()
    conn.close()
    db = Database(path)
    db.init()
    st = Store(db)
    sid = await st.insert_session(**_session(state="stopped", created_at=0.0, stopped_at=3600.0, rate_cu_h=1.5,
                                             gpu_class="L4"))
    rid = await st.insert_run(session_id=sid, engine="llamacpp", alias="m", kv="q4_0", started_at=0.0, ended_at=3600.0)
    await st.insert_request({"run_id": rid, "ts": 10.0, "method": "POST", "path": "/v1/chat/completions",
                             "status": 200, "stream": 1, "prompt_tokens": 5000, "prefill_tokens": 5000,
                             "completion_tokens": 100, "engine_prefill_ms": 1000.0, "engine_gen_ms": 8000.0,
                             "ttft_ms": 1100.0, "total_ms": 9100.0, "client": "colabinet-playground"})
    row = await db.fetchone("SELECT * FROM request_stats")
    assert (row["gpu"], row["model"], row["prompt_bucket"], row["is_playground"]) == ("L4", "m", "4-32K", 1)
    assert (row["gen_tps"], row["prefill_tps"]) == (12.5, 5000.0)
    assert round(row["e2e_tps"], 2) == round(100 / 9.1, 2)
    assert (await db.fetchone("SELECT cu FROM run_cu"))["cu"] == 1.5
    assert (await db.fetchone("PRAGMA user_version"))["user_version"] == 4
    db.close()


async def test_v1_to_v2_migration_preserves_rows(tmp_path):
    from colabinet.db import SCHEMA_V1, Database

    database_path = tmp_path / "populated-v1.db"
    with sqlite3.connect(database_path) as connection:
        connection.executescript(SCHEMA_V1)
        connection.execute("PRAGMA user_version = 1")
        connection.execute(
            "INSERT INTO sessions (id, colab_name, gpu_requested, proxy_port, hidden_port, vm_port, state, "
            "state_since, created_at, idle_min, rate_cu_h, rate_source) "
            "VALUES (5, 'preserved', 'L4', 3080, 23000, 8090, 'stopped', 0, 0, 30, 1.54, 'test')"
        )
        connection.execute("INSERT INTO runs (id, session_id, engine, started_at) VALUES (7, 5, 'llamacpp', 0)")
        connection.execute(
            "INSERT INTO requests (id, run_id, ts, method, path, status) VALUES (9, 7, 5, 'GET', '/v1/models', 200)"
        )
    database = Database(database_path)
    try:
        database.init()
        database.init()
        assert (await database.fetchone("SELECT colab_name FROM sessions WHERE id = 5"))["colab_name"] == "preserved"
        assert (await database.fetchone("SELECT session_id FROM runs WHERE id = 7"))["session_id"] == 5
        assert (await database.fetchone("SELECT path FROM requests WHERE id = 9"))["path"] == "/v1/models"
        assert (await database.fetchone("PRAGMA user_version"))["user_version"] == 4
        assert (await database.fetchone("SELECT account_id FROM sessions WHERE id = 5"))["account_id"] == "default"
    finally:
        database.close()


async def test_saved_queries_update_and_delete(store):
    query_id = await store.save_query("speed", '{"time":"30d"}', 1.0)
    assert await store.save_query("speed", '{"time":"7d"}', 2.0) == query_id
    queries = await store.saved_queries()
    assert len(queries) == 1 and queries[0]["spec_json"] == '{"time":"7d"}'
    await store.delete_query(query_id)
    assert await store.saved_queries() == []


async def test_v3_failure_migration_retains_error_and_marks_pending_run_failed(tmp_path):
    from colabinet.db import SCHEMA_V1, SCHEMA_V2, SCHEMA_V3, Database
    from colabinet.store import Store

    path = tmp_path / "old-failure.db"
    with sqlite3.connect(path) as connection:
        connection.executescript(SCHEMA_V1 + SCHEMA_V2 + SCHEMA_V3)
        fields = _session(state="failed", stopped_at=1.0, account_id="owner")
        connection.execute(f"INSERT INTO sessions ({', '.join(fields)}) VALUES ({', '.join('?' * len(fields))})",
                           tuple(fields.values()))
        connection.execute("INSERT INTO runs (session_id, engine, started_at, ended_at, status) "
                           "VALUES (1, 'llamacpp', 1, 1, 'pending')")
        connection.execute("INSERT INTO events (session_id, ts, kind, detail_json) VALUES "
                           "(1, 1, 'finish_message', ?)",
                           ('{"message":"Backend rejected L4 access_token=old-secret"}',))
        connection.execute("PRAGMA user_version = 3")
    db = Database(path)
    db.init()
    try:
        store = Store(db)
        assert (await store.current_run(1))["status"] == "failed"
        assert (await store.session(1))["account_id"] == "owner"
        diagnostic = await store.creation_diagnostics(1)
        assert diagnostic["output"] == ""
        assert "Backend rejected L4" in diagnostic["error"]
        assert "old-secret" not in diagnostic["error"]
    finally:
        db.close()


async def test_saved_query_upsert_returns_its_id_after_another_insert(store):
    first_id = await store.save_query("first", '{"time":"30d"}', 1.0)
    second_id = await store.save_query("second", '{"time":"7d"}', 2.0)
    updated_id = await store.save_query("first", '{"time":"24h"}', 3.0)
    assert updated_id == first_id != second_id
    first = next(query for query in await store.saved_queries() if query["name"] == "first")
    assert first["id"] == updated_id and first["spec_json"] == '{"time":"24h"}'


async def test_replacing_a_run_is_atomic_if_the_insert_fails(store):
    session_id = await store.insert_session(**_session(state="ready"))
    first_id = await store.insert_run(session_id=session_id, engine="llamacpp", started_at=1.0)
    with pytest.raises(sqlite3.IntegrityError):
        await store.replace_run(session_id, 2.0, engine=None, started_at=2.0)
    assert (await store.current_run(session_id))["ended_at"] is None
    new_id = await store.replace_run(session_id, 2.0, engine="exllamav3", started_at=2.0)
    assert (await store.current_run(session_id))["id"] == new_id != first_id
    first = await store.db.fetchone("SELECT ended_at FROM runs WHERE id = ?", (first_id,))
    assert first["ended_at"] == 2.0


async def test_aggregates_exist_on_reader_and_writer_connections(store):
    def aggregate(connection):
        return tuple(connection.execute(
            "SELECT median(value), p90(value), percentile(value, 50) FROM "
            "(SELECT 1 AS value UNION ALL SELECT 3 UNION ALL SELECT 2 UNION ALL SELECT NULL)"
        ).fetchone())

    assert await store.db.read(aggregate) == (2.0, 2.8, 2.0)
    assert await store.db.write(aggregate) == (2.0, 2.8, 2.0)
