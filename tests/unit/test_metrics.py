import pytest

from colabinet.config import Clock
from colabinet.db import Database
from colabinet.metrics import (
    QueryError,
    QuerySpec,
    build_sql,
    cu_by_group,
    percentile,
    register_aggregates,
    run_query,
    run_sql,
    to_csv,
)
from colabinet.store import Store

CLOCK = Clock()
MID = CLOCK.day_bounds(1_800_000_000)[1]
D0, D1 = CLOCK.day_key(MID - 1), CLOCK.day_key(MID)


@pytest.fixture
async def seeded(tmp_path):
    db = Database(tmp_path / "m.db", on_connect=register_aggregates)
    db.init()
    st = Store(db)

    l4 = await st.insert_session(colab_name="llm", gpu_requested="L4", gpu_class="L4", proxy_port=3080,
                                 hidden_port=23000, vm_port=8090, state="stopped", state_since=0, idle_min=30,
                                 created_at=MID - 3600, stopped_at=MID + 3600, rate_cu_h=1.5, rate_source="t")
    cpu = await st.insert_session(colab_name="cpu", gpu_requested="CPU", gpu_class="CPU", proxy_port=3081,
                                  hidden_port=23001, vm_port=8090, state="stopped", state_since=0, idle_min=30,
                                  created_at=MID, stopped_at=MID + 1800, rate_cu_h=0.08, rate_source="t")
    r1 = await st.insert_run(session_id=l4, engine="llamacpp", alias="qwen", kv="q4_0", started_at=MID - 3600,
                             ended_at=MID + 3600)
    await st.insert_run(session_id=cpu, engine="llamacpp", alias="tiny", kv="f16", started_at=MID,
                        ended_at=MID + 1800)
    for index, (tps_ms, client, status) in enumerate([(1000, "pi", 200), (2000, "pi", 200), (500, "curl", 200),
                                                  (800, "colabinet-playground", 200), (0, "pi", 500)]):
        await st.insert_request({"run_id": r1, "ts": MID + index, "method": "POST", "path": "/v1/chat/completions",
                                 "status": status, "stream": 1, "prompt_tokens": 100, "completion_tokens": 10,
                                 "engine_gen_ms": tps_ms or None, "ttft_ms": 300.0, "total_ms": 2000.0,
                                 "client": client})
    yield db
    db.close()


def test_percentile():
    assert percentile([], 0.5) is None
    assert percentile([3, 1, 2], 0.5) == 2
    assert percentile([0, 10], 0.9) == 9


def test_spec_validation():
    with pytest.raises(QueryError, match="group-by"):
        QuerySpec.parse({"group_by": ["gpu; DROP TABLE x"]})
    with pytest.raises(QueryError, match="measure"):
        QuerySpec.parse({"measures": ["evil"]})
    with pytest.raises(QueryError, match="filter"):
        QuerySpec.parse({"filters": {"bogus": 1}})
    with pytest.raises(QueryError, match="list of strings"):
        QuerySpec.parse({"filters": {"gpu": "L4"}})


def test_build_sql_is_parameterised():
    spec = QuerySpec.parse({"filters": {"gpu": ["L4'; --"], "status_class": "2xx"}, "group_by": ["gpu"]})
    sql, params = build_sql(spec, now=1000.0)
    assert "L4'; --" not in sql and "L4'; --" in params
    assert "is_playground = 0" in sql and "status >= ?" in sql


async def test_speed_by_gpu_model_excludes_playground(seeded):
    spec = QuerySpec.parse({"time": "all", "group_by": ["gpu", "model_kv"],
                            "measures": ["requests", "tokens_out", "gen_tps_p50", "errors"]})
    res = await run_query(seeded, spec, now=MID + 10**6, clock=CLOCK)
    assert res.columns == ["gpu", "model_kv", "requests", "tokens_out", "gen_tps_p50", "errors"]
    assert res.rows == [["L4", "qwen · q4_0", 4, 40, 10.0, 1]]


async def test_prefill_p90_uses_successful_timed_requests_and_exports(seeded):
    await seeded.execute("UPDATE requests SET prefill_tokens = 100, engine_prefill_ms = engine_gen_ms")
    spec = QuerySpec.parse({"time": "all", "group_by": ["gpu"],
                            "measures": ["prefill_tps_p50", "prefill_tps_p90"]})
    result = await run_query(seeded, spec, now=MID + 10**6, clock=CLOCK)
    assert result.columns == ["gpu", "prefill_tps_p50", "prefill_tps_p90"]
    assert result.rows == [["L4", 100.0, 180.0]]
    assert to_csv(result).splitlines() == ["gpu,prefill_tps_p50,prefill_tps_p90", "L4,100.0,180.0"]
    spec.filters = {"exclude_playground": False}
    assert (await run_query(seeded, spec, now=MID + 10**6, clock=CLOCK)).rows == [["L4", 112.5, 177.5]]


async def test_cu_by_day_includes_groups_without_requests(seeded):
    spec = QuerySpec.parse({"time": "all", "group_by": ["gpu", "day"], "measures": ["requests", "cu"]})
    res = await run_query(seeded, spec, now=MID + 10**6, clock=CLOCK)
    got = {(row[0], row[1]): (row[2], row[3]) for row in res.rows}
    assert got[("L4", D0)] == (None, 1.5)
    assert got[("L4", D1)] == (4, 1.5)
    assert got[("CPU", D1)] == (None, 0.04)
    assert res.cu_available and "playground" in res.note


async def test_cu_unavailable_for_request_level_groups(seeded):
    spec = QuerySpec.parse({"time": "all", "group_by": ["client"], "measures": ["requests", "cu", "out_per_cu"]})
    res = await run_query(seeded, spec, now=MID + 10**6, clock=CLOCK)
    assert not res.cu_available and all(row[2] is None and row[3] is None for row in res.rows)
    assert "run-level" in res.note


def test_cu_by_group_clips_to_window():
    runs = [{"started_at": 0.0, "ended_at": None, "rate_cu_h": 3.6, "gpu": "L4", "model": "m", "kv": "f16",
             "engine": "e", "session": "s"}]
    got = cu_by_group(runs, ["gpu"], start=1000.0, end=2000.0, now=5000.0, day_bounds=CLOCK.day_bounds,
                      day_key=CLOCK.day_key)
    assert got == {("L4",): pytest.approx(1.0)}


def test_sql_tab_is_read_only(seeded):
    path = seeded.path
    res = run_sql(path, "SELECT gpu, median(gen_tps) AS p50 FROM request_stats GROUP BY gpu")
    assert res.columns == ["gpu", "p50"] and res.rows == [["L4", 11.25]]
    for bad in ("DELETE FROM requests", "UPDATE sessions SET state='x'", "ATTACH DATABASE 'x.db' AS x",
                "PRAGMA user_version = 9", "CREATE TABLE t (x)", "SELECT 1; DELETE FROM requests"):
        with pytest.raises(QueryError):
            run_sql(path, bad)
    assert run_sql(path, "SELECT COUNT(*) FROM requests").rows == [[5]]


def test_sql_tab_time_limit(seeded):
    slow = ("WITH RECURSIVE c(x) AS (SELECT 1 UNION ALL SELECT x + 1 FROM c) SELECT COUNT(*) FROM c")
    with pytest.raises(QueryError, match="longer than"):
        run_sql(seeded.path, slow, timeout_s=0.3)


def test_row_cap_and_csv(seeded):
    res = run_sql(seeded.path, "SELECT id, client FROM requests ORDER BY id", limit=2)
    assert len(res.rows) == 2 and "first 2" in res.note
    assert to_csv(res).splitlines()[0] == "id,client"


def test_saved_time_object_and_malformed_query_shapes():
    import json

    query = QuerySpec.parse({"time": {"preset": "30d"}})
    assert query.time == "30d"
    assert QuerySpec.parse({"time": "30d"}).time == "30d"
    assert json.loads(query.to_json())["time"] == {"preset": "30d"}


@pytest.mark.parametrize("raw", [
    None, [], {"time": []}, {"time": {"preset": "tomorrow"}}, {"time": {}},
    {"filters": []}, {"filters": None}, {"group_by": "gpu"}, {"group_by": [None]},
    {"measures": "requests"}, {"measures": [1]}, {"chart": []},
    {"filters": {"status_class": []}}, {"filters": {"exclude_playground": "false"}},
])
def test_malformed_query_returns_query_error(raw):
    with pytest.raises(QueryError):
        QuerySpec.parse(raw)
