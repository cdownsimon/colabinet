import asyncio
import os
import signal

import httpx
import pytest

from colabinet.billing import RECLAIM_S
from colabinet.supervisor import CreateRefused
from tests.integration.conftest import FakeEngineAdapter, make_world

pytestmark = pytest.mark.slow


async def chat(port, **body):
    async with httpx.AsyncClient() as c:
        return await c.post(f"http://127.0.0.1:{port}/v1/chat/completions",
                            json={"model": "m", "messages": [{"role": "user", "content": "hi"}], **body}, timeout=10)


async def test_happy_path_create_serve_record_stop(world):
    sid = await world.sup.create(world.request())
    row = await world.wait_state(sid, "ready")
    assert row["endpoint"] and row["gpu_class"] == "L4" and row["gpu_name"] == "NVIDIA L4"
    r = await chat(row["proxy_port"], stream=True, x_tokens=3)
    assert r.status_code == 200 and "data: [DONE]" in r.text
    await world.wait_for(lambda: _count(world, "requests"))
    run = await world.store.current_run(sid)
    assert run["status"] == "ready" and run["install_s"] is not None and run["ready_at"]
    await world.sup.stop(sid)
    row = await world.store.session(sid)
    assert row["state"] == "stopped" and row["stop_reason"] == "user" and not row["stop_unconfirmed"]
    assert row["stopped_at"] and (await world.store.current_run(sid))["ended_at"] == row["stopped_at"]
    assert world.fake.server == {}
    status_calls = [c for c in world.fake.calls("exec") if c[4].endswith("status.py")]
    assert sum("WITH_TORCH=1" in c for c in status_calls) == 1  # only the GPU check imports torch...
    assert all("WITH_TORCH=0" in c for c in status_calls[1:])  # ...later polls reset the sticky kernel env


async def _count(world, table):
    return (await world.store.db.fetchone(f"SELECT COUNT(*) AS n FROM {table}"))["n"] > 0


async def test_proxy_answers_503_while_setting_up(world):
    sid = await world.sup.create(world.request())
    row = await world.store.session(sid)
    r = await chat(row["proxy_port"])
    assert r.status_code == 503 and "session llm is" in r.json()["error"]["message"]
    await world.wait_state(sid, "ready")


async def test_not_available_fails_at_zero_cu_and_frees_the_name(world):
    world.fake.fault(new="not_available")
    sid = await world.sup.create(world.request(gpu="A100"))
    row = await world.wait_state(sid, "failed")
    assert row["stopped_at"] == row["created_at"] and not row["name_reserved"]
    assert not await world.store.name_in_use("llm")


async def test_crash_after_assign_raises_orphan_alert(world):
    world.fake.fault(new="crash_after_assign")
    sid = await world.sup.create(world.request())
    await world.wait_state(sid, "failed")
    assert any(a.kind == "create_orphan" for a in world.sup.alerts.list())


async def test_refuses_a_name_that_is_already_listed(world):
    proc = await world.sup.cli.spawn_new("llm", "CPU")
    await proc.communicate()
    with pytest.raises(CreateRefused, match="already listed"):
        await world.sup.create(world.request())
    await world.sup.cli.stop("llm")


async def test_create_needs_confirm_and_valid_port(world):
    req = world.request()
    req.confirm = False
    with pytest.raises(CreateRefused, match="confirm"):
        await world.sup.create(req)
    with pytest.raises(CreateRefused, match="8080"):
        await world.sup.create(_with(world.request(), vm_port=8080))


def _with(req, **kw):
    for k, v in kw.items():
        setattr(req, k, v)
    return req


async def test_gpu_mismatch_then_continue_uses_detected_rate(world):
    world.fake.fault(assign_accel="L4")
    sid = await world.sup.create(world.request(gpu="A100"))
    row = await world.wait_state(sid, "gpu_mismatch")
    assert row["gpu_class"] == "L4" and row["rate_cu_h"] == pytest.approx(1.54, rel=0.3)
    assert any(a.kind == "gpu_mismatch" for a in world.sup.alerts.list())
    await world.sup.decide(sid, "continue")
    await world.wait_state(sid, "ready")
    await world.sup.stop(sid)


async def test_gpu_mismatch_grace_stops(world):
    world.fake.fault(assign_accel="L4")
    sid = await world.sup.create(world.request(gpu="A100"))
    row = await world.wait_state(sid, "stopped", timeout=30)
    assert row["stop_reason"] == "gpu_mismatch"


async def test_setup_failure_then_grace_stop(tmp_path):
    w = await make_world(tmp_path, engine=FakeEngineAdapter(fail=True))
    try:
        sid = await w.sup.create(w.request())
        await w.wait_state(sid, "setup_failed")
        assert "404" in (await w.store.current_run(sid))["error"]
        row = await w.wait_state(sid, "stopped")
        assert row["stop_reason"] == "setup_failed"
    finally:
        await w.sup.shutdown()
        w.fake.kill_all()


async def test_reclaim_is_detected_and_billed_only_until_last_exec_plus_25min(world):
    sid = await world.sup.create(world.request())
    row = await world.wait_state(sid, "ready")
    st = world.fake.read()
    st["server"].pop(row["endpoint"])  # Colab reclaimed the VM
    world.fake.write(st)
    row = await world.wait_state(sid, "stopped", timeout=15)
    assert row["stop_reason"] == "reclaimed"
    assert row["stopped_at"] <= row["last_exec_ok"] + RECLAIM_S


async def test_hidden_auth_failure_does_not_count_as_reclaim(world):
    sid = await world.sup.create(world.request())
    await world.wait_state(sid, "ready")
    st = world.fake.read()
    saved = st["store"]
    st["store"], st["faults"]["hide_when_store_empty"] = {}, True
    world.fake.write(st)
    for _ in range(3):
        await world.sup.reconcile_once()
    assert (await world.store.session(sid))["state"] in ("ready", "engine_down")
    st = world.fake.read()
    st["store"], st["faults"]["hide_when_store_empty"] = saved, False
    world.fake.write(st)
    await world.sup.stop(sid)


async def test_stop_unconfirmed_is_retried_until_confirmed(world):
    sid = await world.sup.create(world.request())
    row = await world.wait_state(sid, "ready")
    world.fake.fault(stop_noop=True)
    await world.sup.stop(sid)
    row = await world.store.session(sid)
    assert row["stop_unconfirmed"] and row["stopped_at"] is None
    assert any(a.kind == "stop_unconfirmed" for a in world.sup.alerts.list())
    assert await world.store.name_in_use("llm")  # the name stays reserved
    world.fake.fault(stop_noop=False)
    with world.fake.edit() as state:
        state["store"]["llm"] = row["endpoint"]

    async def confirmed():
        return not (await world.store.session(sid))["stop_unconfirmed"]

    await world.wait_for(confirmed)
    assert (await world.store.session(sid))["stopped_at"] is not None


async def test_daily_cap_stops_everything_and_blocks_creates(world):
    sid = await world.sup.create(world.request())
    await world.wait_state(sid, "ready")
    await world.store.set_setting("daily_cap_cu", "0.00001")
    row = await world.wait_state(sid, "stopped")
    assert row["stop_reason"] == "daily_cap"
    with pytest.raises(CreateRefused, match="daily cap"):
        await world.sup.create(world.request(name="llm2"))
    await world.sup.extend_daily_cap(5)
    assert await world.sup.cap_today() == pytest.approx(5.00001)


async def test_idle_guard_stops(world):
    sid = await world.sup.create(world.request())
    await world.wait_state(sid, "ready")
    world.sup._activity.pop(sid)
    await world.store.update_session(sid, idle_min=1, last_activity=world.sup.now() - 61)
    row = await world.wait_state(sid, "stopped")
    assert row["stop_reason"] == "idle"


async def test_recovery_reattaches_after_restart(tmp_path):
    w = await make_world(tmp_path)
    sid = await w.sup.create(w.request())
    row = await w.wait_state(sid, "ready")
    await w.sup.shutdown()  # quit without stopping: the VM keeps running
    w2 = await make_world(tmp_path, fake=w.fake, db=w.store.db)
    try:
        await w2.wait_state(sid, "ready")

        async def serving():
            try:
                return (await chat(row["proxy_port"], x_tokens=1)).status_code == 200
            except httpx.HTTPError:
                return False

        await w2.wait_for(serving)
        await w2.sup.stop(sid)
        assert (await w2.store.session(sid))["state"] == "stopped"
    finally:
        await w2.sup.shutdown()
        w.fake.kill_all()
    assert not [h for h in w.fake.hazards if h.startswith(("auto_create", "overwrite"))]


async def test_foreign_vm_under_our_name_is_stopped(world):
    sid = await world.sup.create(world.request())
    row = await world.wait_state(sid, "ready")
    with world.fake.edit() as state:
        state["server"].pop(row["endpoint"])
        state["server"]["m-s-99"] = {"accel": "NONE", "shape": "STANDARD", "pids": []}
        state["store"]["llm"] = "m-s-99"
    row = await world.wait_state(sid, "stopped", timeout=15)
    assert "m-s-99" not in world.fake.server
    assert any(a.kind == "foreign_vm" for a in world.sup.alerts.list())
    await asyncio.sleep(0)


async def test_abandon_keeps_the_name_reserved_until_colab_new_exits(world):
    world.fake.fault(new="hang")  # assigned on the server, name never saved: the orphan case
    sid = await world.sup.create(world.request())

    async def slow():
        return any(a.kind == "create_slow" for a in world.sup.alerts.list())

    await world.wait_for(slow)
    await world.sup.abandon(sid)
    row = await world.store.session(sid)
    assert row["state"] == "failed" and row["stopped_at"] == row["created_at"]
    assert await world.store.name_in_use("llm")  # `colab new` is still running
    assert any(a.kind == "create_orphan" for a in world.sup.alerts.list())
    os.kill(row["new_pid"], signal.SIGKILL)  # stands in for the process finishing on its own

    async def released():
        return not await world.store.name_in_use("llm")

    await world.wait_for(released)


async def test_more_assignments_than_listed_rows_raises_an_alert(world, monkeypatch):
    from colabinet.cli_parse import Listing, Usage

    async def listing_with(assignments):
        return Listing([], Usage(190.0, 1.54, assignments), trusted=True, taken_at=world.sup.now())

    monkeypatch.setattr(world.sup, "fresh_listing", lambda: listing_with(2))  # 2 billing, none listed
    await world.sup.reconcile_once()
    assert any(a.key == "assignments_hidden" for a in world.sup.alerts.list())
    monkeypatch.setattr(world.sup, "fresh_listing", lambda: listing_with(0))
    await world.sup.reconcile_once()
    assert not any(a.key == "assignments_hidden" for a in world.sup.alerts.list())


async def test_reconfigure_starts_a_new_run_on_the_same_vm(world):
    sid = await world.sup.create(world.request())
    row = await world.wait_state(sid, "ready")
    first = await world.store.current_run(sid)
    new_run = await world.sup.reconfigure(sid, ctx=4096, kv="q8_0")
    assert (await world.store.session(sid))["state"] == "reconfiguring"
    r = await chat(row["proxy_port"])
    assert r.status_code == 503
    await world.wait_state(sid, "ready")
    run = await world.store.current_run(sid)
    assert run["id"] == new_run != first["id"] and (run["ctx"], run["kv"]) == (4096, "q8_0")
    old = await world.store.db.fetchone("SELECT ended_at FROM runs WHERE id = ?", (first["id"],))
    assert old["ended_at"] is not None and (await world.store.session(sid))["endpoint"] == row["endpoint"]
    assert (await chat(row["proxy_port"], x_tokens=2)).status_code == 200
    await world.wait_for(lambda: _count_run(world, new_run))
    with pytest.raises(CreateRefused):
        await world.sup.reconfigure(sid + 99)
    await world.sup.stop(sid)


async def _count_run(world, run_id):
    return (await world.store.db.fetchone("SELECT COUNT(*) AS n FROM requests WHERE run_id = ?", (run_id,)))["n"] > 0


async def test_ssh_429_raises_an_alert(world, monkeypatch):
    sid = await world.sup.create(world.request())
    row = await world.wait_state(sid, "ready")

    monkeypatch.setattr(world.sup.tunnels, "last_error", lambda name: "Already-active SSH session (HTTP 429)")
    monkeypatch.setattr(world.sup.tunnels, "alive", lambda pid, name: False)
    await world.sup.tunnels.stop(row["tunnel_pid"], "llm")

    async def alerted():
        return any(a.kind == "ssh_429" for a in world.sup.alerts.list())

    await world.wait_for(alerted)
    monkeypatch.undo()

    async def cleared():
        return not await alerted()

    await world.wait_for(cleared)
    await world.sup.stop(sid)


async def test_settings_persist_and_feed_defaults(world):
    values = await world.sup.update_settings(idle_min_default=7, daily_cap_cu=4.5, notify=False, bogus=1)
    assert (values["idle_min_default"], values["daily_cap_cu"], values["notify"]) == (7, 4.5, False)
    assert world.sup.alerts.notify_enabled is False
    assert await world.sup.cap_today() == 4.5
    with pytest.raises(CreateRefused, match="negative"):
        await world.sup.update_settings(daily_cap_cu=-1)
    sid = await world.sup.create(world.request())
    assert (await world.store.session(sid))["idle_min"] == 7
    await world.wait_state(sid, "ready")
    await world.sup.stop(sid)


async def test_exllamav3_session_gets_streamed_usage_injected(world):
    sid = await world.sup.create(world.request(preset="qwen38-27b-abl-exl3-4bpw"))
    row = await world.wait_state(sid, "ready")
    assert (await world.store.current_run(sid))["engine"] == "exllamav3"
    r = await chat(row["proxy_port"], stream=True, x_tokens=3, x_shape="tabby")
    assert r.status_code == 200 and '"usage"' in r.text

    async def recorded():
        row_ = await world.store.db.fetchone("SELECT tokens_source, completion_tokens FROM requests")
        return row_ is not None and row_["tokens_source"] == "engine" and row_["completion_tokens"] == 3

    await world.wait_for(recorded)
    await world.sup.stop(sid)


async def test_reconfigure_switches_both_engines_without_a_new_vm(world):
    sid = await world.sup.create(world.request())
    first = await world.wait_state(sid, "ready")
    for preset, engine in [("qwen38-27b-abl-exl3-4bpw", "exllamav3"),
                           ("qwen38-27b-abl-q4k", "llamacpp")]:
        await world.sup.reconfigure(sid, preset_id=preset)
        row = await world.wait_state(sid, "ready")
        assert row["endpoint"] == first["endpoint"]
        assert row["proxy_port"] == first["proxy_port"]
        assert (await world.store.current_run(sid))["engine"] == engine
        assert world.sup.route(sid).inject_usage is (engine == "exllamav3")
    assert len(await world.store.live_sessions()) == 1
    await world.sup.stop(sid)


async def test_parallel_slots_propagate_to_the_run(world):
    sid = await world.sup.create(world.request(parallel=2))
    await world.wait_state(sid, "ready")
    assert (await world.store.current_run(sid))["parallel"] == 2
    await world.sup.stop(sid)


async def test_invalid_reconfigure_does_not_close_the_current_run(world):
    sid = await world.sup.create(world.request())
    await world.wait_state(sid, "ready")
    first = await world.store.current_run(sid)
    with pytest.raises(CreateRefused):
        await world.sup.reconfigure(sid, ctx=0)
    assert (await world.store.current_run(sid))["id"] == first["id"]
    assert (await world.store.current_run(sid))["ended_at"] is None
    await world.sup.stop(sid)


async def test_concurrent_reconfigure_cannot_leave_overlapping_runs(world, monkeypatch):
    sid = await world.sup.create(world.request())
    await world.wait_state(sid, "ready")
    monkeypatch.setattr(world.sup, "_start_driver", lambda session_id: None)
    first, second = await asyncio.gather(world.sup.reconfigure(sid, ctx=4096),
                                         world.sup.reconfigure(sid, ctx=8192), return_exceptions=True)
    assert sum(isinstance(result, int) for result in (first, second)) == 1
    assert sum(isinstance(result, CreateRefused) for result in (first, second)) == 1
    open_runs = await world.store.db.fetchall("SELECT id FROM runs WHERE session_id = ? AND ended_at IS NULL", (sid,))
    assert len(open_runs) == 1
    await world.sup.stop(sid)


async def test_stop_during_reconfigure_cannot_resurrect_the_session(world, monkeypatch):
    sid = await world.sup.create(world.request())
    await world.wait_state(sid, "ready")
    for task in world.sup._loops:
        task.cancel()
    await asyncio.gather(*world.sup._loops, return_exceptions=True)
    monkeypatch.setattr(world.sup, "_start_driver", lambda session_id: None)
    read_run, release = asyncio.Event(), asyncio.Event()
    original = world.store.current_run

    async def pause_run_read(session_id):
        run = await original(session_id)
        read_run.set()
        await release.wait()
        return run

    monkeypatch.setattr(world.store, "current_run", pause_run_read)
    reconfigure = asyncio.create_task(world.sup.reconfigure(sid, ctx=4096))
    await asyncio.wait_for(read_run.wait(), 5)
    stop = asyncio.create_task(world.sup.stop(sid))
    try:
        await asyncio.wait({stop}, timeout=3.0)
    finally:
        release.set()
    await asyncio.gather(reconfigure, stop)
    assert (await world.store.session(sid))["state"] == "stopped"
    assert not await world.store.db.fetchall("SELECT id FROM runs WHERE session_id = ? AND ended_at IS NULL", (sid,))
