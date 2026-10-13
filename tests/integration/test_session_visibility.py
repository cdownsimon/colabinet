import asyncio

import httpx
import pytest

from colabinet import supervisor
from colabinet.api import create_app
from tests.integration.conftest import make_world
from tests.integration.test_console import CSRF

pytestmark = pytest.mark.slow


@pytest.fixture
async def console(tmp_path):
    world = await make_world(tmp_path)
    world.settings.hidden_port_range = (39000, 39999)
    try:
        sid = await world.sup.create(world.request())
        await world.wait_state(sid, "ready")
        app = create_app(world.sup, "api-test-token", 7860, csrf=CSRF)
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),
                                    base_url="http://127.0.0.1:7860") as client:
            yield world, client, sid
    finally:
        await world.sup.shutdown()
        for row in await world.store.sessions():
            await world.sup.tunnels.stop(row["tunnel_pid"], row["colab_name"])
        world.fake.kill_all()
        world.store.db.close()


async def test_session_resources_are_available_on_every_detail_tab(console):
    world, client, sid = console
    for tab in ("overview", "runs", "requests", "logs"):
        response = await client.get(f"/sessions/{sid}?tab={tab}")
        assert response.status_code == 200
        assert 'id="session-resources"' in response.text
        assert 'id="session-controls"' in response.text
        assert 'id="session-watch"' in response.text
    response = await client.get(f"/ui/sessions/{sid}/resources")
    assert response.status_code == 200
    for label in ("CPU", "RAM", "Disk", "GPU", "VRAM", "Last received"):
        assert label in response.text


async def test_control_actions_keep_stable_ids_across_partial_refreshes(console):
    world, client, sid = console
    action_ids = ("reconfigure", "restart", "keep", "playground", "stop")
    for _ in range(2):
        response = await client.get(f"/ui/sessions/{sid}/controls")
        assert response.status_code == 200
        for action_id in action_ids:
            assert response.text.count(f'id="session-action-{action_id}"') == 1


async def test_watch_refreshes_a_sample_without_extending_idle(console):
    world, client, sid = console
    world.settings.keepalive_s = 100
    await world.sup._cancel_driver(sid)
    row = await world.store.session(sid)
    last_activity = row["last_activity"]
    previous_exec = row["last_exec_ok"]
    world.sup._last_exec[sid] = world.sup.now() - 20
    world.sup._start_driver(sid)
    response = await client.post(f"/ui/sessions/{sid}/watch", headers={"X-CSRF-Token": CSRF})
    assert response.status_code == 204

    async def refreshed():
        return (await world.store.session(sid))["last_exec_ok"] > previous_exec

    await world.wait_for(refreshed)
    assert (await world.store.session(sid))["last_activity"] == last_activity
    response = await client.get(f"/ui/sessions/{sid}/resources")
    assert "NVIDIA L4" in response.text
    assert "Live" in response.text


async def test_browser_polling_does_not_exec_once_per_request(console):
    world, client, sid = console
    world.settings.keepalive_s = 100
    await world.sup._cancel_driver(sid)
    world.sup._last_exec[sid] = world.sup.now()
    world.sup._start_driver(sid)
    previous_calls = len(world.fake.calls("exec"))
    responses = await asyncio.gather(*(client.post(f"/ui/sessions/{sid}/watch",
        headers={"X-CSRF-Token": CSRF}) for _ in range(8)))
    assert all(response.status_code == 204 for response in responses)
    for _ in range(3):
        assert (await client.get(f"/ui/sessions/{sid}/resources")).status_code == 200
    await asyncio.sleep(0.2)
    assert len(world.fake.calls("exec")) == previous_calls


@pytest.mark.parametrize("observed", [False, True])
async def test_upstream_failures_probe_engine_health_before_sampling_is_due(console, observed):
    world, client, sid = console
    world.settings.keepalive_s = 100
    await world.sup._cancel_driver(sid)
    world.sup._last_exec[sid] = world.sup.now()
    world.fake.kill_all()
    row = await world.store.session(sid)
    previous_calls = len(world.fake.calls("exec"))
    states = []
    for _ in range(2):
        response = await world.sup.http.post(f"http://127.0.0.1:{row['proxy_port']}/v1/chat/completions",
            json={"model": "m", "messages": [{"role": "user", "content": "hi"}]})
        assert response.status_code == 502
        if observed:
            response = await client.post(f"/ui/sessions/{sid}/watch", headers={"X-CSRF-Token": CSRF})
            assert response.status_code == 204
        states.append(await world.sup._ready(row))
    assert states == ["ready", "engine_down"]
    assert len(world.fake.calls("exec")) == previous_calls + 2


async def test_monitoring_requires_csrf_and_a_local_origin(console):
    world, client, sid = console
    assert (await client.post(f"/ui/sessions/{sid}/watch")).status_code == 403
    response = await client.post(f"/ui/sessions/{sid}/watch", headers={
        "X-CSRF-Token": CSRF, "Origin": "https://other.example",
    })
    assert response.status_code == 403
    response = await client.post(f"/ui/sessions/{sid}/watch", headers={"X-CSRF-Token": CSRF})
    assert response.status_code == 204


async def test_stopped_session_freezes_resources_and_removes_live_controls(console):
    world, client, sid = console
    await world.sup.stop(sid)
    previous_calls = len(world.fake.calls("exec"))
    response = await client.post(f"/ui/sessions/{sid}/watch", headers={"X-CSRF-Token": CSRF})
    assert response.status_code == 204
    response = await client.get(f"/ui/sessions/{sid}/resources")
    assert "Session ended" in response.text
    response = await client.get(f"/ui/sessions/{sid}/controls")
    assert "stopped" in response.text
    assert "Restart engine" not in response.text
    assert "cbConfirmStop" not in response.text
    assert len(world.fake.calls("exec")) == previous_calls


async def test_log_tools_download_retained_lines_and_events(console):
    world, client, sid = console
    await world.sup._cancel_driver(sid)
    row = await world.store.session(sid)
    logfile = world.fake.root / "vm" / row["endpoint"] / "colabinet" / "engine.log"
    with logfile.open("a") as output:
        output.write("<script>not HTML</script>\n")
    await world.sup._status(row)
    await world.store.event(world.sup.now(), "vm:download", session_id=sid, gb=1)
    response = await client.get(f"/sessions/{sid}?tab=logs")
    assert "Pause" in response.text and "Auto-scroll" in response.text and "Download" in response.text
    response = await client.get(f"/ui/sessions/{sid}/logs")
    assert "&lt;script&gt;not HTML&lt;/script&gt;" in response.text
    response = await client.get(f"/ui/sessions/{sid}/logs.txt")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/plain")
    assert "attachment" in response.headers["content-disposition"]
    assert "<script>not HTML</script>" in response.text
    assert "vm_created" in response.text
    assert 'vm:download {"gb": 1}' in response.text


async def test_new_session_partials_reject_unknown_ids(console):
    world, client, _ = console
    for part in ("resources", "controls", "logs.txt"):
        assert (await client.get(f"/ui/sessions/999/{part}")).status_code == 404
    response = await client.post("/ui/sessions/999/watch", headers={"X-CSRF-Token": CSRF})
    assert response.status_code == 404


async def test_observation_lease_expires_back_to_normal_keepalive(console, monkeypatch):
    world, client, sid = console
    monkeypatch.setattr(supervisor, "OBSERVE_INTERVAL_S", 0.15, raising=False)
    monkeypatch.setattr(supervisor, "OBSERVE_LEASE_S", 2.0, raising=False)
    world.settings.keepalive_s = 100
    await world.sup._cancel_driver(sid)
    world.sup._start_driver(sid)
    initial_calls = len(world.fake.calls("exec"))
    response = await client.post(f"/ui/sessions/{sid}/watch", headers={"X-CSRF-Token": CSRF})
    assert response.status_code == 204

    async def sampled_twice():
        return len(world.fake.calls("exec")) >= initial_calls + 2

    await world.wait_for(sampled_twice, timeout=4)
    await asyncio.sleep(2.5)
    previous_calls = len(world.fake.calls("exec"))
    await asyncio.sleep(0.4)
    assert len(world.fake.calls("exec")) == previous_calls


async def test_observed_sampling_requires_a_trusted_matching_endpoint(console):
    world, client, sid = console
    world.settings.keepalive_s = 100
    await world.sup._cancel_driver(sid)
    world.sup._last_exec[sid] = world.sup.now() - 20
    world.fake.fault(usage_fail=True)
    previous_calls = len(world.fake.calls("exec"))
    world.sup._start_driver(sid)
    response = await client.post(f"/ui/sessions/{sid}/watch", headers={"X-CSRF-Token": CSRF})
    assert response.status_code == 204
    await asyncio.sleep(0.6)
    assert len(world.fake.calls("exec")) == previous_calls


async def test_resources_are_marked_stale_without_a_fresh_sample(console):
    world, client, sid = console
    await world.sup._cancel_driver(sid)
    future = world.sup.now() + 60
    world.sup.clock.now = lambda: future
    response = await client.get(f"/ui/sessions/{sid}/resources")
    assert response.status_code == 200
    assert "Stale" in response.text


async def test_observed_sampling_does_not_exec_a_reused_session_name(console):
    world, client, sid = console
    world.settings.keepalive_s = 100
    world.settings.reconcile_s = 100
    await world.sup._cancel_driver(sid)
    world.sup._last_exec[sid] = world.sup.now() - 20
    with world.fake.edit() as state:
        state["server"]["foreign-endpoint"] = {"accel": "NONE", "shape": "STANDARD", "pids": []}
        state["store"]["llm"] = "foreign-endpoint"
    previous_calls = len(world.fake.calls("exec"))
    world.sup._start_driver(sid)
    response = await client.post(f"/ui/sessions/{sid}/watch", headers={"X-CSRF-Token": CSRF})
    assert response.status_code == 204
    await asyncio.sleep(0.6)
    assert len(world.fake.calls("exec")) == previous_calls


async def test_current_gpu_temperature_reaches_the_resource_panel(console):
    world, client, sid = console
    response = await client.get(f"/ui/sessions/{sid}/resources")
    assert response.status_code == 200
    assert "61 °C" in response.text
