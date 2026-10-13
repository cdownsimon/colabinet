"""The whole plan-1 stack in one process: serve() + ctl + proxy, with fake colab / fake engine / local tunnel."""

import asyncio
import dataclasses

import httpx
import pytest

from colabinet import ctl
from colabinet.api import parse_deadline
from colabinet.net import first_free
from colabinet.server import AlreadyRunning, serve
from colabinet.tunnel import LocalTunnels
from tests.fakes import FakeColab
from tests.integration.conftest import FakeEngineAdapter, fast_settings

pytestmark = pytest.mark.slow


@pytest.fixture
async def running(tmp_path):
    fake = FakeColab(tmp_path)
    settings = dataclasses.replace(fast_settings(tmp_path, fake), ui_port=first_free(31000, 31999))
    stop, ready = asyncio.Event(), asyncio.Event()
    task = asyncio.create_task(serve(settings, tunnels=LocalTunnels(), engines={"llamacpp": FakeEngineAdapter()},
                                     stop_event=stop, ready=ready))
    await asyncio.wait_for(ready.wait(), 10)
    yield settings, fake
    stop.set()
    await asyncio.wait_for(task, 30)
    fake.kill_all()
    assert not [h for h in fake.hazards if h.startswith(("auto_create", "overwrite", "new_killed"))]


async def ctl_run(settings, *argv) -> tuple[int, str]:
    lines: list[str] = []
    rc = await asyncio.to_thread(ctl.run, list(argv), settings, lines.append)
    return rc, "\n".join(lines)


async def test_ctl_create_use_stop(running):
    settings, fake = running
    vm_port = first_free(30000, 30999)
    rc, out = await ctl_run(settings, "create", "--name", "llm", "--gpu", "L4", "--preset", "qwen38-27b-abl-q4k",
                            "--vm-port", str(vm_port), "--yes")
    assert rc == 0 and "created session" in out
    sid = int(out.split()[-1])
    for _ in range(300):
        rc, out = await ctl_run(settings, "list")
        if " ready " in out:
            break
        await asyncio.sleep(0.1)
    assert " ready " in out
    port = int(out.split("http://127.0.0.1:")[1].split("/")[0])
    async with httpx.AsyncClient() as c:
        r = await c.post(f"http://127.0.0.1:{port}/v1/chat/completions", json={"x_tokens": 2},
                         headers={"User-Agent": "pi/0.99", "Authorization": "Bearer none"})
    assert r.status_code == 200
    await asyncio.sleep(0.3)
    rc, out = await ctl_run(settings, "stats")
    assert rc == 0 and "L4" in out
    rc, _ = await ctl_run(settings, "stop", str(sid), "--yes")
    assert rc == 0
    rc, out = await ctl_run(settings, "list")
    assert " stopped " in out and fake.server == {}


async def test_api_rejects_missing_token_bad_host_and_cross_site(running):
    settings, _ = running
    base = f"http://127.0.0.1:{settings.ui_port}"
    token = settings.token_path.read_text()
    body = {"name": "x", "gpu": "L4", "preset": "qwen38-27b-abl-q4k", "confirm": True}
    async with httpx.AsyncClient(base_url=base) as c:
        assert (await c.get("/api/health")).status_code == 200
        assert (await c.post("/api/sessions", json=body)).status_code == 403
        assert (await c.post("/api/sessions", json=body, headers={"X-Colabinet-Token": token,
                                                                  "Origin": "https://evil.example"})).status_code == 403
        assert (await c.get("/api/health", headers={"Host": "evil.example"})).status_code == 403
        r = await c.post("/api/sessions", json={**body, "confirm": False}, headers={"X-Colabinet-Token": token})
        assert r.status_code == 409 and "confirm" in r.json()["error"]
    assert settings.token_path.stat().st_mode & 0o777 == 0o600


async def test_second_instance_refuses_to_start(running):
    settings, _ = running
    with pytest.raises(AlreadyRunning):
        await serve(settings)


def test_parse_deadline_rolls_to_tomorrow():
    from datetime import datetime

    now = datetime(2026, 10, 1, 18, 0)
    assert datetime.fromtimestamp(parse_deadline("18:30", now)).day == 1
    assert datetime.fromtimestamp(parse_deadline("09:00", now)).day == 2
    assert parse_deadline(None) is None
