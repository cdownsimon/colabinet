import asyncio
import json
import time

import httpx
import pytest

from colabinet.activity import Activity
from colabinet.net import first_free, start_server, stop_server
from colabinet.proxy import ProxyManager, Route, SessionProxy
from tests.fakes import fake_engine


@pytest.mark.parametrize("options", [{}, {"include_usage": False}, None])
def test_explicit_usage_options_are_preserved(options):
    from colabinet.proxy import with_usage_option

    body = json.dumps({"stream": True, "stream_options": options}).encode()
    assert with_usage_option(body) == body


@pytest.fixture
async def engine():
    port = first_free(24000, 24999)
    server, task = await start_server(fake_engine.app, port)
    yield port
    await stop_server(server, task)


@pytest.fixture
async def proxied(engine):
    records, activity = [], Activity()
    route = {"r": Route("forward", engine, run_id=7)}
    port = first_free(25000, 25999)
    client = httpx.AsyncClient(timeout=httpx.Timeout(5, read=None))

    async def record(rec, run_id):
        records.append((rec, run_id))

    manager = ProxyManager()
    await manager.bind(1, port, SessionProxy(port, lambda: route["r"], record, activity, client, time.time))
    yield port, records, activity, route
    await manager.close()
    await client.aclose()


async def test_streaming_passthrough_and_record(proxied):
    port, records, activity, _ = proxied
    async with httpx.AsyncClient() as c:
        async with c.stream("POST", f"http://127.0.0.1:{port}/v1/chat/completions",
                            json={"model": "m", "stream": True, "x_tokens": 4}) as r:
            lines = [ln async for ln in r.aiter_lines() if ln.startswith("data:")]
    assert r.status_code == 200 and lines[-1] == "data: [DONE]" and len(lines) == 7
    await asyncio.sleep(0.05)
    rec, run_id = records[0]
    assert run_id == 7 and rec.stream and rec.completion_tokens == 4 and rec.prefill_tokens == 10
    assert rec.ttft_ms is not None and activity.last > 0


async def test_models_passthrough_is_recorded_but_not_activity(proxied):
    port, records, activity, _ = proxied
    async with httpx.AsyncClient() as c:
        r = await c.get(f"http://127.0.0.1:{port}/v1/models")
    assert r.json()["data"][0]["id"] == "fake-model"
    await asyncio.sleep(0.05)
    assert records[0][0].path == "/v1/models" and activity.last == 0


async def test_503_while_setting_up(proxied):
    port, records, activity, route = proxied
    route["r"] = Route("unavailable", message="session llm is setting up (download 45%)", run_id=7)
    async with httpx.AsyncClient() as c:
        r = await c.post(f"http://127.0.0.1:{port}/v1/chat/completions", json={})
    assert r.status_code == 503 and "download 45%" in r.json()["error"]["message"]
    assert activity.last == 0


async def test_502_when_upstream_unreachable(proxied):
    port, records, activity, route = proxied
    route["r"] = Route("forward", first_free(26000, 26999), run_id=7)
    async with httpx.AsyncClient() as c:
        r = await c.post(f"http://127.0.0.1:{port}/v1/chat/completions", json={})
    assert r.status_code == 502
    await asyncio.sleep(0.05)
    assert records[0][0].status == 502 and activity.last == 0


async def test_cross_site_origin_and_bad_host_rejected(proxied):
    port, records, _, _ = proxied
    async with httpx.AsyncClient() as c:
        r1 = await c.post(f"http://127.0.0.1:{port}/v1/chat/completions", json={},
                          headers={"Origin": "https://evil.example"})
        r2 = await c.post(f"http://127.0.0.1:{port}/v1/chat/completions", json={}, headers={"Host": "evil:80"})
        r3 = await c.post(f"http://127.0.0.1:{port}/v1/chat/completions", json={"x_tokens": 1},
                          headers={"Origin": "http://localhost:7860"})
    assert (r1.status_code, r2.status_code, r3.status_code) == (403, 403, 200)


async def test_client_disconnect_recorded_as_499(proxied):
    port, records, _, _ = proxied
    async with httpx.AsyncClient() as c:
        async with c.stream("POST", f"http://127.0.0.1:{port}/v1/chat/completions",
                            json={"stream": True, "x_tokens": 50, "x_delay": 0.02}) as r:
            n = 0
            async for _ in r.aiter_lines():
                n += 1
                if n > 6:
                    break
    for _ in range(50):
        if records:
            break
        await asyncio.sleep(0.05)
    rec = records[0][0]
    assert rec.status == 499 and rec.completion_tokens and rec.completion_tokens < 50


async def test_usage_injection_for_engines_that_need_it(proxied):
    port, records, _, route = proxied
    route["r"] = Route("forward", route["r"].upstream_port, run_id=7, inject_usage=True)
    async with httpx.AsyncClient() as c:
        r = await c.post(f"http://127.0.0.1:{port}/v1/chat/completions",
                         json={"stream": True, "x_tokens": 3, "x_shape": "tabby"})
        assert '"usage"' in r.text
        await c.post(f"http://127.0.0.1:{port}/v1/chat/completions",
                     json={"stream": True, "x_tokens": 3, "x_shape": "tabby", "stream_options": {}})
    await asyncio.sleep(0.1)
    first, second = records[0][0], records[1][0]
    assert (first.completion_tokens, first.tokens_source, first.engine_gen_ms) == (3, "engine", pytest.approx(300.0))
    assert second.tokens_source == "chunks"


def test_with_usage_option_rules():
    from colabinet.proxy import with_usage_option

    assert json.loads(with_usage_option(b'{"stream": true}')) == {"stream": True,
                                                                  "stream_options": {"include_usage": True}}
    assert with_usage_option(b'{"stream": false}') == b'{"stream": false}'
    assert with_usage_option(b"not json") == b"not json"
