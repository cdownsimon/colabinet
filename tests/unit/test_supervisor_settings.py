import asyncio
import json

import httpx
import pytest

from colabinet.activity import Activity
from colabinet.alerts import Alerts, NullNotifier
from colabinet.config import Settings
from colabinet.presets import PresetError
from colabinet.proxy import Route, SessionProxy
from colabinet.supervisor import CreateRefused, Supervisor


@pytest.fixture
def supervisor(store, clock, tmp_path):
    return Supervisor(settings=Settings(data_dir=tmp_path), store=store, cli=None, tunnels=None, proxies=None,
                      alerts=Alerts(NullNotifier(), clock.now), clock=clock, presets=PresetError("unused"),
                      http=object())


@pytest.mark.parametrize("key", ["daily_cap_cu", "low_balance_cu"])
@pytest.mark.parametrize("value", ["nan", "inf", "-inf"])
async def test_nonfinite_settings_leave_guards_unchanged(supervisor, key, value):
    before = await supervisor.settings_values()
    with pytest.raises(CreateRefused):
        await supervisor.update_settings(**{key: value})
    assert await supervisor.settings_values() == before
    assert await supervisor.cap_today() == before["daily_cap_cu"]


async def test_invalid_settings_batch_leaves_every_setting_unchanged(supervisor):
    before = await supervisor.settings_values()
    with pytest.raises(CreateRefused):
        await supervisor.update_settings(daily_cap_cu=5, notify=not before["notify"], idle_min_default=-1)
    assert await supervisor.settings_values() == before
    assert supervisor.alerts.notify_enabled == before["notify"]


async def test_consecutive_streams_do_not_reuse_stale_upstream_connection(store, clock, tmp_path, monkeypatch):
    body = b'data: {"choices":[{"delta":{"content":"ok"}}]}\n\ndata: [DONE]\n\n'

    class Stream:
        sent = False

        async def read(self, max_bytes, timeout=None):
            if self.sent:
                return b""
            self.sent = True
            return (b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nContent-Length: "
                    + str(len(body)).encode() + b"\r\n\r\n" + body)

        async def write(self, buffer, timeout=None):
            pass

        async def aclose(self):
            pass

        def get_extra_info(self, info):
            # Simulate a closure that becomes visible only on the next read.
            return False if info == "is_readable" else None

    class Backend:
        connections = 0

        async def connect_tcp(self, *args, **kwargs):
            self.connections += 1
            return Stream()

        async def sleep(self, seconds):
            await asyncio.sleep(seconds)

    backend = Backend()
    init_transport = httpx.AsyncHTTPTransport.__init__

    def transport_with_backend(self, *args, **kwargs):
        init_transport(self, *args, **kwargs)
        # Keep the real HTTPcore pool/protocol behavior; replace only network I/O.
        self._pool._network_backend = backend

    monkeypatch.setattr(httpx.AsyncHTTPTransport, "__init__", transport_with_backend)
    sup = Supervisor(settings=Settings(data_dir=tmp_path), store=store, cli=None, tunnels=None, proxies=None,
                     alerts=Alerts(NullNotifier(), clock.now), clock=clock, presets=PresetError("unused"))
    records = []

    async def record(rec, run_id):
        records.append(rec)

    proxy = SessionProxy(3080, lambda: Route("forward", 23000, run_id=1), record, Activity(), sup.http, clock.now)
    async with sup.http, httpx.AsyncClient(transport=httpx.ASGITransport(app=proxy),
                                         base_url="http://127.0.0.1:3080") as client:
        responses = [await client.post("/v1/chat/completions", content=json.dumps({"stream": True}))
                     for _ in range(2)]
    assert [response.status_code for response in responses] == [200, 200]
    assert all(response.content == body for response in responses)
    assert [rec.error_short for rec in records] == [None, None]
    assert backend.connections == 2
