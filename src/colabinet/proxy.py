"""Per-session metering proxy (spec §7): pass any request through unchanged, stream the response back as it arrives,
and record one row per request from a Tap on the side."""

from __future__ import annotations

import asyncio
import contextlib
import itertools
import json
import logging
from dataclasses import dataclass
from typing import Awaitable, Callable
from urllib.parse import urlsplit

import httpx

from .activity import Activity, is_inference
from .net import QuietServer, start_server, stop_server
from .stats import RequestRecord, Tap

log = logging.getLogger("colabinet.proxy")

HOP_BY_HOP = {b"connection", b"keep-alive", b"proxy-authenticate", b"proxy-authorization", b"te", b"trailers",
              b"transfer-encoding", b"upgrade", b"host"}
LOCAL_HOSTS = {"127.0.0.1", "localhost"}


@dataclass
class Route:
    kind: str  # forward | unavailable (503) | down (502)
    upstream_port: int | None = None
    message: str = ""
    run_id: int | None = None
    inject_usage: bool = False


def with_usage_option(body: bytes) -> bytes:
    """Ask for streamed usage, but only if the client asked for a stream and didn't set stream_options itself."""
    try:
        obj = json.loads(body)
    except (ValueError, UnicodeDecodeError):
        return body
    if not isinstance(obj, dict) or not obj.get("stream") or "stream_options" in obj:
        return body
    obj["stream_options"] = {"include_usage": True}
    return json.dumps(obj).encode()


def host_ok(headers: dict[bytes, bytes], port: int) -> bool:
    host = headers.get(b"host", b"").decode("latin-1")
    return host in (f"127.0.0.1:{port}", f"localhost:{port}")


def origin_ok(headers: dict[bytes, bytes]) -> bool:
    origin = headers.get(b"origin")
    if origin is None:
        return True  # non-browser clients (pi, curl, SDKs) send no Origin
    return urlsplit(origin.decode("latin-1")).hostname in LOCAL_HOSTS


async def _read_body(receive) -> bytes | None:
    body = b""
    while True:
        m = await receive()
        if m["type"] == "http.disconnect":
            return None
        body += m.get("body", b"")
        if not m.get("more_body"):
            return body


async def _send_json(send, status: int, message: str, kind: str) -> None:
    data = json.dumps({"error": {"message": message, "type": kind}}).encode()
    await send({"type": "http.response.start", "status": status,
                "headers": [(b"content-type", b"application/json"), (b"content-length", str(len(data)).encode())]})
    await send({"type": "http.response.body", "body": data})


class SessionProxy:
    """ASGI app for one session's local port."""

    _ids = itertools.count(1)

    def __init__(self, port: int, route: Callable[[], Route],
                 record: Callable[[RequestRecord, int], Awaitable[None]], activity: Activity,
                 client: httpx.AsyncClient, now: Callable[[], float],
                 on_upstream_error: Callable[[], None] | None = None) -> None:
        self.port = port
        self.route = route
        self.record = record
        self.activity = activity
        self.client = client
        self.now = now
        self.on_upstream_error = on_upstream_error or (lambda: None)

    async def _record(self, rec: RequestRecord, run_id: int | None) -> None:
        if run_id is None:
            return
        try:
            await self.record(rec, run_id)
        except Exception:  # a stats failure must never affect the proxied response
            log.exception("could not record request")

    async def __call__(self, scope, receive, send) -> None:
        if scope["type"] != "http":
            return
        headers = dict(scope["headers"])
        if not host_ok(headers, self.port) or not origin_ok(headers):
            return await _send_json(send, 403, "colabinet: request rejected (Host/Origin)", "forbidden")
        body = await _read_body(receive)
        if body is None:
            return
        method, path = scope["method"], scope["path"]
        t0 = self.now()
        tap = Tap(method, path, t0, headers.get(b"user-agent", b"").decode("latin-1"), body)
        route = self.route()
        if route.kind != "forward":
            status = 503 if route.kind == "unavailable" else 502
            await _send_json(send, status, f"colabinet: {route.message}", "unavailable")
            tap.rec.error_short = route.message[:200]
            return await self._record(tap.finish(self.now(), status), route.run_id)

        inference = is_inference(method, path)
        key = next(self._ids)
        if inference:
            self.activity.begin(key, t0)
        query = scope.get("query_string", b"").decode("latin-1")
        url = f"http://127.0.0.1:{route.upstream_port}{path}" + (f"?{query}" if query else "")
        if route.inject_usage and method == "POST" and path.rstrip("/") in ("/v1/chat/completions", "/v1/completions"):
            body = with_usage_option(body)
        fwd_headers = [(k, v) for k, v in scope["headers"]
                       if k.lower() not in HOP_BY_HOP and k.lower() != b"content-length"]
        try:
            resp = await self.client.send(self.client.build_request(method, url, headers=fwd_headers, content=body),
                                          stream=True)
        except httpx.TransportError as e:
            self.activity.end(key, self.now(), useful=False)
            self.on_upstream_error()
            await _send_json(send, 502, "colabinet: engine unreachable through the tunnel", "bad_gateway")
            tap.rec.error_short = f"upstream: {type(e).__name__}"
            return await self._record(tap.finish(self.now(), 502), route.run_id)

        tap.start(resp.status_code, resp.headers.get("content-type"))
        disconnected = False

        async def watch() -> None:
            while True:
                if (await receive())["type"] == "http.disconnect":
                    return

        async def pump() -> None:
            await send({"type": "http.response.start", "status": resp.status_code,
                        "headers": [(k, v) for k, v in resp.headers.raw if k.lower() not in HOP_BY_HOP]})
            async for chunk in resp.aiter_raw():
                t = self.now()
                tap.feed(chunk, t)
                self.activity.touch(key, t)
                await send({"type": "http.response.body", "body": chunk, "more_body": True})
            await send({"type": "http.response.body", "body": b"", "more_body": False})

        watcher, pumper = asyncio.create_task(watch()), asyncio.create_task(pump())
        try:
            done, _ = await asyncio.wait({watcher, pumper}, return_when=asyncio.FIRST_COMPLETED)
            if pumper in done:
                watcher.cancel()
                exc = pumper.exception()
                if exc is not None:
                    tap.rec.error_short = f"upstream: {type(exc).__name__}"
            else:
                disconnected = True
                pumper.cancel()
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await pumper
        finally:
            await resp.aclose()
        t_end = self.now()
        self.activity.end(key, t_end, useful=inference)
        await self._record(tap.finish(t_end, 499 if disconnected else None), route.run_id)


class ProxyManager:
    """Starts and stops one listener per session port."""

    def __init__(self) -> None:
        self._servers: dict[int, tuple[QuietServer, asyncio.Task, int]] = {}

    async def bind(self, sid: int, port: int, app) -> None:
        if sid in self._servers:
            return
        server, task = await start_server(app, port)
        self._servers[sid] = (server, task, port)

    async def unbind(self, sid: int) -> None:
        entry = self._servers.pop(sid, None)
        if entry:
            await stop_server(entry[0], entry[1])

    def bound(self, sid: int) -> bool:
        return sid in self._servers

    def ports(self) -> set[int]:
        return {p for _, _, p in self._servers.values()}

    async def close(self) -> None:
        for sid in list(self._servers):
            await self.unbind(sid)
