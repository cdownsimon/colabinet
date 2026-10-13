"""A fake OpenAI-compatible engine with llama.cpp-shaped (default) or TabbyAPI-shaped responses.

Request-body knobs: "x_tokens" (default 5), "x_delay" seconds per token, "x_error" (HTTP 500),
"x_shape": "tabby" (usage only, no llama.cpp timings). Run standalone: python fake_engine.py --port N --alias A
"""

from __future__ import annotations

import argparse
import asyncio
import json

ALIAS = "fake-model"


async def _body(receive) -> bytes:
    body = b""
    while True:
        m = await receive()
        body += m.get("body", b"")
        if not m.get("more_body"):
            return body


async def _json(send, status: int, obj: dict) -> None:
    data = json.dumps(obj).encode()
    await send({"type": "http.response.start", "status": status,
                "headers": [(b"content-type", b"application/json"), (b"content-length", str(len(data)).encode())]})
    await send({"type": "http.response.body", "body": data})


def _numbers(n: int, shape: str) -> dict:
    usage = {"prompt_tokens": 12, "completion_tokens": n, "total_tokens": 12 + n,
             "prompt_tokens_details": {"cached_tokens": 2}}
    if shape == "tabby":
        usage.update(prompt_time=0.05, completion_time=0.1 * n, prompt_tokens_per_sec="Indeterminate",
                     completion_tokens_per_sec=10.0, total_time=0.05 + 0.1 * n)
    out = {"usage": usage}
    if shape != "tabby":
        out["timings"] = {"prompt_n": 10, "prompt_ms": 50.0, "cache_n": 2, "predicted_n": n,
                          "predicted_ms": 100.0 * n}
    return out


async def app(scope, receive, send):
    if scope["type"] == "lifespan":
        while True:
            m = await receive()
            if m["type"] == "lifespan.startup":
                await send({"type": "lifespan.startup.complete"})
            elif m["type"] == "lifespan.shutdown":
                await send({"type": "lifespan.shutdown.complete"})
                return
    path, method = scope["path"], scope["method"]
    raw = await _body(receive)
    if path == "/health":
        return await _json(send, 200, {"status": "ok"})
    if path == "/v1/models":
        return await _json(send, 200, {"object": "list", "data": [{"id": ALIAS, "object": "model"}]})
    if method != "POST" or path not in ("/v1/chat/completions", "/v1/completions"):
        return await _json(send, 404, {"error": {"message": f"no route {method} {path}"}})
    req = json.loads(raw or b"{}")
    if req.get("x_error"):
        return await _json(send, 500, {"error": {"message": "boom: template raised", "type": "server_error"}})
    n, delay, shape = int(req.get("x_tokens", 5)), float(req.get("x_delay", 0)), req.get("x_shape", "llamacpp")
    if not req.get("stream"):
        await asyncio.sleep(delay * n)
        return await _json(send, 200, {
            "id": "c1", "object": "chat.completion", "model": ALIAS,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": " ".join(["tok"] * n)},
                         "finish_reason": "stop"}], **_numbers(n, shape)})
    await send({"type": "http.response.start", "status": 200, "headers": [(b"content-type", b"text/event-stream")]})

    async def event(obj) -> None:
        await send({"type": "http.response.body", "body": f"data: {json.dumps(obj)}\n\n".encode(),
                    "more_body": True})

    await event({"choices": [{"index": 0, "delta": {"role": "assistant"}}]})
    for i in range(n):
        await asyncio.sleep(delay)
        await event({"choices": [{"index": 0, "delta": {"content": f"tok{i} "}}]})
    final = {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}
    if shape != "tabby" or req.get("stream_options", {}).get("include_usage"):
        final.update(_numbers(n, shape))
    await event(final)
    await send({"type": "http.response.body", "body": b"data: [DONE]\n\n", "more_body": False})


if __name__ == "__main__":
    import uvicorn

    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--alias", default=ALIAS)
    a = ap.parse_args()
    ALIAS = a.alias
    uvicorn.run(app, host="127.0.0.1", port=a.port, log_level="warning")
