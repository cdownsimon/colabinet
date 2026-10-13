"""Shared console and JSON API operations; no additional runtime or proxy path."""

from __future__ import annotations

import asyncio
import dataclasses
import time
from typing import Literal

import httpx
from fastapi import Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field, StrictBool, ValidationError

from ..metrics import QueryError, QuerySpec, run_query, run_sql
from ..supervisor import CreateRefused


def require_confirmation(value):
    if value is not True:
        raise CreateRefused('this action spends or stops compute: send {"confirm": true}')


async def read_json(request: Request):
    try:
        return await request.json()
    except ValueError:
        raise QueryError("invalid JSON body") from None


async def query_metrics(sup, raw):
    return await run_query(sup.store.db, QuerySpec.parse(raw), sup.now(), sup.clock)


async def query_sql(sup, text):
    if not isinstance(text, str):
        raise QueryError("SQL must be text")
    return await asyncio.to_thread(run_sql, sup.store.db.path, text)


async def save_query(sup, name, raw):
    if not isinstance(name, str) or not name.strip():
        raise QueryError("give the query a name")
    return await sup.store.save_query(name.strip(), QuerySpec.parse(raw).to_json(), sup.now())


def result_json(result):
    return dataclasses.asdict(result)


class PlaygroundBody(BaseModel):
    session_id: int
    prompt: str = ""
    stream: StrictBool = True
    max_tokens: int = Field(default=256, gt=0)
    thinking: Literal["none", "low", "medium", "high", "xhigh"] = "none"


async def send_playground(sup, raw):
    try:
        body = PlaygroundBody.model_validate(raw)
    except ValidationError:
        raise QueryError("invalid playground input") from None
    row = await sup.store.session(body.session_id)
    if not row or row["state"] != "ready":
        raise CreateRefused("pick a session that is ready")
    run = await sup.store.current_run(row["id"])
    payload = {"model": run["alias"], "stream": body.stream, "max_tokens": body.max_tokens,
               "messages": [{"role": "user", "content": body.prompt}],
               "chat_template_kwargs": {"enable_thinking": body.thinking != "none"}}
    if body.thinking != "none":
        payload["reasoning_effort"] = body.thinking
        payload["chat_template_kwargs"]["reasoning_effort"] = body.thinking
    url = f"http://127.0.0.1:{row['proxy_port']}/v1/chat/completions"
    headers = {"User-Agent": "colabinet-playground"}
    try:
        if not body.stream:
            start = time.time()
            response = await sup.http.post(url, json=payload, headers=headers)
            try:
                data = response.json()
            except ValueError:
                data = {"error": "upstream returned a non-JSON response"}
            return JSONResponse({"status": response.status_code, "total_ms": (time.time() - start) * 1000,
                                 "body": data, "request": payload})
        request = sup.http.build_request("POST", url, json=payload, headers=headers)
        upstream = await sup.http.send(request, stream=True)
    except httpx.HTTPError:
        raise CreateRefused("session proxy is not reachable") from None

    async def relay():
        try:
            async for chunk in upstream.aiter_raw():
                yield chunk
        finally:
            await upstream.aclose()

    return StreamingResponse(relay(), status_code=upstream.status_code,
                             media_type=upstream.headers.get("content-type", "text/event-stream"))
