"""Read token counts and speeds from a response without changing it (spec §7).

The proxy feeds every response chunk to a Tap. For SSE streams it parses events on the side; for JSON bodies it
keeps up to MAX_BODY bytes and parses once at the end. Engine-reported numbers (llama.cpp `timings`, OpenAI or
Anthropic `usage`) are preferred; the proxy's own timings (TTFT, total) are always recorded.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Any

MAX_BODY = 8 * 1024 * 1024


class SSEParser:
    """Incremental text/event-stream parser. feed() returns the `data:` payloads of completed events."""

    def __init__(self) -> None:
        self._buf = b""
        self._data: list[str] = []

    def feed(self, chunk: bytes) -> list[str]:
        self._buf += chunk
        out: list[str] = []
        while True:
            nl = self._buf.find(b"\n")
            if nl < 0:
                break
            line = self._buf[:nl].rstrip(b"\r").decode("utf-8", "replace")
            self._buf = self._buf[nl + 1:]
            if line == "":
                if self._data:
                    out.append("\n".join(self._data))
                    self._data = []
            elif line.startswith("data:"):
                self._data.append(line[5:].lstrip(" "))
        return out


def _content_delta(ev: dict[str, Any]) -> bool:
    """True if this streamed event carries generated text (content or reasoning), not just a role."""
    for choice in ev.get("choices") or []:
        delta = choice.get("delta") or {}
        if delta.get("content") or delta.get("reasoning_content") or delta.get("reasoning"):
            return True
        if choice.get("text"):
            return True
    if ev.get("type") == "content_block_delta":  # Anthropic shape
        d = ev.get("delta") or {}
        return bool(d.get("text") or d.get("thinking"))
    return bool(ev.get("content"))  # llama.cpp native /completion stream


def extract(ev: dict[str, Any]) -> dict[str, Any]:
    """Numbers found in one response object or streamed event."""
    out: dict[str, Any] = {}
    usage = ev.get("usage") or (ev.get("message") or {}).get("usage") or {}
    if "prompt_tokens" in usage:
        out["prompt_tokens"] = usage.get("prompt_tokens")
        out["completion_tokens"] = usage.get("completion_tokens")
        out["cached_tokens"] = (usage.get("prompt_tokens_details") or {}).get("cached_tokens")
        out["reasoning_tokens"] = (usage.get("completion_tokens_details") or {}).get("reasoning_tokens")

        prompt_s, completion_s = usage.get("prompt_time"), usage.get("completion_time")
        if isinstance(prompt_s, (int, float)) and isinstance(completion_s, (int, float)):
            cached = out["cached_tokens"] or 0
            out["usage_timings"] = {"prompt_n": (usage.get("prompt_tokens") or 0) - cached,
                                    "prompt_ms": prompt_s * 1000, "predicted_n": usage.get("completion_tokens"),
                                    "predicted_ms": completion_s * 1000, "cache_n": cached}
    elif "input_tokens" in usage or "output_tokens" in usage:
        if "input_tokens" in usage:
            out["prompt_tokens"] = usage["input_tokens"]
        if "output_tokens" in usage:
            out["completion_tokens"] = usage["output_tokens"]
    t = ev.get("timings")
    if isinstance(t, dict):
        out["timings"] = t
    for choice in ev.get("choices") or []:
        if choice.get("finish_reason"):
            out["finish_reason"] = choice["finish_reason"]
    if ev.get("stop_reason"):
        out["finish_reason"] = ev["stop_reason"]
    if (ev.get("delta") or {}).get("stop_reason"):
        out["finish_reason"] = ev["delta"]["stop_reason"]
    return {k: v for k, v in out.items() if v is not None}


@dataclass
class RequestRecord:
    ts: float
    method: str
    path: str
    status: int = 0
    stream: bool = False
    client: str | None = None
    model_req: str | None = None
    prompt_tokens: int | None = None
    cached_tokens: int | None = None
    prefill_tokens: int | None = None
    completion_tokens: int | None = None
    reasoning_tokens: int | None = None
    engine_prefill_ms: float | None = None
    engine_gen_ms: float | None = None
    ttft_ms: float | None = None
    total_ms: float | None = None
    finish_reason: str | None = None
    tokens_source: str | None = None
    error_short: str | None = None

    def as_row(self, run_id: int) -> dict[str, Any]:
        row = asdict(self)
        row["stream"] = int(self.stream)
        row["run_id"] = run_id
        return row


def request_meta(body: bytes) -> tuple[str | None, bool]:
    """(model, stream) from a JSON request body, if it is one."""
    try:
        obj = json.loads(body) if body else None
    except (ValueError, UnicodeDecodeError):
        return None, False
    if not isinstance(obj, dict):
        return None, False
    model = obj.get("model")
    return (str(model)[:200] if model is not None else None), bool(obj.get("stream"))


class Tap:
    def __init__(self, method: str, path: str, t_start: float, client: str | None, request_body: bytes) -> None:
        model, _ = request_meta(request_body)
        self.rec = RequestRecord(ts=t_start, method=method, path=path, client=(client or "")[:120] or None,
                                 model_req=model)
        self.t_start = t_start
        self._sse: SSEParser | None = None
        self._body = bytearray()
        self._found: dict[str, Any] = {}
        self._chunks = 0

    def start(self, status: int, content_type: str | None) -> None:
        self.rec.status = status
        if content_type and "text/event-stream" in content_type:
            self.rec.stream = True
            self._sse = SSEParser()

    def feed(self, chunk: bytes, t: float) -> None:
        if self._sse is None:
            if len(self._body) + len(chunk) <= MAX_BODY:
                self._body.extend(chunk)
            return
        for data in self._sse.feed(chunk):
            if data == "[DONE]":
                continue
            try:
                ev = json.loads(data)
            except ValueError:
                continue
            if not isinstance(ev, dict):
                continue
            if _content_delta(ev):
                self._chunks += 1
                if self.rec.ttft_ms is None:
                    self.rec.ttft_ms = (t - self.t_start) * 1000
            self._found.update(extract(ev))

    def finish(self, t_end: float, status: int | None = None) -> RequestRecord:
        rec = self.rec
        if status is not None:
            rec.status = status
        rec.total_ms = (t_end - self.t_start) * 1000
        if self._sse is None and self._body:
            try:
                obj = json.loads(bytes(self._body))
            except ValueError:
                obj = None
            if isinstance(obj, dict):
                self._found.update(extract(obj))
                if rec.status >= 400:
                    err = obj.get("error")
                    msg = err.get("message") if isinstance(err, dict) else (err or obj.get("message"))
                    rec.error_short = str(msg)[:200] if msg else None
        f = self._found
        timings = f.get("timings") or f.get("usage_timings") or {}
        prompt_n, cache_n = timings.get("prompt_n"), timings.get("cache_n")
        rec.prompt_tokens = f.get("prompt_tokens")
        if rec.prompt_tokens is None and prompt_n is not None:
            rec.prompt_tokens = prompt_n + (cache_n or 0)
        rec.cached_tokens = f.get("cached_tokens", cache_n)
        if prompt_n is not None:
            rec.prefill_tokens = prompt_n
        elif rec.prompt_tokens is not None:
            rec.prefill_tokens = rec.prompt_tokens - (rec.cached_tokens or 0)
        rec.completion_tokens = f.get("completion_tokens", timings.get("predicted_n"))
        rec.reasoning_tokens = f.get("reasoning_tokens")
        rec.engine_prefill_ms = timings.get("prompt_ms")
        rec.engine_gen_ms = timings.get("predicted_ms")
        rec.finish_reason = f.get("finish_reason")
        if rec.completion_tokens is not None:
            rec.tokens_source = "engine"
        elif rec.stream and self._chunks:
            rec.completion_tokens = self._chunks
            rec.tokens_source = "chunks"
        return rec
