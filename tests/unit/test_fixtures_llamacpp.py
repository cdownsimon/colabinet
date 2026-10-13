"""Stats extraction against real llama-server responses captured from a real L4 session (tests/fixtures/llamacpp/)."""

import json
from pathlib import Path

import pytest

from colabinet.stats import SSEParser, Tap

FIX = Path(__file__).resolve().parents[1] / "fixtures" / "llamacpp"
pytestmark = pytest.mark.skipif(not (FIX / "stream.sse").exists(), reason="captured from a real session")


def test_real_stream_has_engine_timings():
    raw = (FIX / "stream.sse").read_bytes()
    events = [json.loads(d) for d in SSEParser().feed(raw) if d != "[DONE]"]
    predicted = next(e["timings"]["predicted_n"] for e in reversed(events) if "timings" in e)
    tap = Tap("POST", "/v1/chat/completions", 0.0, "curl", b'{"stream": true}')
    tap.start(200, "text/event-stream")
    tap.feed(raw, 1.0)
    r = tap.finish(2.0)
    assert r.tokens_source == "engine" and r.completion_tokens == predicted
    assert r.prefill_tokens is not None and r.engine_gen_ms and r.engine_prefill_ms
    assert r.ttft_ms is not None


def test_real_non_stream_has_usage_and_timings():
    body = (FIX / "nonstream.json").read_bytes()
    tap = Tap("POST", "/v1/chat/completions", 0.0, "curl", b"{}")
    tap.start(200, "application/json")
    tap.feed(body, 1.0)
    r = tap.finish(1.0)
    assert r.prompt_tokens and r.completion_tokens and r.engine_gen_ms and r.finish_reason
