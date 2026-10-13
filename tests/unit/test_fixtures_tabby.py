"""Stats extraction against real TabbyAPI responses captured from a real L4 session (tests/fixtures/tabby/)."""

from pathlib import Path

import pytest

from colabinet.stats import Tap

FIX = Path(__file__).resolve().parents[1] / "fixtures" / "tabby"
pytestmark = pytest.mark.skipif(not (FIX / "stream.sse").exists(), reason="captured from a real session")


def test_real_tabby_stream_with_injected_usage():
    tap = Tap("POST", "/v1/chat/completions", 0.0, "curl", b'{"stream": true}')
    tap.start(200, "text/event-stream")
    tap.feed((FIX / "stream.sse").read_bytes(), 1.0)
    row = tap.finish(2.0)
    assert row.tokens_source == "engine" and row.completion_tokens and row.prompt_tokens
    assert row.engine_gen_ms and row.ttft_ms is not None


def test_real_tabby_non_stream():
    tap = Tap("POST", "/v1/chat/completions", 0.0, "curl", b"{}")
    tap.start(200, "application/json")
    tap.feed((FIX / "nonstream.json").read_bytes(), 1.0)
    row = tap.finish(1.0)
    assert row.prompt_tokens and row.completion_tokens and row.engine_gen_ms and row.finish_reason
