import json
import random

from colabinet.stats import SSEParser, Tap, extract

LLAMA_FINAL = {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
               "timings": {"prompt_n": 10, "prompt_ms": 50.0, "cache_n": 2, "predicted_n": 3, "predicted_ms": 300.0}}


def sse(*events) -> bytes:
    return b"".join(f"data: {json.dumps(e)}\n\n".encode() for e in events) + b"data: [DONE]\n\n"


def test_sse_parser_survives_any_split():
    stream = sse({"a": 1}, {"b": 2}).replace(b"\n", b"\r\n")
    for _ in range(50):
        p, got, i = SSEParser(), [], 0
        while i < len(stream):
            j = i + random.randint(1, 7)
            got += p.feed(stream[i:j])
            i = j
        assert got == ['{"a": 1}', '{"b": 2}', "[DONE]"]


def test_extract_shapes():
    assert extract({"usage": {"prompt_tokens": 5, "completion_tokens": 7}}) == \
        {"prompt_tokens": 5, "completion_tokens": 7}
    assert extract({"type": "message_delta", "usage": {"output_tokens": 9}, "delta": {"stop_reason": "end_turn"}}) \
        == {"completion_tokens": 9, "finish_reason": "end_turn"}
    assert extract(LLAMA_FINAL)["timings"]["predicted_n"] == 3


def test_streamed_llamacpp_record():
    tap = Tap("POST", "/v1/chat/completions", 100.0, "pi/0.99", b'{"model": "m", "stream": true}')
    tap.start(200, "text/event-stream")
    tap.feed(sse({"choices": [{"delta": {"role": "assistant"}}]})[:-14], 100.2)  # role chunk: not a token
    tap.feed(sse({"choices": [{"delta": {"content": "Hi"}}]})[:-14], 100.5)
    tap.feed(sse({"choices": [{"delta": {"reasoning_content": "hm"}}]}, LLAMA_FINAL), 101.0)
    r = tap.finish(101.2)
    assert r.stream and r.status == 200 and r.model_req == "m" and r.client == "pi/0.99"
    assert r.ttft_ms == 500.0 and round(r.total_ms) == 1200
    assert (r.prompt_tokens, r.cached_tokens, r.prefill_tokens, r.completion_tokens) == (12, 2, 10, 3)
    assert (r.engine_prefill_ms, r.engine_gen_ms, r.finish_reason, r.tokens_source) == (50.0, 300.0, "stop", "engine")


def test_stream_without_usage_falls_back_to_chunk_count():
    tap = Tap("POST", "/v1/chat/completions", 0.0, None, b"{}")
    tap.start(200, "text/event-stream; charset=utf-8")
    tap.feed(sse(*({"choices": [{"delta": {"content": "x"}}]} for _ in range(4))), 1.0)
    r = tap.finish(2.0)
    assert (r.completion_tokens, r.tokens_source, r.prompt_tokens) == (4, "chunks", None)


def test_non_stream_json_and_error():
    body = {"choices": [{"message": {"content": "x"}, "finish_reason": "length"}],
            "usage": {"prompt_tokens": 30, "completion_tokens": 8, "prompt_tokens_details": {"cached_tokens": 10}}}
    tap = Tap("POST", "/v1/chat/completions", 0.0, None, b'{"model":"m"}')
    tap.start(200, "application/json")
    data = json.dumps(body).encode()
    tap.feed(data[:10], 0.5)
    tap.feed(data[10:], 0.6)
    r = tap.finish(1.0)
    assert (r.prompt_tokens, r.cached_tokens, r.prefill_tokens, r.completion_tokens, r.finish_reason) == \
        (30, 10, 20, 8, "length")
    assert r.ttft_ms is None
    err = Tap("POST", "/v1/chat/completions", 0.0, None, b"")
    err.start(500, "application/json")
    err.feed(b'{"error": {"message": "' + b"x" * 500 + b'"}}', 0.1)
    e = err.finish(0.2)
    assert e.status == 500 and len(e.error_short) == 200


def test_tabby_usage_times_become_engine_timings():
    usage = {"prompt_tokens": 30, "completion_tokens": 20, "prompt_tokens_details": {"cached_tokens": 10},
             "prompt_time": 0.5, "completion_time": 2.0, "prompt_tokens_per_sec": "Indeterminate"}
    tap = Tap("POST", "/v1/chat/completions", 0.0, None, b"{}")
    tap.start(200, "text/event-stream")
    tap.feed(sse({"choices": [{"delta": {"content": "x"}}]}, {"choices": [{"delta": {}, "finish_reason": "stop"}],
                                                            "usage": usage}), 1.0)
    r = tap.finish(3.0)
    assert (r.prompt_tokens, r.cached_tokens, r.prefill_tokens, r.completion_tokens) == (30, 10, 20, 20)
    assert (r.engine_prefill_ms, r.engine_gen_ms, r.tokens_source) == (500.0, 2000.0, "engine")
