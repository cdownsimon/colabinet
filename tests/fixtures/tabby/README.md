# Real TabbyAPI fixtures

Captured through colabinet's metering proxy from a real L4 session on 2026-10-02.
These are deliberately requested diagnostic responses, not application chat-history storage.

- TabbyAPI: `be74bf0a00bcb3a518e6feb7606f150c189be637`.
- ExLlamaV3: `1.5.2`, cu132/torch2.11 wheel; live torch remained `2.11.0+cu130`.
- Model: `turboderp/Huihui-Qwen3.8-27B-abliterated-exl3`, revision `4.00bpw`,
  commit `718be252b757f8b7bda082d3cca746b8de314fd3`; ctx 65536, FP16, one slot.
- Prompt: `Say hello in five words.`
- `stream.sse`: stream enabled, `reasoning_effort=low`, max 200 tokens. The client omitted
  `stream_options`; the proxy injected `include_usage`. The final event contains real usage/timings.
- `nonstream.json`: stream disabled, `reasoning_effort=high`, max 400 tokens.
- Both requests explicitly enabled thinking and passed effort through `chat_template_kwargs`.
- User agent: `paid-exllamav3-check/1` (distinct from Playground traffic).

Both responses were HTTP 200 with finish reason `stop`. Line endings/final newlines are normalized;
response content is otherwise unchanged. No credentials or runtime proxy tokens are included.

## Vision acceptance (2026-10-03)

`vision-20261003/` contains two synthetic PNGs, their actual nonstreamed responses,
image/text SSE captures and `summary.json` from a real L4 session. It used the same
engine versions, model repo/revision, 65536-token FP16 cache and one slot, with
`vision=true` and `vision_offload=true`. Requests went through the real colabinet proxy.

Both images received the identical prompt recorded in `summary.json`, without their
colors or shapes supplied in text. The model correctly returned red circle / blue
square / green triangle for image A and yellow triangle / purple circle for image B.
Text before and after image requests, image streaming, an image-history follow-up and
two images in one message also passed. Both SSE captures ended with real usage and
`[DONE]`; the client omitted `stream_options` so the proxy injected `include_usage`.

The setup event and `/props` both reported `modalities={"vision": true}`;
`/v1/model` reported `parameters.use_vision=true`. All seven completions returned
HTTP 200 and were metered successfully.
