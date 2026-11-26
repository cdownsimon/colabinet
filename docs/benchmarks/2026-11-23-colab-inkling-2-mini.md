# Inkling 2 Mini on Colab

The November 23, 2026 run used Strata v0.2.19 (`c6f6427`) and `strata-inkling` 0.1.65.
Each native pack used 8K context, FP16 KV, integrated vision, embedded DSpark and one request slot.
The [v0.2.20 DSpark comparison](2026-11-23-colab-inkling-dspark.md) measures drafting off and on.


| Colab class | Pack | Generation (tok/s) | Warm samples | Prefill (tok/s) | VRAM used / total at readiness (MiB) | Image check |
|---|---|---:|---:|---:|---:|---|
| T4 | SQ-10 | 16.9 | 3 | 395 | 12,551 / 15,360 | Pass (215.6 s) |
| T4 | SQ-15 | 5.9 | 3 | 384 | 14,623 / 15,360 | Pass (243.7 s) |
| T4-HIGHMEM | SQ-15 | 8.1 | 3 | 402 | 14,623 / 15,360 | Pass (71.4 s) |
| L4 | SQ-15 | 32.2 | 2 | 926 | 16,340 / 23,034 | Incomplete |
| L4 | SQ-20 | 31.3 | 3 | 788 | 21,096 / 23,034 | Pass (41.2 s) |
| A100 | SQ-15 | 58.8 | 2 | 2,518 | 16,574 / 40,960 | Pass (39.6 s) |
| A100 | SQ-20 | 74.0 | 2 | 2,519 | 21,330 / 40,960 | Pass (39.1 s) |
| A100-80 | SQ-20 | 78.5 | 2 | 2,552 | 21,330 / 81,920 | Pass (38.6 s) |
| G4 | SQ-20 | — | — | — | — | — |

## Method

- Generation counts engine output tokens, including reasoning, and takes the median of warm streamed requests
  that returned visible text. Runs use the same short writing prompt at temperature 1.0. Prefill is the median
  of completed engine timings from two distinct long prompts of about 4,800 tokens with fewer than 125 cached tokens.
- Early requests reached the output cap during model reasoning. The follow-up raised the cap to 768 tokens on
  short text, 192 on long-prompt processing, and 384 on vision. The generation median uses requests with visible
  replies. Completed input timings remain valid when the output cap is reached.
- The vision check expects a yellow triangle and a purple circle in the reply. Image time is end-to-end and
  uses the output cap of its pass. The L4 SQ-15 image reply reached its cap before naming both objects. All
  loaded rows passed health, model-list and metrics checks for the requested pack, context, vision, KV and DSpark.
- VRAM is the device reading immediately after readiness. Maximum context and peak inference memory were outside this benchmark.
- G4 returned HTTP 503 during allocation, leaving its benchmark pending.

The Qwen3.6 and Qwen3.8 Flash-Next IQ2_XS presets also passed streamed text and vision on L4 with Strata 0.1.65.

Colab’s balance decreased by 4.52 CU. The CLI confirmed 0 active sessions and 0.00 CU/h after teardown.

[Strata v0.2.19](https://github.com/Niko1221/Strata/releases/tag/v0.2.19) · [Model files](https://huggingface.co/strata-engine/Inkling-2-Mini-Strata)
