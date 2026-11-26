# Inkling 2 Mini DSpark benchmark on Colab

Runs on November 23 and 24, 2026 used Strata v0.2.20, Inkling engine 0.1.66 and official Strata model packs.
Each GPU used one active sequence, 8,192 tokens of context, FP16 KV, temperature 1.0 and the same streamed HTTP requests in both arms.
Each arm restarted the server, made one warm-up request, then measured four short prompts and two 4,385-token prompts.
Generation and visible first-token latency are medians over the short prompts. Prefill is the median over the two long prompts.
Short requests were capped at 384 output tokens. Both DSpark settings use the same model build.

| GPU | Pack | Generation off / on (tok/s) | Prefill off / on (tok/s) | First visible token off / on (ms) | Enabled draft acceptance | Status |
|---|---|---:|---:|---:|---:|---|
| T4 | SQ-10 | 19.9 / 22.9 | 443 / 444 | 461 / 508 | 35.3% | Measured |
| T4-HIGHMEM | SQ-10 | 20.8 / 21.9 | 412 / 414 | 431 / 525 | 35.3% | Measured |
| L4 | SQ-15 | 34.1 / 38.0 | 935 / 909 | 175 / 184 | 30.3% | Measured |
| L4 | SQ-20 | — / 39.0 | — / 816 | — / 193 | 35.1% | DSpark on measured |
| A100 | SQ-20 | — / — | — / — | — / — | — | Allocation unavailable (Nov. 26) |
| A100-80 | SQ-20 | — / — | — / — | — / — | — | Allocation unavailable (Nov. 26) |
| G4 | SQ-20 | — / — | — / — | — / — | — | Allocation unavailable (Nov. 26) |

The Nov. 24 L4 SQ-20 run used DSpark on and 21,144 MiB of VRAM at readiness. Its request rates and aggregate counters were recovered from the saved benchmark output.

On Nov. 26 (Hong Kong time), one CLI allocation request each for A100, A100-80 and G4 returned HTTP 503. The attempts used 0 CU and left zero active assignments. An earlier A100 assignment could not be reached by the benchmark.

[Per-request metrics](2026-11-23-colab-inkling-dspark.json) · [Strata release](https://github.com/Niko1221/Strata/releases/tag/v0.2.20)
