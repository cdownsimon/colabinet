# Colab preset performance matrix

The November 19, 2026 sweep accounted for 63 preset and runtime pairs. It measured 35 pairs across 6 Colab classes and recorded 19 fit blocks.
Table values are output generation / prompt processing in tokens per second.

| Preset | CPU | T4 | T4-HIGHMEM | L4 | A100 | A100-80 | G4 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Qwen3.8-27B abliterated Q4_K (+vision) | Fit blocked | 1.5 / 177 | 4.3 / 208 | 14.4 / 808 | 46.1 / 1,303 | 48.2 / 1,311 | Allocation unavailable |
| Qwen3.8-27B UD-Q4_K_M (+vision) | 0.028 / —¹ | 1.4 / 186 | 4.0 / 222 | 14.8 / 842 | 48.8 / 1,308 | 51.9 / 1,318 | Allocation unavailable |
| Qwen3.8-27B abliterated EXL3 4.0 bpw (+vision) | Fit blocked | Fit blocked | Fit blocked | 16.5 / 588 | 50.2 / 1,860 | 51.6 / 1,883 | Allocation unavailable |
| Qwen3.8-27B EXL3 4.0 bpw (+vision) | Fit blocked | Fit blocked | Fit blocked | 16.5 / 712 | 50.2 / 2,757 | 52.3 / 2,844 | Allocation unavailable |
| Qwen3.8-Flash-Next abliterated UD-Q4_K_XL (+vision, CPU offload) | Fit blocked | Fit blocked | Fit blocked | Fit blocked | 19.5 / 84 | 55.0 / 1,667 | Allocation unavailable |
| Qwen3.8-Flash-Next Strata IQ2_XS (+vision) | Fit blocked | Fit blocked | 28.1 / 483 | 47.6 / 1,277 | 79.1 / 2,683 | 88.8 / 2,698 | Allocation unavailable |
| Qwen3.8-Flash-Next Strata IQ3_XXS (+vision) | Fit blocked | Fit blocked | 19.0 / 74 | 33.1 / 339 | 76.1 / 2,745 | 71.9 / 2,941 | Allocation unavailable |
| Qwen3.8-Flash-Next Strata IQ3_S (+vision) | Fit blocked | Fit blocked | 11.2 / 55 | 24.0 / 69 | 70.0 / 2,082 | 69.0 / 2,731 | Allocation unavailable |
| Qwen3.6-35B-A3B Strata UD-Q4_K_XL (+vision) | Fit blocked | Fit blocked | 53.3 / 760 | 65.4 / 1,611 | 135.1 / 6,403 | 139.8 / 6,471 | Allocation unavailable |

¹ Bounded CPU measurement with up to four output tokens.

Colab returned Service Unavailable while allocating G4 on 23 requests.
The backend also rejected 2 G4 requests using the CLI's high-memory flag.
The CLI also recorded 1 connection failure during G4 allocation.
The table records the allocation failure for 9 pairs. Earlier measurements in the catalog carry their October date.

## Environment and billing

The run used llama.cpp `b12359`, ExLlamaV3 `1.6.7` with TabbyAPI `4728158`, and Strata `v0.2.15`.
Strata ran engine `0.1.60` on L4, T4-HIGHMEM.
Strata ran engine `0.1.61` on A100, A100-80.

| Class | GPU | VRAM MiB | RAM GB | Driver | Session estimate CU/h |
|---|---|---:|---:|---|---:|
| CPU | — | — | 13.6 | — | 0.08 |
| T4 | Tesla T4 | 15360 | 13.6 | 580.82.07 | 1.15 |
| T4-HIGHMEM | Tesla T4 | 15360 | 54.8 | 580.82.07 | 1.27 |
| T4-HIGHMEM | Tesla T4 | 15360 | 54.8 | 580.82.07 | 1.57 |
| L4 | NVIDIA L4 | 23034 | 56.9 | 580.82.07 | 1.54 |
| A100 | NVIDIA A100-SXM4-40GB | 40960 | 89.6 | 580.82.07 | 5.30 |
| A100-80 | NVIDIA A100-SXM4-80GB | 81920 | 179.4 | 580.82.07 | 6.77 |

The isolated Colabinet instance estimated 24.56 CU over all allocations.
The Colab balance decreased by 29.65 CU against the authorized 80 CU cap.
Colab reported 0 active assignments at 0.00 CU/h after teardown.

## Method

- The sweep used the preset's context, KV type, vision settings, and drafting configuration for the actual runtime class.
- The first streamed generation after readiness is recorded as cold. The table uses the median of three subsequent short text requests capped at 160 output tokens. For a cold generation below 2 tok/s, it uses one warm request capped at 64 tokens to bound runtime.
- Prompt processing uses uncached token counts and engine-reported timing on two distinct approximately 4K-token texts (one for the slow profiles). Request-level proxy and client time to first token are in the JSON.
- 34 of 34 image requests both completed and named the two expected colors and shapes. Each pair used the same image and question. The vision limit was raised from 32 to 96 output tokens after early replies were truncated. The counts use the last image request per pair.
- The sweep started serially and later ran up to three independent VMs with distinct GPU classes. Within a group, Colabinet switched presets on the same VM and reused installs. Each VM avoided background model downloads during its timed requests.
- Fit-blocked entries use Colabinet's fit rules and reference RAM. The bounded CPU control retried the original 27B Q4_K_M because October measured it despite disk paging.
- The October run used different engine versions and a mostly cold single pass. The table describes current Colab performance. The difference between tables also includes the changed workload and warm-up method.

[Per-request results and provenance](2026-11-19-colab-preset-matrix.json) · [October baseline](2026-10-13-qwen38-preset-matrix.md)
