# Qwen3.8 presets on every Colab class — 2026-10-13

Every preset in `presets.toml` was run on all seven Colab runtime classes through colabinet's own console API,
to check which classes New session should grey out. The verdicts it now shows match every outcome below.
[Per-request metrics and provenance](2026-10-13-qwen38-preset-matrix.json) contain metrics only, with no
prompt or response bodies. The speeds below are recorded under `[preset.measured]` in `presets.toml`, which the
New session GPU choices, its review and the Models page display.

## Method

- **Sessions:** one VM per class, created with the first preset and switched with Reconfigure for each next one,
  so engines, downloads and builds were reused. Each preset used its per-GPU context and KV defaults; classes
  without their own defaults use the generic ones (32K, f16 / FP16 / int8). Strata ran with upstream's automatic
  low-RAM selection, except the explicit `--low-ram on` IQ3_XXS step on L4.
- **Requests:** three streamed chat requests per model at temperature 0 with thinking disabled: a 26-token
  prompt with 160 tokens out, a 4,223-token prompt with up to 32 out, and an image with a one-word colour
  question. If the first took over four minutes, the other two were skipped. CPU requests were capped at
  4 tokens out.
- **Measures:** generation and prefill speeds are engine-reported; time to first token is measured at
  colabinet's metering proxy. Each figure is **one cold pass on a freshly loaded engine**, with no warm-up
  excluded, so it understates warmed speeds, Strata's in particular. The exception is footnote 3's warmed reruns.
  The original 27B EXL3's prefill came from the VM's first ExLlamaV3 long prompt and includes kernel warm-up; the
  abliterated build, loaded next on the same VM, shows the warmed figure.
- **Prefetch:** a low-priority worker on each VM staged later models while the current one was tested. On L4 it
  visibly slowed two measurements (27B abliterated Q4_K 6.6 tok/s, Strata IQ2_XS 2.2 tok/s); both were rerun
  without it and the reruns are reported. Early measurements on other classes may include some of its contention.
- **Engines:** llama.cpp `b11273`, ExLlamaV3 `1.5.2` with TabbyAPI `be74bf0`, and Strata `d6708a4`.

| Class | GPU | VRAM | Host RAM | Rate |
|---|---|---:|---:|---:|
| CPU | — | — | 13.6 GB | 0.08 CU/h |
| T4 | Tesla T4 | 15,360 MiB | 13.6 GB | 1.07 CU/h |
| T4-HIGHMEM | Tesla T4 | 15,360 MiB | 54.8 GB | 1.27 CU/h |
| L4 | NVIDIA L4 | 23,034 MiB | 56.9 GB | 1.54 CU/h |
| A100 | A100-SXM4-40GB | 40,960 MiB | 89.6 GB | 5.30 CU/h |
| A100-80 | A100-SXM4-80GB | 81,920 MiB | 179.4 GB | 6.77 CU/h |
| G4 | RTX PRO 6000 Blackwell Server Edition | 97,887 MiB | 189.9 GB | 8.90 CU/h |

All GPU classes reported driver 580.82.07.

## Results

Generation speed on the 26-token prompt / prefill speed on the 4,223-token prompt, in tokens per second.
**✗** marks a configuration that does not run; the reasons follow the table. Every configuration that reached
the image request answered it correctly.

| Preset | CPU | T4 | T4-HIGHMEM | L4 | A100 | A100-80 | G4 |
|---|---|---|---|---|---|---|---|
| 27B Q4_K_M (original) | 0.024 / –¹ | 1.2 / –² | 3.9 / 200³ | 14.6 / 715 | 47.2 / 1,219³ | 49.4 / 1,230³ | 71.2 / 3,652 |
| 27B Q4_K (abliterated) | ✗ OOM | 1.0 / 140 | 3.0 / 167 | 13.1 / 660 | 43.0 / 1,194 | 45.2 / 1,214 | 67.8 / 3,527 |
| 27B EXL3 4.0 bpw (original) | ✗ GPU | ✗ VRAM | ✗ VRAM | 16.3 / 525 | 47.5 / 795 | 46.5 / 727 | 65.3 / 2,346 |
| 27B EXL3 4.0 bpw (abliterated) | ✗ GPU | ✗ VRAM | ✗ VRAM | 16.2 / 736 | 46.9 / 2,639 | 48.9 / 2,673 | 67.8 / 4,854 |
| Flash-Next GGUF (abliterated) | ✗ OOM | 0.24 / –² | 1.8 / 9 | 3.0 / 11 | 8.0 / 90 | 38.7 / 886 | 50.8 / 1,366 |
| Strata IQ2_XS | ✗ GPU | ✗ RAM | 14.4 / 346 | 28.4 / 1,075 | 42.2 / 1,625 | 66.6 / 1,924 | 87.9 / 3,746 |
| Strata IQ3_XXS | ✗ GPU | ✗ RAM | 3.4 / 57 | 5.6 / 82⁴ | 37.9 / 1,786 | 41.9 / 1,834 | 48.5 / 3,150 |
| Strata IQ3_S | ✗ GPU | ✗ RAM | 2.8 / 35 | 4.3 / 53 | 36.7 / 1,602 | 43.9 / 1,810 | 53.4 / 4,018 |

1. 4 tokens out; 101 s to the first token.
2. The first request took over four minutes, so the 4K and image requests were skipped.
3. Rerun on a fresh VM after an unmeasured short and 4K-token warm-up. The first pass, the VM's first llama.cpp
   requests, measured 1.2, 12.0 and 9.3 tok/s, slowed by CUDA JIT compilation (finding 6) and, on T4-HIGHMEM, by
   the prefetch worker's engine builds. The measured 4K request then reused the warm-up's cached prompt, so the
   prefill shown is the warm-up's own uncached 4K request as the console recorded it.
4. With `--low-ram on`. In auto mode upstream refused it: "Strata IQ3_XXS requires about 60 GiB host RAM".

Context and KV cache as run:

- **27B GGUFs:** 256K q4_0 on L4, 256K q8_0 on A100, 256K f16 on A100-80 and G4, 32K f16 on T4 classes.
- **27B EXL3s:** 64K FP16 on L4, 256K FP16 on A100s and G4, 32K FP16 on T4 classes.
- **Flash-Next GGUF:** 8K f16 on A100s and G4, and the generic 32K f16 on T4 classes and L4.
- **Strata:** 128K int8 on A100s, G4 and L4 (IQ2_XS); 32K int8 on L4 (IQ3) and T4-HIGHMEM.

### Configurations that do not run

- **✗ GPU:** ExLlamaV3 and Strata on a CPU runtime. colabinet refuses them before any compute is spent.
- **✗ VRAM:** both 27B EXL3 presets on T4 and T4-HIGHMEM. TabbyAPI exits with
  `RuntimeError: Insufficient VRAM in split for model and cache`, including with a Q8 cache at 32K on a T4.
- **✗ RAM:** every Strata quant on a T4: "cannot fit 12.7 GiB host RAM and 15.0 GiB VRAM under upstream's
  low-RAM policy".
- **✗ OOM:** on the CPU runtime, llama-server was killed (signal 9) loading the 27B abliterated Q4_K
  and Flash-Next GGUF. The slightly smaller 27B Q4_K_M loaded but generated 0.024 tok/s.

## Findings

1. **The greying matches the hardware.** New session blocks exactly the ✗ cells, plus llama.cpp configurations
   that need more than VRAM plus host RAM. Those still start but page weights from disk. Flash-Next GGUF did
   that at 0.24 tok/s on a T4; on a T4-HIGHMEM and an L4 it generated 1.8 and 3.0 tok/s, but its 4,223-token
   prompt waited 485 s and 377 s for a first token.
2. **ExLlamaV3 does not fit a 16 GB T4.** The 27B 4.0 bpw weights take 13.2 GB on GPU, and ExLlamaV3 reserves
   working memory beyond weights and cache; colabinet's fit now adds 2 GB for it.
3. **Strata follows upstream's host-RAM policy exactly.** Reading host RAM as Strata does (GiB), the policy
   predicted every refusal and every low-RAM selection: normal mode for IQ2_XS on T4-HIGHMEM and the
   80 GB+ classes, and low-RAM mode for IQ3 on T4-HIGHMEM and L4.
4. **L4 needs `--low-ram on` for IQ3_XXS.** At 56.9 GB the auto mode chooses normal mode and refuses the quant;
   forced low-RAM mode runs it. The L4 IQ3 presets now set it.
5. **The Flash-Next GGUF path had moved upstream.** The shards are now under `unsloth-UD-Q4_K_XL/`; the old
   `UD-Q4_K_XL/` path no longer resolves. The preset uses the new path.
6. **First requests on T4 and A100 wait for CUDA JIT.** The b11273 llama.cpp binary contains GPU code for
   sm_86, sm_89 and sm_120a only, with PTX for older architectures. The first request after the first model
   load waited 131–177 s for its first token on T4, T4-HIGHMEM, A100 and A100-80, versus 1 s on L4 and G4. The
   A100 VM's CUDA compute cache held 158.5 MB afterwards, and later llama.cpp models on the A100s answered within
   1–2 s. The compile also slowed generation: the 27B Q4_K_M measured 12.0 tok/s on its first A100 pass and
   47.2 tok/s after a warm-up. T4-class first tokens stay slow for another reason: the 27B GGUFs offload layers
   to the CPU there.
7. **Colab reports more host RAM than the old references:** 13.6 / 54.8 / 56.9 / 179.4 / 189.9 GB for
   CPU and T4 / T4-HIGHMEM / L4 / A100-80 / G4. The references in `config.py` and `colab-facts.md` now use them.

## Cost

These are colabinet's per-session estimates, from the create request to the confirmed stop at each class's
rate. The account's balance fell by less than their total over the same window, so they are upper bounds.

| Class | CU | Session time |
|---|---:|---:|
| CPU | 0.17 | 127 min |
| T4 | 1.13 | 63 min |
| T4-HIGHMEM | 3.08 | 108 + 28 + 9 min (IQ3_S after a disk-space failure; warmed Q4_K_M rerun) |
| L4 | 3.49 | 61 + 75 min (second VM for the clean reruns and remaining steps) |
| A100 | 8.01 | 84 + 7 min (warmed Q4_K_M rerun) |
| A100-80 | 9.14 | 74 + 7 min (warmed Q4_K_M rerun) |
| G4 | 11.62 | 78 min |
| **Total** | **36.62** | |

Two creates were refused by Colab at no cost: T4 once with "Service Unavailable", and L4 once at a capacity
limit. Both succeeded on retry.
