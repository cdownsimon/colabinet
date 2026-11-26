# Engines, presets and GPU suitability

colabinet runs one of three inference engines on the Colab VM, each behind the same OpenAI-compatible endpoint:

| engine | model format | notes |
|---|---|---|
| [llama.cpp](https://github.com/ggml-org/llama.cpp) `llama-server` | GGUF | prebuilt CUDA 12.8 binary; offloads layers to the CPU when VRAM runs out; vision through an `mmproj` projector |
| [ExLlamaV3](https://github.com/turboderp-org/exllamav3) via [TabbyAPI](https://github.com/theroyallab/tabbyAPI) | EXL3 | GPU only; fast, but weights and cache must fit in VRAM |
| [Strata](https://github.com/Niko1221/Strata) | Qwen GGUF, native Inkling `.strata` | places experts in GPU and host memory. Colabinet serves one sequence at a time. |

Engine versions are pinned in the `[engines]` table of `presets.toml`.

## Presets

`presets.toml` holds the models offered in the console. Edit it and select **Reload presets** on the Models page;
a malformed file is reported there and never stops the console.

| preset | engine | weights | runs on |
|---|---|---:|---|
| `qwen38-27b-q4km` | llama.cpp | 16.5 GB | T4 classes (CPU offload), L4, A100, A100-80, G4 |
| `qwen38-27b-abl-q4k` | llama.cpp | 16.8 GB | T4 classes (CPU offload), L4, A100, A100-80, G4 |
| `qwen38-27b-exl3-4bpw` | ExLlamaV3 | 13.2 GB on GPU | L4, A100, A100-80, G4 |
| `qwen38-27b-abl-exl3-4bpw` | ExLlamaV3 | 13.2 GB on GPU | L4, A100, A100-80, G4 |
| `qwen38-flash-next-abl-q4k-xl` | llama.cpp | 111.3 GB | A100, A100-80, G4 (CPU offload) |
| `qwen38-flash-next-strata-iq2-xs` | Strata | 68.0 GB download | T4-HIGHMEM, L4, A100, A100-80, G4 |
| `qwen38-flash-next-strata-iq3-xxs` | Strata | 75.8 GB download | T4-HIGHMEM, L4, A100, A100-80, G4 |
| `qwen38-flash-next-strata-iq3-s` | Strata | 83.6 GB download | T4-HIGHMEM, L4, A100, A100-80, G4 |
| `qwen36-35b-a3b-strata-ud-q4-k-xl` | Strata | 23.75 GB with vision | T4-HIGHMEM, L4, A100, A100-80 |
| `inkling-2-mini-sq-10` | Strata | 10.47 GB + vision | T4 and larger |
| `inkling-2-mini-sq-15` | Strata | 14.35 GB + vision | T4 and larger |
| `inkling-2-mini-sq-20` | Strata | 19.35 GB + vision | L4 and larger |

Qwen3.8 presets with `abl` in their ID use Huihui abliterated weights. The Models page shows each model's licence.

### `presets.toml` reference

```toml
[[preset]]
id = "my-model"                 # unique; used by `ctl create --preset`
engine = "llamacpp"             # llamacpp | exllamav3 | strata
label = "My model Q4_K_M"
repo = "org/My-Model-GGUF"      # Hugging Face repository
file = "My-Model-Q4_K_M.gguf"   # GGUF/Strata: the (first shard's) file; EXL3 uses `revision` instead
mmproj = "mmproj-F16.gguf"      # optional llama.cpp vision projector
template = "qwen38_effort_map.jinja"  # optional chat template from src/colabinet/chat_templates/
alias = "org/my-model"          # model name the endpoint serves
weights_gb = 16.5               # used by the fit estimate
kv_bytes_per_token_f16 = 65536  # optional; makes the KV cache estimate exact
licence = "Apache-2.0"
parallel = 1                    # parallel request slots
vision = false                  # EXL3/Strata bundled vision
vision_offload = false          # EXL3: keep vision weights in host RAM

[preset.gpu.L4]                 # per-class defaults; T4-HIGHMEM falls back to T4, A100-80 to A100
ctx = 262144
kv = "q4_0"
mmproj_gpu = true
extra_args = ["--low-ram", "on"]  # engine args for this class

[preset.measured]               # speeds measured on real runtimes, shown in the console
L4 = { gen_tps = 14.6, prefill_tps = 715, date = "2026-10-13", note = "optional caveat" }

[preset.evidence.L4]            # Strata only: recorded runs that replace the fit estimate
status = "tested"               # tested | reported | build_only
message = "What ran, with which settings."
performance = "Optional benchmark summary."
```

Custom models can also be launched from **New session** without a preset: choose the engine, repository and file
(or EXL3 revision). Models shows a TOML snippet for recently used custom models.

## GPU suitability

New session assesses every offered GPU class with the form's model, context, KV cache and engine args, and greys
out the classes that cannot run it. Host RAM is the latest reading from a recorded session, else the reference
value in `config.py`.

- **ExLlamaV3 and Strata on a CPU runtime.** Creating one is refused before anything is billed.
- **ExLlamaV3 past VRAM.** Weights, cache and about 2 GB of working memory must fit; ExLlamaV3 does not offload
  dense layers. Both 27B 4.0 bpw presets fail "Insufficient VRAM in split for model and cache" on a 16 GB T4,
  even with a Q8 cache.
- **llama.cpp past VRAM plus host RAM.** llama.cpp would page weights from disk: Flash-Next GGUF generated
  0.24 / 1.8 / 3.0 tok/s on T4 / T4-HIGHMEM / L4 that way and was killed for memory on a CPU runtime.
- **Strata where upstream's host-RAM policy refuses the quant** with the form's `--low-ram` setting, such as every
  quant on a T4's 13.6 GB. Setup still applies the same policy on the VM before downloading.

llama.cpp offload that fits in host RAM stays selectable with a warning. A class chosen explicitly, such as from
the command menu, stays selected and is flagged with a suggested alternative. Without one, the form starts on L4
unless it cannot run the model. The Models page shows the same verdicts, plus each preset's measured speeds. The
[2026-10-13 preset test matrix](benchmarks/2026-10-13-qwen38-preset-matrix.md) ran every preset on every class.

## llama.cpp

- Split GGUF downloads fetch every shard, and size checks count them all. Model and projector caches are
  separated by repository, so publishers' identical projector filenames never collide.
- **Flash-Next GGUF** (111.3 GB in four shards) needs CPU offload on every class and sufficient host RAM. Its
  default context is 8K; the fit estimate leaves its hybrid KV size unknown. Its vision projector runs on the CPU.
- **First requests on T4 and A100:** the pinned b11273 binary ships GPU code for sm_86, sm_89 and sm_120a only,
  so CUDA JIT-compiles the PTX for sm_75/sm_80 on first use. The first request on a new VM waited 131–177 s for
  its first token, and the 27B Q4_K_M generated 12.0 tok/s on its first A100 pass against 47.2 once warmed.
  The VM keeps the compiled cache for later models. L4 and G4 run the shipped code directly.

## ExLlamaV3

- The Qwen3.8-27B EXL3 presets set `vision = true` and `vision_offload = true`: vision weights stay in host RAM
  and stream to the GPU for image encoding. The fit estimate does not include image-encoding workspace.
- Setup reads the loaded model's `/props` modalities and fails if requested vision cannot be confirmed.
- Extra engine args `model.vision=false` disable vision; `model.vision_offload=false` keeps vision weights on the
  GPU. Custom EXL3 models default to vision disabled; their presets or extra args can enable it.
- Existing runs keep their saved settings after a preset reload; reconfigure a run to apply new ones.

## Strata

Strata is pinned to [v0.2.19](https://github.com/Niko1221/Strata/releases/tag/v0.2.19) at `c6f6427`.
It needs an NVIDIA GPU, an AVX2 host CPU and driver 580+, and serves one sequence at a time.

### Qwen3.8 Flash-Next

The three Flash-Next presets use the original
[ISTA-DASLab GSQ-RCO weights](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF).

- **Host RAM.** Strata keeps experts in host RAM and caches a subset in VRAM, so download size is not required
  VRAM. Upstream's normal-mode guidance is about 48/60/62 GiB of host RAM for IQ2_XS/IQ3_XXS/IQ3_S; smaller
  allocations use its low-RAM policy where it fits. The IQ3 presets pass `--low-ram on` on L4. IQ3_XXS
  also uses that setting on T4-HIGHMEM to retain the memory-mapped mode used in the earlier benchmark.
- **Setup selector:** Strata extra args are native engine arguments, except `--low-ram auto|on|off`, which
  colabinet forwards to setup. `auto` is the default; `on` chooses upstream's memory-mapped expert policy while
  keeping its fit check; `off` requires resident-expert headroom. In the CLI, pass it as
  `--extra-arg=--low-ram --extra-arg=on`; in the console, as `--low-ram on`.
- **Drafting.** Use `--drafting off|dflash2|mtp|auto` in extra args. Colabinet defaults to `mtp`.
  `off` skips draft downloads, `dflash2` prepares the upstream stock DFlash2 pack, and `auto` follows
  upstream's hardware and model policy, which can select MTP. Off and DFlash2 skip the roughly 5 GB MTP download.
- **PLE table I/O:** direct I/O is the safe default. On allocations with ample host RAM, `--ple-io ram` warms
  the shared 28.8 GB lookup table after the model is ready, verifies residency and then publishes the endpoint;
  it is page cache, not locked memory. Avoid `--ple-io mmap` on memory-tight L4 allocations, where a long-prompt
  trial aborted.
- **Setup** prepares the engine, model and selected drafter concurrently. On compatible Linux x86-64 VMs,
  it verifies and caches the release's CUDA 13 engine and vision package (0.1.65). Unsupported GPUs or a
  package download or validation failure use a build from the pinned source. A failed setup branch
  cancels its siblings. Downloads and builds consume CU while the runtime is active.
- **Monitoring** checks the loaded model too, because upstream `/health` can stay HTTP 200 after the native
  engine dies.
- **Experimental refusal projection:** off by default. Enable it in New session or Reconfigure, with
  `--refusal-projection` in the CLI, or `refusal_projection: true` in JSON. The
  [upstream vector documentation](https://github.com/Niko1221/Strata/blob/c77572806996349b49995e4f98d41b5294d36a21/data/experimental-speed-projection/README.md)
  describes removing a refusal direction from layers 4–44. It reduces refusals and changes ordinary answers; it
  is not a separate checkpoint or a guarantee of unrestricted output. A request can set
  `"experimental_speed_projection": false` to disable it for that request.

Models and New session show **GPU evidence** for Strata instead of a full-weight VRAM estimate: recorded runs per
GPU class, from `[preset.evidence.<class>]`. Pairs without a record show **Not benchmarked**, not a known failure.
Evidence is keyed to the exact class. The October 2026 Strata measurements used engine 0.1.29.

```bash
uv run colabinet ctl create --name strata --gpu A100 \
  --preset qwen38-flash-next-strata-iq2-xs --idle 20 --cap 5
```

### Qwen3.6-35B-A3B

`qwen36-35b-a3b-strata-ud-q4-k-xl` uses the pinned
[Unsloth MTP GGUF and projector](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-MTP-GGUF/tree/5bc3e238d916f48a861bac2f8a1990a0e9b7e98d)
and defaults to 8K context, FP32 KV and CPU vision. With vision it downloads 23.75 GB. The VM checks for at least
32 GiB of host RAM and uses the upstream Qwen3.6 context fit check. Strata places experts automatically across VRAM
and host RAM and reads MTP from the GGUF. The Nov. 19 Colab L4 smoke test verified the default configuration,
streamed text with usage, and an image request through the CPU encoder.

### Inkling 2 Mini

The `inkling-2-mini-sq-10`, `inkling-2-mini-sq-15`, and `inkling-2-mini-sq-20` presets use the
[native Strata packs](https://huggingface.co/strata-engine/Inkling-2-Mini-Strata) and a separately verified
`strata-inkling` 0.1.65 CUDA engine. They launch with 8K context, FP16 KV, embedded DSpark and the 0.64 GB vision
pack. The VM checks expert placement against available VRAM and host RAM, and rejects configurations that would
spill experts to disk. The setup checks the loaded pack, context, vision capability, KV type and draft mode before
the session becomes ready.

## Vision requests

Send images as OpenAI `image_url` content parts to the session's `/v1/chat/completions` endpoint. The
Playground accepts text prompts only.
