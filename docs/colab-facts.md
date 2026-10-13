# Colab facts colabinet relies on

What colabinet's design assumes about Google Colab and `google-colab-cli`. Most of it was measured on a paid
Colab account between 2026-09-28 and 2026-10-13. Items marked *(third-party)* come from public sources and were
not measured here.

Re-check this list whenever Colab or the CLI changes. The last column says what breaks if a fact changes.

## CLI (google-colab-cli 0.7.4)

| fact | why colabinet cares |
|---|---|
| Commands: `new -s NAME [--gpu L4\|A100\|G4\|T4] [--high-mem]` (no `--gpu` = CPU), `upload`, `download` (single file, via the Contents API), `exec -f FILE --env K=V`, `ssh [--proxy-mode]`, `sessions`, `status`, `usage`, `stop -s NAME`, `drivemount` | the only control surface |
| Older CLI builds orphaned VMs after ~60 min (bug #147, fixed 2026-09-25) | require ≥ 0.7.4 (Settings page checks the version) |
| A long `colab exec` can drop its websocket (`RuntimeError: Connection was lost.`), after which the local CLI hangs | never run long work in the foreground: upload a script, start it detached (`bg.py`), poll it (`status.py`); every CLI call has a watchdog |
| `colab usage` shows the balance and the current hourly rate, and lags a few minutes after a stop | show the balance with its age; per-session cost is an estimate |
| After every session, `colab sessions` must be empty (anything listed is billing) | stops are confirmed against `colab sessions`; the reconcile loop looks for unmanaged sessions |
| Eligible GPUs depend on the account; the measured account offered G4, A100, L4 and T4 (no H100) | GPU choices in the UI; Settings narrows them |
| A100 is often "not available". The rate card notes it can fall back to L4 *(third-party)* | never auto-fallback; verify the real GPU with `nvidia-smi` after create and alert if it differs from the request |
| `drivemount` needs a browser consent on every new runtime | colabinet doesn't use Drive |
| **`colab upload` fails with HTTP 500 when the remote parent directory doesn't exist.** The Jupyter Contents API doesn't create directories. The failed upload also printed the runtime proxy token inside the URL | `vm/prepare.py` creates `/content/colabinet` (+ `templates/`) before any upload; CLI output is scrubbed of `colab-runtime-proxy-token` values |
| `colab exec -f FILE` sends the **local** file's text to the kernel, so nothing has to be uploaded to run it | `bg.py`, `status.py` and `prepare.py` are exec'd from the local machine, never uploaded; only the setup script and templates are uploaded, because `bg.py` starts them as files on the VM |
| Real output formats match the 0.7.4 source: `[name] endpoint \| Hardware: CPU \| Shape: Standard \| Variant: DEFAULT`, `Usage rate: 0.08/hr`, `Active assignments: N`, and `[colab] Session READY.` from `new` (a CPU create took a few seconds) | parser fixtures in `tests/fixtures/cli/` |

## Runtimes and rates

Rates (CU/h) come from `colab usage` unless marked. The *(third-party)* ones are from the 2026-09-24 rate card in
`hcsolakoglu/colab-cli-agent-skill`. RAM is the total (decimal GB) each class reported to colabinet during the
2026-10-13 preset test matrix.

| `colab new …` | CU/h | GPU / VRAM | RAM | notes |
|---|---|---|---|---|
| (no flag) CPU | 0.08 | — | 13.6 GB | 2 vCPU, ~206 GB free disk |
| `--gpu T4` | 1.07 *(third-party)* | T4 16 GB | 13.6 GB | no bf16/fp8 |
| `--gpu T4 --high-mem` | 1.27 *(third-party)* | T4 16 GB | 54.8 GB | |
| `--gpu L4` | **1.54** | L4 22.5 GB (23,034 MiB), fp8 | 56.9 GB | 12 vCPU; usually assigned first try |
| `--gpu A100` | 5.30 | A100 40 GB | 89.6 GB | often not available |
| `--gpu A100 --high-mem` | 6.77 | A100 **80 GB** | 179.4 GB | the RAM choice decides the card *(third-party)* |
| `--gpu G4` | 8.90 | RTX PRO 6000 Blackwell 96 GB, fp8 + nvfp4 | 189.9 GB | |

- **All runtimes:** Ubuntu 24.04, Python 3.13.15, ~236 GB ephemeral disk (~189 GB free on L4), `uv` and the
  `hf` CLI preinstalled. Everything is wiped on stop, and there's no way to change a runtime's GPU.
- **GPU image (2026-09-30):** torch 2.11.0+**cu128**, cuDNN 9.19, driver 580.82.07. Colab rolled a CUDA 13
  image out and back on 2026-09-24 (colabtools #6111/#6112) and has promised a second rollout. That
  matters to anything installed against torch's CUDA version, such as the ExLlamaV3 wheel. llama.cpp's
  prebuilt CUDA 12.8 binary ships its own cudart, so it works either way.
- **CU grants expire** 90 days after purchase or grant *(third-party)*.
- **Running out of balance:** paid users whose balance hits zero drop to free-tier rules, and **SSH is
  banned on free-tier runtimes**. colabinet warns well before the balance runs out.

## Networking and SSH

| fact | why colabinet cares |
|---|---|
| Colab FAQ bans "connecting to remote proxies" for everyone (cloudflared, ngrok, localtunnel) | only SSH through `colab ssh`; all listeners on 127.0.0.1 |
| SSH and web UIs are allowed for paid users with a positive balance | the tunnel is within the rules |
| Verified tunnel: `ssh -N -L 127.0.0.1:LOCAL:127.0.0.1:REMOTE -o ProxyCommand="colab ssh --proxy-mode -s NAME" -o ExitOnForwardFailure=yes -o ServerAliveInterval=15 -o ServerAliveCountMax=4 -o ConnectTimeout=90 -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR -o IdentitiesOnly=yes -i ~/.ssh/id_ed25519 root@colab-runtime` | the tunnel command, as used |
| ed25519 keys work; RSA keys are rejected | require an Ed25519 key (Settings shows whether it exists) |
| `colab ssh --proxy-mode -s NAME` **creates a new paid VM if NAME is not live** | start a tunnel only after `colab sessions` lists NAME |
| One SSH connection per runtime; a second gets `Already-active SSH session (HTTP 429)` | one ssh process per session carrying every forward |
| Tunnel overhead ≈ 0.1 s per request; SSE streaming passes through (first token 0.65 s, then full speed) | the proxy's end-to-end numbers include ~0.1 s |
| **A tunnel survives past the ~1 h runtime token:** one `ssh -N` ran 75.9 min on a CPU runtime with no drop or restart | no special handling needed; restart-on-exit stays as a safety net |
| Port 8080 is taken on Colab VMs | engine port defaults to 8090; 8080 is rejected |
| Piping `ssh -f` into another process (e.g. `tee`) hangs the shell | colabinet runs ssh as a child process with `-N`, output to its own log |

## Idle reclaim

- Third-party observations saw idle reclaim 22–25 min after the last kernel execution. That does not
  establish a fixed Colab timeout, and background servers and tunnel traffic are not kernel executions.
- Runtimes with a kernel execution at least every ~15 min lived 3.7 h and 7 h. colabinet's keep-alive runs
  `colab exec status.py` every 100 s.
- **Idle runtimes still bill until reclaimed or stopped.** Stop unmanaged runtimes explicitly rather than
  relying on reclamation.

## Measured setup times

| engine / model | GPU | create → ready | notes |
|---|---|---:|---|
| llama.cpp, 19 MB test model | CPU | 90 s | `colab new` 5 s, GPU check 13 s, setup 54 s, tunnel 18 s |
| llama.cpp, 27B Q4_K (17.7 GB) | L4 | 10.7 min | download 9 min at 31.6 MB/s; Hugging Face speed varies with time of day |
| ExLlamaV3, 27B 4.0 bpw (16.5 GB) | L4 | 4.8–7.5 min | install 23–103 s, download 71–190 s, load 93–96 s |
| Strata IQ2_XS, cold parallel setup | L4 | 20.6 min | engine build, model pack and MTP branches run concurrently (916 s joined) |

- **llama.cpp on a CPU runtime:** the CUDA 12.8 build falls back to CPU there; `nvidia-smi` isn't installed
  and torch is `2.11.0+cpu`.
- **Stream stats:** llama-server b11273 streaming puts `timings` (`prompt_n`, `prompt_ms`, `predicted_n`,
  `predicted_ms`, `cache_n`) in the **final chunk** and sends no `usage` in the stream. Non-streamed responses
  carry both `usage` and `timings`. TabbyAPI streams omit usage unless `stream_options.include_usage` is set,
  so the proxy injects it.
- **ExLlamaV3 1.5.2 wheel:** there is no cu130-named wheel; the
  [pinned build workflow](https://github.com/turboderp-org/exllamav3/blob/v1.5.2/.github/workflows/build.yml)
  builds the cu132/torch2.11 wheel against PyTorch's cu130 distribution, so setup maps that combination to it.
  The package has no `__version__`; setup reads it through `importlib.metadata`.
- **Vision:** the 27B EXL3 build answered image questions correctly through the OpenAI `image_url` API
  (captures in `tests/fixtures/tabby/vision-20261003/`), at 16.2 tok/s on L4.

## Downloads

- **HF to VM disk:** 28–240 MB/s depending on time of day. `HF_HUB_DISABLE_XET=1` was the steadiest at ~100 MB/s.
  Downloads without an HF token work; whether a token speeds them up is untested.
- **Google Drive to VM:** cold reads ran at 39.9 MB/s against HF's 99.8 MB/s on the same VM, so Drive isn't used.

## llama.cpp measurements (L4, b11273, Qwen3.8-27B abliterated Q4_K, `-np 1`)

| context / KV cache | prompt t/s | output t/s | VRAM |
|---|---|---|---|
| 262K / f16 (`--fit` spills layers to CPU) | 19 | 4.4 | |
| 262K / q8_0 | 31 | 6.4 | |
| **262K / q4_0** | 118 | **13.0** | 21.3 GB |
| 32K / f16 | 156 | 14.2 | 17.7 GB |
| 262K / q4_0, ~40K-token prompt | 453 | 10.7 | 21.3 GB |

- **Qwen3.8-27B cache:** 16 of its 64 layers are full attention, with 4 KV heads × 256 dims, which gives
  **64 KiB of f16 KV per token**.
- **Qwen3.8 chat template:** `reasoning_effort` accepts only `low`, `medium` and `xhigh`. The stock
  template returns **HTTP 500 for `high` or `max`**. `chat_templates/qwen38_effort_map.jinja` maps
  minimal/low→low, medium→medium and high/xhigh/max→xhigh. `reasoning_effort: "none"` turns thinking off.
- **Clients that need an API key:** some OpenAI-compatible clients only list a custom provider's models when it
  has an `apiKey`; any dummy value works, since the local endpoint ignores it.

## Shell pitfalls

- `pkill -f '<literal>'` also kills any watchdog whose own command line contains that literal, so build
  patterns at run time or track PIDs. colabinet owns its child processes and doesn't need `pkill` locally.
- A liveness check that lists jobs by name once missed a job and stopped a healthy session. Match
  generically, and never stop a session on a liveness miss alone.
- macOS has no GNU `timeout`.
