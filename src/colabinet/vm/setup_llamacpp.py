# Runs ON the Colab VM, started detached by bg.py.
# Idempotent: skips the install and downloads that are already present, stops all three engines first (so a
# reconfigure can switch engines), then starts llama-server and waits for /health. Progress goes to $EVENTS.
import json
import os
import re
import shlex
import socket
import subprocess
import sys
import tarfile
import threading
import time
import urllib.request
from urllib.parse import quote

from gguf_files import gguf_files

E = os.environ.get
ROOT = E("COLABINET_ROOT", "/content/colabinet")
EVENTS = E("EVENTS", os.path.join(ROOT, "setup.jsonl"))
ENGINE_LOG = E("ENGINE_LOG", os.path.join(ROOT, "engine.log"))
MODELS = E("MODELS_DIR", "/content/models")
LLAMA_DIR = E("LLAMA_DIR", "/content/llama")
T0 = time.time()


def event(kind, **kw):
    rec = {"t": round(time.time() - T0, 1), "event": kind, **kw}
    with open(EVENTS, "a") as f:
        f.write(json.dumps(rec) + "\n")
    print(json.dumps(rec), flush=True)


def sh(cmd, env=None, timeout=None):
    p = subprocess.run(cmd, shell=True, capture_output=True, text=True, env=env, timeout=timeout)
    return p.returncode, p.stdout + p.stderr


def model_directory(repo, models):
    """Repository-specific cache: different publishers reuse projector filenames."""
    return os.path.join(models, quote(repo, safe=""))


def build_command(env, server, models=MODELS):
    """llama-server argv from the LLM_* environment."""
    models = model_directory(env["LLM_REPO"], models)
    cmd = [server, "-m", os.path.join(models, env["LLM_FILE"]), "--host", "127.0.0.1", "--port", env["LLM_PORT"],
           "-c", env.get("LLM_CTX", "32768"), "-np", env.get("LLM_PARALLEL", "1"),
           "-ctk", env.get("LLM_KV", "f16"), "-ctv", env.get("LLM_KV", "f16"), "--jinja"]
    if env.get("LLM_ALIAS"):
        cmd += ["--alias", env["LLM_ALIAS"]]
    if env.get("LLM_MMPROJ_FILE"):
        cmd += ["--mmproj", os.path.join(models, env["LLM_MMPROJ_FILE"])]
        if env.get("LLM_MMPROJ_GPU") != "1":
            cmd += ["--no-mmproj-offload"]
    if env.get("LLM_TEMPLATE"):
        cmd += ["--chat-template-file", env["LLM_TEMPLATE"]]
    cmd += json.loads(env.get("LLM_EXTRA_ARGS", "[]"))
    return cmd


def port_busy(port):
    with socket.socket() as s:
        s.settimeout(1)
        return s.connect_ex(("127.0.0.1", int(port))) == 0


def stop_engines(port):
    # A shell command containing these patterns can kill its own shell before reaching Tabby.
    patterns = ("[l]lama-server", "[t]abbyAPI/main.py", "[s]erve/server.py.*--engine strata",
                "[/]engine/strata([ -]|$)", "[/]engine/strata-vision([ ]|$)")
    for pattern in patterns:
        subprocess.run(["pkill", "-f", pattern], capture_output=True, text=True)
    time.sleep(2)
    # The listening socket can disappear while Strata's model/vision children still own RAM and VRAM.
    for pattern in patterns:
        subprocess.run(["pkill", "-f", pattern, "--signal", "KILL"], capture_output=True, text=True)
    for _ in range(30):
        if not port_busy(port):
            return True
        time.sleep(1)
    return False


def install(release):
    base = f"https://github.com/ggml-org/llama.cpp/releases/download/{release}/"
    suffix = f"{release}-bin-ubuntu-cuda-12.8-x64.tar.gz"
    names = [f"llama-{suffix}", f"cudart-llama-{suffix}"]
    t = time.time()
    marker = os.path.join(LLAMA_DIR, f".installed-{release}")
    skipped = os.path.exists(marker)
    if not skipped:
        os.makedirs(LLAMA_DIR, exist_ok=True)
        for name in names:
            path = os.path.join(LLAMA_DIR, name)
            urllib.request.urlretrieve(base + name, path)
            with tarfile.open(path) as tf:
                tf.extractall(LLAMA_DIR)
        open(marker, "w").close()
    server = sh(f"find {LLAMA_DIR} -name llama-server -type f | head -1")[1].strip()
    libs = sorted({os.path.dirname(p) for p in sh(f"find {LLAMA_DIR} -name 'lib*.so*'")[1].split()})
    event("install_done", engine="llamacpp", release=release, seconds=round(time.time() - t, 1), skipped=skipped)
    return server, libs


def dir_bytes(path):
    total = 0
    for root, _, files in os.walk(path):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(root, f))
            except OSError:
                pass
    return total


def download(repo, name):
    models = model_directory(repo, MODELS)
    shards = gguf_files(name)
    missing = [shard for shard in shards if not os.path.exists(os.path.join(models, shard))]

    def total_bytes():
        return sum(os.path.getsize(os.path.join(models, shard)) for shard in shards)

    if not missing:
        event("download_done", file=name, gb=round(total_bytes() / 1e9, 2), seconds=0, skipped=True)
        return
    os.makedirs(models, exist_ok=True)
    env = {**os.environ, "HF_HUB_DISABLE_XET": "1", "HF_HUB_DISABLE_PROGRESS_BARS": "1"}
    start_bytes, t = dir_bytes(models), time.time()
    done = threading.Event()

    def progress():
        while not done.wait(10):
            got = dir_bytes(models) - start_bytes
            event("download", file=name, done_gb=round(got / 1e9, 2), mb_s=round(got / (time.time() - t) / 1e6, 1))

    threading.Thread(target=progress, daemon=True).start()
    # One Hub request lets hf fetch independent shards concurrently and resume partial downloads.
    files = " ".join(shlex.quote(shard) for shard in missing)
    rc, out = sh(f"hf download {shlex.quote(repo)} {files} --local-dir {shlex.quote(models)} --max-workers 4", env=env)
    done.set()
    remaining = [shard for shard in shards if not os.path.exists(os.path.join(models, shard))]
    if rc != 0 or remaining:
        raise RuntimeError(f"download of {name} failed (missing {remaining}): {out[-500:]}")
    size, dt = total_bytes(), time.time() - t
    transferred = sum(os.path.getsize(os.path.join(models, shard)) for shard in missing)
    event("download_done", file=name, gb=round(size / 1e9, 2), seconds=round(dt, 1),
          mb_s=round(transferred / dt / 1e6, 1))


def log_tail(n=2000):
    try:
        with open(ENGINE_LOG, "rb") as f:
            f.seek(0, 2)
            f.seek(max(0, f.tell() - n))
            return f.read().decode("utf-8", "replace")
    except OSError:
        return ""


def log_size():
    try:
        return os.path.getsize(ENGINE_LOG)
    except OSError:
        return 0


def exit_reason(returncode, log_start):
    """Why llama-server stopped while loading: SIGKILL (measured on a CPU runtime loading a model larger than its
    13.6 GB RAM), or the last error line this run logged."""
    if returncode == -9:
        return "llama-server was killed (SIGKILL), usually by the out-of-memory killer: not enough host RAM"
    try:
        with open(ENGINE_LOG, "rb") as f:
            f.seek(log_start)
            lines = f.read().decode("utf-8", "replace").splitlines()
    except OSError:
        lines = []
    errors = [line.strip() for line in lines if re.search(r"\berror\b", line, re.IGNORECASE)]
    return f"llama-server exited with {returncode}" + (f": {errors[-1][:300]}" if errors else "")


def main():
    env = dict(os.environ)
    port = env["LLM_PORT"]
    rc, out = sh("nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader,nounits")
    rc2, torch = sh("python3 -c 'import torch; print(torch.__version__, torch.version.cuda)'", timeout=120)
    event("gpu_info", nvidia_smi=out.strip(), torch=torch.strip())
    if not stop_engines(port):
        raise RuntimeError(f"port {port} is still busy after stopping the engines")
    server, libs = install(env["LLM_RELEASE"])
    for name in [env["LLM_FILE"]] + ([env["LLM_MMPROJ_FILE"]] if env.get("LLM_MMPROJ_FILE") else []):
        download(env["LLM_REPO"], name)
    run_env = {**env, "LD_LIBRARY_PATH": ":".join([os.path.dirname(server), *libs, "/usr/lib64-nvidia",
                                                   env.get("LD_LIBRARY_PATH", "")])}
    cmd = build_command(env, server)
    t = time.time()
    log_start = log_size()
    proc = subprocess.Popen(cmd, env=run_env, stdout=open(ENGINE_LOG, "a"), stderr=subprocess.STDOUT,
                            start_new_session=True)
    while time.time() - t < float(env.get("LOAD_TIMEOUT", "600")):
        if proc.poll() is not None:
            raise RuntimeError(exit_reason(proc.returncode, log_start))
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=5) as r:
                if r.status == 200:
                    break
        except Exception:
            pass
        time.sleep(3)
    else:
        raise RuntimeError("llama-server did not become healthy in time")
    props = {}
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/props", timeout=10) as r:
            props = json.loads(r.read())
    except Exception:
        pass
    used = sh("nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits")[1].strip()
    event("server_ready", seconds=round(time.time() - t, 1), vram_used_mib=int(used) if used.isdigit() else None,
          modalities=props.get("modalities"), cmd=" ".join(cmd[1:]))


if __name__ == "__main__":
    try:
        main()
    except Exception as e:  # report every failure as an event the supervisor can show
        event("error", message=str(e)[:1000], log_tail=log_tail())
        sys.exit(1)
