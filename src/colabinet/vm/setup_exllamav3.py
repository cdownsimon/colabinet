import json
import os
import re
import shutil
import socket
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

env_get = os.environ.get
ROOT = env_get("COLABINET_ROOT", "/content/colabinet")
EVENTS = env_get("EVENTS", os.path.join(ROOT, "setup.jsonl"))
ENGINE_LOG = env_get("ENGINE_LOG", os.path.join(ROOT, "engine.log"))
TABBY_DIR = env_get("TABBY_DIR", "/content/tabbyAPI")
TABBY_GIT = "https://github.com/theroyallab/tabbyAPI"
WHEEL_URL = ("https://github.com/turboderp-org/exllamav3/releases/download/v{version}/"
             "exllamav3-{version}%2B{cuda}.torch{torch}-{py}-{py}-linux_x86_64.whl")
PY_TAG = f"cp{sys.version_info.major}{sys.version_info.minor}"
VENV = Path(ROOT) / f"exllamav3-venv-{PY_TAG}"
DEFAULT_TORCH = "2.11.0+cu128"
# ExLlamaV3's CUDA 13.2 extension is built against PyTorch's CUDA 13.0 distribution.
WHEEL_CUDA = {"2.11.0+cu128": "cu128", "2.11.0+cu130": "cu132"}
T0 = time.time()


def event(kind, **kw):
    rec = {"t": round(time.time() - T0, 1), "event": kind, **kw}
    with open(EVENTS, "a") as filters:
        filters.write(json.dumps(rec) + "\n")
    print(json.dumps(rec), flush=True)


def sh(cmd, cwd=None, env=None, timeout=None):
    preset = subprocess.run(cmd, capture_output=True, text=True, cwd=cwd, env=env, timeout=timeout)
    return preset.returncode, preset.stdout + preset.stderr


def wheel_url(version, torch_version, py_tag):
    """The exllamav3 release wheel for this torch ("2.11.0+cu128") and Python ("cp313"), or an error message."""
    if torch_version not in WHEEL_CUDA:
        return None, f"no known exllamav3 wheel pairing for torch {torch_version} (supported: {', '.join(WHEEL_CUDA)})"
    if py_tag not in {"cp312", "cp313", "cp314"}:
        return None, f"no known exllamav3 wheel for torch {torch_version} and {py_tag}"
    return WHEEL_URL.format(version=version, cuda=WHEEL_CUDA[torch_version],
                            torch=torch_version.split("+")[0], py=py_tag), None


def model_dir_name(repo, revision):
    return f"{repo.split('/')[-1]}-{revision}"


def build_config(env, model_name, template_name):
    """Effective TabbyAPI settings, including extra-option overrides."""
    cfg = {
        "network": {"host": "127.0.0.1", "port": int(env["EXL_PORT"]), "disable_auth": True,
                    "api_servers": ["OAI"]},
        "logging": {"log_prompt": False, "log_requests": False, "log_live_status": False},
        "model": {
            "model_dir": "models", "model_name": model_name, "backend": "exllamav3",
            "max_seq_len": int(env["EXL_MAX_SEQ_LEN"]), "cache_size": int(env["EXL_MAX_SEQ_LEN"]),
            "cache_mode": env.get("EXL_CACHE_MODE", "FP16"), "chunk_size": int(env.get("EXL_CHUNK_SIZE", "2048")),
            "max_batch_size": int(env.get("EXL_PARALLEL", "1")), "warmup": True,
            "vision": env.get("EXL_VISION", "0") == "1",
            "vision_offload": env.get("EXL_VISION_OFFLOAD", "0") == "1",
            "inline_model_loading": False,
        },
        "draft_model": {"draft_mode": env.get("EXL_DRAFT_MODE", "disabled")},
        "sampling": {"override_preset": "safe_defaults"},
    }
    if template_name:
        cfg["model"]["prompt_template"] = template_name
    for key, value in json.loads(env.get("EXL_EXTRA", "{}")).items():
        section, _, name = key.partition(".")
        if section not in ("model", "draft_model", "sampling") or not name.isidentifier():
            raise ValueError(f"unsupported extra option {key!r}")
        try:
            value = json.loads(value)
        except (TypeError, ValueError):
            pass
        cfg.setdefault(section, {})[name] = value
    for name in ("vision", "vision_offload"):
        if not isinstance(cfg["model"][name], bool):
            raise ValueError(f"model.{name} must be a boolean (true or false)")
    return cfg


def render_config(env, model_name, template_name):
    """TabbyAPI config.yml. JSON scalars are valid YAML, so values are emitted with json.dumps."""
    cfg = build_config(env, model_name, template_name)
    lines = []
    for section, values in cfg.items():
        lines.append(f"{section}:")
        lines += [f"  {key}: {json.dumps(value)}" for key, value in values.items()]
    return "\n".join(lines) + "\n"


def read_modalities(port):
    """Query the loaded model, rather than infer capabilities from requested settings."""
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/props", timeout=10) as response:
            props = json.loads(response.read())
    except (OSError, ValueError):
        return None
    modalities = props.get("modalities") if isinstance(props, dict) else None
    if isinstance(modalities, dict) and isinstance(modalities.get("vision"), bool):
        return modalities
    return None


def port_busy(port):
    with socket.socket() as statement:
        statement.settimeout(1)
        return statement.connect_ex(("127.0.0.1", int(port))) == 0


def stop_engines(port):
    # A shell command containing these patterns can kill its own shell before reaching Tabby.
    patterns = ("[l]lama-server", "[t]abbyAPI/main.py", "[s]erve/server.py.*--engine strata",
                "[/]engine/strata([ -]|$)", "[/]engine/strata-vision([ ]|$)", "[/]inkling-engine/strata-inkling([ ]|$)")
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


def private_env():
    env = dict(os.environ)
    # Colab can inject its site-packages into otherwise isolated interpreters.
    env.pop("PYTHONPATH", None)
    env.pop("PYTHONHOME", None)
    env.update(VIRTUAL_ENV=str(VENV), PATH=f"{VENV / 'bin'}:{env['PATH']}")
    return env


def install(env):
    commit, version = env["EXL_TABBY_COMMIT"], env["EXL_WHEEL"]
    torch_version = env.get("EXL_TORCH", DEFAULT_TORCH)
    url, err = wheel_url(version, torch_version, PY_TAG)
    if err:
        raise RuntimeError(err)
    started = time.time()
    python = str(VENV / "bin/python")
    marker = VENV / ".installed"
    identity = f"{commit} {version} {torch_version} {PY_TAG}"
    wheel_version = f"{version}+{WHEEL_CUDA[torch_version]}.torch{torch_version.split('+')[0]}"
    probe = [python, "-c", """
import sys
from importlib.metadata import version
import torch
import exllamav3_ext
import exllamav3

actual = [torch.__version__, version("exllamav3")]
if actual != sys.argv[1:]:
    sys.exit(f"expected {sys.argv[1:]}, got {actual}")
""", torch_version, wheel_version]

    def run(args, action, cwd=None):
        rc, out = sh(args, cwd=cwd, env=env)
        if rc:
            raise RuntimeError(f"{action} failed: {out[-500:]}")

    skipped = False
    if marker.is_file() and marker.read_text() == identity and os.path.isfile(python) and os.path.isdir(TABBY_DIR):
        rc, head = sh(["git", "rev-parse", "HEAD"], cwd=TABBY_DIR, env=env)
        if rc == 0 and head.strip() == commit:
            skipped = sh(probe, env=env)[0] == 0
    if not skipped:
        marker.unlink(missing_ok=True)
        if not os.path.isfile(python):
            # Colab does not ship ensurepip. Bootstrap pip directly into the new interpreter.
            run([sys.executable, "-m", "venv", "--without-pip", str(VENV)], "creating the ExLlamaV3 venv")
        if sh([python, "-m", "pip", "--version"], env=env)[0]:
            run([sys.executable, "-m", "pip", "--isolated", "--python", python, "install", "pip"], "bootstrapping pip")
        if not os.path.isdir(os.path.join(TABBY_DIR, ".git")):
            run(["git", "clone", "-q", TABBY_GIT, TABBY_DIR], "cloning TabbyAPI")
        run(["git", "fetch", "-q", "origin", commit], f"fetching TabbyAPI {commit}", cwd=TABBY_DIR)
        run(["git", "checkout", "-q", "--detach", commit], f"checking out TabbyAPI {commit}", cwd=TABBY_DIR)
        pip = [python, "-m", "pip", "--isolated", "install", "-q", "--no-cache-dir"]
        torch_pin = f"torch=={torch_version}"
        index = "https://download.pytorch.org/whl/" + torch_version.split("+")[1]
        run([*pip, "--index-url", index, torch_pin], f"installing {torch_pin}")
        run([*pip, torch_pin, url, TABBY_DIR], f"installing TabbyAPI and exllamav3 {version} for {torch_pin}/{PY_TAG}")
        run(probe, "verifying private torch and exllamav3")
        marker.write_text(identity)
    event("install_done", engine="exllamav3", release=f"{version}+{commit[:7]}", skipped=skipped,
          seconds=round(time.time() - started, 1), torch=torch_version)


def dir_bytes(path):
    total = 0
    for root, _, files in os.walk(path):
        for filters in files:
            try:
                total += os.path.getsize(os.path.join(root, filters))
            except OSError:
                pass
    return total


def download(repo, revision, dest, env):
    if os.path.exists(os.path.join(dest, ".complete")):
        event("download_done", file=f"{repo}@{revision}", gb=round(dir_bytes(dest) / 1e9, 2), seconds=0,
              skipped=True)
        return
    os.makedirs(dest, exist_ok=True)
    cursor, done = time.time(), threading.Event()

    def progress():
        while not done.wait(10):
            got = dir_bytes(dest)
            event("download", file=f"{repo}@{revision}", done_gb=round(got / 1e9, 2),
                  mb_s=round(got / (time.time() - cursor) / 1e6, 1))

    threading.Thread(target=progress, daemon=True).start()
    env = {**env, "HF_HUB_DISABLE_XET": "1", "HF_HUB_DISABLE_PROGRESS_BARS": "1"}
    code = ("import sys; from huggingface_hub import snapshot_download as s; "
            "s(sys.argv[1], revision=sys.argv[2], local_dir=sys.argv[3])")
    preset = subprocess.run([VENV / "bin/python", "-c", code, repo, revision, dest],
                            capture_output=True, text=True, env=env)
    done.set()
    if preset.returncode:
        raise RuntimeError(f"download of {repo}@{revision} failed: {(preset.stdout + preset.stderr)[-500:]}")
    open(os.path.join(dest, ".complete"), "w").close()
    size, dt = dir_bytes(dest), time.time() - cursor
    event("download_done", file=f"{repo}@{revision}", gb=round(size / 1e9, 2), seconds=round(dt, 1),
          mb_s=round(size / dt / 1e6, 1))


def log_tail(count=2000):
    try:
        with open(ENGINE_LOG, "rb") as filters:
            filters.seek(0, 2)
            filters.seek(max(0, filters.tell() - count))
            return filters.read().decode("utf-8", "replace")
    except OSError:
        return ""


def log_size():
    try:
        return os.path.getsize(ENGINE_LOG)
    except OSError:
        return 0


def exit_reason(returncode, log_start):
    """Why TabbyAPI stopped while loading: SIGKILL, or the last exception this run logged, such as ExLlamaV3's
    "RuntimeError: Insufficient VRAM in split for model and cache" (measured on a T4)."""
    if returncode == -9:
        return "TabbyAPI was killed (SIGKILL), usually by the out-of-memory killer: not enough host RAM"
    try:
        with open(ENGINE_LOG, "rb") as filters:
            filters.seek(log_start)
            lines = filters.read().decode("utf-8", "replace").splitlines()
    except OSError:
        lines = []
    errors = [line.strip() for line in lines if re.match(r"\s*[\w.]*(Error|Exception): ", line)]
    return f"TabbyAPI exited with {returncode}" + (f": {errors[-1][:300]}" if errors else "")


def main():
    env = private_env()
    port = env["EXL_PORT"]
    _, smi = sh(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader,nounits"])
    # Keep host versions in diagnostics. Install and launch use the private pair.
    _, versions = sh([sys.executable, "-c", "from importlib.metadata import version; "
                      "print(version('torch'), version('triton'))"])
    event("gpu_info", nvidia_smi=smi.strip(), torch=versions.strip()[-120:])
    if not stop_engines(port):
        raise RuntimeError(f"port {port} is still busy after stopping the engines")
    install(env)
    name = model_dir_name(env["EXL_REPO"], env["EXL_REVISION"])
    download(env["EXL_REPO"], env["EXL_REVISION"], os.path.join(TABBY_DIR, "models", name), env)
    template_name = None
    if env.get("EXL_TEMPLATE") and os.path.exists(env["EXL_TEMPLATE"]):
        template_name = os.path.splitext(os.path.basename(env["EXL_TEMPLATE"]))[0]
        os.makedirs(os.path.join(TABBY_DIR, "templates"), exist_ok=True)
        shutil.copy(env["EXL_TEMPLATE"], os.path.join(TABBY_DIR, "templates", template_name + ".jinja"))
    config = build_config(env, name, template_name)
    with open(os.path.join(TABBY_DIR, "config.yml"), "w") as filters:
        filters.write(render_config(env, name, template_name))
    cursor = time.time()

    log_start = log_size()
    proc = subprocess.Popen([VENV / "bin/python", os.path.join(TABBY_DIR, "main.py")], cwd=TABBY_DIR, env=env,
                            stdout=open(ENGINE_LOG, "a"), stderr=subprocess.STDOUT, start_new_session=True)
    while time.time() - cursor < float(env.get("LOAD_TIMEOUT", "900")):
        if proc.poll() is not None:
            raise RuntimeError(exit_reason(proc.returncode, log_start))
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=5) as row:
                if row.status == 200:
                    break
        except Exception:
            pass
        time.sleep(3)
    else:
        raise RuntimeError(f"TabbyAPI did not answer on port {port} in time (if the port was busy it moves to +1)")
    modalities = read_modalities(port)
    if config["model"]["vision"] and (modalities is None or not modalities["vision"]):
        raise RuntimeError("vision was requested but TabbyAPI did not confirm a loaded vision component via /props")
    used = sh(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"])[1].strip()
    event("server_ready", seconds=round(time.time() - cursor, 1), vram_used_mib=int(used) if used.isdigit() else None,
          modalities=modalities, cmd=f"tabbyAPI {name} {env.get('EXL_CACHE_MODE', 'FP16')}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        event("error", message=str(exc)[:1000], log_tail=log_tail())
        sys.exit(1)
