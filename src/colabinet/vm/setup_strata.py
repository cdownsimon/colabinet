# Runs on the Colab VM via bg.py. Upstream owns formats and inference; this script owns parallel setup/lifecycle.
import ctypes
import hashlib
import importlib.util
import json
import os
import platform
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import tarfile
import tempfile
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

E = os.environ.get
ROOT = Path(E("COLABINET_ROOT", "/content/colabinet"))
RELEASE = "c77572806996349b49995e4f98d41b5294d36a21"
SOURCE = ROOT / f"Strata-{RELEASE[:7]}"
DATA = ROOT / "strata-data"
EVENTS = E("EVENTS", str(ROOT / "setup.jsonl"))
ENGINE_LOG = E("ENGINE_LOG", str(ROOT / "engine.log"))
REPO = "ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF"
# v0.2.15's release feed reuses this portable CUDA 13 build from v0.2.14.
ENGINE_VERSION = "0.1.60"
ENGINE_URL = ("https://github.com/Niko1221/Strata/releases/download/v0.2.14/"
              f"strata-engine-{ENGINE_VERSION}-linux-x86_64-cuda-cuda13.tar.gz")
ENGINE_SHA256 = "a72c194c8f6de96629ed53079b5e344ab81223d45667a53f6a6d79398895aa50"
ENGINE_ARCHS = {75, 86, 89, 120}
ENGINE_PACKAGE_MARKER = ".colabinet-package-sha256"
T0 = time.monotonic()
STOP_PATTERNS = ("[l]lama-server", "[t]abbyAPI/main.py", "[s]erve/server.py.*--engine strata",
                 "[/]engine/strata([ -]|$)", "[/]engine/strata-vision([ ]|$)")
EVENT_LOCK = threading.Lock()
OUTPUT_LOCK = threading.Lock()


def event(kind, **kw):
    record = {"t": round(time.monotonic() - T0, 1), "event": kind, **kw}
    line = json.dumps(record)
    with EVENT_LOCK:
        with open(EVENTS, "a") as handle:
            handle.write(line + "\n")
        print(line, flush=True)


def log_tail(count=4000):
    try:
        with open(ENGINE_LOG, "rb") as handle:
            handle.seek(0, 2)
            handle.seek(max(0, handle.tell() - count))
            return handle.read().decode("utf-8", "replace")
    except OSError:
        return ""


def run(cmd, cwd=None, env=None, timeout=3600, cancel=None, label=None):
    """Stream setup output while a guard cancels the entire child process group on timeout or sibling failure."""
    prefix = f"[{label}] " if label else ""
    with OUTPUT_LOCK:
        print("+ " + prefix + " ".join(map(str, cmd)), flush=True)
    with open(ENGINE_LOG, "a") as log:
        proc = subprocess.Popen(cmd, cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, start_new_session=True)
        finished, timed_out, sibling_failed = threading.Event(), threading.Event(), threading.Event()

        def guard():
            deadline = time.monotonic() + timeout
            while not finished.wait(0.2):
                if cancel is not None and cancel.is_set():
                    sibling_failed.set()
                    terminate(proc)
                    return
                if time.monotonic() >= deadline:
                    timed_out.set()
                    terminate(proc)
                    return

        watcher = threading.Thread(target=guard, name=f"guard-{label or cmd[0]}", daemon=True)
        watcher.start()
        output = ""
        try:
            for line in proc.stdout:
                output = (output + line)[-4000:]
                with OUTPUT_LOCK:
                    print(prefix + line, end="", flush=True)
                    log.write(prefix + line)
                    log.flush()
            proc.wait()
        except BaseException:
            terminate(proc)
            raise
        finally:
            finished.set()
            watcher.join()
            proc.stdout.close()
        if timed_out.is_set() or sibling_failed.is_set() or proc.returncode:
            failure = ("timed out" if timed_out.is_set() else "cancelled after another setup branch failed"
                       if sibling_failed.is_set() else f"failed: exit {proc.returncode}")
            raise RuntimeError(f"{cmd[0]} {failure}: {output[-1500:]}")
        return output


def terminate(proc):
    # The upstream engine and vision encoder inherit this session; kill the entire group even if the server exited.
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(proc.pid, sig)
        except ProcessLookupError:
            break
        if sig == signal.SIGTERM:
            time.sleep(2)
    proc.wait()


def port_busy(port):
    with socket.socket() as sock:
        sock.settimeout(1)
        return sock.connect_ex(("127.0.0.1", int(port))) == 0


def stop_engines(port):
    for pattern in STOP_PATTERNS:
        subprocess.run(["pkill", "-f", pattern], capture_output=True, text=True)
    time.sleep(2)
    # A closed listening socket does not mean the engine/vision children have exited.
    for pattern in STOP_PATTERNS:
        subprocess.run(["pkill", "-f", pattern, "--signal", "KILL"], capture_output=True, text=True)
    for _ in range(30):
        if not port_busy(port):
            return True
        time.sleep(1)
    return False


def checkout(release):
    if release != RELEASE:
        raise ValueError(f"unsupported Strata source: expected {RELEASE}")
    SOURCE.mkdir(parents=True, exist_ok=True)
    if not (SOURCE / ".git").is_dir():
        run(["git", "init", "-q", str(SOURCE)])
    head = subprocess.run(["git", "-C", str(SOURCE), "rev-parse", "HEAD"], capture_output=True, text=True)
    if head.returncode or head.stdout.strip() != release:
        run(["git", "fetch", "-q", "--depth", "1", "https://github.com/Niko1221/Strata.git", release], cwd=SOURCE)
        run(["git", "checkout", "-q", "--detach", release], cwd=SOURCE)


def select_drafting(requested, choice_of, quant, gpu, args):
    if requested not in ("auto", "off", "dflash2", "mtp"):
        raise ValueError("Strata drafting must be auto, off, dflash2 or mtp")
    selected = choice_of(requested, model_name=f"qwen3.8-flash-next-{quant.lower()}",
                         family="qwen", backend="cuda", archs=[f"sm_{gpu['arch']}"],
                         vram_mib=gpu["vram_gb"] * 1024, args=args)
    if requested == "dflash2" and selected != requested:
        raise RuntimeError(f"DFlash2 is unavailable for Strata {quant} on {gpu['name']}")
    return selected


def check_hardware(upstream, quant, ctx, low_ram, choose_drafting):
    """Use the pinned upstream's GPU, CPU and low-RAM policies before installing/downloading anything large."""
    gpu = upstream.gpu_info()
    if not gpu:
        raise RuntimeError("Strata requires an NVIDIA GPU; CPU sessions are unsupported")
    problem = upstream.gpu_problem(gpu)
    if problem:
        raise RuntimeError(f"Strata GPU {gpu['name']}: {problem}")
    if upstream.driver_major(gpu) < upstream.MIN_DRIVER:
        raise RuntimeError(f"Strata requires NVIDIA driver {upstream.MIN_DRIVER}+; detected {gpu['driver']}")
    cpu, avx2, _ = upstream.cpu_info()
    if not avx2:
        raise RuntimeError(f"Strata requires an AVX2 CPU; detected {cpu}")
    ram = upstream.ram_gb()
    drafting = choose_drafting(gpu)
    if low_ram not in ("auto", "on", "off"):
        raise ValueError("Strata low-RAM mode must be auto, on or off")
    low = low_ram == "on" or (low_ram == "auto" and upstream.low_ram_needed(quant, ram))
    if low and not upstream.low_ram_fits(quant, ram, gpu["vram_gb"], mtp=drafting != "off"):
        raise RuntimeError(f"Strata {quant} cannot fit {ram:.1f} GiB host RAM and {gpu['vram_gb']:.1f} GiB VRAM "
                           "under upstream's low-RAM policy")
    if not low and ram < upstream.MODELS[quant]["ram_gb"] - 4:
        raise RuntimeError(f"Strata {quant} requires about {upstream.MODELS[quant]['ram_gb']} GiB host RAM")
    needed = upstream.MODELS[quant]["arena_gb"] + ctx * 13 * 1056 / 1e9 + 24
    if quant in ("IQ3_XXS", "IQ3_S") and ctx > 131072 and ram < needed:
        raise RuntimeError(f"Strata host RAM guard rejects requested context {ctx} with {ram:.1f} GiB RAM; "
                           f"about {needed:.1f} GiB is required")
    event("gpu_info", nvidia_smi=f"{gpu['name']}, {gpu['vram_gb'] * 1024:.0f}, {gpu['driver']}",
          ram_gb=round(ram, 1), low_ram=low)
    return low, gpu, drafting


def prepare(env):
    """Validate the VM and install the prerequisites shared by the setup branches."""
    started = time.monotonic()
    checkout(env["STRATA_RELEASE"])
    module_spec = importlib.util.spec_from_file_location("strata_setup", SOURCE / "setup.py")
    upstream = importlib.util.module_from_spec(module_spec)
    sys.path.insert(0, str(SOURCE))
    try:
        module_spec.loader.exec_module(upstream)
        from strata_control.drafting import choice_of
    finally:
        sys.path.pop(0)
    requested = env["STRATA_DRAFTING"]
    quant = env["STRATA_QUANT"]
    args = ["--max-context", env["STRATA_CTX"], "--kv", env["STRATA_KV"],
            *json.loads(env["STRATA_EXTRA_ARGS"])]
    low_ram, gpu, drafting = check_hardware(upstream, quant, int(env["STRATA_CTX"]), env["STRATA_LOW_RAM"],
                                           lambda gpu: select_drafting(requested, choice_of, quant, gpu, args))
    env["STRATA_DRAFTING"] = drafting
    event("drafting_selected", requested=requested, selected=drafting)
    check_storage(upstream, env, low_ram)
    python = SOURCE / ".venv" / "bin" / "python"
    if not python.exists():
        # Colab omits ensurepip. Upstream setup.sh accepts an already bootstrapped private venv.
        run([sys.executable, "-m", "venv", "--without-pip", str(SOURCE / ".venv")])
    pip = subprocess.run([str(python), "-m", "pip", "--version"], capture_output=True, text=True)
    if pip.returncode:
        bootstrap = ROOT / "get-pip.py"
        urllib.request.urlretrieve("https://bootstrap.pypa.io/get-pip.py", bootstrap)
        try:
            run([str(python), str(bootstrap), "--quiet"], timeout=600)
        finally:
            bootstrap.unlink(missing_ok=True)
    run([str(python), "-u", "-c", "import setup as S; "
         "S.pip_install(S.runtime_packages(), 'python packages')"],
        cwd=SOURCE, env=env, timeout=float(env.get("INSTALL_TIMEOUT", "3600")))
    event("setup_prepare_done", seconds=round(time.monotonic() - started, 1), low_ram=low_ram)
    return str(python), low_ram, gpu, started


def dir_bytes(path):
    total = 0
    for root, _, files in os.walk(path):
        for name in files:
            try:
                total += os.path.getsize(os.path.join(root, name))
            except OSError:
                pass
    return total


def model_artifacts(env):
    models = DATA / "models"
    first = models / env["STRATA_FILE"]
    second = first.with_name(first.name.replace("-00001-of-00002.gguf", "-00002-of-00002.gguf"))
    files = [first, second]
    if env["STRATA_VISION"] != "none":
        files.append(models / "mmproj-Qwen3.8-Flash-Next-BF16.gguf")
    return files


def finished_file(path):
    marker = path.with_name(path.name + ".done")
    return path.is_file() and path.stat().st_size > 0 and marker.is_file()


def pack_complete(quant, low_ram):
    pack = DATA / "packs" / quant.lower()
    required = [pack / "native_experts.txt", pack / "tokenizer" / "vocab.json"]
    if low_ram:
        required.append(pack / "experts.bin")
    return all(path.is_file() and path.stat().st_size > 0 for path in required)


def check_storage(upstream, env, low_ram):
    """Apply upstream's free-space policy before any parallel branch starts writing."""
    DATA.mkdir(parents=True, exist_ok=True)
    quant = env["STRATA_QUANT"]
    artifacts = model_artifacts(env)
    model_gb = 0 if all(finished_file(path) for path in artifacts[:2]) else upstream.MODELS[quant]["download_gb"]
    vision_gb = 1 if len(artifacts) == 3 and not finished_file(artifacts[2]) else 0
    pack_gb = upstream.MODELS[quant]["arena_gb"] + 1 if low_ram and not pack_complete(quant, True) else 0
    needed = model_gb + vision_gb + pack_gb + 8
    free = shutil.disk_usage(DATA).free / 1e9
    if free < needed:
        raise RuntimeError(
            f"Strata {quant} parallel setup needs about {needed:.0f} GB free; {free:.0f} GB is available")
    event("storage_check", free_gb=round(free, 1), needed_gb=round(needed, 1))


def prebuilt_supported(gpu):
    driver = tuple(int(part) for part in gpu["driver"].split(".")[:2] if part.isdecimal())
    return (sys.platform == "linux" and platform.machine() == "x86_64"
            and int(gpu["arch"]) in ENGINE_ARCHS and driver >= (580, 65))


def probe_prebuilt(engine, env, cancel):
    meta = json.loads((engine / "BUILD.json").read_text())
    if meta.get("version") != ENGINE_VERSION:
        raise RuntimeError(f"Strata package has unexpected engine version {meta.get('version')}")
    executables = [engine / "strata"]
    if env["STRATA_VISION"] == "gpu":
        executables.append(engine / "strata-vision")
    libs = os.pathsep.join(filter(None, (str(engine), env.get("LD_LIBRARY_PATH"))))
    probe_env = {**env, "LD_LIBRARY_PATH": libs}
    for executable in executables:
        if not executable.is_file() or not os.access(executable, os.X_OK):
            raise RuntimeError(f"Strata package is missing executable {executable.name}")
        output = run(["ldd", str(executable)], env=probe_env, timeout=30, cancel=cancel, label="engine")
        if "not found" in output:
            raise RuntimeError(f"Strata package cannot load the libraries for {executable.name}")
    output = run([str(executables[0]), "--families"], env=probe_env, timeout=30, cancel=cancel, label="engine")
    if "qwen4exp" not in output:
        raise RuntimeError("Strata package does not advertise Qwen3.8 Flash-Next support")


def install_prebuilt(engine, env, cancel):
    cached = engine / ENGINE_PACKAGE_MARKER
    if cached.is_file() and cached.read_text() == ENGINE_SHA256:
        probe_prebuilt(engine, env, cancel)
        return True
    with tempfile.TemporaryDirectory(prefix=".engine-install-", dir=SOURCE) as folder:
        archive = Path(folder) / "engine.tar.gz"
        run(["curl", "-fL", "--silent", "--show-error", "--retry", "2", "--connect-timeout", "20",
             "--output", str(archive), ENGINE_URL],
            timeout=min(600, float(env.get("INSTALL_TIMEOUT", "3600"))), cancel=cancel, label="engine")
        with archive.open("rb") as stream:
            if hashlib.file_digest(stream, "sha256").hexdigest() != ENGINE_SHA256:
                raise RuntimeError("Strata engine archive failed SHA-256 verification")
        staged = Path(folder) / "engine"
        with tarfile.open(archive, "r:gz") as package:
            for member in package:
                if cancel.is_set():
                    raise RuntimeError("Strata package extraction cancelled")
                package.extract(member, staged, filter="data")
        probe_prebuilt(staged, env, cancel)
        (staged / ENGINE_PACKAGE_MARKER).write_text(ENGINE_SHA256)
        if engine.exists():
            shutil.rmtree(engine)
        staged.rename(engine)
    return False


def build_engine(python, env, gpu, cancel):
    started = time.monotonic()
    engine = SOURCE / "engine"
    stamp = engine / "BUILD.json"
    if prebuilt_supported(gpu):
        try:
            skipped = install_prebuilt(engine, env, cancel)
            return {"seconds": round(time.monotonic() - started, 1), "skipped": skipped, "source": "prebuilt"}
        except (OSError, ValueError, RuntimeError, tarfile.TarError) as exc:
            if cancel.is_set():
                raise
            event("engine_prebuilt_fallback", reason=str(exc)[:1000])
    else:
        event("engine_prebuilt_fallback", reason=f"CUDA package does not cover {gpu['name']} driver {gpu['driver']}")
    if (engine / ENGINE_PACKAGE_MARKER).exists():
        shutil.rmtree(engine)
    before = stamp.stat().st_mtime_ns if stamp.exists() else None
    run([python, "-u", "-c", "import setup as S; "
         "S.build_engine(S.gpu_info(), " + repr(env["STRATA_VISION"]) + ", True, S.llama_override())"],
        cwd=SOURCE, env=env, timeout=float(env.get("INSTALL_TIMEOUT", "3600")), cancel=cancel, label="engine")
    after = stamp.stat().st_mtime_ns if stamp.exists() else None
    return {"seconds": round(time.monotonic() - started, 1), "skipped": before is not None and before == after,
            "source": "source"}


def prepare_model(python, env, low_ram, cancel):
    started = time.monotonic()
    artifacts = model_artifacts(env)
    initial = dir_bytes(DATA / "models")
    skipped = all(finished_file(path) for path in artifacts)
    progress_done = threading.Event()

    def progress():
        while not progress_done.wait(10):
            got = max(0, dir_bytes(DATA / "models") - initial)
            event("download", file=env["STRATA_FILE"], done_gb=round(got / 1e9, 2),
                  mb_s=round(got / max(1, time.monotonic() - started) / 1e6, 1))

    reporter = threading.Thread(target=progress, name="strata-download-progress", daemon=True)
    reporter.start()
    try:
        download = """
import os
import sys
from pathlib import Path
import setup as S

quant, models_dir, vision = sys.argv[1:]
models = Path(models_dir)
family = S.FAMILIES["qwen"]
shards = [models / quant / family["file"].format(q=quant, i=i) for i in (1, 2)]
if not shards[1].exists():
    for candidate in models.glob("*/Qwen3.8-Flash-Next-GSQ-RCO-*-00002-of-00002.gguf"):
        if candidate != shards[1] and S.done(candidate):
            try:
                shards[1].parent.mkdir(parents=True, exist_ok=True)
                os.link(candidate, shards[1])
                S.mark(shards[1])
                break
            except OSError:
                pass
for shard in shards:
    S.download(family["hf"].format(q=quant) + shard.name, shard)
try:
    S.check_shards(shards)
except BaseException:
    for shard in shards:
        shard.with_name(shard.name + ".done").unlink(missing_ok=True)
    raise
artifacts = list(shards)
if vision != "none":
    projector = models / family["mmproj"]
    S.download(family["mmproj_hf"] + projector.name, projector, "vision encoder")
    artifacts.append(projector)
for artifact in artifacts:
    if not artifact.is_file() or artifact.stat().st_size == 0 or not S.done(artifact):
        artifact.with_name(artifact.name + ".done").unlink(missing_ok=True)
        raise RuntimeError(f"incomplete model artifact: {artifact}")
"""
        run([python, "-u", "-c", download, env["STRATA_QUANT"], str(DATA / "models"),
             env["STRATA_VISION"]], cwd=SOURCE, env=env,
            timeout=float(env.get("DOWNLOAD_TIMEOUT", "7200")), cancel=cancel, label="model")
    finally:
        progress_done.set()
        reporter.join()

    quant = env["STRATA_QUANT"]
    pack = DATA / "packs" / quant.lower()
    if not pack_complete(quant, low_ram):
        temporary = pack.with_name(pack.name + ".parallel-part")
        shutil.rmtree(temporary, ignore_errors=True)
        try:
            run([python, "-u", "tools/iq_pack.py", "--gguf", str(artifacts[0]), "--out", str(temporary)],
                cwd=SOURCE, env=env, timeout=float(env.get("PACK_TIMEOUT", "3600")),
                cancel=cancel, label="model")
            if low_ram:
                run([python, "-u", "tools/iq_pack.py", "--gguf", str(artifacts[0]), "--out", str(temporary),
                    "--experts-bin"], cwd=SOURCE, env=env, timeout=float(env.get("PACK_TIMEOUT", "3600")),
                    cancel=cancel, label="model")
            required = [temporary / "native_experts.txt", temporary / "tokenizer" / "vocab.json"]
            if low_ram:
                required.append(temporary / "experts.bin")
            if any(not path.is_file() or path.stat().st_size == 0 for path in required):
                raise RuntimeError("Strata model pack did not produce every required artifact")
            shutil.rmtree(pack, ignore_errors=True)
            os.replace(temporary, pack)
        except BaseException:
            shutil.rmtree(temporary, ignore_errors=True)
            raise
    seconds = round(time.monotonic() - started, 1)
    return {"seconds": seconds, "skipped": skipped,
            "gb": round(sum(path.stat().st_size for path in artifacts[:2]) / 1e9, 2)}


def prepare_mtp(python, env, cancel):
    started = time.monotonic()
    mtp = DATA / "mtp"
    runtime = mtp / "rt"
    if (runtime / "experts.bin").is_file() and (runtime / "experts.bin").stat().st_size > 0:
        return {"seconds": 0, "skipped": True, "download_bytes": 0}
    mtp.mkdir(parents=True, exist_ok=True)
    packed = mtp / "mtp-q2_0.gguf"
    packed_part = mtp / "mtp-q2_0.gguf.parallel-part"
    runtime_part = mtp / "rt.parallel-part"
    packed_part.unlink(missing_ok=True)
    shutil.rmtree(runtime_part, ignore_errors=True)
    try:
        initial = dir_bytes(mtp / "tensors")
        run([python, "-u", "tools/mtp_fetch.py", "fetch", "--out", str(mtp)], cwd=SOURCE, env=env,
            timeout=float(env.get("MTP_TIMEOUT", "3600")), cancel=cancel, label="mtp")
        download_bytes = max(0, dir_bytes(mtp / "tensors") - initial)
        run([python, "-u", "tools/mtp_pack.py", "--src", str(mtp), "--experts", "q2_0",
             "--out", str(packed_part)], cwd=SOURCE, env=env,
            timeout=float(env.get("MTP_TIMEOUT", "3600")), cancel=cancel, label="mtp")
        run([python, "-u", "tools/mtp_rt.py", "--gguf", str(packed_part), "--out", str(runtime_part)],
            cwd=SOURCE, env=env, timeout=float(env.get("MTP_TIMEOUT", "3600")),
            cancel=cancel, label="mtp")
        experts = runtime_part / "experts.bin"
        if not experts.is_file() or experts.stat().st_size == 0:
            raise RuntimeError("Strata MTP preparation did not produce runtime experts")
        os.replace(packed_part, packed)
        shutil.rmtree(runtime, ignore_errors=True)
        os.replace(runtime_part, runtime)
    except BaseException:
        packed_part.unlink(missing_ok=True)
        shutil.rmtree(runtime_part, ignore_errors=True)
        raise
    return {"seconds": round(time.monotonic() - started, 1), "skipped": False, "download_bytes": download_bytes}


def prepare_dflash2(python, env, cancel):
    started = time.monotonic()
    # ensure() validates the pinned checkpoint and deletes its source files once packed. Count
    # new payload bytes at the transfer boundary, before the source files are removed.
    script = """
import json
import sys
from strata_control import drafting, drafting_fetch

download = drafting_fetch.transfer.download
download_bytes = 0

def record(url, path, **kwargs):
    global download_bytes
    before = path.stat().st_size if path.exists() else 0
    result = download(url, path, **kwargs)
    download_bytes += max(0, path.stat().st_size - before)
    return result

skipped = drafting.pack_ready(sys.argv[1])
drafting_fetch.transfer.download = record
drafting_fetch.ensure(sys.argv[1])
print(json.dumps({"skipped": skipped, "download_bytes": download_bytes}))
"""
    output = run([python, "-u", "-c", script, str(DATA)], cwd=SOURCE, env=env,
                 timeout=float(env.get("MTP_TIMEOUT", "3600")), cancel=cancel, label="dflash2")
    return {**json.loads(output.splitlines()[-1]), "seconds": round(time.monotonic() - started, 1)}


def parallel_setup(python, env, low_ram, gpu, install_started):
    """Run the selected cold-setup branches and cancel siblings after the first failure."""
    started = time.monotonic()
    cancel = threading.Event()
    failures = []
    failure_lock = threading.Lock()

    def execute(name, function):
        task_started = time.monotonic()
        event("setup_task_start", task=name)
        try:
            result = function()
        except BaseException as exc:
            with failure_lock:
                primary = not failures
                failures.append((name, exc))
                cancel.set()
            event("setup_task_failed", task=name, primary=primary, seconds=round(time.monotonic() - task_started, 1),
                  error=str(exc)[:1000])
            raise
        result["finished_at"] = time.monotonic()
        event("setup_task_done", task=name, seconds=result["seconds"], skipped=result.get("skipped", False))
        return result

    drafting = env["STRATA_DRAFTING"]
    tasks = {
        "engine": lambda: build_engine(python, env, gpu, cancel),
        "model": lambda: prepare_model(python, env, low_ram, cancel),
    }
    draft_task = {"mtp": prepare_mtp, "dflash2": prepare_dflash2}.get(drafting)
    if draft_task:
        tasks[drafting] = lambda: draft_task(python, env, cancel)
    event("parallel_setup_start", tasks=list(tasks), drafting=drafting)
    pool = ThreadPoolExecutor(max_workers=len(tasks), thread_name_prefix="strata-setup")
    futures = {pool.submit(execute, name, function): name for name, function in tasks.items()}
    results = {}
    try:
        for future in as_completed(futures):
            if future.exception() is None:
                results[futures[future]] = future.result()
    except BaseException:
        cancel.set()
        raise
    finally:
        pool.shutdown(wait=True, cancel_futures=True)
    if failures:
        name, exc = failures[0]
        raise RuntimeError(f"parallel Strata setup failed in {name}: {exc}") from exc
    engine, model = results["engine"], results["model"]
    event("install_done", engine="strata", release=env["STRATA_RELEASE"],
          seconds=round(engine["finished_at"] - install_started, 1), skipped=engine["skipped"], source=engine["source"])
    event("download_done", file=env["STRATA_FILE"], gb=model["gb"], seconds=model["seconds"],
          skipped=model["skipped"])
    draft = results.get(drafting, {"seconds": 0, "skipped": True, "download_bytes": 0})
    event("drafting_done", mode=drafting, seconds=draft["seconds"], skipped=draft["skipped"],
          download_bytes=draft["download_bytes"])
    event("parallel_setup_done", seconds=round(time.monotonic() - started, 1),
          branches={name: results[name]["seconds"] for name in tasks})
    return results


def argument(args, flag, default=None):
    for i in range(len(args) - 1, -1, -1):
        if args[i] == flag:
            if i + 1 == len(args):
                raise ValueError(f"missing value for {flag}")
            return args[i + 1]
    return default


def configure(config, env):
    """Check upstream choices and apply only settings upstream's CLI cannot express."""
    args = config["args"]
    if int(argument(args, "--max-context", "0")) != int(env["STRATA_CTX"]):
        raise RuntimeError("Strata setup changed the requested context; refusing silently clamped configuration")
    # setup omits --kv below 8K, but the engine supports all three KV formats at every context length.
    if "--kv" in args:
        if argument(args, "--kv") != env["STRATA_KV"]:
            raise RuntimeError("Strata setup changed the requested KV format")
    else:
        args += ["--kv", env["STRATA_KV"]]
    native = Path(argument(args, "--native", ""))
    if native.name != Path(env["STRATA_FILE"]).name:
        raise RuntimeError("Strata setup selected a different model shard")
    wants_vision = env["STRATA_VISION"] == "gpu"
    if wants_vision:
        if config.get("vision", {}).get("gpu") is not True or "--vision" not in args:
            raise RuntimeError("Strata setup did not enable the requested GPU vision encoder")
    elif config.get("vision") or "--vision" in args:
        raise RuntimeError("Strata setup retained vision while vision was disabled")
    projection = env["STRATA_PROJECTION"] == "on"
    loaded = "--control-vector-scaled" in args
    if projection != loaded:
        raise RuntimeError("Strata setup did not honor refusal_projection on/off")
    if projection:
        layer = args.index("--control-vector-layer-range") if "--control-vector-layer-range" in args else -1
        if (layer < 0 or args[layer + 1:layer + 3] != ["4", "44"] or argument(args, "--cvec-mode") != "project"
                or argument(args, "--cvec-dir") != "per-layer"):
            raise RuntimeError("Strata setup did not use the bundled per-layer projection profile")
    # Direct PLE I/O is the safe default on low-RAM Colab machines; explicit --ple-io overrides remain effective.
    args += ["--ple-io", "direct", *json.loads(env.get("STRATA_EXTRA_ARGS", "[]"))]
    drafting = env["STRATA_DRAFTING"]
    if ("--mtp" in args) != (drafting == "mtp"):
        raise RuntimeError("Strata setup did not honor the requested MTP layer setting")
    config["drafting"] = drafting
    config["mtp"] = drafting == "mtp"
    config["parallel"] = 1
    config.pop("user_args", None)
    config["model_name"] = env["STRATA_ALIAS"]
    # stderr reaches the persistent tee below, so native, vision and API logs go to engine.log AND setup.out.
    config["log"] = "/dev/stderr"
    config["host"] = "127.0.0.1"
    config["port"] = int(env["STRATA_PORT"])
    config["api_key"] = ""
    return config


def finalize_model(python, env, low_ram, engine_source):
    """Let pinned upstream verify the completed branches and write the authoritative launch configuration."""
    started = time.monotonic()
    quant = env["STRATA_QUANT"]
    path = SOURCE / f"strata-{quant.lower()}.json"
    path.unlink(missing_ok=True)
    path.with_suffix(".shared-settings.json").unlink(missing_ok=True)
    mtp = env["STRATA_DRAFTING"] == "mtp"
    script = "import setup as S\n"
    if engine_source == "prebuilt":
        # The checked portable package ships its CUDA libs. Upstream's zip installer normally gets them from pip.
        script += "S.pip_cuda_libs = lambda *args: None\n"
    if not mtp:
        # v0.2.15 otherwise repairs a cached MTP layer even when --mtp off was explicit.
        script += "S.mtp_corrupt = lambda *args: False\nS.refresh_draft_vocab = lambda *args: None\n"
    script += "raise SystemExit(S.main())\n"
    engine_args = ["--prebuilt", "", "--cuda", "13"] if engine_source == "prebuilt" else ["--build"]
    run([python, "-u", "-c", script, "--family", "qwen", "--model", quant,
         "--context", env["STRATA_CTX"], "--kv", env["STRATA_KV"], "--vision", env["STRATA_VISION"],
         "--experimental-speed-projection", env["STRATA_PROJECTION"], "--low-ram", "mmap" if low_ram else "off",
         "--mtp", "on" if mtp else "off", "--parallel", "1", "--port", env["STRATA_PORT"],
         "--host", "127.0.0.1", "--data-dir", str(DATA), *engine_args, "--yes", "--no-start"],
        cwd=SOURCE, env=env, timeout=float(env.get("FINALIZE_TIMEOUT", "1800")), label="finalize")
    shards = model_artifacts(env)[:2]
    if any(not path.is_file() for path in shards):
        raise RuntimeError("Strata setup did not retain both official GGUF shards")
    config = configure(json.loads(path.read_text()), env)
    if engine_source == "prebuilt":
        engine = SOURCE / "engine"
        meta = json.loads((engine / "BUILD.json").read_text())
        if Path(config["exe"]) != engine / "strata" or meta.get("version") != ENGINE_VERSION:
            raise RuntimeError("Strata setup did not retain the checked prebuilt engine")
        config["lib_dirs"] = [str(engine)]
    path.write_text(json.dumps(config, indent=1))
    event("setup_finalize_done", seconds=round(time.monotonic() - started, 1))
    return path, config


def read_json(port, path):
    with urllib.request.urlopen(f"http://127.0.0.1:{port}{path}", timeout=10) as response:
        return json.loads(response.read())


def confirm_capabilities(health, models, env, metrics):
    if health.get("status") != "ok" or health.get("model") != env["STRATA_ALIAS"]:
        raise RuntimeError("Strata health did not confirm the requested model alias")
    if health.get("loaded") is False:
        raise RuntimeError("Strata health did not confirm a loaded model")
    if health.get("max_context") != int(env["STRATA_CTX"]):
        raise RuntimeError(f"Strata loaded context {health.get('max_context')} "
                           f"differs from requested {env['STRATA_CTX']}")
    model = next((m for m in models.get("data", []) if m.get("id") == env["STRATA_ALIAS"]), None)
    if (not model or model.get("status", {}).get("value") == "unloaded"
            or model.get("meta", {}).get("n_ctx") != int(env["STRATA_CTX"])):
        raise RuntimeError("Strata model metadata did not confirm the requested loaded model/context")
    wants_vision = env["STRATA_VISION"] == "gpu"
    images = "image" in model.get("architecture", {}).get("input_modalities", [])
    if health.get("images") is not wants_vision or images != wants_vision:
        raise RuntimeError("Strata health/model metadata did not confirm the requested vision capability")
    engine = metrics.get("engine", {})
    if engine.get("kv") != env["STRATA_KV"]:
        raise RuntimeError("Strata engine did not confirm the requested loaded KV format")
    expected = "project:4-44" if env["STRATA_PROJECTION"] == "on" else "0"
    if str(engine.get("cvec")) != expected:
        raise RuntimeError("Strata engine did not confirm the requested refusal projection profile")
    if engine.get("drafting") != env["STRATA_DRAFTING"]:
        raise RuntimeError(f"Strata loaded drafting mode {engine.get('drafting')!r}; "
                           f"expected {env['STRATA_DRAFTING']}: {engine.get('drafting_note', '')}")
    return {"vision": wants_vision}


def cached_bytes(path):
    """Measure Linux file-cache residency, not merely bytes read during warmup."""
    libc = ctypes.CDLL(None, use_errno=True)
    libc.mmap.restype = ctypes.c_void_p
    libc.mmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_long]
    libc.mincore.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p]
    libc.munmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    with open(path, "rb") as handle:
        size = os.fstat(handle.fileno()).st_size
        page = os.sysconf("SC_PAGE_SIZE")
        address = libc.mmap(None, size, 1, 1, handle.fileno(), 0)
        if address == ctypes.c_void_p(-1).value:
            raise OSError(ctypes.get_errno(), "cannot map table to check cache residency")
        try:
            vector = (ctypes.c_ubyte * ((size + page - 1) // page))()
            if libc.mincore(address, size, vector):
                raise OSError(ctypes.get_errno(), "cannot check table cache residency")
            return min(size, sum(v & 1 for v in vector) * page)
        finally:
            libc.munmap(address, size)


def warm_table(config, proc, deadline=None):
    if argument(config["args"], "--ple-io") != "ram":
        return
    path = Path(argument(config["args"], "--ple-gguf"))
    size = path.stat().st_size
    with open("/proc/meminfo") as handle:
        available = next(int(line.split()[1]) * 1024 for line in handle if line.startswith("MemAvailable:"))
    if available + cached_bytes(path) < size:
        raise RuntimeError("--ple-io ram requested but available host RAM cannot retain the PLE table after model load")
    started = time.monotonic()
    buffer = bytearray(16 * 2**20)
    total = 0
    with path.open("rb", buffering=0) as handle:
        while True:
            if deadline is not None and time.monotonic() >= deadline:
                raise RuntimeError("Strata load timed out during PLE table warmup")
            if proc.poll() is not None:
                raise RuntimeError(f"Strata server exited with {proc.returncode} during table warmup")
            count = handle.readinto(buffer)
            if not count:
                break
            total += count
    if total != size or path.stat().st_size != size:
        raise RuntimeError("PLE table changed during warmup")
    resident = cached_bytes(path)
    event("table_warm", bytes=total, cached_bytes=resident, seconds=round(time.monotonic() - started, 1))
    if resident != size:
        raise RuntimeError(f"--ple-io ram warmup completed but only {resident}/{size} bytes remain cached")


def serve(python, path, config, env):
    started = time.monotonic()
    deadline = started + float(env.get("LOAD_TIMEOUT", "1800"))
    # tee stays alive with the server after setup exits. No daemon thread owns a pipe needed by inference.
    command = [python, "-u", "serve/server.py", "--engine", "strata", "--config", str(path),
               "--host", "127.0.0.1", "--port", env["STRATA_PORT"]]
    pipeline = shlex.join(command) + " 2>&1 | tee -a " + shlex.quote(ENGINE_LOG)
    proc = subprocess.Popen(["bash", "-o", "pipefail", "-c", pipeline], cwd=SOURCE, env=env,
                            stderr=subprocess.STDOUT, start_new_session=True)
    try:
        while time.monotonic() < deadline:
            if proc.poll() is not None:
                raise RuntimeError(f"Strata server exited with {proc.returncode}: {log_tail()}")
            try:
                health = read_json(env["STRATA_PORT"], "/health")
            except (OSError, ValueError):
                time.sleep(3)
                continue
            modalities = confirm_capabilities(health, read_json(env["STRATA_PORT"], "/v1/models"), env,
                                             read_json(env["STRATA_PORT"], "/metrics"))
            warm_table(config, proc, deadline)
            # Warmup may be lengthy; recheck the live service rather than announce a dead process.
            if proc.poll() is not None:
                raise RuntimeError(f"Strata server exited with {proc.returncode} before readiness")
            confirm_capabilities(read_json(env["STRATA_PORT"], "/health"), read_json(env["STRATA_PORT"], "/v1/models"),
                                 env, read_json(env["STRATA_PORT"], "/metrics"))
            if time.monotonic() >= deadline:
                raise RuntimeError("Strata load timed out before capability confirmation completed")
            event("server_ready", seconds=round(time.monotonic() - started, 1), modalities=modalities,
                  refusal_projection=env["STRATA_PROJECTION"] == "on", cmd=" ".join(config["args"]))
            return
        raise RuntimeError(f"Strata health/load timed out after {env.get('LOAD_TIMEOUT', '1800')}s: {log_tail()}")
    except BaseException:
        terminate(proc)
        raise


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    if env["STRATA_REPO"] != REPO or env["STRATA_QUANT"] not in ("IQ2_XS", "IQ3_XXS", "IQ3_S"):
        raise ValueError("Strata requires the original official repository and supported IQ quant")
    quant = env["STRATA_QUANT"]
    if env["STRATA_FILE"] != f"{quant}/Qwen3.8-Flash-Next-GSQ-RCO-{quant}-00001-of-00002.gguf":
        raise ValueError("Strata requires the exact official first-shard path")
    if not stop_engines(env["STRATA_PORT"]):
        raise RuntimeError(f"port {env['STRATA_PORT']} is still busy after stopping all engines")
    # Isolate upstream's persistent setup choices, and clear request auth inherited from Colab exec.
    # Upstream rejects an explicitly empty STRATA_API_KEY: remove the cleared Colab variable in the child env.
    env.pop("STRATA_API_KEY", None)
    env.pop("STRATA_PREBUILT_URL", None)
    env.update(HF_HUB_DISABLE_XET="1", HF_HUB_DISABLE_PROGRESS_BARS="1",
               XDG_CONFIG_HOME=str(ROOT / "strata-settings"))
    python, low_ram, gpu, install_started = prepare(env)
    branches = parallel_setup(python, env, low_ram, gpu, install_started)
    path, config = finalize_model(python, env, low_ram, branches["engine"]["source"])
    serve(python, path, config, env)


if __name__ == "__main__":
    def interrupted(signum, frame):
        raise RuntimeError(f"Strata setup interrupted by signal {signum}")

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    try:
        main()
    except Exception as exc:
        event("error", message=str(exc)[:1000], log_tail=log_tail())
        sys.exit(1)
