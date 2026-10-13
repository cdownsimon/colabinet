# Runs ON the Colab VM via `colab exec` (GPU check, setup polling, keep-alive). Stateless: the caller passes the
# byte offsets it has already read. Prints exactly one line: COLABINET_STATUS {json}.
# Each exec also counts as kernel activity, which keeps Colab from reclaiming the VM as idle.
import json
import os
import shutil
import subprocess
import urllib.request

E = os.environ.get
ROOT = E("COLABINET_ROOT", "/content/colabinet")


def read_new(path, offset, max_lines=200):
    """Complete lines appended since `offset`; returns (lines, new_offset). Restarts if the file shrank."""
    try:
        size = os.path.getsize(path)
    except OSError:
        return [], 0
    if size < offset:
        offset = 0
    with open(path, "rb") as f:
        f.seek(offset)
        data = f.read(4 * 1024 * 1024)
    cut = data.rfind(b"\n") + 1
    lines = data[:cut].decode("utf-8", "replace").splitlines()
    return lines[-max_lines:], offset + cut


def gpu():
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,memory.used,driver_version,utilization.gpu,temperature.gpu",
             "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=15,
        ).stdout.strip().splitlines()
    except (OSError, subprocess.TimeoutExpired):
        return None
    if not out:
        return None
    parts = [p.strip() for p in out[0].split(",")]
    info = {"name": parts[0]}
    if len(parts) > 1 and parts[1].isdigit():
        info["vram_mib"] = int(parts[1])
    if len(parts) > 2 and parts[2].isdigit():
        info["used_mib"] = int(parts[2])
    if len(parts) > 3:
        info["driver"] = parts[3]
    for index, key in ((4, "utilization_pct"), (5, "temperature_c")):
        if len(parts) > index and parts[index].isdigit():
            info[key] = int(parts[index])
    return info


def cpu():
    info = {"percent": None, "cores": os.cpu_count()}
    try:
        with open("/proc/stat") as stat_file:
            ticks = [int(value) for value in stat_file.readline().split()[1:9]]
        total, idle = sum(ticks), sum(ticks[3:5])
        previous_path = os.path.join(ROOT, "cpu-sample.json")
        try:
            with open(previous_path) as previous_file:
                previous = json.load(previous_file)
            elapsed, idle_delta = total - previous["total"], idle - previous["idle"]
            if elapsed > 0 and 0 <= idle_delta <= elapsed:
                info["percent"] = round(100 * (elapsed - idle_delta) / elapsed, 1)
        except (OSError, ValueError, KeyError, TypeError):
            pass
        with open(previous_path, "w") as sample_file:
            json.dump({"total": total, "idle": idle}, sample_file)
    except (OSError, ValueError):
        pass
    return info


def ram():
    total = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    info = {"total_bytes": total, "used_bytes": None, "available_bytes": None}
    try:
        with open("/proc/meminfo") as memory_file:
            values = {line.split()[0].rstrip(":"): int(line.split()[1]) * 1024 for line in memory_file}
        total, available = values["MemTotal"], values["MemAvailable"]
        info.update(total_bytes=total, used_bytes=max(0, total - available), available_bytes=available)
    except (OSError, ValueError, KeyError, IndexError):
        pass
    return info


def disk():
    try:
        usage = shutil.disk_usage(os.path.dirname(ROOT) or "/")
        return {"total_bytes": usage.total, "used_bytes": usage.used, "free_bytes": usage.free}
    except OSError:
        return None


def healthy(port, engine="", model=""):
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=3) as r:
            if r.status != 200:
                return False
        if engine == "strata":
            # /health covers the frontend, which survives native engine death; require a loaded model too.
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/models", timeout=3) as r:
                if r.status != 200:
                    return False
                data = json.load(r).get("data")
            return isinstance(data, list) and any(isinstance(entry, dict) and entry.get("id")
                and (not model or entry["id"] == model) for entry in data)
        return True
    except Exception:
        return False


def marker():
    try:
        with open(os.path.join(ROOT, "session.json")) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


events_lines, events_offset = read_new(E("EVENTS", os.path.join(ROOT, "setup.jsonl")), int(E("EVENTS_OFFSET", "0")))
log_lines, log_offset = read_new(E("ENGINE_LOG", os.path.join(ROOT, "engine.log")), int(E("LOG_OFFSET", "0")))
events = []
for line in events_lines:
    try:
        events.append(json.loads(line))
    except ValueError:
        pass
resources = {"cpu": cpu(), "ram": ram(), "disk": disk(), "gpu": gpu()}
status = {
    "events": events,
    "events_offset": events_offset,
    "log_lines": log_lines,
    "log_offset": log_offset,
    "engine_healthy": healthy(int(E("VM_PORT", "8090")), E("ENGINE_KIND", ""), E("ENGINE_MODEL", "")),
    "marker": marker(),
    "gpu": resources["gpu"],
    "ram_gb": round(resources["ram"]["total_bytes"] / 1e9, 1),
    "disk_free_gb": round(resources["disk"]["free_bytes"] / 1e9, 1) if resources["disk"] else None,
    "resources": resources,
}
if E("WITH_TORCH") == "1":
    try:
        out = subprocess.run(["python3", "-c", "import torch; print(torch.__version__, torch.version.cuda)"],
                             capture_output=True, text=True, timeout=60).stdout.split()
        status["torch"], status["torch_cuda"] = (out + [None, None])[:2]
    except (OSError, subprocess.TimeoutExpired):
        pass
print("COLABINET_STATUS " + json.dumps(status))
