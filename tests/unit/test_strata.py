import builtins
import importlib.util
import io
import json
import re
import signal
import threading
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from colabinet.engines.base import VM_DIR, RunSpec
from colabinet.engines.strata import FILES, RELEASE, REPO, StrataEngine, validate_spec

module_spec = importlib.util.spec_from_file_location("setup_strata", VM_DIR / "setup_strata.py")
vm = importlib.util.module_from_spec(module_spec)
module_spec.loader.exec_module(vm)


def spec(**changes):
    return replace(RunSpec(engine="strata", repo=REPO, file=next(iter(FILES)), alias="qwen-original",
                           ctx=131072, kv="int8", release=RELEASE, vision=True), **changes)


@pytest.mark.parametrize("changes, message", [
    ({"repo": REPO.replace("ISTA-DASLab", "other")}, "original"),
    ({"file": next(iter(FILES)).replace("00001", "00002")}, "first-shard"),
    ({"file": "IQ4_XS/model.gguf"}, "first-shard"),
    ({"parallel": 2}, "parallel=1"),
    ({"kv": "Q8"}, "KV"),
    ({"ctx": 0}, "positive integer"),
    ({"ctx": 262145}, "positive integer"),
    ({"ctx": True}, "positive integer"),
    ({"template": "custom.jinja"}, "template/mmproj"),
    ({"mmproj": "different.gguf"}, "template/mmproj"),
    ({"vision_offload": True}, "offload"),
    ({"release": "main"}, "pinned"),
    ({"revision": "some-hf-revision"}, "revision"),
    ({"extra_args": ("--max-context", "512")}, "managed"),
    ({"extra_args": ("--vision",)}, "managed"),
    ({"extra_args": ("--control-vector-scaled=other.gguf:1",)}, "managed"),
    ({"extra_args": ("--host", "0.0.0.0")}, "managed"),
    ({"extra_args": ("--ple-io", "invalid")}, "requires"),
    ({"extra_args": ("--ple-io=ram",)}, "separate"),
    ({"extra_args": ("--low-ram",)}, "requires"),
    ({"extra_args": ("--low-ram", "invalid")}, "requires"),
    ({"extra_args": ("--low-ram", "on", "--low-ram", "off")}, "requires"),
    ({"extra_args": ("--low-ram=on",)}, "separate"),
])
def test_unsupported_requests_fail_before_setup(changes, message):
    with pytest.raises(ValueError, match=message):
        validate_spec(spec(**changes))


def config(vision=True, projection=False, ctx=131072):
    result = {
        "exe": "/content/colabinet/Strata/engine/strata", "model_name": "upstream-name", "args": [
            "--native", "/models/" + Path(next(iter(FILES))).name,
            "--max-context", str(ctx), "--kv", "int8", "--ple-gguf", "/models/table.gguf",
        ],
    }
    if vision:
        result["vision"] = {"gpu": True, "exe": "/engine/strata-vision"}
        result["args"] += ["--vision"]
    if projection:
        result["args"] += ["--control-vector-scaled", "/bundled/vector.gguf:1.0",
                           "--control-vector-layer-range", "4", "44", "--cvec-mode", "project",
                           "--cvec-dir", "per-layer"]
    return result


def runtime_env(**changes):
    return StrataEngine().setup_env(spec(**changes), 8090, 1)


def test_low_context_retains_requested_quantized_kv():
    upstream = config(ctx=4096)
    index = upstream["args"].index("--kv")
    del upstream["args"][index:index + 2]
    effective = vm.configure(upstream, runtime_env(ctx=4096, kv="k8v4"))
    assert vm.argument(effective["args"], "--kv") == "k8v4"


@pytest.mark.parametrize("edit, message", [
    (lambda c: c["args"].__setitem__(c["args"].index("--max-context") + 1, "65536"), "context"),
    (lambda c: c["args"].__setitem__(c["args"].index("--kv") + 1, "q4_0"), "KV"),
    (lambda c: c["vision"].__setitem__("gpu", False), "GPU vision"),
])
def test_setup_cannot_silently_change_requested_configuration(edit, message):
    upstream = config()
    edit(upstream)
    with pytest.raises(RuntimeError, match=message):
        vm.configure(upstream, runtime_env())


def test_projection_switching_rejects_stale_or_wrong_profile():
    with pytest.raises(RuntimeError, match="on/off"):
        vm.configure(config(projection=True), runtime_env(refusal_projection=False))
    wrong = config(projection=True)
    wrong["args"][wrong["args"].index("--cvec-mode") + 1] = "add"
    with pytest.raises(RuntimeError, match="per-layer"):
        vm.configure(wrong, runtime_env(refusal_projection=True))
    with pytest.raises(RuntimeError, match="retained vision"):
        vm.configure(config(), runtime_env(vision=False))


def capabilities(vision=True, projection=False):
    return ({"status": "ok", "model": "qwen-original", "max_context": 131072, "images": vision},
            {"data": [{"id": "qwen-original", "meta": {"n_ctx": 131072},
                       "architecture": {"input_modalities": ["text", "image"] if vision else ["text"]}}]},
            {"engine": {"kv": "int8", "cvec": "project:4-44" if projection else 0}})


@pytest.mark.parametrize("part, edit, message", [
    (0, lambda d: d.update(images=False), "vision"),
    (1, lambda d: d["data"][0]["architecture"].update(input_modalities=["text"]), "vision"),
    (1, lambda d: d.update(data=[]), "loaded model"),  # /health stays 200 after the upstream native process dies.
    (0, lambda d: d.update(max_context=65536), "context"),
    (2, lambda d: d["engine"].update(kv="q4_0"), "KV"),
    (2, lambda d: d["engine"].update(cvec="project:4-44"), "projection"),
])
def test_readiness_requires_actual_loaded_capabilities(part, edit, message):
    data = list(capabilities())
    edit(data[part])
    with pytest.raises(RuntimeError, match=message):
        vm.confirm_capabilities(data[0], data[1], runtime_env(), data[2])


@pytest.mark.parametrize("mode", ["capability", "timeout", "exit", "warmup"])
def test_startup_failure_cleans_entire_server_group(tmp_path, monkeypatch, mode):
    proc = SimpleNamespace(pid=4242, returncode=17 if mode == "exit" else None,
                           poll=lambda: 17 if mode == "exit" else None, wait=lambda: None)
    monkeypatch.setattr(vm.subprocess, "Popen", lambda *a, **kw: proc)
    killed = []
    monkeypatch.setattr(vm.os, "killpg", lambda pid, sig: killed.append((pid, sig)))
    monkeypatch.setattr(vm.time, "sleep", lambda seconds: None)
    emitted = []
    monkeypatch.setattr(vm, "event", lambda kind, **kw: emitted.append(kind))
    env = runtime_env()
    if mode == "timeout":
        env["LOAD_TIMEOUT"] = "0"
    health, models, metrics = capabilities()
    if mode == "capability":
        health["images"] = False
    monkeypatch.setattr(vm, "read_json", lambda port, path: {"/health": health, "/v1/models": models,
                                                           "/metrics": metrics}[path])
    if mode == "warmup":
        def fail_warm(*args):
            raise RuntimeError("table cannot remain resident")
        monkeypatch.setattr(vm, "warm_table", fail_warm)
    with pytest.raises(RuntimeError):
        vm.serve("python", tmp_path / "config.json", config(), env)
    assert (4242, signal.SIGTERM) in killed and (4242, signal.SIGKILL) in killed
    assert "server_ready" not in emitted


def test_reconfigure_clears_only_managed_shared_settings(tmp_path, monkeypatch):
    source, data = tmp_path / "Strata", tmp_path / "data"
    source.mkdir()
    model_dir = data / "models" / "IQ2_XS"
    model_dir.mkdir(parents=True)
    for shard in (1, 2):
        (model_dir / f"Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS-0000{shard}-of-00002.gguf").write_bytes(b"gguf")
    path = source / "strata-iq2_xs.json"
    path.write_text(json.dumps(config(projection=True)))
    sidecar = source / "strata-iq2_xs.shared-settings.json"
    sidecar.write_text('{"experimental_speed_projection":false}')
    unrelated = source / "strata-iq3_s.shared-settings.json"
    unrelated.write_text('{"temperature":0.7}')
    monkeypatch.setattr(vm, "SOURCE", source)
    monkeypatch.setattr(vm, "DATA", data)
    monkeypatch.setattr(vm, "run", lambda *a, **kw: None)
    monkeypatch.setattr(vm, "event", lambda *a, **kw: None)
    vm.finalize_model("python", runtime_env(refusal_projection=True))
    assert not sidecar.exists()
    assert unrelated.read_text() == '{"temperature":0.7}'


def test_storage_check_reserves_space_for_low_ram_pack(tmp_path, monkeypatch):
    monkeypatch.setattr(vm, "DATA", tmp_path)
    monkeypatch.setattr(vm, "event", lambda *args, **kwargs: None)
    monkeypatch.setattr(vm.shutil, "disk_usage", lambda path: SimpleNamespace(free=100e9))
    upstream = SimpleNamespace(MODELS={"IQ2_XS": {"download_gb": 68.0, "arena_gb": 35.5}})
    env = runtime_env()

    vm.check_storage(upstream, env, False)
    with pytest.raises(RuntimeError, match="114 GB free"):
        vm.check_storage(upstream, env, True)


def test_cold_setup_runs_engine_model_and_mtp_concurrently(monkeypatch):
    barrier = threading.Barrier(3, timeout=2)
    entered = []
    emitted = []

    def branch(name, result):
        def execute(*args):
            entered.append(name)
            barrier.wait()
            return result
        return execute

    monkeypatch.setattr(vm, "build_engine", branch("engine", {"seconds": 3.0, "skipped": False}))
    monkeypatch.setattr(vm, "prepare_model",
                        branch("model", {"seconds": 2.0, "skipped": False, "gb": 68.0}))
    monkeypatch.setattr(vm, "prepare_mtp", branch("mtp", {"seconds": 1.0, "skipped": False}))
    monkeypatch.setattr(vm, "event", lambda kind, **kw: emitted.append({"event": kind, **kw}))

    result = vm.parallel_setup("python", runtime_env(), False, vm.time.monotonic())

    assert set(entered) == {"engine", "model", "mtp"}
    assert set(result) == {"engine", "model", "mtp"}
    assert {row["task"] for row in emitted if row["event"] == "setup_task_start"} == set(entered)
    assert next(row for row in emitted if row["event"] == "parallel_setup_done")["branches"] == {
        "engine": 3.0, "model": 2.0, "mtp": 1.0}


def test_parallel_setup_cancels_siblings_and_reports_primary_failure(monkeypatch):
    barrier = threading.Barrier(3, timeout=2)
    cancelled = []

    def fail_engine(python, env, cancel):
        barrier.wait()
        raise RuntimeError("compiler exploded")

    def wait_for_cancel(name):
        def execute(*args):
            cancel = args[-1]
            barrier.wait()
            assert cancel.wait(2)
            cancelled.append(name)
            return {"seconds": 1.0, "skipped": False, **({"gb": 68.0} if name == "model" else {})}
        return execute

    monkeypatch.setattr(vm, "build_engine", fail_engine)
    monkeypatch.setattr(vm, "prepare_model", wait_for_cancel("model"))
    monkeypatch.setattr(vm, "prepare_mtp", wait_for_cancel("mtp"))
    monkeypatch.setattr(vm, "event", lambda *args, **kwargs: None)

    with pytest.raises(RuntimeError, match="engine: compiler exploded"):
        vm.parallel_setup("python", runtime_env(), False, vm.time.monotonic())

    assert set(cancelled) == {"model", "mtp"}


@pytest.mark.parametrize("available, message", [(0, "available host RAM"), (999999, "remain cached")])
def test_ram_warmup_refuses_insufficient_or_evicted_cache(tmp_path, monkeypatch, available, message):
    table = tmp_path / "table.gguf"
    table.write_bytes(b"table")
    original_open = builtins.open
    monkeypatch.setattr(builtins, "open", lambda path, *a, **kw: io.StringIO(f"MemAvailable: {available} kB\n")
                        if path == "/proc/meminfo" else original_open(path, *a, **kw))
    monkeypatch.setattr(vm, "cached_bytes", lambda path: 0)
    monkeypatch.setattr(vm, "event", lambda *a, **kw: None)
    cfg = {"args": ["--ple-io", "ram", "--ple-gguf", str(table)]}
    with pytest.raises(RuntimeError, match=message):
        vm.warm_table(cfg, SimpleNamespace(poll=lambda: None))


@pytest.mark.parametrize("process", [
    "/content/colabinet/Strata/.venv/bin/python -u serve/server.py --engine strata --config cfg.json",
    "/content/colabinet/Strata/engine/strata --serve --pack model",
    "/content/colabinet/Strata/engine/strata-vision --mmproj projector",
    "/content/llama/llama-server -m model", "/content/tabbyAPI/main.py",
])
def test_engine_switch_matches_all_serving_processes(process, monkeypatch):
    commands = []
    monkeypatch.setattr(vm.subprocess, "run", lambda cmd, **kw: commands.append((cmd, kw)))
    monkeypatch.setattr(vm, "port_busy", lambda port: False)
    monkeypatch.setattr(vm.time, "sleep", lambda seconds: None)
    assert vm.stop_engines("8090")
    matching = [cmd for cmd, kw in commands if re.search(cmd[2], process)]
    assert any("KILL" in cmd for cmd in matching)
    assert all(not kw.get("shell") for cmd, kw in commands)


@pytest.mark.parametrize("fault, message", [
    ("cpu_session", "NVIDIA GPU"), ("old_driver", "driver"), ("no_avx2", "AVX2"),
    ("memory", "cannot fit"), ("context_clamp", "clamp"),
])
def test_hardware_policy_refuses_unsupported_vm_before_large_downloads(monkeypatch, fault, message):
    gpu = {"name": "A100", "vram_gb": 40, "driver": "580"}
    policy = SimpleNamespace(
        gpu_info=lambda: None if fault == "cpu_session" else gpu,
        gpu_problem=lambda g: None, driver_major=lambda g: 570 if fault == "old_driver" else 580,
        MIN_DRIVER=580, cpu_info=lambda: ("CPU", fault != "no_avx2", False),
        ram_gb=lambda: 64 if fault == "context_clamp" else 12 if fault == "memory" else 84,
        low_ram_needed=lambda q, ram: ram < 50,
        low_ram_fits=lambda q, ram, vram: fault != "memory",
        MODELS={"IQ3_S": {"ram_gb": 62, "arena_gb": 50.3}},
    )
    monkeypatch.setattr(vm, "event", lambda *a, **kw: None)
    with pytest.raises(RuntimeError, match=message):
        vm.check_hardware(policy, "IQ3_S", 262144, "auto")


@pytest.mark.parametrize("mode, ram", [("auto", 52.96), ("on", 12), ("off", 52)])
def test_explicit_expert_memory_mode_preserves_fit_guards(monkeypatch, mode, ram):
    policy = SimpleNamespace(
        gpu_info=lambda: {"name": "L4", "vram_gb": 22.5, "driver": "580"},
        gpu_problem=lambda gpu: None, driver_major=lambda gpu: 580, MIN_DRIVER=580,
        cpu_info=lambda: ("CPU", True, False), ram_gb=lambda: ram,
        low_ram_needed=lambda quant, ram: ram < 52.9,
        low_ram_fits=lambda quant, ram, vram: ram - 6 + max(0, vram - 5) >= 42.9,
        MODELS={"IQ3_XXS": {"ram_gb": 60, "arena_gb": 42.9}},
    )
    monkeypatch.setattr(vm, "event", lambda *args, **kwargs: None)
    with pytest.raises(RuntimeError):
        vm.check_hardware(policy, "IQ3_XXS", 32768, mode)
