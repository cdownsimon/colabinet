import builtins
import io
import runpy
import shutil
import subprocess
from types import SimpleNamespace

import pytest

from colabinet.engines.base import VM_DIR


def vm_status(tmp_path, monkeypatch, stat, memory=None, gpu_output=None, disk_error=False):
    original_open = builtins.open

    def read_file(path, *args, **kwargs):
        if str(path) == "/proc/stat":
            if stat is None:
                raise FileNotFoundError(path)
            return io.StringIO(stat)
        if str(path) == "/proc/meminfo":
            if memory is None:
                raise FileNotFoundError(path)
            return io.StringIO(memory)
        return original_open(path, *args, **kwargs)

    def disk_usage(path):
        if disk_error:
            raise OSError("unavailable disk")
        return SimpleNamespace(total=214748364800, used=128849018880, free=85899345920)

    monkeypatch.setattr(builtins, "open", read_file)
    monkeypatch.setattr(shutil, "disk_usage", disk_usage)
    monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: subprocess.CompletedProcess(
        args[0], 0 if gpu_output else 1, stdout=gpu_output or "", stderr=""))
    monkeypatch.setenv("COLABINET_ROOT", str(tmp_path))
    monkeypatch.setenv("VM_PORT", "1")
    monkeypatch.setenv("WITH_TORCH", "0")
    return runpy.run_path(str(VM_DIR / "status.py"))["status"]


def test_cpu_usage_is_interval_based_and_excludes_guest_ticks(tmp_path, monkeypatch):
    first = vm_status(tmp_path, monkeypatch, "cpu 100 0 50 800 50 0 0 0 10 5\n")
    assert first.get("resources", {}).get("cpu", {}).get("percent") is None
    second = vm_status(tmp_path, monkeypatch, "cpu 140 0 70 870 70 0 0 0 20 10\n")
    assert second.get("resources", {}).get("cpu", {}).get("percent") == 40.0


def test_ram_usage_excludes_available_cache_and_disk_has_capacity(tmp_path, monkeypatch):
    status = vm_status(tmp_path, monkeypatch, None,
                       "MemTotal: 8388608 kB\nMemAvailable: 2097152 kB\nMemFree: 1048576 kB\n")
    resources = status.get("resources", {})
    assert resources.get("ram") == {
        "total_bytes": 8589934592, "used_bytes": 6442450944, "available_bytes": 2147483648,
    }
    assert resources.get("disk") == {
        "total_bytes": 214748364800, "used_bytes": 128849018880, "free_bytes": 85899345920,
    }


def test_gpu_utilization_temperature_and_vram_are_current(tmp_path, monkeypatch):
    status = vm_status(tmp_path, monkeypatch, None,
                       gpu_output="NVIDIA L4, 23034, 12000, 550.54, 72, 61\n")
    assert status["gpu"].get("utilization_pct") == 72
    assert status["gpu"].get("temperature_c") == 61
    assert status["gpu"]["used_mib"] == 12000
    assert status.get("resources", {}).get("gpu") == status["gpu"]


def test_unavailable_resources_are_unknown_not_zero(tmp_path, monkeypatch):
    status = vm_status(tmp_path, monkeypatch, None, disk_error=True)
    resources = status.get("resources")
    assert resources is not None
    assert resources["cpu"]["percent"] is None
    assert resources["ram"]["used_bytes"] is None
    assert resources["disk"] is None
    assert resources["gpu"] is None
    assert status["engine_healthy"] is False


@pytest.mark.parametrize("next_stat", ["cpu 5 0 5 10 0 0 0 0\n", "cpu 100 0 50 800 50 0 0 0\n"])
def test_reset_or_unchanged_cpu_counters_do_not_invent_usage(tmp_path, monkeypatch, next_stat):
    vm_status(tmp_path, monkeypatch, "cpu 100 0 50 800 50 0 0 0\n")
    status = vm_status(tmp_path, monkeypatch, next_stat)
    assert status.get("resources") is not None
    assert status["resources"]["cpu"]["percent"] is None


def test_optional_gpu_counters_do_not_hide_the_detected_gpu(tmp_path, monkeypatch):
    status = vm_status(tmp_path, monkeypatch, None,
                       gpu_output="NVIDIA L4, 23034, 0, 550.54, N/A, [Not Supported]\n")
    assert status["gpu"]["name"] == "NVIDIA L4"
    assert status["gpu"]["used_mib"] == 0
    assert status["gpu"].get("utilization_pct") is None
    assert status.get("resources") is not None
