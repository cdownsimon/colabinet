import importlib.util
import io
import json
import os
import sys
from dataclasses import replace
from types import SimpleNamespace

import pytest

from colabinet.config import REPO_ROOT
from colabinet.engines import ENGINES, RunSpec
from colabinet.engines.base import VM_DIR
from colabinet.fit import estimate, kv_ratio
from colabinet.presets import PresetError, load_presets

module_spec = importlib.util.spec_from_file_location("setup_exllamav3", VM_DIR / "setup_exllamav3.py")
sx = importlib.util.module_from_spec(module_spec)
module_spec.loader.exec_module(sx)


COMMIT = "47281587fafbd69e50aaf77c7626f390811126b6"


@pytest.mark.parametrize("version,torch,cuda", [
    ("1.6.7", "2.11.0+cu128", "cu128"),
    ("1.6.7", "2.11.0+cu130", "cu132"),
    ("1.5.2", "2.11.0+cu130", "cu132"),
])
def test_wheel_pairings(version, torch, cuda):
    url, err = sx.wheel_url(version, torch, "cp313")
    assert err is None
    assert url == (f"https://github.com/turboderp-org/exllamav3/releases/download/v{version}/"
                   f"exllamav3-{version}%2B{cuda}.torch2.11.0-cp313-cp313-linux_x86_64.whl")


@pytest.mark.parametrize("torch,tag", [("2.14.1+cu130", "cp313"), ("2.11.0+cpu", "cp313"),
                                       ("2.11.0", "cp313"), ("2.11.0+cu128", "cp315")])
def test_unsupported_wheel_pairing(torch, tag):
    url, err = sx.wheel_url("1.6.7", torch, tag)
    assert url is None and torch in err


def test_install_isolated_from_host_and_rechecks_cached_pair(tmp_path, monkeypatch):
    venv = tmp_path / "private env"
    tabby = tmp_path / "tabby"
    (tabby / ".git").mkdir(parents=True)
    (tabby / f".installed-{COMMIT[:12]}-1.6.7").touch()  # Old host install.
    host = tmp_path / "host"
    host.mkdir()
    (host / "torch.py").write_text("__version__ = '2.14.1+cu130'")
    monkeypatch.setenv("PYTHONPATH", str(host))
    monkeypatch.setenv("EXL_TABBY_COMMIT", COMMIT)
    monkeypatch.setenv("EXL_WHEEL", "1.6.7")
    monkeypatch.delenv("EXL_TORCH", raising=False)
    monkeypatch.setattr(sx, "VENV", venv)
    monkeypatch.setattr(sx, "TABBY_DIR", str(tabby))
    monkeypatch.setattr(sx, "PY_TAG", "cp313")
    env = sx.private_env()
    assert "PYTHONPATH" not in env and env["PATH"].split(os.pathsep)[0] == str(venv / "bin")

    # Use a real private interpreter for the import check. Stub network installs with tiny distributions.
    site = venv / f"lib/python{sys.version_info.major}.{sys.version_info.minor}/site-packages"
    python = str(venv / "bin/python")
    version = "1.6.7+cu128.torch2.11.0"
    calls = []
    original_sh = sx.sh

    def run(args, **kwargs):
        calls.append(args)
        if args[:3] == [sys.executable, "-m", "venv"] or args[:2] == [python, "-c"]:
            return original_sh(args, **kwargs)
        if args == ["git", "rev-parse", "HEAD"]:
            return 0, COMMIT
        if args == [python, "-m", "pip", "--version"]:
            return 1, "No module named pip"
        if args[-1] == str(tabby):
            (site / "torch.py").write_text("__version__ = '2.11.0+cu128'")
            (site / "exllamav3_ext.py").touch()
            (site / "exllamav3.py").touch()  # Distribution metadata is the version authority.
            metadata = site / "exllamav3-1.6.7+cu128.torch2.11.0.dist-info"
            metadata.mkdir(exist_ok=True)
            (metadata / "METADATA").write_text(f"Name: exllamav3\nVersion: {version}\n")
        return 0, ""

    monkeypatch.setattr(sx, "sh", run)
    events = []
    monkeypatch.setattr(sx, "event", lambda kind, **data: events.append((kind, data)))
    sx.install(env)
    assert events[-1][0] == "install_done" and events[-1][1]["skipped"] is False
    assert [sys.executable, "-m", "venv", "--without-pip", str(venv)] in calls
    assert [sys.executable, "-m", "pip", "--isolated", "--python", python, "install", "pip"] in calls
    installs = [args for args in calls if args[:3] == [python, "-m", "pip"] and "install" in args]
    assert len(installs) == 2
    assert installs[0][-3:] == ["--index-url", "https://download.pytorch.org/whl/cu128", "torch==2.11.0+cu128"]
    assert installs[1][-3:] == ["torch==2.11.0+cu128", sx.wheel_url("1.6.7", "2.11.0+cu128", "cp313")[0], str(tabby)]

    calls.clear()
    sx.install(env)
    assert events[-1][1]["skipped"] is True
    assert len(calls) == 2 and calls[-1][:2] == [python, "-c"]

    version = "1.6.6+cu128.torch2.11.0"
    (site / "exllamav3_ext.py").unlink()
    with pytest.raises(RuntimeError, match="verifying private torch and exllamav3"):
        sx.install(env)
    assert not (venv / ".installed").exists()
    version = "1.6.7+cu128.torch2.11.0"
    sx.install(env)
    assert events[-1][1]["skipped"] is False


def test_download_uses_private_interpreter(tmp_path, monkeypatch):
    monkeypatch.setattr(sx, "VENV", tmp_path / "venv")
    monkeypatch.setattr(sx, "event", lambda *args, **kwargs: None)
    monkeypatch.setenv("PYTHONPATH", "/host/site-packages")
    dest = str(tmp_path / "model")

    def snapshot(args, env, **kwargs):
        assert args[:2] == [sx.VENV / "bin/python", "-c"]
        assert args[-3:] == ["org/model", "4bpw", dest]
        assert env["VIRTUAL_ENV"] == str(sx.VENV) and "PYTHONPATH" not in env
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(sx.subprocess, "run", snapshot)
    sx.download("org/model", "4bpw", dest, sx.private_env())
    assert (tmp_path / "model/.complete").exists()


def test_render_config():
    env = {"EXL_PORT": "8090", "EXL_MAX_SEQ_LEN": "65536", "EXL_CACHE_MODE": "8,8",
           "EXL_EXTRA": '{"model.chunk_size": "4096", "draft_model.draft_mode": "mtp"}'}
    text = sx.render_config(env, "Huihui-Qwen3.8-27B-abliterated-exl3-4.00bpw", "qwen38_effort_map")
    for line in ("network:", "  host: \"127.0.0.1\"", "  port: 8090", "  disable_auth: true",
                 "  max_seq_len: 65536", "  cache_size: 65536", "  cache_mode: \"8,8\"", "  chunk_size: 4096",
                 "  max_batch_size: 1", "  inline_model_loading: false", "  prompt_template: \"qwen38_effort_map\"",
                 "  draft_mode: \"mtp\"", "  warmup: true"):
        assert line in text.splitlines(), line


def test_model_dir_name():
    assert sx.model_dir_name("turboderp/Qwen3.8-27B-exl3", "4.00bpw") == "Qwen3.8-27B-exl3-4.00bpw"


def test_exl3_preset_env_and_uploads():
    presets = load_presets(REPO_ROOT / "presets.toml")
    spec = RunSpec.from_preset(presets.presets["qwen38-27b-abl-exl3-4bpw"], "L4", presets.engines)
    assert (spec.engine, spec.revision, spec.ctx, spec.kv) == ("exllamav3", "4.00bpw", 65536, "FP16")
    assert spec.release == f"1.6.7@{COMMIT}"
    eng = ENGINES["exllamav3"]
    env = eng.setup_env(spec, 8090, run_id=3)
    assert env["SCRIPT"] == "/content/colabinet/setup_exllamav3.py" and env["EXL_REVISION"] == "4.00bpw"
    assert (env["EXL_WHEEL"], env["EXL_TABBY_COMMIT"]) == ("1.6.7", COMMIT)
    assert eng.setup_env(replace(spec, release=None), 8090, 3) == env
    assert env["EXL_TEMPLATE"] == "/content/colabinet/templates/qwen38_effort_map.jinja"
    assert eng.inject_usage_option and all(local.exists() for local, _ in eng.uploads(spec))
    assert RunSpec.from_preset(presets.presets["qwen38-27b-abl-exl3-4bpw"], "A100-80", presets.engines).ctx == 262144


def test_exl3_preset_needs_revision(tmp_path):
    preset_file = tmp_path / "presets.toml"
    preset_file.write_text("[[preset]]\nid='a'\nengine='exllamav3'\nlabel='x'\nrepo='r'\n")
    with pytest.raises(PresetError, match="revision"):
        load_presets(preset_file)


def test_fit_with_tabby_cache_modes():
    assert kv_ratio("8,8") == 0.5 and kv_ratio("4,6") == pytest.approx(10 / 32) and kv_ratio("Q4") == 0.25
    assert estimate("L4", 13.2, 65536, "FP16", 65536).verdict == "fits"
    assert estimate("L4", 13.2, 262144, "FP16", 65536).verdict == "spills"


@pytest.mark.parametrize("key", ["network.host", "network.port", "network.api_servers", "logging.log_prompt"])
def test_config_rejects_unsafe_extra_options(key):
    env = {"EXL_PORT": "8090", "EXL_MAX_SEQ_LEN": "65536", "EXL_EXTRA": '{"' + key + '": "true"}'}
    with pytest.raises(ValueError):
        sx.render_config(env, "model", None)


def test_optional_env_clears_previous_template():
    presets = load_presets(REPO_ROOT / "presets.toml")
    spec = RunSpec.from_preset(presets.presets["qwen38-27b-abl-exl3-4bpw"], "L4", presets.engines)
    spec = replace(spec, template=None)
    assert ENGINES["exllamav3"].setup_env(spec, 8090, 1)["EXL_TEMPLATE"] == ""


@pytest.mark.parametrize("gpu", ["L4", "A100", "A100-80", "G4"])
def test_qwen_preset_enables_vision_with_weights_in_system_ram(gpu):
    presets = load_presets(REPO_ROOT / "presets.toml")
    spec = RunSpec.from_preset(presets.presets["qwen38-27b-abl-exl3-4bpw"], gpu, presets.engines)
    env = ENGINES["exllamav3"].setup_env(spec, 8090, 1)
    config = sx.render_config(env, "model", "qwen38_effort_map")
    assert "  vision: true" in config.splitlines()
    assert "  vision_offload: true" in config.splitlines()


def test_custom_text_model_explicitly_clears_vision_environment():
    spec = RunSpec(engine="exllamav3", repo="org/text-model", file=None, alias="text", ctx=8192, kv="FP16")
    env = ENGINES["exllamav3"].setup_env(spec, 8090, 1)
    assert env["EXL_VISION"] == "0"
    assert env["EXL_VISION_OFFLOAD"] == "0"


@pytest.mark.parametrize("key", ["model.vision", "model.vision_offload"])
def test_vision_options_reject_non_boolean_overrides(key):
    env = {"EXL_PORT": "8090", "EXL_MAX_SEQ_LEN": "65536", "EXL_EXTRA": json.dumps({key: "no"})}
    with pytest.raises(ValueError, match="boolean"):
        sx.render_config(env, "model", None)


@pytest.mark.parametrize("key", ["vision", "vision_offload"])
@pytest.mark.parametrize("value", ["'false'", "1"])
def test_preset_rejects_non_boolean_vision_settings(tmp_path, key, value):
    preset_file = tmp_path / "presets.toml"
    preset_file.write_text("[[preset]]\nid='a'\nengine='exllamav3'\nlabel='x'\nrepo='r'\nrevision='main'\n"
                           f"{key}={value}\n")
    with pytest.raises(PresetError, match=f"{key} must be a boolean"):
        load_presets(preset_file)


@pytest.fixture
def setup_vm(tmp_path, monkeypatch):
    presets = load_presets(REPO_ROOT / "presets.toml")
    spec = RunSpec.from_preset(presets.presets["qwen38-27b-abl-exl3-4bpw"], "L4", presets.engines)
    monkeypatch.setattr(sx, "ROOT", str(tmp_path))
    monkeypatch.setattr(sx, "TABBY_DIR", str(tmp_path / "tabby"))
    monkeypatch.setattr(sx, "ENGINE_LOG", str(tmp_path / "engine.log"))
    monkeypatch.setattr(sx, "EVENTS", str(tmp_path / "events.jsonl"))
    (tmp_path / "tabby").mkdir()
    monkeypatch.setattr(sx, "sh", lambda *args, **kwargs: (0, "1024"))
    monkeypatch.setattr(sx, "stop_engines", lambda port: True)
    monkeypatch.setattr(sx, "install", lambda env: None)
    monkeypatch.setattr(sx, "download", lambda *args: None)

    def spawn(cmd, **kwargs):
        assert cmd == [sx.VENV / "bin/python", str(tmp_path / "tabby/main.py")]
        assert kwargs["env"]["VIRTUAL_ENV"] == str(sx.VENV) and "PYTHONPATH" not in kwargs["env"]
        kwargs["stdout"].close()
        return SimpleNamespace(poll=lambda: None)

    monkeypatch.setattr(sx.subprocess, "Popen", spawn)

    def run(extra_args, reported_vision):
        env = ENGINES["exllamav3"].setup_env(replace(spec, extra_args=extra_args), 8090, 1)
        for key, value in env.items():
            monkeypatch.setenv(key, value)

        def urlopen(url, timeout):
            if url == "http://127.0.0.1:8090/health":
                response = io.BytesIO(b'{"status":"healthy"}')
                response.status = 200
                return response
            if url == "http://127.0.0.1:8090/props":
                props = {"total_slots": 1, "model_path": "/models/qwen", "chat_template": None,
                         "default_generation_settings": {"n_ctx": 65536}, "modalities": reported_vision}
                return io.BytesIO(json.dumps(props).encode())
            raise AssertionError(f"unexpected engine URL: {url}")

        monkeypatch.setattr(sx.urllib.request, "urlopen", urlopen)
        sx.main()
        return [json.loads(line) for line in (tmp_path / "events.jsonl").read_text().splitlines()]

    return run


@pytest.mark.parametrize("extra_args,reported,expected", [
    ((), {"vision": True}, {"vision": True}),
    (("model.vision=false",), {"vision": True}, {"vision": True}),
    (("model.vision=false",), {"vision": False}, {"vision": False}),
    (("model.vision=false",), None, None),
    (("model.vision=false",), {"vision": "false"}, None),
])
def test_setup_reports_loaded_model_modalities(setup_vm, extra_args, reported, expected):
    events = setup_vm(extra_args, reported)
    ready = next(event for event in events if event["event"] == "server_ready")
    assert ready["modalities"] == expected


@pytest.mark.parametrize("reported", [{"vision": False}, None])
def test_setup_refuses_to_report_ready_when_requested_vision_is_unavailable(setup_vm, reported):
    with pytest.raises(RuntimeError, match="vision"):
        setup_vm((), reported)


@pytest.mark.parametrize("error", [OSError("connection refused"), TimeoutError("timed out"),
                                  ValueError("invalid JSON")])
def test_unreachable_or_invalid_props_leave_modalities_unknown(monkeypatch, error):
    def urlopen(url, timeout):
        raise error

    monkeypatch.setattr(sx.urllib.request, "urlopen", urlopen)
    assert sx.read_modalities(8090) is None
