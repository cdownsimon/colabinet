import contextlib
import importlib.metadata
import importlib.util
import io
import json
import shlex
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


def test_install_checks_distribution_version_without_module_attribute(tmp_path, monkeypatch):
    monkeypatch.setattr(sx, "ROOT", str(tmp_path))
    monkeypatch.setattr(sx, "TABBY_DIR", str(tmp_path / "tabby"))
    (tmp_path / "tabby" / ".git").mkdir(parents=True)
    for name, module in {"torch": SimpleNamespace(__version__="2.11.0+cu130"),
                         "exllamav3": SimpleNamespace(), "exllamav3_ext": SimpleNamespace()}.items():
        monkeypatch.setitem(sys.modules, name, module)
    original_version = importlib.metadata.version
    monkeypatch.setattr(importlib.metadata, "version",
                        lambda name: "1.5.2" if name == "exllamav3" else original_version(name))

    def run(command, **kwargs):
        parts = shlex.split(command)
        if len(parts) < 3 or parts[1] != "-c":
            return 0, ""
        output = io.StringIO()
        try:
            with contextlib.redirect_stdout(output):
                exec(parts[parts.index("-c") + 1], {})
        except AttributeError as error:
            return 1, str(error)
        return 0, output.getvalue()

    monkeypatch.setattr(sx, "sh", run)
    recorded = []
    monkeypatch.setattr(sx, "event", lambda kind, **values: recorded.append((kind, values)))
    sx.install({"EXL_TABBY_COMMIT": "be74bf0a00bcb3a518e6feb7606f150c189be637", "EXL_WHEEL": "1.5.2"})
    assert recorded[-1][0] == "install_done"


def test_wheel_url_matches_the_vm_torch():
    url, err = sx.wheel_url("1.5.2", "2.11.0+cu128", "cp313")
    assert err is None and url == ("https://github.com/turboderp-org/exllamav3/releases/download/v1.5.2/"
                                   "exllamav3-1.5.2%2Bcu128.torch2.11.0-cp313-cp313-linux_x86_64.whl")
    assert sx.wheel_url("1.5.2", "2.11.0+cpu", "cp313")[1].endswith("not a CUDA build")
    assert "no CUDA tag" in sx.wheel_url("1.5.2", "2.11.0", "cp313")[1]


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


def test_cuda13_torch211_uses_the_documented_cuda132_build():
    url, error = sx.wheel_url("1.5.2", "2.11.0+cu130", "cp313")
    assert error is None
    assert url.endswith("exllamav3-1.5.2%2Bcu132.torch2.11.0-cp313-cp313-linux_x86_64.whl")


def test_exl3_preset_env_and_uploads():
    presets = load_presets(REPO_ROOT / "presets.toml")
    spec = RunSpec.from_preset(presets.presets["qwen38-27b-abl-exl3-4bpw"], "L4", presets.engines)
    assert (spec.engine, spec.revision, spec.ctx, spec.kv) == ("exllamav3", "4.00bpw", 65536, "FP16")
    assert spec.release == "1.5.2@be74bf0a00bcb3a518e6feb7606f150c189be637"
    eng = ENGINES["exllamav3"]
    env = eng.setup_env(spec, 8090, run_id=3)
    assert env["SCRIPT"] == "/content/colabinet/setup_exllamav3.py" and env["EXL_REVISION"] == "4.00bpw"
    assert (env["EXL_WHEEL"], env["EXL_TABBY_COMMIT"][:7]) == ("1.5.2", "be74bf0")
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

    def spawn(*args, **kwargs):
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
