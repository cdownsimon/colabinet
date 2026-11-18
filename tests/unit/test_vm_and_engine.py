import io
import json
import re
import runpy
import shlex
import subprocess
import sys
import tarfile
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from colabinet.cli_parse import parse_status
from colabinet.config import REPO_ROOT
from colabinet.engines import ENGINES, RunSpec
from colabinet.engines.base import VM_DIR
from colabinet.presets import load_presets

sys.path.insert(0, str(VM_DIR))
import setup_exllamav3  # noqa: E402
import setup_llamacpp  # noqa: E402

sys.path.pop(0)


def test_retry_clears_events_before_spawning(tmp_path, monkeypatch):
    events = tmp_path / "run-1.jsonl"
    events.write_text('{"event":"error","message":"old attempt"}\n')
    monkeypatch.setenv("EVENTS", str(events))
    monkeypatch.setenv("SCRIPT", str(tmp_path / "setup.py"))
    monkeypatch.setenv("OUT", str(tmp_path / "setup.log"))

    def spawn(*args, **kwargs):
        assert events.read_text() == ""
        kwargs["stdout"].close()
        return SimpleNamespace(pid=123)

    monkeypatch.setattr(subprocess, "Popen", spawn)
    runpy.run_path(str(VM_DIR / "bg.py"))


def run_status(tmp_path, **env):
    full = {"COLABINET_ROOT": str(tmp_path), "PATH": "/usr/bin:/bin", **{k: str(v) for k, v in env.items()}}
    out = subprocess.run([sys.executable, str(VM_DIR / "status.py")], env=full, capture_output=True, text=True)
    return parse_status(out.stdout)


def test_status_reads_new_complete_lines_only(tmp_path):
    ev = tmp_path / "run-1.jsonl"
    ev.write_text('{"event": "gpu_info"}\n{"event": "install_done"}\n{"event": "part')
    s = run_status(tmp_path, EVENTS=ev, VM_PORT=1)
    assert [e["event"] for e in s["events"]] == ["gpu_info", "install_done"]
    assert s["engine_healthy"] is False and s["marker"] is None and s["gpu"] is None and s["ram_gb"] > 0
    with ev.open("a") as f:
        f.write('ial"}\n')
    s2 = run_status(tmp_path, EVENTS=ev, EVENTS_OFFSET=s["events_offset"], VM_PORT=1)
    assert [e["event"] for e in s2["events"]] == ["partial"]


def test_status_marker_and_log(tmp_path):
    (tmp_path / "session.json").write_text(json.dumps({"session_id": 3, "endpoint": "ep"}))
    (tmp_path / "engine.log").write_text("loading\nready\n")
    s = run_status(tmp_path, VM_PORT=1)
    assert s["marker"] == {"session_id": 3, "endpoint": "ep"} and s["log_lines"] == ["loading", "ready"]


def test_build_command_matches_verified_flags():
    env = {"LLM_REPO": "org/repo", "LLM_FILE": "m.gguf", "LLM_PORT": "8090", "LLM_CTX": "262144",
           "LLM_KV": "q4_0", "LLM_PARALLEL": "1",
           "LLM_ALIAS": "a/b", "LLM_MMPROJ_FILE": "mm.gguf", "LLM_MMPROJ_GPU": "0",
           "LLM_TEMPLATE": "/content/colabinet/templates/t.jinja", "LLM_EXTRA_ARGS": '["--reasoning-budget", "4096"]'}
    cmd = setup_llamacpp.build_command(env, "/x/llama-server", models="/content/models")
    assert cmd[:3] == ["/x/llama-server", "-m", "/content/models/org%2Frepo/m.gguf"]
    joined = " ".join(cmd)
    for part in ("--host 127.0.0.1 --port 8090", "-c 262144", "-ctk q4_0 -ctv q4_0", "--alias a/b", "--jinja",
                 "--mmproj /content/models/org%2Frepo/mm.gguf --no-mmproj-offload",
                 "--chat-template-file /content/colabinet/templates/t.jinja", "--reasoning-budget 4096"):
        assert part in joined


@pytest.mark.parametrize("module", [setup_llamacpp, setup_exllamav3])
@pytest.mark.parametrize("process", [
    "/content/tabbyAPI/main.py",
    "/content/colabinet/Strata/.venv/bin/python -u serve/server.py --engine strata --config cfg.json",
    "/content/colabinet/Strata/engine/strata --serve",
    "/content/colabinet/Strata/engine/strata-vision --mmproj image.gguf",
])
def test_engine_switch_stops_other_engines(module, process, monkeypatch):
    running = True
    commands = []

    def pkill(command, **kwargs):
        nonlocal running
        commands.append((command, kwargs))
        if kwargs.get("shell"):
            # The first broad pkill matches its parent `sh -c` command, so the second never runs.
            return subprocess.CompletedProcess(command, -15, stdout="", stderr="")
        assert command[:2] == ["pkill", "-f"]
        matched = re.search(command[2], process) is not None
        if matched:
            running = False
        return subprocess.CompletedProcess(command, 0 if matched else 1, stdout="", stderr="")

    monkeypatch.setattr(module.subprocess, "run", pkill)
    monkeypatch.setattr(module, "port_busy", lambda port: running)
    monkeypatch.setattr(module.time, "sleep", lambda seconds: None)
    assert module.stop_engines("8090") is True
    for command, kwargs in commands:
        assert not kwargs.get("shell")
        assert not re.search(command[2], "pkill -f " + command[2])


def test_tabbyapi_exit_reports_this_runs_exception(tmp_path, monkeypatch):
    log = tmp_path / "engine.log"
    log.write_text("RuntimeError: an earlier run's failure\n")
    monkeypatch.setattr(setup_exllamav3, "ENGINE_LOG", str(log))
    start = setup_exllamav3.log_size()
    log.write_text(log.read_text() + (
        'INFO:     Loading model\n  File "model_ls.py", line 298, in _load_autosplit\n'
        '    raise RuntimeError("Insufficient VRAM in split for model and cache")\n'
        "RuntimeError: Insufficient VRAM in split for model and cache\n"
        "INFO:     exllamav3 version: 1.5.2\n"))
    assert setup_exllamav3.exit_reason(1, start) == (
        "TabbyAPI exited with 1: RuntimeError: Insufficient VRAM in split for model and cache")
    assert setup_exllamav3.exit_reason(1, log.stat().st_size) == "TabbyAPI exited with 1"
    assert "out-of-memory killer" in setup_exllamav3.exit_reason(-9, start)


def test_llama_server_exit_names_an_oom_kill_or_its_last_error(tmp_path, monkeypatch):
    log = tmp_path / "engine.log"
    log.write_text("main: loading model\nllama_model_load: error loading model: tensor data is not within bounds\n"
                   "srv    load_model: failed to load model\n")
    monkeypatch.setattr(setup_llamacpp, "ENGINE_LOG", str(log))
    assert setup_llamacpp.log_size() == log.stat().st_size
    assert setup_llamacpp.exit_reason(1, 0) == (
        "llama-server exited with 1: llama_model_load: error loading model: tensor data is not within bounds")
    # Measured on a CPU runtime: a 27B GGUF larger than its 13.6 GB RAM was killed with -9 while loading.
    assert setup_llamacpp.exit_reason(-9, 0) == (
        "llama-server was killed (SIGKILL), usually by the out-of-memory killer: not enough host RAM")


def test_llamacpp_env_from_preset():
    p = load_presets(REPO_ROOT / "presets.toml")
    spec = RunSpec.from_preset(p.presets["qwen38-27b-abl-q4k"], "L4", p.engines, ctx=65536)
    assert (spec.kv, spec.ctx, spec.release, spec.mmproj_gpu) == ("q4_0", 65536, "b12359", False)
    eng = ENGINES["llamacpp"]
    env = eng.setup_env(spec, 8090, run_id=4)
    assert env["SCRIPT"] == "/content/colabinet/setup_llamacpp.py"
    assert env["EVENTS"] == "/content/colabinet/run-4.jsonl" and env["LLM_KV"] == "q4_0"
    assert env["LLM_RELEASE"] == "b12359"
    assert eng.setup_env(replace(spec, release=None), 8090, run_id=4)["LLM_RELEASE"] == "b12359"
    assert eng.setup_env(replace(spec, release="b11273"), 8090, run_id=4)["LLM_RELEASE"] == "b11273"
    assert env["LLM_TEMPLATE"] == "/content/colabinet/templates/qwen38_effort_map.jinja"
    remote = [r for _, r in eng.uploads(spec)]
    assert remote == ["/content/colabinet/setup_llamacpp.py", "/content/colabinet/gguf_files.py",
                      "/content/colabinet/templates/qwen38_effort_map.jinja"]
    assert all(local.exists() for local, _ in eng.uploads(spec))


@pytest.fixture
def llama_archives(tmp_path, monkeypatch):
    downloads = []
    events = []

    def download(url, destination):
        release, filename = url.rsplit("/", 2)[-2:]
        suffix = f"{release}-bin-ubuntu-cuda-12.8-x64"
        if filename == f"llama-{suffix}.tar.gz":
            names = [f"llama-{release}/llama-server", f"llama-{release}/libllama.so"]
        else:
            assert filename == f"cudart-llama-{suffix}.tar.gz"
            names = [f"cudart-llama-{suffix}/libcudart.so.12"]
        with tarfile.open(destination, "w:gz") as archive:
            for name in names:
                data = release.encode()
                member = tarfile.TarInfo(name)
                member.size = len(data)
                archive.addfile(member, io.BytesIO(data))
        downloads.append(filename)

    monkeypatch.setattr(setup_llamacpp, "LLAMA_DIR", str(tmp_path))
    monkeypatch.setattr(setup_llamacpp, "event", lambda kind, **data: events.append((kind, data)))
    monkeypatch.setattr(setup_llamacpp.urllib.request, "urlretrieve", download)
    return downloads, events


def test_llamacpp_install_and_rollback_select_only_requested_release(tmp_path, llama_archives):
    downloads, events = llama_archives
    legacy_binary = tmp_path / "llama-b11273" / "llama-server"
    legacy_binary.parent.mkdir()
    legacy_binary.write_text("legacy")
    (tmp_path / ".installed-b11273").touch()

    for release in ("b11273", "b12359", "b11273"):
        directory = tmp_path / release
        server, libraries = setup_llamacpp.install(release)
        assert Path(server) == directory / f"llama-{release}" / "llama-server"
        assert Path(server).read_text() == release
        assert [Path(path) for path in libraries] == [
            directory / f"cudart-llama-{release}-bin-ubuntu-cuda-12.8-x64", directory / f"llama-{release}",
        ]
        for library in directory.rglob("lib*.so*"):
            assert library.read_text() == release

    assert legacy_binary.read_text() == "legacy"
    assert len(downloads) == 4
    assert [(kind, data["release"], data["skipped"]) for kind, data in events] == [
        ("install_done", "b11273", False),
        ("install_done", "b12359", False),
        ("install_done", "b11273", True),
    ]


def test_llamacpp_interrupted_install_can_retry(tmp_path, llama_archives, monkeypatch):
    downloads, events = llama_archives
    setup_llamacpp.install("b11273")
    target = tmp_path / "b12359"
    staging = tmp_path / ".b12359.partial"
    # Model a previous VM process dying before it could clean up.
    staging.mkdir()
    (staging / "llama-server").write_text("incomplete")
    download = setup_llamacpp.urllib.request.urlretrieve

    def interrupt_runtime_download(url, destination):
        if "/cudart-" in url:
            assert not target.exists()
            destination.write_bytes(b"partial archive")
            raise InterruptedError("download interrupted")
        download(url, destination)

    with monkeypatch.context() as interrupted:
        interrupted.setattr(setup_llamacpp.urllib.request, "urlretrieve", interrupt_runtime_download)
        with pytest.raises(InterruptedError, match="download interrupted"):
            setup_llamacpp.install("b12359")
    assert not target.exists()
    assert not staging.exists()
    assert [data["release"] for _, data in events] == ["b11273"]

    server, _ = setup_llamacpp.install("b12359")
    assert Path(server).read_text() == "b12359" and len(downloads) == 5
    assert not staging.exists()
    assert not list(target.rglob("*.tar.gz"))
    old_server, _ = setup_llamacpp.install("b11273")
    assert Path(old_server).read_text() == "b11273" and events[-1][1]["skipped"]


def test_split_download_fetches_missing_companions_when_first_shard_exists(tmp_path, monkeypatch):
    first = "Q4/model-00001-of-00003.gguf"
    repo_dir = tmp_path / "org%2Frepo"
    (repo_dir / "Q4").mkdir(parents=True)
    (repo_dir / first).write_bytes(b"first")
    monkeypatch.setattr(setup_llamacpp, "MODELS", str(tmp_path))
    monkeypatch.setattr(setup_llamacpp, "event", lambda *a, **kw: None)

    def download(cmd, **kw):
        args = shlex.split(cmd)
        assert args[:3] == ["hf", "download", "org/repo"]
        paths = args[3:args.index("--local-dir")]
        assert paths == ["Q4/model-00002-of-00003.gguf", "Q4/model-00003-of-00003.gguf"]
        for path in paths:
            (Path(args[args.index("--local-dir") + 1]) / path).write_bytes(b"companion")
        return 0, ""

    monkeypatch.setattr(setup_llamacpp, "sh", download)
    setup_llamacpp.download("org/repo", first)
    assert sorted(p.name for p in (repo_dir / "Q4").iterdir()) == [
        "model-00001-of-00003.gguf", "model-00002-of-00003.gguf", "model-00003-of-00003.gguf"]


def test_split_download_stops_on_missing_companion(tmp_path, monkeypatch):
    first = "model-00001-of-00002.gguf"
    repo_dir = tmp_path / "org%2Frepo"
    repo_dir.mkdir()
    (repo_dir / first).write_bytes(b"first")
    monkeypatch.setattr(setup_llamacpp, "MODELS", str(tmp_path))
    monkeypatch.setattr(setup_llamacpp, "event", lambda *a, **kw: None)
    monkeypatch.setattr(setup_llamacpp, "sh", lambda *a, **kw: (1, "missing shard"))
    with pytest.raises(RuntimeError, match="model-00002-of-00002.gguf"):
        setup_llamacpp.download("org/repo", first)


def test_same_projector_filename_is_isolated_between_repositories(tmp_path, monkeypatch):
    monkeypatch.setattr(setup_llamacpp, "MODELS", str(tmp_path))
    monkeypatch.setattr(setup_llamacpp, "event", lambda *a, **kw: None)
    downloaded = []

    def download(cmd, **kw):
        args = shlex.split(cmd)
        repo = args[2]
        dest = Path(args[args.index("--local-dir") + 1])
        dest.mkdir(parents=True, exist_ok=True)
        (dest / "mmproj-model-bf16.gguf").write_bytes(repo.encode())
        downloaded.append(repo)
        return 0, ""

    monkeypatch.setattr(setup_llamacpp, "sh", download)
    projectors = []
    for repo in ("org/27b", "org/flash"):
        setup_llamacpp.download(repo, "mmproj-model-bf16.gguf")
        env = {"LLM_REPO": repo, "LLM_FILE": "model.gguf", "LLM_PORT": "8090",
               "LLM_MMPROJ_FILE": "mmproj-model-bf16.gguf"}
        cmd = setup_llamacpp.build_command(env, "/x/llama-server", models=str(tmp_path))
        projector = Path(cmd[cmd.index("--mmproj") + 1])
        assert projector.read_bytes() == repo.encode()
        projectors.append(projector)
    assert downloaded == ["org/27b", "org/flash"]
    assert projectors[0] != projectors[1]


def test_prepare_creates_dirs_and_marker(tmp_path):
    root = tmp_path / "colabinet"
    marker = json.dumps({"session_id": 5, "endpoint": "e"})
    env = {"COLABINET_ROOT": str(root), "PATH": "/usr/bin:/bin", "MARKER": marker}
    out = subprocess.run([sys.executable, str(VM_DIR / "prepare.py")], env=env, capture_output=True, text=True)
    assert parse_status(out.stdout) == {"prepared": str(root), "marker": True}
    assert (root / "templates").is_dir()
    assert json.loads((root / "session.json").read_text()) == {"session_id": 5, "endpoint": "e"}
