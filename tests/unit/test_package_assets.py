import subprocess
import sys
import tarfile
import zipfile

import pytest

from colabinet.config import REPO_ROOT


@pytest.fixture(scope="module")
def wheel(tmp_path_factory):
    directory = tmp_path_factory.mktemp("wheel")
    result = subprocess.run(["uv", "build", "--offline", "--wheel", "--out-dir", str(directory)],
                            cwd=REPO_ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    return next(directory.glob("*.whl"))


def test_source_distribution_excludes_runtime_data(wheel):
    directory = wheel.parent
    result = subprocess.run(["uv", "build", "--offline", "--sdist", "--out-dir", str(directory)],
                            cwd=REPO_ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    with tarfile.open(next(directory.glob("*.tar.gz"))) as archive:
        paths = [member.name.split("/", 1)[-1] for member in archive.getmembers()]
        assert "pyproject.toml" in paths and "presets.toml" in paths
        assert not any(path.startswith((".superpowers/", ".worktrees/", "data/", ".venv/")) for path in paths)


def test_wheel_contains_console_assets(wheel):
    paths = ["templates/base.html", "templates/macros.html", "chat_templates/qwen38_effort_map.jinja",
             "static/app.css", "static/app.js", "presets.toml"]
    paths += [f"templates/pages/{page}.html" for page in
              ("overview", "sessions", "session", "new", "playground", "metrics", "models", "settings")]
    paths += [f"static/vendor/{asset}" for asset in
              ("basecoat-nova.cdn.min.css", "basecoat.all.min.js", "basecoat.chart.min.js",
               "chart.umd.min.js", "htmx.min.js", "LICENSE.basecoat.md")]
    with zipfile.ZipFile(wheel) as archive:
        for path in paths:
            assert archive.read("colabinet/" + path), path


def test_installed_package_has_usable_default_presets(wheel, tmp_path):
    with zipfile.ZipFile(wheel) as archive:
        archive.extractall(tmp_path / "installed")
    code = (
        "import sys; sys.path.insert(0, sys.argv[1]); "
        "from colabinet.config import Settings; from colabinet.presets import load_presets; "
        "settings = Settings(); assert settings.presets_path.is_file(); "
        "assert 'qwen38-27b-abl-exl3-4bpw' in load_presets(settings.presets_path).presets"
    )
    result = subprocess.run([sys.executable, "-I", "-c", code, str(tmp_path / "installed")],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
