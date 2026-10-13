"""Exercise first-run commands with an isolated CLI; no Google credentials or runtimes."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from colabinet.config import Settings
from colabinet.doctor import run


@pytest.fixture
def installation(tmp_path):
    binary = tmp_path / "custom colab"
    journal = tmp_path / "commands.jsonl"
    binary.write_text(
        f"#!{sys.executable}\n"
        "import json, os, sys\n"
        f"with open({str(journal)!r}, 'a') as log:\n"
        "    log.write(json.dumps(sys.argv[1:]) + '\\n')\n"
        "command = sys.argv[-1]\n"
        "if command == 'version':\n"
        "    print('Version: ' + os.environ.get('TEST_COLAB_VERSION', '0.7.4'))\n"
        "elif command == 'usage':\n"
        "    if os.environ.get('TEST_LOGIN_PROMPT'):\n"
        "        answer = input('Enter the authorization code: ')\n"
        "        if answer != 'test-code': sys.exit(3)\n"
        "    if os.environ.get('TEST_AUTH_FAIL'):\n"
        "        print('fake-sensitive-authorization-url')\n"
        "        sys.exit(1)\n"
        "    print('Current balance: 10.00 compute units\\nUsage rate: 0.00/hr\\nActive assignments: 0')\n"
        "else:\n"
        "    sys.exit(4)\n"
    )
    binary.chmod(0o755)
    key = tmp_path / "test identity"
    subprocess.run(["ssh-keygen", "-q", "-t", "ed25519", "-N", "", "-f", str(key)], check=True)
    return binary, key, journal


def invoke(*args, env=None, input=None, cwd=None):
    return subprocess.run(
        [sys.executable, "-m", "colabinet", *map(str, args)], capture_output=True, text=True,
        env={**os.environ, **(env or {})}, input=input, **({"stdin": subprocess.DEVNULL} if input is None else {}),
        timeout=15, cwd=cwd,
    )


def commands(journal):
    return [json.loads(line) for line in journal.read_text().splitlines()] if journal.exists() else []


def test_doctor_checks_account_without_provisioning_or_creating_console_state(installation, tmp_path):
    binary, key, journal = installation
    data = tmp_path / "console-state"
    result = invoke("doctor", "--colab-bin", binary, "--ssh-key", key, "--colab-auth", "adc", "--data-dir", data)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "ready" in result.stdout.lower()
    assert commands(journal) == [["--auth=adc", "version"], ["--auth=adc", "usage"]]
    assert not data.exists()


def test_offline_doctor_skips_account_checks(installation):
    binary, key, journal = installation
    result = invoke("doctor", "--offline", "--colab-bin", binary, "--ssh-key", key)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "skipped" in result.stdout.lower()
    assert commands(journal) == [["--auth=oauth2", "version"]]


@pytest.mark.parametrize("command", ["doctor", "login", "serve"])
def test_missing_cli_has_installation_instructions_instead_of_a_traceback(tmp_path, command):
    result = invoke(command, "--colab-bin", tmp_path / "missing")
    assert result.returncode == 1
    assert "uv tool install" in result.stdout + result.stderr
    assert "Traceback" not in result.stdout + result.stderr


@pytest.mark.parametrize("version", ["0.7.3", "0.7.4rc1", "development"])
def test_doctor_rejects_unsupported_cli_before_account_access(installation, version):
    binary, key, journal = installation
    result = invoke("doctor", "--colab-bin", binary, "--ssh-key", key, env={"TEST_COLAB_VERSION": version})
    assert result.returncode == 1
    assert "0.7.4" in result.stdout
    assert commands(journal) == [["--auth=oauth2", "version"]]


def test_doctor_reports_login_needed_without_printing_captured_auth_data(installation):
    binary, key, _ = installation
    result = invoke("doctor", "--colab-bin", binary, "--ssh-key", key, env={"TEST_AUTH_FAIL": "1"})
    assert result.returncode == 1
    assert "colabinet login" in result.stdout
    assert "fake-sensitive" not in result.stdout + result.stderr


@pytest.mark.parametrize("kind", ["missing", "rsa", "encrypted"])
def test_doctor_rejects_keys_that_cannot_be_used_unattended(installation, tmp_path, kind):
    binary, _, _ = installation
    key = tmp_path / kind
    if kind != "missing":
        subprocess.run(["ssh-keygen", "-q", "-t", "rsa" if kind == "rsa" else "ed25519",
                        "-N", "test-passphrase" if kind == "encrypted" else "", "-f", str(key)], check=True)
    result = invoke("doctor", "--offline", "--colab-bin", binary, "--ssh-key", key)
    assert result.returncode == 1
    assert "SSH" in result.stdout and "Ed25519" in result.stdout


def test_login_preserves_terminal_input_and_output_and_only_reads_usage(installation):
    binary, _, journal = installation
    result = invoke("login", "--colab-bin", binary, input="test-code\n", env={"TEST_LOGIN_PROMPT": "1"})
    assert result.returncode == 0, result.stdout + result.stderr
    assert "Enter the authorization code:" in result.stdout
    assert "Current balance:" in result.stdout
    assert commands(journal) == [["--auth=oauth2", "usage"]]


def test_login_propagates_cli_failure(installation):
    binary, _, _ = installation
    result = invoke("login", "--colab-bin", binary, env={"TEST_AUTH_FAIL": "1"})
    assert result.returncode == 1


def test_cli_installed_on_path_is_discovered(installation):
    binary, key, journal = installation
    binary = binary.rename(binary.with_name("colab"))
    result = invoke("doctor", "--offline", "--ssh-key", key,
                    env={"PATH": str(binary.parent) + os.pathsep + os.environ["PATH"]})
    assert result.returncode == 0, result.stdout + result.stderr
    assert str(binary) in result.stdout
    assert commands(journal) == [["--auth=oauth2", "version"]]


@pytest.mark.parametrize("path_has_another_colab", [False, True])
def test_explicit_relative_cli_path_does_not_select_another_executable(installation, tmp_path, path_has_another_colab):
    binary, key, journal = installation
    binary.rename(tmp_path / "colab")
    tools = tmp_path / "other-tools"
    tools.mkdir()
    if path_has_another_colab:
        decoy = tools / "colab"
        decoy.write_text("#!/bin/sh\nexit 7\n")
        decoy.chmod(0o755)
    result = invoke("doctor", "--offline", "--colab-bin", "./colab", "--ssh-key", key,
                    env={"PATH": f"{tools}:/usr/bin:/bin"}, cwd=tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr
    assert commands(journal) == [["--auth=oauth2", "version"]]


@pytest.mark.parametrize("loaded_in_agent", [False, True])
async def test_doctor_checks_agent_for_an_encrypted_default_key(installation, tmp_path, monkeypatch, loaded_in_agent):
    binary, _, _ = installation
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    key = tmp_path / ".ssh/id_ed25519"
    key.parent.mkdir()
    subprocess.run(["ssh-keygen", "-q", "-t", "ed25519", "-N", "test-passphrase", "-f", str(key)], check=True)
    tools = tmp_path / "agent-tools"
    tools.mkdir()
    ssh_add = tools / "ssh-add"
    pubkey = key.with_name(key.name + ".pub").read_text() if loaded_in_agent else ""
    ssh_add.write_text(f"#!{sys.executable}\nimport sys\nassert sys.argv[1:] == ['-L']\nprint({pubkey!r})\n")
    ssh_add.chmod(0o755)
    monkeypatch.setenv("PATH", str(tools) + os.pathsep + os.environ["PATH"])
    output = []
    rc = await run(Settings(colab_bin=str(binary), ssh_key=key), offline=True, out=output.append)
    assert rc == (0 if loaded_in_agent else 1), "\n".join(output)
    assert ("loaded in ssh-agent" in "\n".join(output)) == loaded_in_agent
