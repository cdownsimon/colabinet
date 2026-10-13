import asyncio
import shlex
import subprocess
import sys
from pathlib import Path

import httpx
import pytest

from colabinet.config import Settings
from colabinet.net import first_free, start_server, stop_server
from colabinet.tunnel import LocalTunnels, kill_if_owned, ssh_command
from tests.fakes import fake_engine


def test_ssh_command_is_the_verified_one():
    cmd = ssh_command("/u/.local/bin/colab", "llm", 23000, 8090, Path("/u/.ssh/id_ed25519"))
    joined = " ".join(cmd)
    assert cmd[:2] == ["ssh", "-N"] and "-f" not in cmd  # a child we own, never forked into the background
    assert "-L 127.0.0.1:23000:127.0.0.1:8090" in joined
    assert "ProxyCommand=/u/.local/bin/colab ssh --proxy-mode -s llm" in joined
    assert "-i /u/.ssh/id_ed25519" in joined and cmd[-1] == "root@colab-runtime"


def test_ssh_proxy_uses_the_selected_auth_and_identity_with_quoted_paths():
    cmd = ssh_command("/tools with spaces/colab", "llm", 23000, 8090, Path("/keys/test identity"), auth="adc")
    proxy = next(arg.removeprefix("ProxyCommand=") for arg in cmd if arg.startswith("ProxyCommand="))
    assert shlex.split(proxy) == ["/tools with spaces/colab", "--auth=adc", "ssh", "--proxy-mode", "-s", "llm",
                                  "-i", "/keys/test identity"]
    assert cmd[cmd.index("-i") + 1] == "/keys/test identity"


def test_default_identity_keeps_the_public_key_path_for_agent_auth(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    key = tmp_path / ".ssh/id_ed25519"
    key.parent.mkdir()
    key.with_name(key.name + ".pub").write_text("ssh-ed25519 test-public-key\n")
    cmd = ssh_command("/tools/colab", "llm", 23000, 8090, key, auth="oauth2")
    proxy = next(arg.removeprefix("ProxyCommand=") for arg in cmd if arg.startswith("ProxyCommand="))
    assert shlex.split(proxy) == ["/tools/colab", "--auth=oauth2", "ssh", "--proxy-mode", "-s", "llm"]
    assert cmd[cmd.index("-i") + 1] == str(key)


@pytest.mark.parametrize("default_public_exists", [False, True])
def test_symlinked_default_identity_uses_the_cli_public_key_location(tmp_path, monkeypatch, default_public_exists):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    default_key = tmp_path / ".ssh/id_ed25519"
    default_key.parent.mkdir()
    target = tmp_path / "private-identity"
    target.write_text("test-private-key-placeholder")
    default_key.symlink_to(target)
    public_key = default_key if default_public_exists else target
    public_key.with_name(public_key.name + ".pub").write_text("ssh-ed25519 test-public-key\n")
    key = Settings(ssh_key=default_key).ssh_key
    cmd = ssh_command("/tools/colab", "llm", 23000, 8090, key, auth="oauth2")
    proxy = next(arg.removeprefix("ProxyCommand=") for arg in cmd if arg.startswith("ProxyCommand="))
    assert ("-i" in shlex.split(proxy)) == (not default_public_exists)


def test_kill_if_owned_checks_the_command_line():
    p = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)", "--proxy-mode -s llm"],
                         start_new_session=True)
    try:
        assert not kill_if_owned(p.pid, "--proxy-mode -s other")  # someone else's process: left alone
        assert p.poll() is None
        assert kill_if_owned(p.pid, "--proxy-mode -s llm")
        assert p.wait(5) is not None
    finally:
        p.kill()
    assert not kill_if_owned(None, "x") and not kill_if_owned(-3, "x")


async def test_local_tunnel_relays_http():
    vm_port, hidden = first_free(24000, 24999), first_free(27000, 27999)
    server, task = await start_server(fake_engine.app, vm_port)
    tunnels = LocalTunnels()
    pid = await tunnels.start("llm", hidden, vm_port)
    try:
        async with httpx.AsyncClient() as c:
            r = await c.get(f"http://127.0.0.1:{hidden}/health")
        assert r.status_code == 200 and tunnels.alive(pid, "llm")
    finally:
        await tunnels.stop(pid, "llm")
        await stop_server(server, task)
    assert not tunnels.alive(pid, "llm")
    await asyncio.sleep(0)
