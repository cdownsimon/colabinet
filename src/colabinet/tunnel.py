"""SSH tunnels to Colab VMs (spec §4 tunnel.py).

One `ssh -N -L 127.0.0.1:HIDDEN:127.0.0.1:VM_PORT` child per session, bridged by `colab ssh --proxy-mode`. The
tunnel module only spawns and kills; the safe tunnel rule (fresh trusted listing showing NAME with the stored
endpoint, allowed states only) is enforced by the supervisor before it calls start(), because `--proxy-mode -s NAME`
creates a new paid VM whenever NAME is missing from the CLI's local store.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import shlex
import signal
import subprocess
from pathlib import Path
from typing import Protocol

from .config import default_ssh_public_key


def ssh_command(colab_bin: str, name: str, hidden_port: int, vm_port: int, key: Path,
                auth: str | None = None, prefix: list[str] | None = None) -> list[str]:
    proxy = [*(prefix if prefix is not None else [colab_bin, *([f"--auth={auth}"] if auth else [])]),
             "ssh", "--proxy-mode", "-s", name]
    if default_ssh_public_key(key) is None:
        proxy += ["-i", str(key)]
    return [
        "ssh", "-N",
        "-L", f"127.0.0.1:{hidden_port}:127.0.0.1:{vm_port}",
        "-o", f"ProxyCommand={shlex.join(proxy)}",
        "-o", "ExitOnForwardFailure=yes",
        "-o", "ServerAliveInterval=15", "-o", "ServerAliveCountMax=4", "-o", "ConnectTimeout=90",
        "-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=/dev/null", "-o", "LogLevel=ERROR",
        "-o", "IdentitiesOnly=yes", "-i", str(key),
        "root@colab-runtime",
    ]


def cmdline(pid: int) -> str | None:
    try:
        out = subprocess.run(["ps", "-o", "command=", "-p", str(pid)], capture_output=True, text=True, timeout=5)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return out.stdout.strip() or None


def kill_if_owned(pid: int | None, needle: str) -> bool:
    """Kill pid's process group only if its command line still contains `needle` (PIDs get reused)."""
    if not pid or pid <= 0:
        return False
    line = cmdline(pid)
    if not line or needle not in line:
        return False
    with contextlib.suppress(ProcessLookupError, PermissionError):
        os.killpg(pid, signal.SIGTERM)
    return True


class Tunnels(Protocol):
    async def start(self, name: str, hidden_port: int, vm_port: int) -> int: ...
    async def stop(self, pid: int | None, name: str) -> None: ...
    def alive(self, pid: int | None, name: str) -> bool: ...
    def last_error(self, name: str) -> str | None: ...


class SshTunnels:
    def __init__(self, colab_bin: str, key: Path, log_dir: Path, auth: str | None = None) -> None:
        self.colab_bin, self.key, self.log_dir = colab_bin, key, log_dir
        self.auth = auth
        self.command_prefix: list[str] | None = None

    @staticmethod
    def needle(name: str) -> str:
        return f"--proxy-mode -s {name}"

    async def start(self, name: str, hidden_port: int, vm_port: int) -> int:
        self.log_dir.mkdir(parents=True, exist_ok=True)
        log = open(self.log_dir / f"tunnel-{name}.log", "ab")
        proc = await asyncio.create_subprocess_exec(
            *ssh_command(self.colab_bin, name, hidden_port, vm_port, self.key, auth=self.auth,
                         prefix=self.command_prefix),
            stdin=asyncio.subprocess.DEVNULL, stdout=log, stderr=log, start_new_session=True,
        )
        log.close()
        return proc.pid

    async def stop(self, pid: int | None, name: str) -> None:
        await asyncio.to_thread(kill_if_owned, pid, self.needle(name))

    def alive(self, pid: int | None, name: str) -> bool:
        line = cmdline(pid) if pid else None
        return bool(line and self.needle(name) in line)

    def last_error(self, name: str) -> str | None:
        """The end of this session's ssh log (e.g. "Already-active SSH session (HTTP 429)")."""
        try:
            data = (self.log_dir / f"tunnel-{name}.log").read_bytes()[-2000:]
        except OSError:
            return None
        return data.decode("utf-8", "replace").strip() or None


class LocalTunnels:
    """In-process TCP relay HIDDEN -> 127.0.0.1:VM_PORT, used by tests (the fake 'VM' runs on this Mac)."""

    def __init__(self) -> None:
        self._servers: dict[int, asyncio.Server] = {}
        self._next = -1

    async def start(self, name: str, hidden_port: int, vm_port: int) -> int:
        async def pipe(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
            try:
                while data := await reader.read(65536):
                    writer.write(data)
                    await writer.drain()
            except (ConnectionError, asyncio.CancelledError):
                pass
            finally:
                writer.close()

        async def handle(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
            try:
                up_r, up_w = await asyncio.open_connection("127.0.0.1", vm_port)
            except OSError:
                writer.close()
                return
            await asyncio.gather(pipe(reader, up_w), pipe(up_r, writer))

        server = await asyncio.start_server(handle, "127.0.0.1", hidden_port)
        pid, self._next = self._next, self._next - 1
        self._servers[pid] = server
        return pid

    async def stop(self, pid: int | None, name: str) -> None:
        server = self._servers.pop(pid, None) if pid is not None else None
        if server:
            server.close()
            with contextlib.suppress(Exception):
                await asyncio.wait_for(server.wait_closed(), 2)

    def alive(self, pid: int | None, name: str) -> bool:
        return pid in self._servers

    def last_error(self, name: str) -> str | None:
        return None
