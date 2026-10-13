"""Async wrapper around the `colab` binary: the only module that runs it.

Background commands except `new` have a watchdog; on timeout the whole process group is killed. `colab new` is
spawned in its own process group and is never killed here (spec 5.2 step 1): killing it after assignment but before
the name is saved would orphan a billing VM. Interactive login inherits the terminal and has no watchdog.
"""

from __future__ import annotations

import asyncio
import os
import re
import shlex
import signal
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .cli_parse import Listing, Usage, make_listing, parse_status, parse_usage
from .config import cli_flags

SECRET_KEY = re.compile(r"(TOKEN|KEY|SECRET|PASSWORD)", re.I)
PROXY_TOKEN = re.compile(r"(colab-runtime-proxy-token=)[^&\s]+")

TIMEOUTS = {"sessions": 30.0, "usage": 30.0, "upload": 120.0, "exec": 90.0, "stop": 60.0}


@dataclass
class CmdResult:
    rc: int
    out: str
    timed_out: bool = False

    @property
    def ok(self) -> bool:
        return self.rc == 0 and not self.timed_out


@dataclass
class ExecResult(CmdResult):
    status: dict | None = None


def scrub(text: str) -> str:
    """Hide runtime proxy tokens that the CLI prints inside URLs (e.g. on a failed upload)."""
    return PROXY_TOKEN.sub(r"\1***", text)


def redact(args: list[str]) -> str:
    shown = []
    for a in args:
        if "=" in a and SECRET_KEY.search(a.split("=", 1)[0]):
            a = a.split("=", 1)[0] + "=***"
        shown.append(a)
    return " ".join(shown)


class ColabCLI:
    def __init__(self, binary: str, now: Callable[[], float], log: Callable[[str], None] | None = None,
                 timeouts: dict[str, float] | None = None, auth: str | None = None,
                 profile: Path | None = None, enabled: bool = True) -> None:
        self.binary = binary
        self.now = now
        self.log = log or (lambda _msg: None)
        self.timeouts = {**TIMEOUTS, **(timeouts or {})}
        self.auth_args = [f"--auth={auth}"] if auth else []
        self.auth, self.profile, self.enabled = auth, profile, enabled

    def for_profile(self, profile: Path | None, *, enabled: bool = True) -> ColabCLI:
        return ColabCLI(self.binary, self.now, self.log, self.timeouts, self.auth, profile, enabled)

    def command(self) -> list[str]:
        if self.profile is None:
            return [self.binary, *self.auth_args]
        try:
            with open(self.binary, "rb") as file:
                header = file.readline(4096).decode().strip()
            interpreter = shlex.split(header[2:]) if header.startswith("#!") else []
            if not interpreter or not any("python" in Path(part).name for part in interpreter):
                raise ValueError
        except (OSError, UnicodeError, ValueError):
            raise OSError("Account profiles require the official Python colab launcher.") from None
        return [*interpreter, str(Path(__file__).with_name("cli_profile.py")), str(self.profile),
                self.binary, *self.auth_args]

    def login(self) -> int:
        """Let the official CLI handle consent in the user's terminal. Usage does not allocate a runtime."""
        if not self.enabled:
            return 1
        return subprocess.run([*self.command(), "usage"], check=False).returncode

    async def spawn_login(self) -> asyncio.subprocess.Process:
        """An owned, interactive usage command for the local web UI. Never log its output or input."""
        if not self.enabled:
            raise OSError("Sign in to a Google account first.")
        return await asyncio.create_subprocess_exec(
            *self.command(), "usage", stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT,
            env={**os.environ, "PYTHONUNBUFFERED": "1"}, start_new_session=True,
        )

    async def identity(self) -> str | None:
        """Read only the account email from the CLI's diagnostic; never expose its raw output."""
        result = await self._run(["whoami"], self.timeouts["usage"])
        match = re.search(r"^Email:\s+([^\s<>]+@[^\s<>]+)\s*$", result.out, re.M) if result.ok else None
        return match[1] if match and len(match[1]) <= 320 else None

    async def _run(self, args: list[str], timeout: float) -> CmdResult:
        if not self.enabled and args[0] != "version":
            return CmdResult(1, "Sign in to a Google account in Settings.")
        self.log(f"colab {redact(args)}")
        proc = await asyncio.create_subprocess_exec(
            *self.command(), *args, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT,
            stdin=asyncio.subprocess.DEVNULL, start_new_session=True,
        )
        try:
            out, _ = await asyncio.wait_for(proc.communicate(), timeout)
        except TimeoutError:
            self._kill(proc)
            await proc.wait()
            self.log(f"colab {args[0]} timed out after {timeout:.0f}s")
            return CmdResult(-9, "", timed_out=True)
        except asyncio.CancelledError:  # app shutting down: don't leave a stray CLI process behind
            self._kill(proc)
            raise
        return CmdResult(proc.returncode or 0, scrub(out.decode(errors="replace")))

    @staticmethod
    def _kill(proc: asyncio.subprocess.Process) -> None:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except OSError:
            pass

    async def listing(self) -> Listing:
        """One pass of `sessions` + `usage` (a "trusted listing" candidate, spec 5.3)."""
        taken = self.now()
        s = await self._run(["sessions"], self.timeouts["sessions"])
        u = await self._run(["usage"], self.timeouts["usage"])
        return make_listing(0 if s.ok else 1, s.out, 0 if u.ok else 1, u.out, taken)

    async def version(self) -> str | None:
        r = await self._run(["version"], self.timeouts["usage"])
        m = re.search(r"Version:\s*(\S+)", r.out) if r.ok else None
        return m[1] if m else None

    async def usage(self) -> Usage | None:
        r = await self._run(["usage"], self.timeouts["usage"])
        return parse_usage(r.out) if r.ok else None

    async def spawn_new(self, name: str, gpu_class: str) -> asyncio.subprocess.Process:
        if not self.enabled:
            raise OSError("Sign in to a Google account first.")
        args = ["new", "-s", name, *cli_flags(gpu_class)]
        self.log(f"colab {redact(args)} (own process group, no watchdog)")
        return await asyncio.create_subprocess_exec(
            *self.command(), *args, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT,
            stdin=asyncio.subprocess.DEVNULL, start_new_session=True,
        )

    async def upload(self, name: str, local: Path, remote: str) -> CmdResult:
        return await self._run(["upload", "-s", name, str(local), remote], self.timeouts["upload"])

    async def exec_file(self, name: str, path: Path, env: dict[str, str] | None = None) -> ExecResult:
        args = ["exec", "-s", name, "-f", str(path)]
        for k, v in (env or {}).items():
            args += ["--env", f"{k}={v}"]
        r = await self._run(args, self.timeouts["exec"])
        return ExecResult(r.rc, r.out, r.timed_out, parse_status(r.out))

    async def stop(self, name: str) -> CmdResult:
        return await self._run(["stop", "-s", name], self.timeouts["stop"])
