"""First-run checks. Never create a runtime, start a console, or inspect credential files."""

from __future__ import annotations

import re
import shutil
import subprocess
import time

from .accounts import Accounts
from .colab_cli import ColabCLI
from .config import Settings, default_ssh_public_key

INSTALL_COMMAND = "uv tool install --upgrade 'google-colab-cli>=0.7.4'"


class InstallationError(RuntimeError):
    pass


def require_colab(settings: Settings) -> None:
    if not shutil.which(settings.colab_bin):
        raise InstallationError(f"Colab CLI not found or executable at {settings.colab_bin}. "
                                f"Install with: {INSTALL_COMMAND}; run uv tool update-shell and reopen your terminal, "
                                "or select its path with --colab-bin. See docs/installation.md.")


def supported_version(version: str | None) -> bool:
    match = re.fullmatch(r"(\d+)\.(\d+)\.(\d+)(?:\.post\d+)?(?:\+[\w.-]+)?", version or "")
    return bool(match and tuple(map(int, match.groups())) >= (0, 7, 4))


async def run(settings: Settings, *, offline: bool = False, out=print) -> int:
    failed = False

    def report(name: str, ok: bool, message: str) -> None:
        nonlocal failed
        failed |= not ok
        out(f"{'OK' if ok else 'FAIL'} {name}: {message}")

    try:
        require_colab(settings)
    except InstallationError as e:
        report("Colab CLI", False, str(e))
        return 1
    cli = ColabCLI(settings.colab_bin, now=time.time, auth=settings.colab_auth)
    try:
        version = await cli.version()
    except OSError:
        version = None
    version_ok = supported_version(version)
    report("Colab CLI", version_ok, f"{settings.colab_bin} ({version or 'version unavailable'})" if version_ok else
           f"A stable release >= 0.7.4 is required (found {version or 'unknown'}). Run: {INSTALL_COMMAND}")

    tools_ok = all(shutil.which(tool) for tool in ("ssh", "ssh-keygen"))
    report("OpenSSH", tools_ok, "ssh and ssh-keygen available" if tools_ok else
           "Install OpenSSH and make ssh and ssh-keygen available on PATH.")
    key_ok = False
    agent_key = False
    if tools_ok and settings.ssh_key.is_file():
        try:
            result = subprocess.run(["ssh-keygen", "-y", "-P", "", "-f", str(settings.ssh_key)],
                                    stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=5)
            key_ok = result.returncode == 0 and result.stdout.startswith(
                ("ssh-ed25519 ", "ecdsa-sha2-nistp256 ", "ecdsa-sha2-nistp384 ", "ecdsa-sha2-nistp521 "))
        except (OSError, subprocess.TimeoutExpired):
            pass
        public_path = default_ssh_public_key(settings.ssh_key)
        if not key_ok and public_path is not None:
            try:
                public = public_path.read_text().split()[:2]
                agent = subprocess.run(["ssh-add", "-L"], stdin=subprocess.DEVNULL,
                                       capture_output=True, text=True, timeout=5)
                agent_key = agent.returncode == 0 and len(public) == 2 and public[0] == "ssh-ed25519" and any(
                    line.split()[:2] == public for line in agent.stdout.splitlines())
                key_ok = agent_key
            except (OSError, subprocess.TimeoutExpired):
                pass
    report("SSH key", key_ok, str(settings.ssh_key) + (" (loaded in ssh-agent)" if agent_key else "") if key_ok else
           f"{settings.ssh_key} is missing, unreadable, encrypted or unsupported. "
           "Load the default Ed25519 key into ssh-agent, or use a dedicated key without a passphrase; "
           "see docs/installation.md and --ssh-key.")

    if offline:
        out("SKIP Google account: account API check skipped (--offline).")
    elif version_ok:
        saved_accounts = settings.data_dir / "accounts" / "accounts.json"
        try:
            if settings.colab_auth == "oauth2" and saved_accounts.is_file():
                accounts = Accounts(settings.data_dir)
                cli = accounts.cli(cli, accounts.active)
            usage = await cli.usage()
        except (OSError, ValueError):
            usage = None
        report("Google account", usage is not None,
               f"Account API access verified ({settings.colab_auth})." if usage else
               "Account API access failed. Sign in or select an account in Settings → Google accounts, "
               "or run colabinet login for an initial shared CLI account; "
               "if already signed in, check your network and the installation guide.")
    else:
        out("SKIP Google account: fix the CLI version first.")

    if failed:
        out("Fix the failed checks, then run colabinet doctor again. See docs/installation.md.")
        return 1
    if offline:
        out("Local checks passed. Run colabinet login, then colabinet doctor to verify account access.")
    else:
        out("Installation ready. Start colabinet with the same CLI/auth/key options and --stop-on-exit; "
            "creating a session is a separate, billable action.")
    return 0
