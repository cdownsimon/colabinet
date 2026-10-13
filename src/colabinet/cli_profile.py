"""Run the installed official Python CLI with account-scoped paths, preserving HOME and the SSH agent.

This file runs under the CLI's own interpreter, so it must use only the standard library and colab_cli.
Upstream 0.7.4 exposes only the session-store override; other stores are redirected before loading its launcher.
"""

import logging
import os
import runpy
import sys
from pathlib import Path


def main() -> None:
    profile, binary, *args = sys.argv[1:]
    profile = Path(profile)
    os.umask(0o077)
    profile.mkdir(mode=0o700, parents=True, exist_ok=True)
    # This script's directory also contains Colabinet's colab_cli.py wrapper, which must not shadow upstream.
    sys.path = [entry for entry in sys.path if Path(entry).resolve() != Path(__file__).resolve().parent]
    # Match Python's normal launcher import path, including test/install-local launchers.
    sys.path.insert(0, str(Path(binary).resolve().parent))
    try:
        from colab_cli import auth, common
        from colab_cli.history import HistoryLogger
        from colab_cli.state import SettingsStore, StateStore
        auth.TOKEN_CONFIG_PATH = str(profile / "token.json")
        common.state._store = StateStore(str(profile / "sessions.json"))
        common.state._settings_store = SettingsStore(str(profile / "settings.json"))
        common.state._history = HistoryLogger(str(profile / "history"))
        # Upstream DEBUG logging includes authenticated URLs. Do not create its shared debug log.
        logging.disable(logging.CRITICAL)
        common.setup_logging = lambda *args, **kwargs: None
    except (ImportError, AttributeError, TypeError):
        sys.exit("Account profiles require the official google-colab-cli Python installation (0.7.4 or newer).")
    sys.argv = [binary, *args]
    runpy.run_path(binary, run_name="__main__")


if __name__ == "__main__":
    main()
