"""Entry point: console, terminal Google login, installation checks, and `colabinet ctl ...`."""

from __future__ import annotations

import argparse
import asyncio
import sys
import time

from . import __version__
from .config import Settings


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--port", type=int, default=7860, help="API port (default 7860)")
    common.add_argument("--data-dir", help="database, token, lock and logs (default <repo>/data)")
    if argv and argv[0] == "ctl":
        opts, rest = common.parse_known_args(argv[1:])
        from .ctl import run

        return run(rest, _settings(opts))
    ap = argparse.ArgumentParser(prog="colabinet", parents=[common])
    ap.add_argument("command", nargs="?", default="serve", choices=["serve", "login", "doctor"],
                    help="serve the console (default), sign in to Google, or check installation")
    ap.add_argument("--colab-bin", help="Colab CLI executable (default: colab on PATH, then ~/.local/bin/colab)")
    ap.add_argument("--colab-auth", choices=["oauth2", "adc"], default="oauth2",
                    help="Google authentication method (default: oauth2; adc uses existing gcloud credentials)")
    ap.add_argument("--ssh-key", help="SSH private key (default ~/.ssh/id_ed25519)")
    ap.add_argument("--offline", action="store_true", help="doctor: skip the Google account API check")
    ap.add_argument("--stop-on-exit", action="store_true", help="stop all managed sessions when quitting")
    ap.add_argument("--no-notify", action="store_true", help="no macOS notifications for critical alerts")
    ap.add_argument("--version", action="version", version=f"colabinet {__version__}")
    opts = ap.parse_args(argv)
    if opts.offline and opts.command != "doctor":
        ap.error("--offline is only supported by doctor")
    settings = _settings(opts)
    settings.stop_on_exit = opts.stop_on_exit
    settings.notify = not opts.no_notify
    from .doctor import InstallationError, require_colab

    try:
        if opts.command == "doctor":
            from .doctor import run

            return asyncio.run(run(settings, offline=opts.offline))
        require_colab(settings)
        if opts.command == "login":
            from .colab_cli import ColabCLI

            print("Google sign-in is handled by the Colab CLI in this terminal. "
                  "No runtime will be created.", flush=True)
            return ColabCLI(settings.colab_bin, now=time.time, auth=settings.colab_auth).login()
        from .server import AlreadyRunning, serve

        try:
            remaining = asyncio.run(serve(settings))
        except AlreadyRunning as e:
            print(f"error: {e}", file=sys.stderr)
            return 1
    except InstallationError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    except OSError as e:
        print(f"error: {e}. Run colabinet doctor to check installation.", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 130
    if remaining:
        print("still running (billing): " + ", ".join(remaining), file=sys.stderr)
    return 0


def _settings(opts: argparse.Namespace) -> Settings:
    kw = {"ui_port": opts.port}
    if opts.data_dir:
        kw["data_dir"] = opts.data_dir
    for name in ("colab_bin", "colab_auth", "ssh_key"):
        if value := getattr(opts, name, None):
            kw[name] = value
    return Settings(**kw)


if __name__ == "__main__":
    sys.exit(main())
