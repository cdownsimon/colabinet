"""Saved account metadata and private CLI profiles. Credential contents stay with the official CLI."""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import uuid
from pathlib import Path

from .colab_cli import ColabCLI

PROFILE_ID = re.compile(r"[a-f0-9]{32}\Z")
log = logging.getLogger("colabinet.accounts")


class Accounts:
    def __init__(self, data_dir: Path) -> None:
        self.root = data_dir / "accounts"
        if self.root.is_symlink():
            raise ValueError("Account storage must not be a symbolic link.")
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.root.chmod(0o700)
        self.path = self.root / "accounts.json"
        self.active: str | None = "default"  # compatible with an existing CLI installation
        self.rows: list[dict] = []
        if self.path.exists():
            if self.path.is_symlink():
                raise ValueError("Account metadata must not be a symbolic link.")
            try:
                saved = json.loads(self.path.read_text())
                self.active, self.rows = saved["active"], saved["accounts"]
                ids, profiles = set(), set()
                for row in self.rows:
                    if (not (row["id"] == "default" or PROFILE_ID.fullmatch(row["id"]))
                            or not (row["profile"] is None or PROFILE_ID.fullmatch(row["profile"]))
                            or (row["profile"] is None and row["id"] != "default")
                            or not isinstance(row["email"], str) or not 0 < len(row["email"]) <= 320
                            or not isinstance(row["signed_in"], bool) or row["id"] in ids
                            or (row["profile"] and row["profile"] in profiles)):
                        raise ValueError
                    ids.add(row["id"])
                    profiles.add(row["profile"])
                if not (self.active is None or self.active in ids or (self.active == "default" and not ids)):
                    raise ValueError
                if self.active and self.rows and not self.get(self.active)["signed_in"]:
                    raise ValueError
                for row in self.rows:
                    if row["profile"] and not (self.profile(row["profile"]) / "token.json").is_file():
                        row["signed_in"] = False
            except (KeyError, TypeError, ValueError):
                raise ValueError("Saved account metadata is invalid. Restore a valid accounts.json.") from None

    def get(self, account_id: str) -> dict:
        row = next((row for row in self.rows if row["id"] == account_id), None)
        if row is None:
            raise ValueError("Unknown saved account. Refresh Settings.")
        return row

    def profile(self, profile_id: str) -> Path:
        if not PROFILE_ID.fullmatch(profile_id):
            raise ValueError("Invalid account profile.")
        path = self.root / profile_id
        if path.is_symlink() or any((path / file).is_symlink()
                                    for file in ("token.json", "sessions.json", "settings.json", "history")):
            raise ValueError("Account profile files must not be symbolic links.")
        path.mkdir(mode=0o700, exist_ok=True)
        path.chmod(0o700)
        return path

    def candidate(self, base: ColabCLI) -> ColabCLI:
        return base.for_profile(self.profile(uuid.uuid4().hex))

    def cli(self, base: ColabCLI, account_id: str | None, *, initial: bool = False) -> ColabCLI:
        if account_id is None:
            return base.for_profile(None, enabled=False)
        row = next((row for row in self.rows if row["id"] == account_id), None)
        if row is None and account_id == "default":
            return base.for_profile(None)
        row = self.get(account_id)
        if not row["signed_in"] and not initial:
            raise ValueError("This account is logged out. Sign in with Google again.")
        path = self.profile(row["profile"]) if row["profile"] else None
        if path and not (path / "token.json").is_file() and not initial:
            raise ValueError("This account's credentials are missing. Sign in with Google again.")
        return base.for_profile(path, enabled=row["signed_in"])

    def _save(self) -> None:
        temporary = self.root / f".{uuid.uuid4().hex}.tmp"
        try:
            fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(fd, "w") as file:
                json.dump({"active": self.active, "accounts": self.rows}, file)
                file.flush()
                os.fsync(file.fileno())
            os.replace(temporary, self.path)
        finally:
            temporary.unlink(missing_ok=True)

    def remember(self, email: str, cli: ColabCLI) -> str:
        rows = [dict(row) for row in self.rows]
        row = next((row for row in rows if row["email"].casefold() == email.casefold()), None)
        if row is None:
            account_id = cli.profile.name if cli.profile else "default"
            row = {"id": account_id, "email": email, "profile": None, "signed_in": True}
            rows.append(row)
        previous = row["profile"]
        row.update(email=email, profile=cli.profile.name if cli.profile else None, signed_in=True)
        self._commit(row["id"], rows)
        if previous and previous != row["profile"]:
            self._cleanup(self.profile(previous) / "token.json")
        return row["id"]

    def _commit(self, active: str | None, rows: list[dict]) -> None:
        old_active, old_rows = self.active, self.rows
        self.active, self.rows = active, rows
        try:
            self._save()
        except OSError:
            self.active, self.rows = old_active, old_rows
            raise

    @staticmethod
    def _cleanup(path: Path) -> None:
        try:
            path.unlink(missing_ok=True)
        except OSError:
            log.warning("Could not remove an unused private credential file.")

    def select(self, account_id: str) -> None:
        self.get(account_id)
        self._commit(account_id, self.rows)

    def repair(self, email: str, candidate: ColabCLI, current: ColabCLI) -> str:
        """Replace only the same owner's credentials, preserving the CLI session store and running tunnels."""
        row = self.get(self.active)
        if row["email"].casefold() != email.casefold():
            raise ValueError("Choose the Google account that owns these sessions.")
        target = current.profile or Path.home() / ".config" / "colab-cli"
        target.mkdir(parents=True, exist_ok=True)
        token = target / "token.json"
        if token.is_symlink():
            raise ValueError("Credential files must not be symbolic links.")
        temporary = target / f".{uuid.uuid4().hex}.tmp"
        backup = target / f".{uuid.uuid4().hex}.backup"
        had_token = token.exists()
        try:
            fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(fd, "wb") as output, (candidate.profile / "token.json").open("rb") as source:
                shutil.copyfileobj(source, output)
                output.flush()
                os.fsync(output.fileno())
            if had_token:
                os.replace(token, backup)
            try:
                os.replace(temporary, token)
                account_id = self.remember(email, current)
            except OSError:
                if had_token:
                    os.replace(backup, token)
                else:
                    token.unlink(missing_ok=True)
                raise
            self._cleanup(backup)
            return account_id
        finally:
            temporary.unlink(missing_ok=True)

    def logout(self) -> None:
        rows = [dict(row) for row in self.rows]
        token = backup = None
        if self.active:
            row = next(row for row in rows if row["id"] == self.active)
            if row["profile"]:
                token = self.profile(row["profile"]) / "token.json"
                if token.exists():
                    backup = token.with_name(f".{uuid.uuid4().hex}.backup")
                    os.replace(token, backup)
            row["signed_in"] = False
        try:
            self._commit(None, rows)
        except OSError:
            if backup:
                os.replace(backup, token)
            raise
        if backup:
            self._cleanup(backup)

    def discard(self, cli: ColabCLI) -> None:
        if cli.profile and not any(row["profile"] == cli.profile.name for row in self.rows):
            shutil.rmtree(self.profile(cli.profile.name))

    def snapshot(self) -> list[dict]:
        return [{"id": row["id"], "email": row["email"], "signed_in": row["signed_in"],
                 "active": row["id"] == self.active, "shared": row["profile"] is None} for row in self.rows]
