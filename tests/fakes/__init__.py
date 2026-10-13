"""Test doubles: a fake `colab` executable and a fake OpenAI-compatible engine."""

from __future__ import annotations

import fcntl
import json
import os
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

FAKES = Path(__file__).resolve().parent


class FakeColab:
    """Handle on a fake `colab` binary whose state lives in a temp dir."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.state_path = root / "colab_state.json"
        self.bin = root / "colab"
        script = FAKES / "fake_colab.py"
        env = f'FAKE_COLAB_STATE="{self.state_path}"'
        self.bin.write_text(f'#!/bin/sh\n{env} exec "{sys.executable}" "{script}" "$@"\n')
        self.bin.chmod(0o755)
        self.write(self.fresh())

    @staticmethod
    def fresh() -> dict:
        sys.path.insert(0, str(FAKES))
        try:
            import fake_colab

            return fake_colab.default_state()
        finally:
            sys.path.pop(0)

    def read(self) -> dict:
        return json.loads(self.state_path.read_text())

    def write(self, st: dict) -> None:
        tmp = self.state_path.with_name(f"{self.state_path.name}.{os.getpid()}.tmp")
        tmp.write_text(json.dumps(st, indent=1))
        os.replace(tmp, self.state_path)

    @contextmanager
    def edit(self) -> Iterator[dict]:
        with self.state_path.with_suffix(".lock").open("w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            state = self.read()
            yield state
            self.write(state)

    def fault(self, **faults) -> None:
        with self.edit() as state:
            state["faults"].update(faults)

    @property
    def hazards(self) -> list[str]:
        return self.read()["hazards"]

    @property
    def server(self) -> dict:
        return self.read()["server"]

    @property
    def store(self) -> dict:
        return self.read()["store"]

    def calls(self, command: str) -> list[list[str]]:
        return [c for c in self.read()["calls"] if c and c[0] == command]

    def kill_all(self) -> None:
        """Stop every process started inside fake VMs (test cleanup)."""
        import signal

        for pid in self.read().get("all_pids", []):
            try:
                os.killpg(pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass
