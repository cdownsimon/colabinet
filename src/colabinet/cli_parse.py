"""Parse `colab` CLI output (google-colab-cli 0.7.4 formats, from its source).

sessions: `[name] endpoint | Hardware: X | Shape: Y | Variant: Z[ | Status: W]`, name "?" = no local name.
usage:    `Current balance: B compute units` / `Usage rate: R/hr` / `Active assignments: N`.
Everything else (update banners, "Pruned N stale local session(s).") is noise.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

_LINE = re.compile(
    r"^\[(?P<name>[^\]]+)\] (?P<endpoint>\S+) \| Hardware: (?P<hw>[^|]+?) \| Shape: (?P<shape>[^|]+?)"
    r" \| Variant: (?P<variant>[^|]+?)(?: \| Status: (?P<status>.+))?$"
)
_EMPTY = "[colab] No active sessions found on server."
_BALANCE = re.compile(r"Current balance: ([0-9.]+) compute units")
_RATE = re.compile(r"Usage rate: ([0-9.]+)/hr")
_COUNT = re.compile(r"Active assignments: (\d+)")
STATUS_PREFIX = "COLABINET_STATUS "


@dataclass(frozen=True)
class Runtime:
    name: str | None  # None for "[?]" rows
    endpoint: str
    hardware: str
    shape: str
    variant: str


@dataclass(frozen=True)
class Usage:
    balance: float
    rate_per_hour: float
    active_assignments: int


@dataclass(frozen=True)
class Listing:
    runtimes: list[Runtime] = field(default_factory=list)
    usage: Usage | None = None
    trusted: bool = False
    taken_at: float = 0.0

    def named(self, name: str) -> Runtime | None:
        return next((r for r in self.runtimes if r.name == name), None)

    @property
    def unnamed(self) -> list[Runtime]:
        return [r for r in self.runtimes if r.name is None]


def parse_sessions(text: str) -> tuple[list[Runtime], bool]:
    """Returns (runtimes, said_empty)."""
    rows: list[Runtime] = []
    said_empty = False
    for raw in text.splitlines():
        line = raw.strip()
        if line == _EMPTY:
            said_empty = True
            continue
        m = _LINE.match(line)
        if m:
            name = m["name"]
            rows.append(Runtime(None if name == "?" else name, m["endpoint"], m["hw"].strip(), m["shape"].strip(),
                                m["variant"].strip()))
    return rows, said_empty


def parse_usage(text: str) -> Usage | None:
    b, r, c = _BALANCE.search(text), _RATE.search(text), _COUNT.search(text)
    if not (b and r and c):
        return None
    return Usage(float(b[1]), float(r[1]), int(c[1]))


def make_listing(sessions_rc: int, sessions_out: str, usage_rc: int, usage_out: str, taken_at: float) -> Listing:
    """Combine one `sessions` and one `usage` call into a Listing.

    An empty sessions list is only trusted when `usage` succeeded in the same pass and reports 0 assignments:
    with an empty local store, `colab sessions` swallows auth failures and prints the "No active sessions" line.
    """
    runtimes, _ = parse_sessions(sessions_out) if sessions_rc == 0 else ([], False)
    usage = parse_usage(usage_out) if usage_rc == 0 else None
    if sessions_rc != 0 or usage is None:
        trusted = False
    elif not runtimes:
        trusted = usage.active_assignments == 0
    else:
        trusted = True
    return Listing(runtimes, usage, trusted, taken_at)


def parse_status(text: str) -> dict | None:
    """The JSON line that vm/status.py prints, or None if the exec didn't get that far."""
    for line in reversed(text.splitlines()):
        if line.startswith(STATUS_PREFIX):
            try:
                return json.loads(line[len(STATUS_PREFIX):])
            except json.JSONDecodeError:
                return None
    return None
