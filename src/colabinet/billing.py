"""CU arithmetic. CU is time-based: rate (CU/h) x time the VM existed."""

from __future__ import annotations

from typing import Any, Iterable

RECLAIM_S = 25 * 60  # Billing estimate from observations, not a confirmed Colab timeout.


def overlap(a0: float, a1: float, b0: float, b1: float) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0))


def billing_end(row: dict[str, Any], now: float) -> float:
    """When this session stopped billing (now if it still bills)."""
    return row["stopped_at"] if row.get("stopped_at") is not None else now


def session_cu(row: dict[str, Any], now: float) -> float:
    return row["rate_cu_h"] * max(0.0, billing_end(row, now) - row["created_at"]) / 3600


def spend_between(rows: Iterable[dict[str, Any]], start: float, end: float, now: float) -> float:
    """CU billed inside [start, end) by the given sessions (e.g. one local day)."""
    total = 0.0
    for row in rows:
        total += row["rate_cu_h"] * overlap(row["created_at"], billing_end(row, now), start, end) / 3600
    return total


def close_time(*, now: float, created_at: float, vm_created: bool, reason: str | None,
               last_exec_ok: float | None) -> float:
    """`stopped_at` for a transition into stopped/failed ("closing the bill", spec 5.2)."""
    if not vm_created:
        return created_at
    if reason in ("reclaimed", "lost", "unreachable"):
        base = last_exec_ok if last_exec_ok is not None else created_at
        return min(now, base + RECLAIM_S)
    return now
