"""Alerts: "Needs attention" items. Logged, listed by `ctl alerts` / GET /api/alerts, and critical ones become macOS
notifications, so the headless core never hides a billing problem (spec §10)."""

from __future__ import annotations

import json
import logging
import subprocess
from dataclasses import dataclass, field
from typing import Callable, Protocol

log = logging.getLogger("colabinet.alerts")

CRITICAL_KINDS = {
    "stop_unconfirmed", "external_runtime", "unknown_runtime", "gpu_mismatch", "daily_cap", "create_slow",
    "engine_down", "low_balance", "create_orphan", "endpoint_missing", "foreign_vm", "ssh_429",
}


class Notifier(Protocol):
    def notify(self, title: str, message: str) -> None: ...


class MacNotifier:
    def notify(self, title: str, message: str) -> None:
        script = f"display notification {json.dumps(message, ensure_ascii=False)} " \
                 f"with title {json.dumps(title, ensure_ascii=False)}"
        try:
            subprocess.run(["osascript", "-e", script], timeout=5, capture_output=True)
        except (OSError, subprocess.TimeoutExpired):
            log.warning("could not show a macOS notification")


class NullNotifier:
    def __init__(self) -> None:
        self.sent: list[tuple[str, str]] = []

    def notify(self, title: str, message: str) -> None:
        self.sent.append((title, message))


@dataclass
class Alert:
    key: str
    kind: str
    level: str  # critical | warning
    message: str
    since: float
    session_id: int | None = None
    actions: list[str] = field(default_factory=list)
    target: str | None = None


class Alerts:
    def __init__(self, notifier: Notifier, now: Callable[[], float]) -> None:
        self.notifier = notifier
        self.now = now
        self.notify_enabled = True
        self._alerts: dict[str, Alert] = {}

    def raise_(self, kind: str, message: str, session_id: int | None = None, key: str | None = None,
               actions: list[str] | None = None, target: str | None = None) -> Alert:
        key = key or (f"{kind}:{session_id}" if session_id is not None else kind)
        level = "critical" if kind in CRITICAL_KINDS else "warning"
        existing = self._alerts.get(key)
        if existing:
            existing.message = message
            return existing
        alert = Alert(key, kind, level, message, self.now(), session_id, actions or [], target)
        self._alerts[key] = alert
        log.warning("ALERT %s: %s", kind, message)
        if level == "critical" and self.notify_enabled:
            self.notifier.notify("colabinet", message)
        return alert

    def clear(self, key: str) -> None:
        self._alerts.pop(key, None)

    def clear_session(self, session_id: int, kinds: set[str] | None = None) -> None:
        for k, a in list(self._alerts.items()):
            if a.session_id == session_id and (kinds is None or a.kind in kinds):
                del self._alerts[k]

    def list(self) -> list[Alert]:
        return sorted(self._alerts.values(), key=lambda a: (a.level != "critical", a.since))
