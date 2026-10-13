"""Credential-safe, bounded text for persistent CLI diagnostics (not machine-readable CLI replies)."""

from __future__ import annotations

import json
import re
from typing import Any

LOG_LIMIT = 65536
TRUNCATED = "[Earlier creation output truncated; showing the retained tail.]\n"
ANSI = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
CONTROLS = re.compile(r"[\x00-\x08\x0b-\x1f\x7f]")
AUTH = re.compile(r"\b(Bearer|Basic)\s+[^\s\"',;}]+", re.I)
COOKIE = re.compile(r"(^\s*(?:Set-)?Cookie\s*:\s*).*$", re.I | re.M)
URL_AUTH = re.compile(r"(https?://)[^/\s@]+@", re.I)
PRIVATE_KEY = re.compile(r"-----BEGIN ([A-Z0-9 ]*PRIVATE KEY)-----.*?(?:-----END \1-----|$)", re.S)
SECRET_NAME = (r"(?:[\w.-]*(?:token|secret|password|api[_-]?key|credential|signature)[\w.-]*"
               r"|code|(?:[\w.-]*[_-])?key|authorization|(?:set[-_]?)?cookie)")
SECRET_FIELD = re.compile(SECRET_NAME, re.I)
JSON_START = re.compile(r"[\[{]")
SECRET = re.compile(
    r"(?<![\w-])([\"']?" + SECRET_NAME + r"[\"']?\s*[:=]\s*)"
    r"(?:\"(?:\\.|[^\"\\])*\"|'(?:\\.|[^'\\])*'|[^\s&;,}\]]+)", re.I,
)


def _json_value(value: Any, depth: int) -> Any:
    if depth >= 12:
        return "***"
    if isinstance(value, dict):
        return {key: "***" if SECRET_FIELD.fullmatch(key) else _json_value(item, depth + 1)
                for key, item in value.items()}
    if isinstance(value, list):
        return [_json_value(item, depth + 1) for item in value]
    return _redact(value, depth + 1) if isinstance(value, str) else value


def _json_fragments(text: str, depth: int) -> str:
    """Decode embedded JSON, including formatted objects after a CLI error prefix."""
    decoder = json.JSONDecoder()
    parts, cursor, search_from = [], 0, 0
    while match := JSON_START.search(text, search_from):
        start = match.start()
        try:
            parsed, end = decoder.raw_decode(text, start)
        except json.JSONDecodeError:
            search_from = start + 1
            continue
        except RecursionError:
            return "".join(parts) + text[cursor:start] + "[Deeply nested diagnostic JSON omitted.]"
        redacted = _json_value(parsed, depth)
        if redacted != parsed:
            parts.extend((text[cursor:start], json.dumps(redacted, ensure_ascii=False)))
            cursor = end
        search_from = end
    return "".join(parts) + text[cursor:]


def _redact(text: str, depth: int = 0) -> str:
    if depth >= 12:
        return "***"
    text = CONTROLS.sub("", ANSI.sub("", text))
    text = PRIVATE_KEY.sub("***", text)
    text = _json_fragments(text, depth + 1)
    text = COOKIE.sub(r"\1***", text)
    text = AUTH.sub(r"\1 ***", text)
    text = URL_AUTH.sub(r"\1***@", text)
    return SECRET.sub(r"\1***", text)


def redact_diagnostic(text: str) -> str:
    """Mask credentials in headers, assignments, JSON and URLs before storing/displaying output."""
    return _redact(text)


def bounded_diagnostic(text: str, limit: int = LOG_LIMIT) -> str:
    """Redact before truncation, so a cut cannot strip a secret's key but retain its value."""
    text = redact_diagnostic(text)
    if len(text) <= limit:
        return text
    marker = TRUNCATED[:limit]
    return marker + text[-(limit - len(marker)):] if limit > len(marker) else marker
