"""What counts as "useful activity" for the idle guard (spec 5.4)."""

from __future__ import annotations

INFERENCE_PATHS = {"/v1/chat/completions", "/v1/completions", "/v1/messages", "/v1/embeddings", "/completion"}
INFLIGHT_MAX_S = 60 * 60  # an in-flight request counts for up to 60 min (covers a 262K prefill)...
BYTE_WINDOW_S = 5 * 60  # ...after that only while upstream bytes arrived in the last 5 min


def is_inference(method: str, path: str) -> bool:
    return method.upper() == "POST" and path.rstrip("/") in INFERENCE_PATHS


class Activity:
    """Per-session activity clock fed by the proxy."""

    def __init__(self, last: float | None = None) -> None:
        self.last = last or 0.0
        self._inflight: dict[int, list[float]] = {}  # key -> [started, last upstream byte]

    def reset(self, t: float) -> None:
        self.last = max(self.last, t)

    def begin(self, key: int, t: float) -> None:
        self._inflight[key] = [t, t]

    def touch(self, key: int, t: float) -> None:
        if key in self._inflight:
            self._inflight[key][1] = t

    def end(self, key: int, t: float, useful: bool) -> None:
        self._inflight.pop(key, None)
        if useful:
            self.last = max(self.last, t)

    def effective(self, now: float) -> float:
        best = self.last
        for started, last_byte in self._inflight.values():
            counts = now - started <= INFLIGHT_MAX_S or now - last_byte <= BYTE_WINDOW_S
            best = max(best, now if counts else last_byte)
        return best
