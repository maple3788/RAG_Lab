"""Simple global rate limiter for LLM / API calls during ingest."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass


@dataclass
class RateLimiter:
    """At most ``max_calls`` per ``period_seconds`` (sliding window simplified as spacing)."""

    min_interval_seconds: float = 0.5
    _lock: threading.Lock = threading.Lock()
    _last: float = 0.0

    def acquire(self) -> None:
        with self._lock:
            now = time.monotonic()
            wait = self.min_interval_seconds - (now - self._last)
            if wait > 0:
                time.sleep(wait)
            self._last = time.monotonic()
