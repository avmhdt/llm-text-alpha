"""Request and byte rate limits shared across threads (NFR-07, owner's 10 MB/s download cap)."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable

DEFAULT_MAX_BYTES_PER_SECOND = 10_000_000.0
"""Owner's cap on any download while live recorders share this PC's connection."""


class TokenBucket:
    """Token bucket holding a single token: requests are spaced at least ``1/rate`` apart.

    A bucket larger than one token would let a burst plus the refill exceed ``rate`` inside
    a one-second window. Each caller reserves the next free slot under a lock and sleeps
    outside it, so threads share the cap without serializing their sleeps.

    Args:
        rate: requests per second.
        clock: monotonic clock in seconds.
        sleep: sleep function in seconds.
    """

    def __init__(
        self,
        rate: float,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if rate <= 0:
            raise ValueError("rate must be positive")
        self.rate = rate
        self._interval = 1.0 / rate
        self._clock = clock
        self._sleep = sleep
        self._lock = threading.Lock()
        self._next_free: float | None = None

    def acquire(self) -> float:
        """Block until a request may go out; return the seconds waited."""
        with self._lock:
            now = self._clock()
            slot = now if self._next_free is None else max(now, self._next_free)
            self._next_free = slot + self._interval
        wait = slot - now
        if wait > 0:
            self._sleep(wait)
        return wait


class ByteThrottle:
    """Keep a stream's average rate at or under ``max_bytes_per_second``.

    Call :meth:`consumed` after each chunk; it sleeps until the bytes so far fit the rate.
    """

    def __init__(
        self,
        max_bytes_per_second: float = DEFAULT_MAX_BYTES_PER_SECOND,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if max_bytes_per_second <= 0:
            raise ValueError("max_bytes_per_second must be positive")
        self.max_bytes_per_second = max_bytes_per_second
        self._clock = clock
        self._sleep = sleep
        self._start: float | None = None
        self.total_bytes = 0

    def consumed(self, n_bytes: int) -> float:
        """Record ``n_bytes`` read; sleep as needed and return the seconds waited."""
        now = self._clock()
        if self._start is None:
            self._start = now
        self.total_bytes += n_bytes
        earliest = self._start + self.total_bytes / self.max_bytes_per_second
        wait = earliest - now
        if wait > 0:
            self._sleep(wait)
            return wait
        return 0.0
