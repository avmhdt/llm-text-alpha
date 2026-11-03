from __future__ import annotations

import threading
import time
from itertools import pairwise

import pytest

from llm_text_alpha.data.ratelimit import ByteThrottle, TokenBucket


class FakeClock:
    """A clock that only moves when something sleeps."""

    def __init__(self) -> None:
        self.now = 0.0
        self.sleeps: list[float] = []

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


def test_requests_are_spaced_by_the_interval() -> None:
    clock = FakeClock()
    bucket = TokenBucket(8, clock=clock, sleep=clock.sleep)
    times = []
    for _ in range(17):
        bucket.acquire()
        times.append(clock.now)
    gaps = [b - a for a, b in pairwise(times)]
    assert times[0] == 0.0
    assert all(gap == pytest.approx(0.125) for gap in gaps)
    assert (len(times) - 1) / (times[-1] - times[0]) == pytest.approx(8.0)


def test_no_window_of_one_second_holds_more_than_rate() -> None:
    clock = FakeClock()
    bucket = TokenBucket(10, clock=clock, sleep=clock.sleep)
    times = []
    for _ in range(50):
        bucket.acquire()
        times.append(clock.now)
    for start in times:
        inside = [t for t in times if start <= t < start + 1.0 - 1e-9]
        assert len(inside) <= 10


def test_idle_time_does_not_build_a_burst() -> None:
    clock = FakeClock()
    bucket = TokenBucket(5, clock=clock, sleep=clock.sleep)
    bucket.acquire()
    clock.now += 100.0
    assert bucket.acquire() == 0.0
    assert bucket.acquire() == pytest.approx(0.2)


def test_threads_share_the_cap() -> None:
    bucket = TokenBucket(100)
    stamps: list[float] = []
    lock = threading.Lock()

    def work() -> None:
        for _ in range(10):
            bucket.acquire()
            with lock:
                stamps.append(time.perf_counter())

    threads = [threading.Thread(target=work) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    stamps.sort()
    assert len(stamps) == 40
    assert (len(stamps) - 1) / (stamps[-1] - stamps[0]) <= 100 * 1.05


def test_rates_must_be_positive() -> None:
    with pytest.raises(ValueError):
        TokenBucket(0)
    with pytest.raises(ValueError):
        ByteThrottle(0)


def test_byte_throttle_keeps_the_average() -> None:
    clock = FakeClock()
    throttle = ByteThrottle(1000, clock=clock, sleep=clock.sleep)
    assert throttle.consumed(500) == pytest.approx(0.5)
    assert throttle.consumed(500) == pytest.approx(0.5)
    assert clock.now == pytest.approx(1.0)
    clock.now += 10
    assert throttle.consumed(100) == 0.0
    assert throttle.total_bytes == 1100
