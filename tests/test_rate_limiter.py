import threading
import time

import pytest

from rate_limiter import RateLimiter


def test_add_call_increments_counter():
    rl = RateLimiter(max_calls=5, period=1.0)
    rl.add_call()
    assert rl.get_current_calls() == 1


def test_old_calls_expire():
    rl = RateLimiter(max_calls=5, period=0.2)
    rl.add_call()
    rl.add_call()
    time.sleep(0.25)
    assert rl.get_current_calls() == 0


def test_wait_returns_immediately_when_below_limit():
    rl = RateLimiter(max_calls=5, period=1.0)
    start = time.time()
    assert rl.wait(timeout=0.5) is True
    assert time.time() - start < 0.2


def test_wait_times_out_when_at_limit():
    rl = RateLimiter(max_calls=2, period=10.0)
    rl.add_call()
    rl.add_call()
    assert rl.wait(timeout=0.2) is False


def test_wait_unblocks_when_slot_expires():
    rl = RateLimiter(max_calls=1, period=0.3)
    rl.add_call()
    start = time.time()
    assert rl.wait(timeout=1.0) is True
    elapsed = time.time() - start
    assert 0.2 < elapsed < 0.8


def test_concurrent_callers_smoke():
    rl = RateLimiter(max_calls=5, period=1.0)
    max_seen = [0]
    samples_lock = threading.Lock()

    def worker():
        rl.wait(timeout=5.0)
        rl.add_call()
        with samples_lock:
            current = rl.get_current_calls()
            if current > max_seen[0]:
                max_seen[0] = current

    threads = [threading.Thread(target=worker) for _ in range(20)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert max_seen[0] <= 10, (
        f"Observed {max_seen[0]} concurrent calls — way over max_calls=5, "
        "indicates a broken limiter"
    )


def test_reserve_never_exceeds_max_under_contention():
    """reserve() is the atomic replacement for wait()+add_call()."""
    rl = RateLimiter(max_calls=5, period=2.0)
    max_seen = [0]
    samples_lock = threading.Lock()

    def worker():
        rl.reserve(timeout=5.0)
        with samples_lock:
            current = rl.get_current_calls()
            if current > max_seen[0]:
                max_seen[0] = current

    threads = [threading.Thread(target=worker) for _ in range(30)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert max_seen[0] <= 5, (
        f"reserve() allowed {max_seen[0]} concurrent calls with max_calls=5"
    )


def test_reserve_times_out():
    rl = RateLimiter(max_calls=1, period=10.0)
    assert rl.reserve(timeout=0.1) is True
    assert rl.reserve(timeout=0.2) is False
