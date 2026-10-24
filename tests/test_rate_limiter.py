import threading
import time
import pytest
from rate_limiter import RateLimiter


def test_reserve_below_limit():
    """Verify slots are granted when below max_calls and rejected when full."""
    rl = RateLimiter(max_calls=2, period=1.0)
    assert rl.reserve(timeout=0.05) is True
    assert rl.reserve(timeout=0.05) is True
    # The 3rd one must fail because the limit of 2 is reached
    assert rl.reserve(timeout=0.05) is False


def test_reserve_times_out_when_full():
    """Reserving when full should block and time out cleanly."""
    rl = RateLimiter(max_calls=1, period=10.0)
    assert rl.reserve(timeout=0.05) is True
    assert rl.reserve(timeout=0.1) is False


def test_reserve_unblocks_when_slot_expires():
    """A thread waiting on a full limiter unblocks as soon as the oldest call expires."""
    rl = RateLimiter(max_calls=1, period=0.2)
    assert rl.reserve(timeout=0.05) is True

    start = time.monotonic()
    # Should block for ~0.2s then succeed
    assert rl.reserve(timeout=1.0) is True
    elapsed = time.monotonic() - start
    assert 0.15 <= elapsed <= 0.4


def test_reserve_never_exceeds_max_under_contention():
    """Under heavy concurrency, exactly max_calls threads must get immediate slots."""
    max_calls = 5
    # Long period so slots don't expire during the test
    rl = RateLimiter(max_calls=max_calls, period=10.0)
    successful_threads = []
    lock = threading.Lock()

    def worker():
        # Short timeout: only the first 5 threads should get a slot
        if rl.reserve(timeout=0.1):
            with lock:
                successful_threads.append(True)

    threads = [threading.Thread(target=worker) for _ in range(25)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # Exactly 5 should succeed, never 6 or more
    assert len(successful_threads) == max_calls


def test_abort_immediately_unblocks_waiting_threads():
    """Calling abort() must wake up sleeping threads immediately with False."""
    rl = RateLimiter(max_calls=1, period=10.0)
    assert rl.reserve() is True

    results = []

    def worker():
        res = rl.reserve(timeout=10.0)
        results.append(res)

    t = threading.Thread(target=worker)
    t.start()
    time.sleep(0.05)  # Let worker enter condition.wait()

    start = time.monotonic()
    rl.abort()
    t.join(timeout=1.0)

    assert not t.is_alive(), "Worker thread hung after abort()"
    assert time.monotonic() - start < 0.2, "Abort took too long to unblock worker"
    assert results == [False]


def test_reset_restores_aborted_limiter():
    """reset() should clear the abort state so new calls can be made."""
    rl = RateLimiter(max_calls=2, period=1.0)
    rl.abort()
    assert rl.reserve(timeout=0.05) is False

    rl.reset()
    assert rl.reserve(timeout=0.05) is True


def test_reset_preserves_active_call_history():
    """Verification of bug fix: reset() must NOT wipe unexpired calls (prevents 429s)."""
    rl = RateLimiter(max_calls=1, period=5.0)
    assert rl.reserve() is True

    # Call reset (e.g. at the start of a batch)
    rl.reset()

    # The call from 5ms ago is still active, so a new call must still be rejected
    assert rl.reserve(timeout=0.05) is False

