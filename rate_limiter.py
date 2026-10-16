"""
Rate limiting module for API call management.

This module provides a thread-safe, high-performance rate limiter
using a sliding window with collections.deque and monotonic clocks.
"""
from collections import deque
import threading
import time


class RateLimiter:
    """
    Thread-safe sliding-window rate limiter.

    Optimizations:
    - Uses collections.deque for O(1) removals and O(1) oldest-call lookup.
    - Uses time.monotonic() to be immune to system clock / NTP adjustments.
    - Uses threading.Condition instead of busy-polling with time.sleep().
    - Supports context manager syntax (`with rate_limiter:`).
    - Supports instant abort/wake-up for graceful batch cancellation.
    """

    def __init__(self, max_calls: int, period: float):
        """
        Initialize rate limiter.

        Args:
            max_calls: Maximum number of calls allowed in the period.
            period: Time period in seconds.
        """
        self.max_calls = max_calls
        self.period = period
        self.calls = deque()
        self.lock = threading.Lock()
        self.condition = threading.Condition(self.lock)
        self._aborted = False

    def _clean_old_calls(self, now: float):
        """Purge calls that have expired outside the sliding window. (Must hold lock)."""
        cutoff = now - self.period
        while self.calls and self.calls[0] <= cutoff:
            self.calls.popleft()

    def reserve(self, timeout: float = None) -> bool:
        """
        Atomically wait for an available slot and record the call.

        Returns:
            bool: True if slot was reserved, False if timed out or aborted.
        """
        start_time = time.monotonic()

        with self.condition:
            while True:
                if self._aborted:
                    return False

                now = time.monotonic()
                self._clean_old_calls(now)

                if len(self.calls) < self.max_calls:
                    self.calls.append(now)
                    return True

                # Oldest call is always at index 0 (O(1) lookup)
                time_until_expire = self.period - (now - self.calls[0])
                wait_time = max(0.001, time_until_expire)

                if timeout is not None:
                    elapsed = now - start_time
                    remaining = timeout - elapsed
                    if remaining <= 0:
                        return False
                    wait_time = min(wait_time, remaining)

                # Suspends thread efficiently; releases lock while waiting
                self.condition.wait(timeout=wait_time)

    def wait(self, timeout: float = None) -> bool:
        """Wait until a slot is available without consuming it."""
        start_time = time.monotonic()

        with self.condition:
            while True:
                if self._aborted:
                    return False

                now = time.monotonic()
                self._clean_old_calls(now)

                if len(self.calls) < self.max_calls:
                    return True

                time_until_expire = self.period - (now - self.calls[0])
                wait_time = max(0.001, time_until_expire)

                if timeout is not None:
                    elapsed = now - start_time
                    remaining = timeout - elapsed
                    if remaining <= 0:
                        return False
                    wait_time = min(wait_time, remaining)

                self.condition.wait(timeout=wait_time)

    def add_call(self):
        """Record a new API call at the current time."""
        with self.condition:
            now = time.monotonic()
            self._clean_old_calls(now)
            self.calls.append(now)

    def get_current_calls(self) -> int:
        """Get the current number of active calls in the period."""
        with self.condition:
            self._clean_old_calls(time.monotonic())
            return len(self.calls)

    def reset(self):
        """Manually reset all call records and wake up any waiting threads."""
        with self.condition:
            self.calls.clear()
            self._aborted = False
            self.condition.notify_all()

    def abort(self):
        """Immediately wake up and release any waiting threads (e.g. on Cancel)."""
        with self.condition:
            self._aborted = True
            self.condition.notify_all()

    def get_status(self) -> dict:
        """Get current rate limiter status."""
        with self.condition:
            self._clean_old_calls(time.monotonic())
            current = len(self.calls)
            return {
                'current_calls': current,
                'max_calls': self.max_calls,
                'remaining': max(0, self.max_calls - current),
                'period_seconds': self.period
            }

    def __enter__(self):
        """Context manager support (`with rate_limiter:`)."""
        if not self.reserve():
            raise TimeoutError("Could not reserve rate limit slot.")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass
