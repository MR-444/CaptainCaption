"""
Rate limiting module for API call management.

Provides a thread-safe sliding-window rate limiter using collections.deque,
monotonic clocks, and condition variables for thread synchronization.
"""
from collections import deque
import threading
import time


class RateLimiter:
    """Thread-safe sliding-window rate limiter with instant abort capability."""

    def __init__(self, max_calls: int, period: float):
        self.max_calls = max_calls
        self.period = period
        self.calls = deque()
        self.lock = threading.Lock()
        self.condition = threading.Condition(self.lock)
        self._aborted = False

    def _clean_old_calls(self, now: float):
        """Purge calls outside the sliding window. Must hold lock."""
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

                # Wait until the oldest recorded call expires
                time_until_expire = self.period - (now - self.calls[0])
                wait_time = max(0.001, time_until_expire)

                if timeout is not None:
                    elapsed = now - start_time
                    remaining = timeout - elapsed
                    if remaining <= 0:
                        return False
                    wait_time = min(wait_time, remaining)

                self.condition.wait(timeout=wait_time)

    def reset_abort(self):
        """Reset the abort state without erasing recent call history."""
        with self.condition:
            self._aborted = False

    def reset(self):
        """Clear the abort state. Does not wipe recent valid call history."""
        with self.condition:
            self._aborted = False
            self._clean_old_calls(time.monotonic())
            self.condition.notify_all()

    def abort(self):
        """Immediately wake up and release any waiting threads (e.g. on Cancel)."""
        with self.condition:
            self._aborted = True
            self.condition.notify_all()
