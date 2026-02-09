"""
Rate limiting module for API call management.

This module provides a thread-safe rate limiter using a sliding window
approach to prevent exceeding API rate limits.
"""
import threading
import time


class RateLimiter:
    """
    Thread-safe rate limiter that tracks API calls and enforces limits.

    Uses a sliding window approach where calls automatically expire after
    the specified period, rather than resetting all at once.
    """

    def __init__(self, max_calls, period):
        """
        Initialize rate limiter.

        Args:
            max_calls: Maximum number of calls allowed in the period
            period: Time period in seconds
        """
        self.max_calls = max_calls
        self.period = period
        self.calls = []  # Store timestamps of calls
        self.lock = threading.Lock()

    def _clean_old_calls(self):
        """Remove calls older than the period (internal, must hold lock)."""
        current_time = time.time()
        cutoff_time = current_time - self.period
        self.calls = [call_time for call_time in self.calls if call_time > cutoff_time]

    def add_call(self):
        """Record a new API call."""
        with self.lock:
            self.calls.append(time.time())
            self._clean_old_calls()

    def get_current_calls(self):
        """Get the current number of calls in the period."""
        with self.lock:
            self._clean_old_calls()
            return len(self.calls)

    def wait(self, timeout=None):
        """
        Wait until a call slot is available.

        Args:
            timeout: Maximum time to wait in seconds (None = wait forever)

        Returns:
            bool: True if slot became available, False if timed out
        """
        start_time = time.time()

        while True:
            with self.lock:
                self._clean_old_calls()

                if len(self.calls) < self.max_calls:
                    return True

                # Calculate how long until the oldest call expires
                if self.calls:
                    oldest_call = min(self.calls)
                    time_until_expire = self.period - (time.time() - oldest_call)
                    sleep_time = max(0.1, min(1.0, time_until_expire))
                else:
                    sleep_time = 0.1

            # Check timeout
            if timeout is not None:
                elapsed = time.time() - start_time
                if elapsed >= timeout:
                    return False

            time.sleep(sleep_time)

    def reset(self):
        """Manually reset all call records (use with caution)."""
        with self.lock:
            self.calls = []

    def get_status(self):
        """
        Get current rate limiter status.

        Returns:
            dict: Status information including current calls and remaining slots
        """
        with self.lock:
            self._clean_old_calls()
            current = len(self.calls)
            return {
                'current_calls': current,
                'max_calls': self.max_calls,
                'remaining': self.max_calls - current,
                'period_seconds': self.period
            }
