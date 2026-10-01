"""Small thread-safe in-memory cache with per-entry TTL.

Expired entries are kept so they can be served as a fallback when a refresh
fails, e.g. when Yahoo rate limits us.
"""

import functools
import logging
import threading
import time
from collections.abc import Callable
from typing import Any

logger = logging.getLogger(__name__)

PRICE_TTL = 15 * 60
FUNDAMENTALS_TTL = 24 * 60 * 60

_MISSING = object()


class TTLCache:
    def __init__(self, clock: Callable[[], float] = time.monotonic):
        self._clock = clock
        self._store: dict[Any, tuple[float, Any]] = {}
        self._lock = threading.Lock()

    def get(self, key, allow_stale: bool = False):
        """Return the cached value, or _MISSING if absent (or expired unless allow_stale)."""
        with self._lock:
            entry = self._store.get(key)
        if entry is None:
            return _MISSING
        expires_at, value = entry
        if allow_stale or self._clock() < expires_at:
            return value
        return _MISSING

    def set(self, key, value, ttl: float) -> None:
        with self._lock:
            self._store[key] = (self._clock() + ttl, value)

    def clear(self) -> None:
        with self._lock:
            self._store.clear()


cache = TTLCache()


def cached(ttl: float, is_valid: Callable[[Any], bool] = lambda value: True):
    """Cache a function's results by its arguments.

    Results failing `is_valid` are not stored; if an expired value exists it is
    returned instead, so a failing upstream degrades to slightly old data.
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args):
            key = (func.__qualname__, args)
            value = cache.get(key)
            if value is not _MISSING:
                return value

            result = func(*args)
            if is_valid(result):
                cache.set(key, result, ttl)
                return result

            stale = cache.get(key, allow_stale=True)
            if stale is not _MISSING:
                logger.warning("Serving stale data for %s%s", func.__name__, args)
                return stale
            return result

        return wrapper

    return decorator
