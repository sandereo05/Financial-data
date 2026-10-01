import pytest

from app import cache as cache_module
from app.cache import _MISSING, TTLCache, cached


class FakeClock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


@pytest.fixture
def clock(monkeypatch):
    clock = FakeClock()
    monkeypatch.setattr(cache_module, "cache", TTLCache(clock))
    return clock


def test_get_and_expire():
    clock = FakeClock()
    store = TTLCache(clock)
    store.set("a", 1, ttl=10)
    assert store.get("a") == 1
    clock.now = 11
    assert store.get("a") is _MISSING
    assert store.get("a", allow_stale=True) == 1
    assert store.get("b") is _MISSING


def test_cached_reuses_result_until_ttl(clock):
    calls = []

    @cached(ttl=10)
    def square(x):
        calls.append(x)
        return x * x

    assert square(3) == 9
    assert square(3) == 9
    assert square(4) == 16
    assert calls == [3, 4]

    clock.now = 11
    assert square(3) == 9
    assert calls == [3, 4, 3]


def test_cached_falls_back_to_stale_value(clock):
    responses = iter([{"price": 1}, {}, {}])

    @cached(ttl=10, is_valid=bool)
    def fetch():
        return next(responses)

    assert fetch() == {"price": 1}
    clock.now = 11
    assert fetch() == {"price": 1}


def test_cached_does_not_store_invalid_results(clock):
    responses = iter([None, 5])

    @cached(ttl=10, is_valid=lambda value: value is not None)
    def fetch():
        return next(responses)

    assert fetch() is None
    assert fetch() == 5
