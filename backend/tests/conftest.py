import pytest
from fastapi.testclient import TestClient

from app.cache import cache
from app.main import app


@pytest.fixture(autouse=True)
def clear_cache():
    cache.clear()
    yield
    cache.clear()


@pytest.fixture
def client():
    return TestClient(app)
