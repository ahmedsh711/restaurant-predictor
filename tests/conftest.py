import pytest
from unittest.mock import MagicMock
from fastapi.testclient import TestClient

SAMPLE_FEATURES = {
    "online_order": "Yes",
    "book_table": "No",
    "votes": 500,
    "cost_for_two": 800.0,
    "location": "BTM",
    "listed_in_city": "BTM",
    "rest_type": "Casual Dining",
    "listed_in_type": "Dine-out",
    "cuisines": "North Indian, Chinese, Biryani",
}

API_HEADERS = {"X-API-Key": "supersecretkey"}


@pytest.fixture(scope="session")
def sample_features() -> dict:
    """A valid, complete set of restaurant features."""
    return SAMPLE_FEATURES.copy()


@pytest.fixture(scope="session")
def trained_model():
    """Session-scoped real model loaded from disk."""
    from src.model import ZomatoSuccessModel
    model = ZomatoSuccessModel("models/restaurant_model.onnx", "models")
    model.load()
    return model


@pytest.fixture(scope="session")
def client():
    """FastAPI TestClient with the real app (model loaded via lifespan)."""
    from src.api import app
    with TestClient(app) as c:
        yield c


@pytest.fixture
def mock_model():
    """A mock model for tests that should not depend on disk artifacts."""
    m = MagicMock()
    m.isloaded = True
    m.predict.return_value = {"success_probability": 0.75, "will_succeed": True}
    return m
