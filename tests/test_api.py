import pytest
from unittest.mock import MagicMock
from fastapi.testclient import TestClient
from src.api import app


API_KEY = "supersecretkey"
HEADERS = {"X-API-Key": API_KEY}

VALID_PAYLOAD = {
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


def test_health_endpoint(client):
    """GET /health returns 200 when model is loaded."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["model_loaded"] is True


def test_metadata_endpoint(client):
    """GET /metadata returns model metadata."""
    response = client.get("/metadata")
    assert response.status_code == 200
    data = response.json()
    assert "model_version" in data
    assert "framework" in data
    assert "feature_count" in data


def test_predict_happy_path(client, sample_features):
    """POST /predict returns a valid prediction."""
    response = client.post("/predict", json=sample_features, headers=HEADERS)
    assert response.status_code == 200
    data = response.json()
    assert "will_succeed" in data
    assert "success_probability" in data
    assert "model_version" in data
    assert "correlation_id" in data
    assert "latency_ms" in data
    assert 0.0 <= data["success_probability"] <= 1.0


def test_predict_invalid_payload_returns_422(client):
    """Invalid payload (negative votes) must return 422."""
    bad_payload = {**VALID_PAYLOAD, "votes": -10}
    response = client.post("/predict", json=bad_payload, headers=HEADERS)
    assert response.status_code == 422


def test_predict_missing_api_key_returns_401(client):
    """Missing API key must return 401."""
    response = client.post("/predict", json=VALID_PAYLOAD)
    assert response.status_code == 401


def test_predict_wrong_api_key_returns_403(client):
    """Wrong API key must return 403."""
    response = client.post("/predict", json=VALID_PAYLOAD, headers={"X-API-Key": "wrongkey"})
    assert response.status_code == 403


def test_predict_unknown_location(client):
    """Unknown location should still return 200 (falls back to freq=0)."""
    payload = {**VALID_PAYLOAD, "location": "UnknownPlace", "listed_in_city": "UnknownCity"}
    response = client.post("/predict", json=payload, headers=HEADERS)
    assert response.status_code == 200


def test_predict_response_schema_matches(client, sample_features):
    """Response schema must include all required fields with correct types."""
    response = client.post("/predict", json=sample_features, headers=HEADERS)
    data = response.json()
    assert isinstance(data["success_probability"], float)
    assert isinstance(data["will_succeed"], bool)
    assert isinstance(data["model_version"], str)
    assert isinstance(data["correlation_id"], str)
    assert isinstance(data["latency_ms"], float)


def test_x_request_id_in_response_header(client, sample_features):
    """Response must include X-Request-ID header."""
    response = client.post("/predict", json=sample_features, headers=HEADERS)
    assert "x-request-id" in response.headers


def test_predict_batch_happy_path(client):
    """POST /predict/batch returns results for all items."""
    batch_payload = {"items": [VALID_PAYLOAD, VALID_PAYLOAD]}
    response = client.post("/predict/batch", json=batch_payload, headers=HEADERS)
    assert response.status_code == 200
    data = response.json()
    assert "results" in data
    assert len(data["results"]) == 2
    for result in data["results"]:
        assert "success_probability" in result
        assert "will_succeed" in result


def test_predict_with_mock_model(client, monkeypatch):
    """Verify API contract without a real model using monkeypatch."""
    mock = MagicMock()
    mock.isloaded = True
    mock.predict_one.return_value = {"success_probability": 0.9, "will_succeed": True}

    # Temporarily replace the loaded model on the running app
    original_model = client.app.state.model
    monkeypatch.setattr(client.app.state, "model", mock)

    response = client.post("/predict", json=VALID_PAYLOAD, headers=HEADERS)

    # Restore original model after test
    monkeypatch.setattr(client.app.state, "model", original_model)

    assert response.status_code == 200
    data = response.json()
    assert data["success_probability"] == 0.9
    assert data["will_succeed"] is True
    mock.predict_one.assert_called_once()
