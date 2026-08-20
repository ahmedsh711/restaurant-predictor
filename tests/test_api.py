from fastapi.testclient import TestClient
from src.api import app


def test_predict_endpoint_success():
    with TestClient(app) as client:
        payload = {
            "online_order": "Yes",
            "book_table": "No",
            "votes": 500,
            "cost_for_two": 800,
            "location": "BTM",
            "listed_in_city": "BTM",
            "rest_type": "Casual Dining",
            "listed_in_type": "Dine-out",
            "cuisines": "North Indian, Chinese, Biryani"
        }
        headers = {"X-API-Key": "supersecretkey"}
        response = client.post("/predict", json=payload, headers=headers)
        assert response.status_code == 200
        data = response.json()
        assert "will_succeed" in data
        assert "success_probability" in data
        assert 0.0 <= data["success_probability"] <= 1.0


def test_predict_validation_error():
    with TestClient(app) as client:
        payload = {
            "online_order": "Yes",
            "book_table": "No",
            "votes": -10,
            "cost_for_two": 800,
            "location": "BTM",
            "listed_in_city": "BTM",
            "rest_type": "Casual Dining",
            "listed_in_type": "Dine-out",
            "cuisines": "North Indian"
        }
        headers = {"X-API-Key": "supersecretkey"}
        response = client.post("/predict", json=payload, headers=headers)
        assert response.status_code == 422


def test_predict_unauthorized():
    with TestClient(app) as client:
        payload = {
            "online_order": "Yes",
            "book_table": "No",
            "votes": 500,
            "cost_for_two": 800,
            "location": "BTM",
            "listed_in_city": "BTM",
            "rest_type": "Casual Dining",
            "listed_in_type": "Dine-out",
            "cuisines": "North Indian"
        }
        response = client.post("/predict", json=payload)
        assert response.status_code == 401


def test_predict_unknown_location():
    """Unknown locations should still work — they get frequency 0."""
    with TestClient(app) as client:
        payload = {
            "online_order": "No",
            "book_table": "No",
            "votes": 10,
            "cost_for_two": 300,
            "location": "SomeUnknownPlace",
            "listed_in_city": "SomeUnknownCity",
            "rest_type": "SomeUnknownType",
            "listed_in_type": "Delivery",
            "cuisines": "North Indian"
        }
        headers = {"X-API-Key": "supersecretkey"}
        response = client.post("/predict", json=payload, headers=headers)
        assert response.status_code == 200
        data = response.json()
        assert "will_succeed" in data