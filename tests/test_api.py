from fastapi.testclient import TestClient
from src.api import app


def test_predict_endpoint_success():
    with TestClient(app) as client:
        payload = {
            "location" : "BTM",
            "cuisine_type" : "cafe",
            "approx_cost_for_two" : 500,
            "online_order" : True
        }
        response = client.post("/predict", json=payload)
        assert response.status_code == 200
        assert "will_succeed" in response.json()

def test_predict_validation_error():
    with TestClient(app) as client:
        payload = {
            "location" : "BTM",
            "cuisine_type" : "cafe",
            "approx_cost_for_two" : -500,
            "online_order" : True
        }
        response = client.post("/predict", json=payload)
        assert response.status_code == 422