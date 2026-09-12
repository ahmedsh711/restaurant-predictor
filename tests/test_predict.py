import pytest
from src.model import ZomatoSuccessModel


def test_prediction_returns_float(trained_model, sample_features):
    """The prediction must return a float probability in [0, 1]."""
    result = trained_model.predict_one(sample_features)
    assert isinstance(result["success_probability"], float)
    assert 0.0 <= result["success_probability"] <= 1.0


def test_prediction_returns_bool(trained_model, sample_features):
    """will_succeed must be a boolean."""
    result = trained_model.predict_one(sample_features)
    assert isinstance(result["will_succeed"], bool)


def test_prediction_deterministic(trained_model, sample_features):
    """Two identical calls must return the exact same probability."""
    r1 = trained_model.predict_one(sample_features)
    r2 = trained_model.predict_one(sample_features)
    assert r1["success_probability"] == r2["success_probability"]


def test_will_succeed_consistent_with_probability(trained_model, sample_features):
    """will_succeed must be True iff success_probability > 0.5."""
    result = trained_model.predict_one(sample_features)
    assert result["will_succeed"] == (result["success_probability"] > 0.5)


def test_prediction_sane_range(trained_model):
    """Probability must be in [0, 1] for various inputs."""
    test_cases = [
        {"online_order": "Yes", "book_table": "Yes", "votes": 5000, "cost_for_two": 5000.0,
         "location": "Indiranagar", "listed_in_city": "Indiranagar",
         "rest_type": "Casual Dining", "listed_in_type": "Dine-out",
         "cuisines": "North Indian, Continental, Italian"},
        {"online_order": "No", "book_table": "No", "votes": 1, "cost_for_two": 50.0,
         "location": "UnknownPlace", "listed_in_city": "UnknownCity",
         "rest_type": "Other", "listed_in_type": "Delivery",
         "cuisines": "Fast Food"},
    ]
    for features in test_cases:
        result = trained_model.predict_one(features)
        assert 0.0 <= result["success_probability"] <= 1.0
