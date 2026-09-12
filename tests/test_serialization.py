"""
Serialization parity and consistency tests.
"""
import pytest
import numpy as np
import pickle
import warnings
from src.model import ZomatoSuccessModel

SAMPLE_FEATURES = [
    {
        "online_order": "Yes", "book_table": "No", "votes": 500,
        "cost_for_two": 800.0, "location": "BTM", "listed_in_city": "BTM",
        "rest_type": "Casual Dining", "listed_in_type": "Dine-out",
        "cuisines": "North Indian, Chinese, Biryani",
    },
    {
        "online_order": "No", "book_table": "No", "votes": 10,
        "cost_for_two": 200.0, "location": "Whitefield", "listed_in_city": "Whitefield",
        "rest_type": "Quick Bites", "listed_in_type": "Delivery",
        "cuisines": "Fast Food",
    },
    {
        "online_order": "Yes", "book_table": "Yes", "votes": 2000,
        "cost_for_two": 3000.0, "location": "Indiranagar", "listed_in_city": "Indiranagar",
        "rest_type": "Casual Dining", "listed_in_type": "Dine-out",
        "cuisines": "North Indian, Continental, Italian, Seafood",
    },
]


@pytest.fixture(scope="module")
def onnx_model():
    model = ZomatoSuccessModel("models/restaurant_model.onnx", "models")
    model.load()
    return model


def test_pickle_onnx_parity(onnx_model):
    """Parity test: Pickle and ONNX models should predict the same probabilities."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with open("models/restaurant_model.pkl", "rb") as f:
            pkl_model = pickle.load(f)

    for features in SAMPLE_FEATURES:
        # Get ONNX prob
        onnx_result = onnx_model.predict_one(features)
        onnx_prob = onnx_result["success_probability"]

        # Get Pickle prob
        input_array = onnx_model._preprocess(features)
        # Pickle model is a LightGBM Classifier, so we use predict_proba
        pkl_prob = float(pkl_model.predict_proba(input_array)[0][1])

        # Assert parity
        assert np.allclose(pkl_prob, onnx_prob, atol=1e-4), (
            f"Parity failure: Pickle={pkl_prob}, ONNX={onnx_prob}"
        )


def test_onnx_model_deterministic(onnx_model):
    """ONNX model must produce identical results on repeated calls."""
    for features in SAMPLE_FEATURES:
        r1 = onnx_model.predict_one(features)
        r2 = onnx_model.predict_one(features)
        assert r1["success_probability"] == r2["success_probability"], (
            f"ONNX model is non-deterministic: {r1} vs {r2}"
        )


def test_onnx_probabilities_in_range(onnx_model):
    """All ONNX probabilities must be in [0, 1]."""
    for features in SAMPLE_FEATURES:
        result = onnx_model.predict_one(features)
        assert 0.0 <= result["success_probability"] <= 1.0


def test_onnx_will_succeed_matches_probability(onnx_model):
    """will_succeed must be True iff success_probability > 0.5."""
    for features in SAMPLE_FEATURES:
        result = onnx_model.predict_one(features)
        assert result["will_succeed"] == (result["success_probability"] > 0.5)


def test_onnx_feature_vector_size(onnx_model):
    """Preprocessing must produce the correct feature count (49) for the ONNX model."""
    vec = onnx_model._preprocess(SAMPLE_FEATURES[0])
    input_shape = onnx_model._session.get_inputs()[0].shape
    expected_features = input_shape[1]
    assert vec.shape == (1, expected_features), (
        f"Feature vector shape {vec.shape} does not match model input {input_shape}"
    )
