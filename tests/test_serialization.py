"""
Serialization parity and consistency tests.

Note: The ONNX model (restaurant_model.onnx) was trained with 49 features
and the pickle (restaurant_model.pkl) wraps a RandomForestClassifier trained
with 48 features — they are from separate training runs and cannot be directly
compared. This test suite verifies each format is internally consistent and
produces deterministic results.
"""
import pytest
import numpy as np
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


def test_onnx_model_deterministic(onnx_model):
    """ONNX model must produce identical results on repeated calls."""
    for features in SAMPLE_FEATURES:
        r1 = onnx_model.predict(features)
        r2 = onnx_model.predict(features)
        assert r1["success_probability"] == r2["success_probability"], (
            f"ONNX model is non-deterministic: {r1} vs {r2}"
        )


def test_onnx_probabilities_in_range(onnx_model):
    """All ONNX probabilities must be in [0, 1]."""
    for features in SAMPLE_FEATURES:
        result = onnx_model.predict(features)
        assert 0.0 <= result["success_probability"] <= 1.0


def test_onnx_will_succeed_matches_probability(onnx_model):
    """will_succeed must be True iff success_probability > 0.5."""
    for features in SAMPLE_FEATURES:
        result = onnx_model.predict(features)
        assert result["will_succeed"] == (result["success_probability"] > 0.5)


def test_onnx_feature_vector_size(onnx_model):
    """Preprocessing must produce the correct feature count (49) for the ONNX model."""
    vec = onnx_model._preprocess(SAMPLE_FEATURES[0])
    input_shape = onnx_model._session.get_inputs()[0].shape
    expected_features = input_shape[1]
    assert vec.shape == (1, expected_features), (
        f"Feature vector shape {vec.shape} does not match model input {input_shape}"
    )


def test_pickle_model_loads():
    """Pickle model file must be loadable without errors."""
    import pickle
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with open("models/restaurant_model.pkl", "rb") as f:
            pkl_model = pickle.load(f)
    assert pkl_model is not None
    assert hasattr(pkl_model, "predict")
