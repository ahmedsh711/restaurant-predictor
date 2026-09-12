import pytest
import numpy as np
from src.model import ZomatoSuccessModel


@pytest.fixture(scope="module")
def model_no_session():
    """Model without loaded ONNX session (for preprocessing tests)."""
    m = ZomatoSuccessModel.__new__(ZomatoSuccessModel)
    m.model_path = "models/restaurant_model.onnx"
    m.artifacts_dir = "models"
    m._session = None
    m._location_freq_map = {"BTM": 100, "Indiranagar": 80}
    m._city_freq_map = {"BTM": 90}
    return m


def test_feature_vector_shape(model_no_session):
    """The preprocessed feature vector must have exactly 49 features."""
    features = {
        "online_order": "Yes", "book_table": "No", "votes": 500,
        "cost_for_two": 800.0, "location": "BTM", "listed_in_city": "BTM",
        "rest_type": "Casual Dining", "listed_in_type": "Dine-out",
        "cuisines": "North Indian, Chinese",
    }
    vec = model_no_session._preprocess(features)
    assert vec.shape == (1, 49), f"Expected (1, 49), got {vec.shape}"


def test_feature_dtype(model_no_session):
    """All features must be float32."""
    features = {
        "online_order": "No", "book_table": "No", "votes": 0,
        "cost_for_two": 100.0, "location": "BTM", "listed_in_city": "BTM",
        "rest_type": "Quick Bites", "listed_in_type": "Delivery",
        "cuisines": "Fast Food",
    }
    vec = model_no_session._preprocess(features)
    assert vec.dtype == np.float32


@pytest.mark.parametrize("online_order,expected", [("Yes", 1.0), ("No", 0.0)])
def test_online_order_encoding(model_no_session, online_order, expected):
    features = {
        "online_order": online_order, "book_table": "No", "votes": 0,
        "cost_for_two": 100.0, "location": "BTM", "listed_in_city": "BTM",
        "rest_type": "Quick Bites", "listed_in_type": "Delivery",
        "cuisines": "Fast Food",
    }
    vec = model_no_session._preprocess(features)
    assert vec[0][0] == expected


def test_unknown_location_gets_zero_freq(model_no_session):
    """An unseen location should map to frequency 0."""
    features = {
        "online_order": "No", "book_table": "No", "votes": 10,
        "cost_for_two": 300.0, "location": "UnknownCity_XYZ",
        "listed_in_city": "UnknownCity_XYZ", "rest_type": "Other",
        "listed_in_type": "Delivery", "cuisines": "Fast Food",
    }
    vec = model_no_session._preprocess(features)
    # location_freq is at index 4 + 6 + 16 = 26
    # Actually it is at index: 4 (base) + 6 (listed_in_type) + 16 (rest_type) = 26
    assert vec[0][26] == 0.0, f"Expected 0.0 for unknown location, got {vec[0][26]}"


@pytest.mark.parametrize("rest_type", ["Quick Bites", "Casual Dining", "Cafe", "UnknownRestaurantType"])
def test_rest_type_one_hot_sum(model_no_session, rest_type):
    """Rest type one-hot columns must always sum to exactly 1."""
    features = {
        "online_order": "Yes", "book_table": "No", "votes": 100,
        "cost_for_two": 500.0, "location": "BTM", "listed_in_city": "BTM",
        "rest_type": rest_type, "listed_in_type": "Delivery",
        "cuisines": "Fast Food",
    }
    vec = model_no_session._preprocess(features)
    # rest_type one-hot: indices 10..25 (16 columns)
    rest_type_slice = vec[0][10:26]
    assert rest_type_slice.sum() == 1.0


def test_zero_votes(model_no_session):
    """Zero votes should produce a valid feature vector without error."""
    features = {
        "online_order": "No", "book_table": "No", "votes": 0,
        "cost_for_two": 100.0, "location": "BTM", "listed_in_city": "BTM",
        "rest_type": "Delivery", "listed_in_type": "Delivery",
        "cuisines": "Fast Food",
    }
    vec = model_no_session._preprocess(features)
    assert vec[0][2] == 0.0  # votes at index 2
