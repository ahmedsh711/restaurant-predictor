from abc import ABC, abstractmethod
import pickle
import logging
import numpy as np
import onnxruntime as rt
from src.utils import timed
from src.logging_conf import get_logger

logger = get_logger(__name__)


# ─── Constants (must match train.py exactly) ──────────────────────────────────

TOP_REST_TYPES = [
    'Quick Bites', 'Casual Dining', 'Cafe', 'Delivery', 'Dessert Parlor',
    'Takeaway, Delivery', 'Casual Dining, Bar', 'Bakery', 'Beverage Shop',
    'Bar', 'Food Court', 'Sweet Shop', 'Bar, Casual Dining', 'Lounge', 'Pub'
]

LISTED_IN_TYPES = [
    'Cafes', 'Delivery', 'Desserts', 'Dine-out',
    'Drinks & nightlife', 'Pubs and bars'
]

TOP_CUISINES = [
    'north_indian', 'chinese', 'south_indian', 'fast_food', 'continental',
    'biryani', 'cafe', 'desserts', 'beverages', 'italian', 'street_food',
    'bakery', 'pizza', 'burger', 'seafood', 'andhra', 'ice_cream',
    'mughlai', 'american', 'asian'
]


class ModelBase(ABC):
    @abstractmethod
    def predict_one(self, features: dict) -> dict:
        pass

    @abstractmethod
    def predict_batch(self, features_list: list[dict]) -> list[dict]:
        pass


class ZomatoSuccessModel(ModelBase):
    """ONNX Runtime Model Loader with full 48-feature preprocessing"""

    def __init__(self, model_path: str, artifacts_dir: str = "models"):
        self.model_path = model_path
        self.artifacts_dir = artifacts_dir
        self._session = None
        self._location_freq_map = {}
        self._city_freq_map = {}

    def load(self):
        try:
            self._session = rt.InferenceSession(self.model_path)
            logging.info(f"Loaded ONNX model from {self.model_path}")
        except Exception as e:
            logging.error(f"Failed to load ONNX model at {self.model_path}: {e}")
            self._session = None

        # Load frequency maps for location_freq / city_freq encoding
        try:
            with open(f"{self.artifacts_dir}/location_freq_map.pkl", "rb") as f:
                self._location_freq_map = pickle.load(f)
            with open(f"{self.artifacts_dir}/city_freq_map.pkl", "rb") as f:
                self._city_freq_map = pickle.load(f)
            logging.info("Loaded frequency maps successfully")
        except Exception as e:
            logging.error(f"Failed to load frequency maps: {e}")

    @property
    def isloaded(self) -> bool:
        return self._session is not None

    def _preprocess(self, features: dict) -> np.ndarray:
        """
        Convert raw user input into the 49-feature vector
        the ONNX model expects. Mirrors train.py preprocessing exactly.
        """
        row = []

        # 1. online_order (binary)
        row.append(1.0 if features['online_order'] == 'Yes' else 0.0)

        # 2. book_table (binary)
        row.append(1.0 if features['book_table'] == 'Yes' else 0.0)

        # 3. votes (numeric)
        row.append(float(features['votes']))

        # 4. cost_for_two (numeric)
        row.append(float(features['cost_for_two']))

        # 5. listed_in(type) one-hot (6 columns)
        listed_type = features['listed_in_type']
        for lt in LISTED_IN_TYPES:
            row.append(1.0 if listed_type == lt else 0.0)

        # 6. rest_type one-hot (16 columns: 15 top + "Other") — sanitized names
        raw_rest_type = features['rest_type']
        rest_type = raw_rest_type if raw_rest_type in TOP_REST_TYPES else 'Other'
        for rt_name in TOP_REST_TYPES + ['Other']:
            row.append(1.0 if rest_type == rt_name else 0.0)

        # 7. location_freq (frequency encoded)
        location = features['location']
        row.append(float(self._location_freq_map.get(location, 0)))

        # 8. city_freq (frequency encoded)
        city = features['listed_in_city']
        row.append(float(self._city_freq_map.get(city, 0)))

        # 9. cuisine_count
        cuisines_raw = features['cuisines'].lower()
        cuisine_list = [c.strip() for c in cuisines_raw.split(',')]
        row.append(float(len(cuisine_list)))

        # 10. cuisine binary flags (20 columns)
        for cuisine in TOP_CUISINES:
            search_term = cuisine.replace('_', ' ')
            row.append(1.0 if search_term in cuisines_raw else 0.0)

        return np.array([row], dtype=np.float32)

    @timed
    def predict_one(self, features: dict) -> dict:
        if not self.isloaded:
            raise RuntimeError("Model not loaded. Call .load() first.")

        # Preprocess raw input → 49-feature vector
        input_array = self._preprocess(features)
        
        # Requirement: DEBUG level for feature vector
        logger.debug("Feature vector generated", input_array=input_array.tolist())

        # Run ONNX inference
        input_name = self._session.get_inputs()[0].name
        pred_onx = self._session.run(None, {input_name: input_array})

        # Extract probability
        probability = float(pred_onx[1][0][1])

        return {
            "success_probability": probability,
            "will_succeed": probability > 0.5
        }
        
    def predict_batch(self, features_list: list[dict]) -> list[dict]:
        """Predict success for a list of restaurants."""
        return [self.predict_one(features) for features in features_list]