import os
import pathlib
import pickle
import logging
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from lightgbm import LGBMClassifier

from skl2onnx import convert_sklearn, update_registered_converter
from skl2onnx.common.data_types import FloatTensorType, StringTensorType, Int64TensorType
from skl2onnx.common.shape_calculator import calculate_linear_classifier_output_shapes
from onnxmltools.convert.lightgbm.operator_converters.LightGbm import convert_lightgbm

logging.basicConfig(level=logging.INFO)

# ─── Constants (aligned with the notebook's preprocessing) ───────────────────

TOP_REST_TYPES = [
    'Quick Bites', 'Casual Dining', 'Cafe', 'Delivery', 'Dessert Parlor',
    'Takeaway, Delivery', 'Casual Dining, Bar', 'Bakery', 'Beverage Shop',
    'Bar', 'Food Court', 'Sweet Shop', 'Bar, Casual Dining', 'Lounge', 'Pub'
]

LISTED_IN_TYPES = [
    'Cafes', 'Delivery', 'Desserts', 'Dine-out',
    'Drinks & nightlife', 'Pubs and bars'
    # 'Buffet' excluded — same as notebook (it's dropped to avoid dummy trap)
]

TOP_CUISINES = [
    'north_indian', 'chinese', 'south_indian', 'fast_food', 'continental',
    'biryani', 'cafe', 'desserts', 'beverages', 'italian', 'street_food',
    'bakery', 'pizza', 'burger', 'seafood', 'andhra', 'ice_cream',
    'mughlai', 'american', 'asian'
]


def clean_rate(x):
    try:
        return float(x.split('/')[0].strip())
    except (ValueError, AttributeError, IndexError):
        return float('nan')


def create_target(x):
    return 1 if x >= 3.75 else 0


# ─── 1. Load & Clean Data ────────────────────────────────────────────────────

logging.info("Loading data...")
dataset_path = pathlib.Path(__file__).parent.parent / "data" / "raw" / "zomato.csv"
df = pd.read_csv(dataset_path)

logging.info("Cleaning data...")
df['rate'] = df['rate'].apply(clean_rate)
df.dropna(subset=['rate'], inplace=True)
df['success'] = df['rate'].apply(create_target)

# Cost cleaning
df['cost_for_two'] = df['approx_cost(for two people)'].str.replace(',', '').astype(float)

# online_order / book_table → binary
df['online_order'] = df['online_order'].map({'Yes': 1, 'No': 0})
df['book_table'] = df['book_table'].map({'Yes': 1, 'No': 0})

# rest_type → keep top categories, bucket the rest into "Other"
df['rest_type'] = df['rest_type'].apply(
    lambda x: x if x in TOP_REST_TYPES else 'Other'
)

# One-hot encode rest_type (notebook uses get_dummies) — sanitize for LightGBM
for rt in TOP_REST_TYPES + ['Other']:
    safe_name = rt.replace(', ', '_').replace(' ', '_')
    col_name = f'rest_type_{safe_name}'
    df[col_name] = (df['rest_type'] == rt).astype(int)

# One-hot encode listed_in(type) — sanitize names for LightGBM
for lt in LISTED_IN_TYPES:
    safe_name = lt.replace(' & ', '_and_').replace(' ', '_').replace('(', '').replace(')', '')
    col_name = f'listed_in_type_{safe_name}'
    df[col_name] = (df['listed_in(type)'] == lt).astype(int)

# Frequency encoding for location and listed_in(city) — exactly like the notebook
location_freq_map = df['location'].value_counts().to_dict()
city_freq_map = df['listed_in(city)'].value_counts().to_dict()
df['location_freq'] = df['location'].map(location_freq_map)
df['city_freq'] = df['listed_in(city)'].map(city_freq_map)

# Cuisine processing — multi-label binarizer style (like the notebook)
df['cuisines_clean'] = df['cuisines'].fillna('').str.lower()
df['cuisine_count'] = df['cuisines_clean'].apply(
    lambda x: len(x.split(', ')) if x else 0
)
for cuisine in TOP_CUISINES:
    search_term = cuisine.replace('_', ' ')
    df[f'cuisine_{cuisine}'] = df['cuisines_clean'].str.contains(search_term, na=False).astype(int)


# ─── 2. Build Feature Matrix (48 features — same as notebook) ────────────────

feature_columns = (
    ['online_order', 'book_table', 'votes', 'cost_for_two']
    + [f'listed_in_type_{lt.replace(" & ", "_and_").replace(" ", "_").replace("(", "").replace(")", "")}' for lt in LISTED_IN_TYPES]
    + [f'rest_type_{rt.replace(", ", "_").replace(" ", "_")}' for rt in TOP_REST_TYPES + ['Other']]
    + ['location_freq', 'city_freq', 'cuisine_count']
    + [f'cuisine_{c}' for c in TOP_CUISINES]
)

logging.info(f"Total features: {len(feature_columns)}")
df = df.dropna(subset=feature_columns + ['success'])
X_train = df[feature_columns].astype(float)
y_train = df['success']
logging.info(f"Training samples: {len(X_train)}")


# ─── 3. Train LightGBM (best hyperparameters from notebook) ──────────────────

logging.info("Training model...")
model = LGBMClassifier(
    learning_rate=0.01,
    max_depth=15,
    n_estimators=500,
    num_leaves=20,
    random_state=42
)
model.fit(X_train, y_train)

accuracy = model.score(X_train, y_train)
logging.info(f"Training accuracy: {accuracy:.4f}")


# ─── 4. Save artifacts ───────────────────────────────────────────────────────

logging.info("Saving model...")
os.makedirs("models", exist_ok=True)

# Save as pickle (for direct Streamlit/Python usage)
with open("models/restaurant_model.pkl", "wb") as f:
    pickle.dump(model, f)

# Save feature names
with open("models/feature_names.pkl", "wb") as f:
    pickle.dump(feature_columns, f)

# Save frequency maps (needed at inference time for location_freq / city_freq)
with open("models/location_freq_map.pkl", "wb") as f:
    pickle.dump(location_freq_map, f)

with open("models/city_freq_map.pkl", "wb") as f:
    pickle.dump(city_freq_map, f)

# Save ONNX version
logging.info("Converting to ONNX...")
update_registered_converter(
    LGBMClassifier,
    'LightGbmLGBMClassifier',
    calculate_linear_classifier_output_shapes,
    convert_lightgbm,
    options={'nocl': [True, False], 'zipmap': [True, False, 'columns']}
)

initial_types = [('features', FloatTensorType([None, len(feature_columns)]))]

onnx_model = convert_sklearn(
    model,
    initial_types=initial_types,
    target_opset={'': 12, 'ai.onnx.ml': 3}
)

with open("models/restaurant_model.onnx", "wb") as f:
    f.write(onnx_model.SerializeToString())

logging.info(f"Done! Saved {len(feature_columns)}-feature model to models/")
logging.info("Artifacts: restaurant_model.onnx, restaurant_model.pkl, feature_names.pkl, location_freq_map.pkl, city_freq_map.pkl")
