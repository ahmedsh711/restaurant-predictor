import joblib
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier

from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import StringTensorType, FloatTensorType

preprocressor = ColumnTransformer(
    [
        ("cat", OneHotEncoder(), ["location", "cuisines_type"]),
    ]
)

pipeline = Pipeline([
    ("preprocessor", preprocressor),
    ("classifier", RandomForestClassifier(n_estimators=100, random_state=42))
])

# pipeline.fit(X_train, y_train)
joblib.dump(pipeline, 'models/restaurant_model.pkl')

# Define input feature types for ONNX (batch size is None, feature dimension is 1)
initial_types = [
    ('location', StringTensorType([None, 1])),
    ('cuisines_type', StringTensorType([None, 1])),
]

# Convert the scikit-learn pipeline to ONNX format
onnx_model = convert_sklearn(pipeline, initial_types=initial_types)

# Save the ONNX model to disk
with open("models/restaurant_model.onnx", "wb") as f:
    f.write(onnx_model.SerializeToString())