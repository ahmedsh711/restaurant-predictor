from abc import ABC, abstractmethod
import os
import joblib
import numpy as np
import pandas as pd
import onnxruntime as rt

class ModelBase(ABC):
    @abstractmethod
    def predict(self, features: dict) -> dict:
        pass

class ZomatoSuccessModel(ModelBase):
    """Joblib / Scikit-Learn Pickle Model Loader"""
    def __init__(self, model_path: str):
        self.model_path = model_path
        self._pipeline = None

    def load(self):
        self._pipeline = joblib.load(self.model_path)

    @property
    def isloaded(self) -> bool:
        return self._pipeline is not None

    def predict(self, features: dict) -> dict:
        if not self.isloaded:
            raise RuntimeError("Model not loaded. Call .load() first.")
        
        df = pd.DataFrame([features])
        probability = float(self._pipeline.predict_proba(df)[0][1])

        return {
            "success_probability": probability,
            "will_succeed": probability >= 0.5,
            "threshold": 3.75
        }

class ZomatoSuccessONNXModel(ModelBase):
    """ONNX Model Loader using ONNX Runtime"""
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.session = None

    def load(self):
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"ONNX model file not found at: {self.model_path}")
        self.session = rt.InferenceSession(self.model_path)

    @property
    def isloaded(self) -> bool:
        return self.session is not None

    def predict(self, features: dict) -> dict:
        if not self.isloaded:
            raise RuntimeError("ONNX Model not loaded. Call .load() first.")
        
        # Prepare inputs mapping for ONNX Runtime session
        inputs = {}
        for inp in self.session.get_inputs():
            name = inp.name
            val = features.get(name, "")
            if isinstance(val, str):
                inputs[name] = np.array([[val]], dtype=object)
            elif isinstance(val, (int, float)):
                inputs[name] = np.array([[val]], dtype=np.float32)
            else:
                inputs[name] = np.array([[val]])

        # Execute model prediction
        outputs = self.session.run(None, inputs)
        
        # Extract prediction probabilities (skl2onnx outputs label at [0], probabilities at [1])
        prob_output = outputs[1]
        if isinstance(prob_output, list) and isinstance(prob_output[0], dict):
            probability = float(prob_output[0].get(1, 0.0))
        elif isinstance(prob_output, np.ndarray):
            probability = float(prob_output[0][1])
        else:
            probability = float(prob_output)

        return {
            "success_probability": probability,
            "will_succeed": probability >= 0.5,
            "threshold": 3.75
        }