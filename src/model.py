from abc import ABC, abstractmethod
import numpy as np
import onnxruntime as rt

class ModelBase(ABC):
    @abstractmethod
    def predict(self, features: dict) -> dict:
        pass

class ZomatoSuccessModel(ModelBase):
    """Joblib / Scikit-Learn Pickle Model Loader"""
    def __init__(self, model_path: str):
        self.model_path = model_path
        self._session = None

    def load(self):
        self._session = rt.InferenceSession(self.model_path)

    @property
    def isloaded(self) -> bool:
        return self._session is not None

    def predict(self, features: dict) -> dict:
        if not self.isloaded:
            raise RuntimeError("Model not loaded. Call .load() first.")
        
        # Prepare inputs for ONNX Runtime:
        inputs = {
            'location': np.array([[features['location']]], dtype=object),
            'cuisine_type' : np.array([[features['cusine_type']]], dtype= 'object'),
            'approx_cost_for_two' : np.array([[features['approx_cost_for_two']]], dtype=np.float32),
            'online_order' : np.array([[features['online_order']]], dtype=np.float32),
        }

        # Run inference
        pred_onx = self._session.run(None, inputs)
        
        # Extract Probability (index 1 contains the probability dictionaries)
        probability = float(pred_onx[1][0][1])

        return{
            "success_probability" : probability,
            "will_succeed" : probability > 0.5,
            "threshold" : 3.75
        }
 