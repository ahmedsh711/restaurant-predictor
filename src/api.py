import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Depends, Security
from fastapi.security import APIKeyHeader
from src.model import ZomatoSuccessModel
from src.schemas import PredictRequest

API_KEY = os.getenv("API_KEY", "supersecretkey")
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=True)


def get_api_key(api_key_header: str = Security(api_key_header)):
    if api_key_header != API_KEY:
        raise HTTPException(status_code=403, detail="Could not validate API key")
    return api_key_header


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load ONNX model + frequency maps on server startup
    model_path = os.getenv("MODEL_PATH", "models/restaurant_model.onnx")
    artifacts_dir = os.getenv("ARTIFACTS_DIR", "models")
    model = ZomatoSuccessModel(model_path, artifacts_dir)
    model.load()
    app.state.model = model
    yield
    app.state.model = None


app = FastAPI(
    title="Zomato Success Prediction API",
    version="0.2.0",
    description="Predicts restaurant success on Zomato using 48 features from LightGBM + ONNX Runtime",
    lifespan=lifespan
)


@app.post("/predict")
def predict(request: PredictRequest, api_key: str = Depends(get_api_key)):
    model = app.state.model
    if model is None or not model.isloaded:
        raise HTTPException(status_code=503, detail="Model is not loaded")

    features = request.model_dump()
    return model.predict(features)