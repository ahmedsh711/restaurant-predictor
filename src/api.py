import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from src.model import ZomatoSuccessModel
from src.schemas import PredictRequest

model: ZomatoSuccessONNXModel | None = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    # Load ONNX model on server startup
    model_path = os.getenv("MODEL_PATH", "models/restaurant_model.onnx")
    model = ZomatoSuccessONNXModel(model_path)
    model.load()
    yield

app = FastAPI(
    title="Zomato Success Prediction API",
    version="0.1.0",
    lifespan=lifespan
)

@app.post("/predict")
async def predict(request: PredictRequest):
    if model is None or not model.isloaded:
        raise HTTPException(status_code=503, detail="Model is not loaded")
    
    features = request.model_dump()
    return model.predict(features)