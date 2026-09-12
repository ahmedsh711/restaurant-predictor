# Zomato Restaurant Success Predictor

Predicts whether a restaurant on Zomato will achieve a rating ≥ 3.75, using a LightGBM model trained on 48 features (location, cuisine, pricing, ordering options) — exported to ONNX Runtime for fast, safe inference.

## Quickstart (3 commands)

```bash
git clone https://github.com/ahmedsh711/zomato-success-api.git && cd zomato-success-api
uv sync
uv run uvicorn src.api:app --reload
```

Visit `http://localhost:8000/docs` to explore the API interactively.

## Example Prediction

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -H "X-API-Key: supersecretkey" \
  -d '{
    "online_order": "Yes",
    "book_table": "No",
    "votes": 500,
    "cost_for_two": 800,
    "location": "BTM",
    "listed_in_city": "BTM",
    "rest_type": "Casual Dining",
    "listed_in_type": "Dine-out",
    "cuisines": "North Indian, Chinese, Biryani"
  }'
```

Expected response:

```json
{
  "success_probability": 0.82,
  "will_succeed": true,
  "model_version": "0.2.0",
  "correlation_id": "a1b2c3d4-...",
  "latency_ms": 1.2
}
```

## Run with Docker

```bash
docker build -f docker/Dockerfile -t zomato-success-api:0.2.0 .
docker run -p 8000:8000 zomato-success-api:0.2.0
```

## Endpoints

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| GET | `/health` | — | 200 if model is loaded in memory |
| GET | `/metadata` | — | Model version, features, artifact hash |
| POST | `/predict` | X-API-Key | Single restaurant prediction |
| POST | `/predict/batch` | X-API-Key | Batch predictions (list in, list out) |
| GET | `/docs` | — | Swagger UI |

## Repository Structure

```
zomato-success-api/
├── src/
│   ├── api.py              # FastAPI app — 4 endpoints + middleware
│   ├── model.py            # ONNX model loading & 48-feature preprocessing
│   ├── schemas.py          # Pydantic request/response models
│   ├── logging_conf.py     # Structured JSON logging with correlation IDs
│   └── train.py            # Training script (LightGBM → ONNX export)
├── tests/
│   ├── conftest.py         # Shared fixtures (sample_features, client, mock_model)
│   ├── test_api.py         # API endpoint tests (all 4 endpoints)
│   ├── test_features.py    # Feature engineering unit tests
│   ├── test_predict.py     # Model prediction tests
│   └── test_serialization.py  # ONNX–pickle parity tests
├── docker/
│   ├── Dockerfile          # Multi-stage build, non-root user, HEALTHCHECK
│   └── docker-compose.yml  # Service composition with env vars & volume
├── models/                 # Trained artifacts (git-ignored)
├── reports/
│   └── module-1.md         # Maturity self-assessment
├── Notebooks/              # Exploratory analysis
├── .dockerignore
├── pyproject.toml
└── README.md
```

## Development

```bash
# Install all dependencies (including dev)
uv sync --all-extras

# Run tests with coverage gate
uv run pytest -v

# Lint
uv run ruff check src tests

# Train model (requires data/raw/zomato.csv)
uv run python src/train.py

# Serve locally
uv run uvicorn src.api:app --reload
```