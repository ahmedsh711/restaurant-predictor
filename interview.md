# 🎤 Zomato Success Prediction API — Interview Reference Guide

> **Project:** Zomato Restaurant Success Prediction API  
> **Author:** Ahmed Al-Shobaki  
> **Stack:** Python · FastAPI · LightGBM · ONNX · Pydantic · uv · Docker  
> **Goal:** Transform a Jupyter notebook ML experiment into a production-ready MLOps microservice.

---

## 1. Project Overview — "What did you build?"

I built an end-to-end **MLOps pipeline** that takes a trained machine learning model from a Jupyter notebook and deploys it as a **REST API** using FastAPI. The API predicts whether a restaurant on Zomato will be **successful** (rated ≥ 3.75/5) based on four features:

| Feature | Type | Example |
|---|---|---|
| `location` | String | `"BTM"`, `"Koramangala"` |
| `cuisine_type` | String | `"North Indian"`, `"cafe"` |
| `approx_cost_for_two` | Float | `500.0` |
| `online_order` | String | `"Yes"` / `"No"` |

The system accepts a JSON request, runs it through the model, and returns:
```json
{
  "success_probability": 0.82,
  "will_succeed": true
}
```

---

## 2. Project Architecture — "How is it structured?"

```
zomato-success-api/
├── data/
│   └── raw/
│       └── zomato.csv           ← Original Kaggle dataset (~51K restaurants)
├── models/
│   └── restaurant_model.onnx    ← Serialized ML model (ONNX format)
├── src/
│   ├── __init__.py
│   ├── train.py                 ← Training pipeline (data → ONNX model)
│   ├── api.py                   ← FastAPI application + routes
│   ├── model.py                 ← ONNX Runtime inference wrapper
│   └── schemas.py               ← Pydantic input validation schemas
├── tests/
│   └── test_api.py              ← Automated API tests (pytest)
├── Notebooks/
│   └── Zomato ML.ipynb          ← Original EDA + model comparison notebook
├── Dockerfile                   ← Multi-stage Docker build
├── pyproject.toml               ← Dependencies & project metadata (uv)
└── uv.lock                      ← Lockfile for reproducibility
```

### Data Flow Diagram

```
User Request (JSON)
       │
       ▼
   FastAPI (/predict)          ← api.py: validates input, checks auth
       │
       ▼
   Pydantic Validation         ← schemas.py: type checks, constraints
       │
       ▼
   ONNX Runtime Inference      ← model.py: feeds data to ONNX session
       │
       ▼
   ONNX Model                  ← restaurant_model.onnx: preprocessing + LightGBM
       │
       ▼
   JSON Response               ← {"success_probability": 0.82, "will_succeed": true}
```

---

## 3. The ML Model — "What model did you use and why?"

### Model Selection Process (from the Notebook)

In the Jupyter notebook, I compared **7 classifiers** using GridSearchCV:

| Model | Best Accuracy |
|---|---|
| Logistic Regression | ~72% |
| Decision Tree | ~78% |
| Random Forest | ~80% |
| SVM | ~74% |
| KNN | ~76% |
| XGBoost | ~82% |
| **LightGBM** | **~83%** ✅ |

**LightGBM won** because it achieved the highest accuracy while training significantly faster than XGBoost.

### Best Hyperparameters (from GridSearchCV)

```python
LGBMClassifier(
    learning_rate=0.01,   # Small steps → better generalization
    max_depth=15,         # Deep trees → captures complex patterns
    n_estimators=500,     # 500 boosting rounds
    num_leaves=20,        # Controls model complexity (< 2^max_depth)
    random_state=42       # Reproducibility
)
```

### Why these hyperparameters matter:

- **`learning_rate=0.01`**: A small learning rate means each tree contributes a small correction. This prevents overfitting and produces smoother convergence, but requires more trees (`n_estimators=500`) to compensate.
- **`max_depth=15`**: Controls the maximum depth of each decision tree. Deeper trees capture more complex feature interactions (e.g., "BTM + cafe + online ordering = high success").
- **`num_leaves=20`**: LightGBM grows trees **leaf-wise** (not level-wise like XGBoost). Setting `num_leaves < 2^max_depth` (20 < 32768) prevents the tree from becoming too complex despite the allowed depth.

### Target Variable

```python
def create_target(x):
    return 1 if x >= 3.75 else 0
```

A restaurant with a Zomato rating ≥ 3.75 out of 5 is classified as **"successful"** (1), otherwise **"not successful"** (0). This is a **binary classification** problem.

---

## 4. Training Pipeline — `train.py` Deep Dive

### Step 1: Data Loading & Cleaning

```python
dataset_path = pathlib.Path(__file__).parent.parent / "data" / "raw" / "zomato.csv"
df = pd.read_csv(dataset_path)
```

**Why `pathlib`?** Using `pathlib.Path(__file__)` constructs an absolute path relative to the script's location. This means the script works correctly regardless of which directory you run it from — a common production best practice.

The raw `rate` column contains messy strings like `"4.1/5"`, `"NEW"`, `"-"`. We clean it:

```python
def clean_rate(x):
    try:
        return float(x.split('/')[0].strip())  # "4.1/5" → 4.1
    except (ValueError, AttributeError, IndexError):
        return float('nan')  # "NEW" → NaN (will be dropped)
```

**Why specific exceptions?** A bare `except:` would silently catch `KeyboardInterrupt` and `SystemExit`, making it impossible to stop the script with Ctrl+C. We only catch the three exceptions that could actually occur.

### Step 2: Preprocessing with `ColumnTransformer`

```python
preprocessor = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown='ignore'), ["location", "cuisine_type"]),
    ("num", StandardScaler(), ["approx_cost_for_two"]),
    ("ord", OrdinalEncoder(categories=[['No', 'Yes']], ...), ["online_order"])
])
```

| Column | Transformer | What It Does |
|---|---|---|
| `location` | OneHotEncoder | Converts "BTM" → `[0, 1, 0, 0, ...]` (one column per unique location) |
| `cuisine_type` | OneHotEncoder | Converts "cafe" → `[0, 0, 1, 0, ...]` (one column per cuisine) |
| `approx_cost_for_two` | StandardScaler | Normalizes cost to mean=0, std=1 (important for gradient-based models) |
| `online_order` | OrdinalEncoder | Converts "No" → 0, "Yes" → 1 |

**Why is `handle_unknown='ignore'` important?** If the API receives a location the model has never seen during training (e.g., `"NewTown"`), the OneHotEncoder will produce all zeros instead of crashing. This makes the model robust to unseen inputs in production.

### Step 3: Scikit-Learn Pipeline

```python
pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", LGBMClassifier(...))
])
```

**Why a Pipeline?** The pipeline bundles preprocessing and the model into a single object. When we export it to ONNX, the preprocessing logic is **embedded inside the model file**. This means:
- The API doesn't need to manually replicate the OneHotEncoder logic.
- There's zero risk of **training-serving skew** (when the API preprocesses data differently from training).
- Deploying a new model is as simple as swapping one `.onnx` file.

### Step 4: ONNX Export

```python
update_registered_converter(
    LGBMClassifier,
    'LightGbmLGBMClassifier',
    calculate_linear_classifier_output_shapes,
    convert_lightgbm,
    ...
)

onnx_model = convert_sklearn(
    pipeline,
    initial_types=initial_types,
    target_opset={'': 12, 'ai.onnx.ml': 3}
)
```

**Why register a custom converter?** `skl2onnx` natively supports Scikit-Learn models, but **LightGBM is not Scikit-Learn**. We register a custom converter from `onnxmltools` that teaches `skl2onnx` how to translate LightGBM's tree structure into ONNX graph operations.

**What is `target_opset`?** ONNX has versioned "operator sets" (opsets). We pin `{'': 12, 'ai.onnx.ml': 3}` for maximum compatibility. The `ai.onnx.ml` domain covers ML-specific operators like tree ensemble classifiers.

---

## 5. Why ONNX? — "Why not just use Pickle?"

| | Pickle / Joblib | ONNX |
|---|---|---|
| **Security** | ❌ Arbitrary code execution on load (pickle deserialization attack) | ✅ Safe — ONNX is a computation graph, not executable code |
| **Language Lock-in** | ❌ Python-only | ✅ Runs in C++, Java, JavaScript, C# via ONNX Runtime |
| **Performance** | ❌ Full Python GIL overhead | ✅ Optimized C++ inference engine, hardware acceleration |
| **Versioning** | ❌ Breaks if scikit-learn version changes | ✅ Standard format, version-independent |
| **Deployment** | ❌ Need entire Python ML stack in production | ✅ Only need onnxruntime (~50MB) |

**Interview talking point:** *"I chose ONNX because it eliminates the security risk of pickle deserialization attacks. A malicious `.pkl` file can execute arbitrary code when loaded. ONNX models are just computation graphs — they can't run arbitrary code."*

---

## 6. The API Layer — `api.py` Deep Dive

### Lifespan Event (Model Loading)

```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    model_path = os.getenv("MODEL_PATH", "models/restaurant_model.onnx")
    model = ZomatoSuccessModel(model_path)
    model.load()
    app.state.model = model    # ← Stored on app state, NOT a global variable
    yield
    app.state.model = None     # ← Cleanup on shutdown
```

**Why `lifespan` instead of `@app.on_event("startup")`?** FastAPI's `lifespan` context manager is the modern replacement for `on_event`. It properly handles both startup and shutdown in a single function using `yield`, following the ASGI standard.

**Why `app.state.model` instead of a global variable?** Global mutable state creates hidden dependencies, makes unit testing difficult (hard to mock), and is not thread-safe. Binding the model to `app.state` is the FastAPI-recommended pattern.

### Synchronous Endpoint (Critical Design Decision)

```python
@app.post("/predict")
def predict(request: PredictRequest, api_key: str = Depends(get_api_key)):
    ...
```

**Why `def` instead of `async def`?** This is a critical production concern:

- `async def` routes run directly on the **main asyncio event loop**.
- ONNX Runtime's `.run()` is a **synchronous, CPU-bound** operation.
- If we used `async def`, the inference would **block the event loop**, freezing all other requests (including health checks) until it finishes.
- With `def`, FastAPI automatically offloads the function to a **background thread pool**, keeping the event loop free to handle concurrent requests.

**Interview talking point:** *"I learned that mixing `async def` with CPU-bound operations causes event loop starvation. This would cause the API to appear unresponsive under load, because a single inference call would block all other requests."*

### API Key Authentication

```python
API_KEY = os.getenv("API_KEY", "supersecretkey")
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=True)

def get_api_key(api_key_header: str = Security(api_key_header)):
    if api_key_header != API_KEY:
        raise HTTPException(status_code=403, detail="Could not validate API key")
    return api_key_header
```

**Why authentication?** ML inference endpoints are computationally expensive. Without protection, anyone could send thousands of requests and overwhelm the server (Application-level DoS attack — OWASP A04:2021). The API key is loaded from an environment variable so it's never hardcoded in source code.

---

## 7. Input Validation — `schemas.py` Deep Dive

```python
class PredictRequest(BaseModel):
    location: str = Field(..., min_length=1, description="Location of the restaurant")
    cuisine_type: str = Field(..., min_length=1, description="Type of cuisine")
    approx_cost_for_two: float = Field(..., gt=0, description="Cost must be > 0")
    online_order: Literal["Yes", "No"] = Field(...)
```

**Why Pydantic?** It provides:
1. **Automatic type coercion** — if someone sends `"500"` (string), Pydantic converts it to `500.0` (float).
2. **Constraint validation** — `gt=0` ensures cost is positive, `min_length=1` ensures no empty strings.
3. **Auto-generated API docs** — FastAPI uses the Pydantic model to generate interactive Swagger docs at `/docs`.
4. **`Literal["Yes", "No"]`** — Restricts the `online_order` field to exactly these two values, preventing invalid inputs from ever reaching the model.

If validation fails, FastAPI returns a **422 Unprocessable Entity** with a detailed error message, without ever touching the ML model.

---

## 8. Model Wrapper — `model.py` Deep Dive

```python
class ZomatoSuccessModel(ModelBase):
    """ONNX Runtime Model Loader"""

    def load(self):
        try:
            self._session = rt.InferenceSession(self.model_path)
            logging.info(f"Loaded ONNX model successfully from {self.model_path}")
        except Exception as e:
            logging.error(f"Failed to load ONNX model at {self.model_path}: {e}")
            self._session = None
```

**Why wrap in try/except?** If the `.onnx` file is missing or corrupted, we don't want the entire server to crash silently. We log the error clearly, set the session to `None`, and the API will gracefully return a `503 Service Unavailable` instead.

**Why an abstract base class (`ModelBase`)?** It enables the **Strategy Pattern**. If we later want to swap ONNX for TensorFlow Serving, PyTorch, or a mock model for testing, we implement a new class inheriting from `ModelBase` without changing the API code at all.

### ONNX Input Preparation

```python
inputs = {
    'location': np.array([[features['location']]], dtype=object),
    'cuisine_type': np.array([[features['cuisine_type']]], dtype=object),
    'approx_cost_for_two': np.array([[features['approx_cost_for_two']]], dtype=np.float32),
    'online_order': np.array([[features['online_order']]], dtype=object),
}
```

ONNX Runtime expects inputs as named NumPy arrays with specific shapes. The `[[value]]` creates a 2D array with shape `(1, 1)` — batch size 1, feature dimension 1. The `dtype=object` is used for string inputs, and `np.float32` for numeric inputs.

---

## 9. Testing — `test_api.py` Deep Dive

```python
def test_predict_endpoint_success():     # ← Happy path: valid input + valid key
def test_predict_validation_error():     # ← Pydantic rejects negative cost (422)
def test_predict_unauthorized():         # ← No API key → 401
```

**Three test categories:**

1. **Happy path** — Valid request with correct API key returns `200` and contains `"will_succeed"` in the response.
2. **Validation** — Sending `approx_cost_for_two: -500` triggers Pydantic validation and returns `422 Unprocessable Entity`.
3. **Security** — Sending a request without the `X-API-Key` header returns `401 Unauthorized`.

**Why `TestClient(app)` as a context manager?** This triggers FastAPI's `lifespan` event, which loads the ONNX model. Without it, `app.state.model` would be `None` and all tests would fail.

---

## 10. Docker — Deployment Strategy

```dockerfile
# Stage 1: Builder
FROM python:3.10-slim-bookworm AS builder
WORKDIR /app
RUN pip install uv
COPY pyproject.toml .
RUN uv sync --no-dev
COPY src/ ./src/

# Stage 2: Runtime
FROM python:3.10-slim AS Runtime
WORKDIR /app
COPY --from=builder /app/.venv /app/.venv
COPY src/ ./src/
COPY models/ ./models/
ENV PATH="/app/.venv/bin:$PATH"
EXPOSE 8080
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8080"]
```

**Why multi-stage build?**
- **Stage 1 (builder)**: Installs all dependencies (including build tools, compilers). This stage is ~1.5GB.
- **Stage 2 (runtime)**: Copies only the `.venv` (pre-installed packages), source code, and model. Build tools are discarded.
- **Result**: Final image is much smaller, more secure (no compilers = smaller attack surface), and faster to deploy.

**Why `uv` instead of `pip`?**
- `uv` is 10–100x faster than pip for dependency resolution.
- It creates a deterministic lockfile (`uv.lock`) for reproducible builds.
- It's written in Rust and handles virtual environments automatically.

---

## 11. Key Design Decisions — "Why did you make these choices?"

### Decision 1: Embedding Preprocessing in the ONNX Model
**Problem:** If preprocessing (OneHotEncoder, StandardScaler) is done in the API code separately from the model, any change to the preprocessing logic requires coordinated updates to both the training script AND the API code. This is called **training-serving skew**.

**Solution:** By using a Scikit-Learn `Pipeline`, the preprocessing is bundled with the model. When exported to ONNX, the entire pipeline (preprocessing + LightGBM) becomes a single artifact. The API sends raw data directly to the ONNX model, which handles everything internally.

### Decision 2: ONNX over Pickle
**Problem:** Python's pickle format has a well-known security vulnerability — loading a malicious pickle file can execute arbitrary code on the server.

**Solution:** ONNX models are computation graphs, not executable code. They cannot execute arbitrary commands when loaded, making them safe for production deployment.

### Decision 3: Synchronous Endpoint
**Problem:** ML inference is CPU-bound. Using `async def` would block FastAPI's event loop, causing all concurrent requests to hang.

**Solution:** Using a regular `def` function tells FastAPI to automatically run it in a thread pool, keeping the async event loop responsive for other requests.

### Decision 4: API Key Authentication
**Problem:** ML inference is computationally expensive. An unprotected endpoint is vulnerable to DoS attacks.

**Solution:** Added API key authentication via the `X-API-Key` header. The key is loaded from an environment variable (`API_KEY`) so it never appears in source code.

---

## 12. Potential Interview Questions & Answers

### Q: "What is training-serving skew?"
**A:** It's when the data preprocessing at training time doesn't match what happens at inference time. For example, if training uses StandardScaler to normalize cost to mean=0 but the API forgets to do this, the model receives unnormalized values and produces garbage predictions. I prevented this by embedding all preprocessing inside the Scikit-Learn Pipeline, which gets exported as part of the ONNX model.

### Q: "Why LightGBM over XGBoost?"
**A:** Both achieved similar accuracy (~82-83%), but LightGBM was chosen because: (1) it trains faster due to its histogram-based algorithm, (2) it grows trees leaf-wise instead of level-wise, which typically produces better accuracy with fewer iterations, and (3) the hyperparameters from GridSearchCV gave the best overall performance.

### Q: "What would you improve next?"
**A:**
1. **Model versioning** — Use MLflow or DVC to track model versions, metrics, and experiments.
2. **CI/CD pipeline** — Auto-run tests and rebuild the Docker image on every git push.
3. **Monitoring** — Add Prometheus metrics to track prediction latency, error rates, and data drift.
4. **A/B testing** — Serve multiple model versions simultaneously and compare performance.
5. **Feature store** — Centralize feature engineering logic so the notebook and the API always share the same transformations.
6. **Rate limiting** — Add per-client rate limiting (e.g., using `slowapi`) beyond the current API key auth.

### Q: "What is ONNX Runtime?"
**A:** ONNX Runtime is a high-performance inference engine developed by Microsoft. It takes an ONNX model file (a standardized computation graph) and executes it efficiently using optimized C++ code. It supports hardware acceleration (GPU, CPU vectorization) and can run models trained in any framework — Scikit-Learn, PyTorch, TensorFlow — as long as they're exported to ONNX format.

### Q: "How does FastAPI handle concurrency?"
**A:** FastAPI is built on ASGI (Asynchronous Server Gateway Interface). It uses an async event loop for I/O-bound tasks (database queries, HTTP calls). For CPU-bound tasks like ML inference, I use synchronous `def` endpoints, which FastAPI automatically runs in a thread pool via `anyio`. This prevents the event loop from being blocked.

### Q: "Explain the Pydantic validation."
**A:** Pydantic is a data validation library that uses Python type hints. In my `PredictRequest` schema:
- `str` with `min_length=1` ensures no empty strings.
- `float` with `gt=0` ensures positive cost values.
- `Literal["Yes", "No"]` restricts online_order to exactly these two values.
If any validation fails, FastAPI automatically returns a `422` error with a human-readable explanation of what went wrong — the model never sees invalid data.

### Q: "Why did you use `app.state` instead of a global variable?"
**A:** Global mutable state creates hidden dependencies between modules, makes unit testing harder (you can't easily inject a mock model), and can cause race conditions in multi-threaded environments. Using `app.state` is the FastAPI-recommended pattern — the model lifecycle is explicitly managed by the lifespan context manager, and endpoints access it via `app.state.model`.

---

## 13. How to Run the Project

```bash
# 1. Install dependencies
uv sync

# 2. Train the model (generates models/restaurant_model.onnx)
uv run python src/train.py

# 3. Run the API server
uv run uvicorn src.api:app --reload

# 4. Test it
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -H "X-API-Key: supersecretkey" \
  -d '{"location": "BTM", "cuisine_type": "cafe", "approx_cost_for_two": 500, "online_order": "Yes"}'

# 5. Run tests
uv run python -m pytest tests/

# 6. Docker build & run
docker build -t zomato-api .
docker run -p 8080:8080 -e API_KEY=mysecretkey zomato-api
```

---

## 14. Technologies Summary

| Technology | Role | Why Chosen |
|---|---|---|
| **Python 3.10+** | Language | ML ecosystem, type hints |
| **FastAPI** | Web framework | Async support, auto-docs, Pydantic integration |
| **Pydantic v2** | Validation | Type-safe input validation with auto-generated docs |
| **LightGBM** | ML model | Best accuracy from notebook experiments (83%) |
| **Scikit-Learn** | Pipeline | Bundles preprocessing + model into single artifact |
| **ONNX** | Model format | Secure, portable, language-agnostic serialization |
| **ONNX Runtime** | Inference engine | Fast C++ inference, no Python ML overhead |
| **uv** | Package manager | 10-100x faster than pip, reproducible lockfiles |
| **pytest** | Testing | Simple, powerful Python test framework |
| **Docker** | Containerization | Consistent deployment across environments |

---

*Good luck with the interview! 💪*
