# Mini Project 1 - Status & Gap Analysis Report

This document evaluates the `zomato-success-api` project against the requirements defined in `mini prject 1 _ Requirements.pdf`.

## 1. Establish the Baseline
* **Status**: ⚠️ **Partial / Deviation**
* **Expected**: Use NYC TLC green taxi data, create `notebooks/00-baseline.ipynb`, engineer specific features, train a baseline model, and record MAE/RMSE in `reports/module-1.md`.
* **Actual**: The project predicts Zomato restaurant success using a different dataset. `reports/module-1.md` and baseline metrics are missing.
* **Missing Actions**:
  - Clarify with your instructor if the Zomato dataset is an acceptable substitute for the TLC dataset.
  - Create `reports/module-1.md` to document baseline model performance.

## 2. Turn it into a real Python package
* **Status**: ⚠️ **Partial**
* **Expected**: `src/` layout with decomposed files (`data.py`, `features.py`, `train.py`, `predict.py`, `config.py`), a `pyproject.toml` with a script entry point, type hints, a custom decorator, config via `pydantic-settings`, and pre-commit hooks.
* **Actual**: `src/` and `pyproject.toml` exist. Type hints are used. OOP is implemented (`ZomatoSuccessModel`).
* **Missing Actions**:
  - Restructure files to match the required names (if strict adherence to the PDF is needed).
  - Add `[project.scripts]` to `pyproject.toml` to create a CLI entry point.
  - Implement a custom decorator (e.g., `@timed` for prediction time).
  - Replace `os.getenv` with `pydantic-settings` for configuration management.
  - Add `.pre-commit-config.yaml` (with `ruff`, `black`, `end-of-file-fixer`).

## 3. Structured Logging
* **Status**: ❌ **Missing**
* **Expected**: JSON logging via `src/logging_conf.py`. Every log line must include a `correlation_id` generated via FastAPI middleware using `contextvars`.
* **Missing Actions**:
  - Create a JSON logging configuration.
  - Implement FastAPI middleware to generate a `uuid4` correlation ID and return it in the `X-Request-ID` header.
  - Replace any standard `print()` statements with appropriate log levels (`DEBUG`, `INFO`, `WARNING`, `ERROR`).

## 4. Serialization
* **Status**: ⚠️ **Partial**
* **Expected**: Export to ONNX, write a parity test (pickle vs ONNX), benchmark latency, and create a comparison table in the report.
* **Actual**: ONNX model is used and loaded successfully via `onnxruntime`.
* **Missing Actions**:
  - Write a parity test script asserting predictions are identical (`np.allclose`) for 500 rows.
  - Benchmark the latency of both formats.
  - Add the serialization comparison table (Pickle vs. ONNX) to `reports/module-1.md`.

## 5. Build the API
* **Status**: ⚠️ **Partial**
* **Expected**: Endpoints `/health`, `/metadata`, `/predict`, `/predict/batch`. Pydantic validation, startup model loading, and custom exception handlers (422, 500).
* **Actual**: `/predict` is implemented. Pydantic validation is used. Lifespan model loading is used.
* **Missing Actions**:
  - Implement `/health` endpoint (checking if the model is loaded in memory).
  - Implement `/metadata` endpoint (returning model version, framework, etc.).
  - Implement `/predict/batch` endpoint.
  - Add explicit exception handlers for 422 (validation errors) and 500 (unexpected errors) status codes to prevent leaking stack traces.

## 6. Test Suite with a Coverage Gate
* **Status**: ❌ **Missing**
* **Expected**: Comprehensive test suite (`conftest.py`, `test_features.py`, `test_predict.py`, `test_api.py`, `test_serialization.py`) with mocking, parametrization, and a 70% coverage gate in `pyproject.toml`.
* **Actual**: Only `test_api.py` exists, but pytest execution fails and configuration is absent.
* **Missing Actions**:
  - Add `[tool.pytest.ini_options]` in `pyproject.toml` with `addopts = "--cov=src --cov-fail-under=70"`.
  - Write the missing test files and fixtures.

## 7. Containerize and Publish
* **Status**: ⚠️ **Partial**
* **Expected**: Multi-stage `Dockerfile`, non-root user, `.dockerignore`, `docker-compose.yml`, pushed to Docker Hub.
* **Actual**: Multi-stage `Dockerfile` exists.
* **Missing Actions**:
  - Add a non-root user to the `Dockerfile` (`useradd`, `USER`).
  - Create a `.dockerignore` file.
  - Create a `docker-compose.yml` file with volume mounts and port mapping.
  - Document the single-stage vs multi-stage image size difference in the report.

## 8. README, Maturity Self-Assessment, Release
* **Status**: ⚠️ **Partial**
* **Expected**: README with 3 commands to run, and a maturity self-assessment in `reports/module-1.md`.
* **Actual**: README has basic instructions.
* **Missing Actions**:
  - Ensure the README clearly allows a stranger to go from zero to prediction in exactly 3 commands.
  - Write the maturity self-assessment in the `reports/module-1.md` file.
