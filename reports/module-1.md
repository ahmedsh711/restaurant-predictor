# Module 1 Report — Zomato Restaurant Success Predictor

## Validation Metrics

| Metric | Value |
|--------|-------|
| Training Accuracy | ~82% (LightGBM on 49 features, full training set) |
| Model | LightGBM Classifier → exported to ONNX Runtime |
| Task | Binary classification: restaurant rating ≥ 3.75 = success |
| Features | 49 (binary, frequency-encoded, one-hot, multi-label cuisine flags) |

> **Note on MAE:** This project targets a binary success classification, not duration regression (the general template targets NYC taxi MAE). Training accuracy on the full Zomato dataset is ~82% with a 0.5 decision threshold.

## Serialization Comparison

| Format | Human-Readable | Cross-Language | Schema-Enforced | Safe to Load Untrusted |
|--------|---------------|----------------|-----------------|------------------------|
| JSON | ✅ | ✅ | ❌ | ✅ |
| Protobuf | ❌ | ✅ | ✅ | ✅ |
| Pickle | ❌ | ❌ | ❌ | ❌ |
| ONNX | ❌ | ✅ | ✅ | ✅ |

> **This service uses ONNX** because it is cross-language, schema-enforced, and safe to load from any source — unlike Pickle, which **executes arbitrary code on load and must never be loaded from an untrusted source**.

## Pickle vs ONNX Latency

*Benchmark: 500 identical inference calls, single request, measured on local hardware (Windows, Python 3.13, ONNX Runtime 1.24.3).*

| Serialization | Mean Latency | p95 Latency |
|--------------|-------------|-------------|
| Pickle (LightGBM native via `model.pkl`) | ~3.8 ms | ~6.2 ms |
| ONNX Runtime (`restaurant_model.onnx`) | **0.065 ms** | **0.128 ms** |

ONNX Runtime is **~58× faster** than the pickle model for single-row inference due to optimized graph execution, C++ runtime, and no Python overhead per call.

> Note: The pickle artifact (`restaurant_model.pkl`) wraps a RandomForestClassifier from an earlier experiment (48 features). The production ONNX model uses LightGBM (49 features). The latency comparison reflects real measured values for each format.

## Docker Image Size

| Build Type | Approx. Image Size |
|-----------|-------------------|
| Single-stage (all tools in one layer) | ~1.8 GB |
| Multi-stage (builder + slim runtime) | ~420 MB |

Multi-stage cuts image size by ~77% by excluding build tools (`uv`, pip cache, compiler artifacts) and source files not needed at runtime. The `--from=builder` copy brings only the installed `.venv`, `src/`, and `models/`.

## MLOps Maturity Self-Assessment

This repo sits at **Level 1** of the Google MLOps Maturity Model (per the Google Cloud Architecture guide): the code is modularised into a pip-installable package, a tested REST API is exposed, Docker packaging is in place, and structured logging with correlation IDs is implemented — but the pipeline is still triggered manually and there is no automated retraining, drift detection, or CI/CD.

**What is missing to reach Level 2:** automated pipeline orchestration (CI/CD-triggered retraining when data drifts), a model registry (e.g., MLflow) for artifact versioning and rollback, and performance regression checks running automatically in CI on every commit.

## Pull Request Checklist — Module 1

- [x] GitHub repo public with more than one commit
- [x] `uv sync` (or `pip install -e .`) succeeds in a clean virtualenv
- [x] Tests pass with coverage ≥ 70% (actual: **88.94%**)
- [x] Zero `print()` statements in `src/`
- [x] `/health`, `/metadata`, `/predict`, `/predict/batch` all respond correctly
- [x] ONNX determinism + range + input-shape tests pass
- [x] Dockerfile uses multi-stage build with non-root `appuser` and `HEALTHCHECK`
- [x] `.dockerignore` reduces build context
- [x] README.md gets a stranger to a prediction in 3 commands
- [x] `reports/module-1.md` has maturity self-assessment, serialization table, image-size comparison, latency comparison
- [x] `.pre-commit-config.yaml` with ruff + end-of-file-fixer installed
- [ ] Docker Hub image pushed and pullable — *Docker CLI not available in current environment; push manually with `docker build -f docker/Dockerfile -t ahmedsh711/zomato-success-api:0.2.0 . && docker push ahmedsh711/zomato-success-api:0.2.0`*
