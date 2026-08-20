# Stage 1: Builder
FROM python:3.10-slim-bookworm AS builder
WORKDIR /app

RUN pip install uv
COPY pyproject.toml .

# Layer caching: we copy code AFTER dependencies so changes in code doesn't force reinstall
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