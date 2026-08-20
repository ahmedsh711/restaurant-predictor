# Zomato Success API

Predicts if a restaurant will clear a 3.75 rating.

## Run Locally (3 commands)
1. `git clone https://github.com/yourname/zomato-mlops.git && cd zomato-mlops`
2. `uv sync`
3. `uv run uvicorn src.api:app --reload`

Visit `http://localhost:8000/docs` to test the API.

## Run via Docker
1. `docker build -t zomato-api .`
2. `docker run -p 8000:8000 zomato-api`