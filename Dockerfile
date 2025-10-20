# ----------- Builder Stage -----------
FROM python:3.11-slim AS builder
WORKDIR /src

# Minimal system deps for building wheels
RUN apt-get update && apt-get install -y build-essential --no-install-recommends \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy full repo
COPY . .

# ARG for model version; default v0.1
ARG MODEL_VERSION=v0.1

# Train model during build
RUN python model/train.py --version ${MODEL_VERSION} --out models/model.joblib --metrics out/metrics.json --seed 42

# ----------- Runtime Stage -----------
FROM python:3.11-slim AS runtime
WORKDIR /app

# Copy installed packages
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages

# Copy app code, model, metrics, and changelog
COPY --from=builder /src/app ./app
COPY --from=builder /src/models ./models
COPY --from=builder /src/out ./out
COPY --from=builder /src/CHANGELOG.md ./CHANGELOG.md

# Environment variables
ENV MODEL_PATH=/app/models/model.joblib
ENV MODEL_VERSION=${MODEL_VERSION}

# Expose FastAPI port
EXPOSE 8000

# Healthcheck for container
HEALTHCHECK --interval=10s --timeout=5s --start-period=5s --retries=3 \
  CMD python -c "import requests, sys; r=requests.get('http://127.0.0.1:8000/health'); sys.exit(0 if r.status_code==200 and r.json().get('status')=='ok' else 1)"

# Start API
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]