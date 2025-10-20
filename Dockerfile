# ----------- Builder Stage (Train Model) -----------
FROM python:3.11-slim AS builder

WORKDIR /src

# Copy dependencies and install
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy source code
COPY model/ ./model
COPY train.py ./train.py
COPY main.py ./main.py
COPY out/ ./out

# Train model
ARG MODEL_VERSION=v0.1
RUN python train.py --version $MODEL_VERSION --out model/model.joblib --metrics out/metrics.json --seed 42

# ----------- Runtime Stage (FastAPI App) -----------
FROM python:3.11-slim

WORKDIR /app

# Copy trained model, metrics, and app
COPY --from=builder /src/model ./model
COPY --from=builder /src/out ./out
COPY --from=builder /src/main.py ./main.py
COPY requirements.txt .

# Install runtime dependencies
RUN pip install --upgrade pip && pip install -r requirements.txt

# Environment variables for model path/version
ENV MODEL_PATH=/app/model/model.joblib
ENV MODEL_VERSION=${MODEL_VERSION}

# Expose FastAPI port
EXPOSE 8000

# Healthcheck for FastAPI container
HEALTHCHECK --interval=10s --timeout=5s --start-period=5s --retries=3 \
  CMD python -c "import requests, sys; r=requests.get('http://127.0.0.1:8000/health'); sys.exit(0 if r.status_code==200 and r.json().get('status')=='ok' else 1)"

# Run FastAPI app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]