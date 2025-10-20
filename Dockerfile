# ----------- Stage 1: Builder (train model) -----------
FROM python:3.11-slim AS builder
WORKDIR /src

# Install dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY model/ ./model
COPY app/ ./app

# Ensure out/ exists
RUN mkdir -p out

# Train model
ARG MODEL_VERSION=v0.1
RUN python model/train.py --version $MODEL_VERSION --out model/model.joblib --metrics out/metrics.json --seed 42

# ----------- Stage 2: Runtime (FastAPI app) -----------
FROM python:3.11-slim AS runtime
WORKDIR /app

# Copy trained model and app from builder
COPY --from=builder /src/model ./model
COPY --from=builder /src/out ./out
COPY --from=builder /src/app ./app

# Install runtime dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

# Expose FastAPI port
EXPOSE 8000

# Healthcheck
HEALTHCHECK --interval=10s --timeout=5s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

# Run FastAPI app
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
