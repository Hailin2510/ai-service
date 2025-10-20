# ----------- Builder Stage (Train Model) -----------
FROM python:3.11-slim AS builder
WORKDIR /src

# Install dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy source code
COPY model/ ./model
COPY app/ ./app

# Train model
ARG MODEL_VERSION=v0.1
RUN python model/train.py --version $MODEL_VERSION --out model/model.joblib --metrics out/metrics.json --seed 42

# ----------- Runtime Stage (FastAPI App) -----------
FROM python:3.11-slim
WORKDIR /app

# Copy model + app + metrics
COPY --from=builder /src/model ./model
COPY --from=builder /src/app ./app
COPY --from=builder /src/out ./out

# Install dependencies for runtime
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Environment variables
ENV MODEL_PATH=/app/model/model.joblib
ENV MODEL_VERSION=$MODEL_VERSION

EXPOSE 8000

# Proper healthcheck using curl
HEALTHCHECK --interval=10s --timeout=5s --start-period=5s --retries=3 \
  CMD curl -f http://127.0.0.1:8000/health || exit 1

# Start FastAPI
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
