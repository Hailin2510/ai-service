# ----------- Builder Stage -----------  
FROM python:3.11-slim AS builder

WORKDIR /src

# Install dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY model/ ./model
COPY train.py ./train.py
COPY main.py ./main.py

# Train model
ARG MODEL_VERSION=v0.1
RUN python train.py --version ${MODEL_VERSION} --out model/model.joblib --metrics out/metrics.json --seed 42

# ----------- Runtime Stage -----------  
FROM python:3.11-slim

WORKDIR /app

# Copy trained model and app
COPY --from=builder /src/model ./model
COPY --from=builder /src/out ./out
COPY --from=builder /src/main.py ./main.py
COPY requirements.txt .

# Install runtime dependencies
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

# Expose FastAPI port
EXPOSE 8000

# Use uvicorn options for CI-friendly fast startup
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--reload", "--workers", "1"]
