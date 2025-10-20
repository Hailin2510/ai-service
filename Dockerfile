# ----------- Builder Stage (Train Model) -----------
FROM python:3.11-slim AS builder

WORKDIR /src

# Install dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy source code
COPY model/train.py ./model/train.py
COPY app/main.py ./app/main.py
COPY model/ ./model

# Only copy out/ if it exists (optional)
# COPY out/ ./out  # remove this; train.py will create it

# Train model
ARG MODEL_VERSION=v0.1
RUN python model/train.py --version $MODEL_VERSION --out model/model.joblib --metrics out/metrics.json --seed 42

# ----------- Runtime Stage (FastAPI App) -----------
FROM python:3.11-slim

WORKDIR /app

# Copy trained model and app
COPY --from=builder /src/model ./model
COPY --from=builder /src/app ./app
COPY --from=builder /src/out ./out

# Copy requirements and install
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# ENV for main.py
ENV MODEL_PATH=/app/model/model.joblib
ENV MODEL_VERSION=${MODEL_VERSION}

EXPOSE 8000

HEALTHCHECK --interval=10s --timeout=5s --start-period=5s --retries=3 \
  CMD python -c "import requests, sys; r=requests.get('http://127.0.0.1:8000/health'); sys.exit(0 if r.status_code==200 and r.json().get('status')=='ok' else 1)"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]