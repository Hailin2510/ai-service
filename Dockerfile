# Dockerfile (multi-stage)
FROM python:3.11-slim AS builder
WORKDIR /src

# system deps for building wheels (kept minimal)
RUN apt-get update && apt-get install -y build-essential --no-install-recommends && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Run training to bake model into image during image build
# default train to v0.1; to build v0.2 pass --build-arg MODEL_VERSION=v0.2
ARG MODEL_VERSION=v0.1
RUN python model/train.py --out models/model.joblib --metrics out/metrics.json --seed 42 --version ${MODEL_VERSION}

# Runtime image
FROM python:3.11-slim AS runtime
WORKDIR /app

COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /src/app /app/app
COPY --from=builder /src/models /app/models
COPY --from=builder /src/out /app/out
COPY --from=builder /src/CHANGELOG.md /app/CHANGELOG.md

ENV MODEL_PATH=/app/models/model.joblib
ENV MODEL_VERSION=${MODEL_VERSION}
EXPOSE 8000

HEALTHCHECK --interval=10s --timeout=5s --start-period=5s --retries=3 \
  CMD python -c "import requests, sys; r=requests.get('http://127.0.0.1:8000/health'); 
  sys.exit(0 if r.status_code==200 and r.json().get('status')=='ok' else 1)"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]