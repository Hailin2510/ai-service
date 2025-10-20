# Virtual Diabetes Clinic - Triage ML Service

## Purpose
Small ML service that predicts short-term "progression index" using sklearn's diabetes dataset as a stand-in.

## Run locally (dev)
1. Create a Python venv and install:
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

2. Train model (v0.1 baseline):
python model/train.py --out models/model.joblib --metrics out/metrics.json --seed 42 --version v0.1

3. Start service:
uvicorn app.main:app --host 0.0.0.0 --port 8000

4. Health:
GET http://127.0.0.1:8000/health
Response: {"status":"ok","model_version":"v0.1"}

5. Predict:
POST http://127.0.0.1:8000/predict
Content-Type: application/json
Body:
{
"age": 0.02, "sex": -0.044, "bmi": 0.06, "bp": -0.03,
"s1": -0.02, "s2": 0.03, "s3": -0.02, "s4": 0.02, "s5": 0.02, "s6": -0.001
}

Response:
{"prediction": 152.3, "model_version":"v0.1"}

## Build Docker image (local)
docker build --build-arg MODEL_VERSION=v0.1 -t ghcr.io/<org>/<repo>:v0.1 .
docker run -p 8000:8000 ghcr.io/<org>/<repo>:v0.1

## GitHub Actions
- `ci.yml` runs on push/PR, runs tests, trains a smoke model and uploads artifacts.
- `release.yml` triggers on tag `v*`: builds and pushes Docker to GHCR and publishes a GitHub Release that includes `out/metrics.json` and `CHANGELOG.md`.

## !IMPORTANT: Docker image URL
`ghcr.io/Hailin2510/diabetes-predictor-ai:v0.2`
Test locally
docker pull ghcr.io/Hailin2510/diabetes-predictor-ai:v0.2
docker run -p 8000:8000 ghcr.io/Hailin2510/diabetes-predictor-ai:v0.2
Then visit:
http://localhost:8000/health
You should see a response like: {"status":"ok"}