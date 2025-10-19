from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field
from typing import Dict
import joblib
import os
import traceback
import json

MODEL_PATH = os.environ.get("MODEL_PATH", "models/model.joblib")
MODEL_VERSION = os.environ.get("MODEL_VERSION", "v0.1")

app = FastAPI(title="Virtual Diabetes Triage - Predictor", version=MODEL_VERSION)

# Exact feature names expected (matches sklearn diabetes dataset frame)
FEATURE_NAMES = ["age","sex","bmi","bp","s1","s2","s3","s4","s5","s6"]

class PatientFeatures(BaseModel):
    age: float
    sex: float
    bmi: float
    bp: float
    s1: float
    s2: float
    s3: float
    s4: float
    s5: float
    s6: float

class PredictionResponse(BaseModel):
    prediction: float
    model_version: str

# Load model at startup
model = None
try:
    model = joblib.load(MODEL_PATH)
except Exception as e:
    # We keep model as None; /health will reflect if model failed to load
    model = None
    load_error = str(e)
    # keep stacktrace in logs (stdout)
    print("Failed to load model:", e)
    traceback.print_exc()

@app.get("/health")
async def health():
    ok = model is not None
    resp = {"status": "ok" if ok else "error", "model_version": MODEL_VERSION}
    if not ok:
        resp["error"] = "model not loaded"
    return resp

@app.post("/predict", response_model=PredictionResponse)
async def predict(payload: PatientFeatures, request: Request):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not available")

    # ensure ordering and convert to 2D array
    try:
        values = [getattr(payload, f) for f in FEATURE_NAMES]
        # Our model expects shape (n_samples, n_features)
        pred = model.predict([values])
        # model.predict returns array-like
        value = float(pred[0])
        return {"prediction": value, "model_version": MODEL_VERSION}
    except Exception as e:
        # Return JSON error per observability requirement
        tb = traceback.format_exc()
        raise HTTPException(status_code=400, detail={"error": str(e), "trace": tb})