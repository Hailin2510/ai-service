from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from typing import List
import joblib
import os
import traceback
import json

# Updated default path to match Dockerfile / train.py output
MODEL_PATH = os.environ.get("MODEL_PATH", "model/model.joblib")
MODEL_VERSION = os.environ.get("MODEL_VERSION", "v0.1")

app = FastAPI(title="Virtual Diabetes Triage - Predictor", version=MODEL_VERSION)

# Exact feature names expected (matches sklearn diabetes dataset)
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
    model = None
    print(f"❌ Failed to load model from {MODEL_PATH}:", e)
    traceback.print_exc()

@app.get("/health")
async def health():
    ok = model is not None
    resp = {"status": "ok" if ok else "error", "model_version": MODEL_VERSION}
    if not ok:
        resp["error"] = f"model not loaded at {MODEL_PATH}"
    return resp

@app.post("/predict", response_model=PredictionResponse)
async def predict(payload: PatientFeatures, request: Request):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not available")

    try:
        values = [getattr(payload, f) for f in FEATURE_NAMES]
        pred = model.predict([values])
        value = float(pred[0])
        return {"prediction": value, "model_version": MODEL_VERSION}
    except Exception as e:
        tb = traceback.format_exc()
        raise HTTPException(status_code=400, detail={"error": str(e), "trace": tb})