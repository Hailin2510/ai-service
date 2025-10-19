"""
Train script for v0.1 (LinearRegression baseline) and v0.2 (RandomForest example).
Usage:
  python model/train.py --out models/model.joblib --metrics out/metrics.json --seed 42 --version v0.1
"""
import argparse
import json
import os
import joblib
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import numpy as np
import random

def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)

def build_pipeline(version):
    # v0.1 baseline: StandardScaler + LinearRegression
    if version == "v0.1":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("model", LinearRegression())
        ])
    # v0.2 improvement: StandardScaler + RandomForest
    elif version == "v0.2":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("model", RandomForestRegressor(n_estimators=200, random_state=SEED, n_jobs=1))
        ])
    else:
        raise ValueError("Unknown version")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="models/model.joblib")
    p.add_argument("--metrics", default="out/metrics.json")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--version", default="v0.1")
    p.add_argument("--test-size", type=float, default=0.2)
    args = p.parse_args()

    OUT = args.out
    METRICS = args.metrics
    SEED = args.seed
    VERSION = args.version

    set_seed(SEED)

    data = load_diabetes(as_frame=True)
    X = data.frame.drop(columns=["target"])
    y = data.frame["target"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=args.test_size, random_state=SEED)

    pipeline = build_pipeline(VERSION)
    pipeline.fit(X_train, y_train)

    preds = pipeline.predict(X_test)
    rmse = float(mean_squared_error(y_test, preds, squared=False))

    # persist model and metrics
    os.makedirs(os.path.dirname(OUT) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(METRICS) or ".", exist_ok=True)
    joblib.dump(pipeline, OUT)

    metrics = {
        "version": VERSION,
        "rmse": rmse,
        "seed": SEED,
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test))
    }

    with open(METRICS, "w") as f:
        json.dump(metrics, f, indent=2)

    print("Saved model to", OUT)
    print("Metrics:", metrics)