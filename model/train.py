# model/train.py
import argparse
import json
import os
from pathlib import Path
import random
import numpy as np
import joblib

from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

def set_seed(seed: int):
    np.random.seed(seed)
    random.seed(seed)

def build_pipeline(version: str, seed: int):
    if version == "v0.1":
        # Baseline: StandardScaler + LinearRegression
        return Pipeline([
            ("scaler", StandardScaler()),
            ("model", LinearRegression())
        ])
    elif version == "v0.2":
        # Improved: StandardScaler + RandomForest
        return Pipeline([
            ("scaler", StandardScaler()),
            ("model", RandomForestRegressor(n_estimators=200, random_state=seed, n_jobs=1))
        ])
    else:
        raise ValueError(f"Unknown version: {version}")

def train_model(version: str = "v0.1", seed: int = 42, test_size: float = 0.2):
    set_seed(seed)
    data = load_diabetes(as_frame=True)
    X = data.frame.drop(columns=["target"])
    y = data.frame["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed
    )

    pipeline = build_pipeline(version, seed)
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    rmse = float(mean_squared_error(y_test, y_pred, squared=False))
    return pipeline, rmse, len(X_train), len(X_test)

def save_model(pipeline, path="model/model.joblib"):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, path)
    print(f"✅ Saved model to {path}")

def save_metrics(rmse, n_train, n_test, version="v0.1", path="out/metrics.json"):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    metrics = {
        "version": version,
        "rmse": rmse,
        "n_train": n_train,
        "n_test": n_test
    }
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"✅ Saved metrics to {path} | RMSE: {rmse:.4f}")

def main():
    parser = argparse.ArgumentParser(description="Train diabetes progression model")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--version", type=str, default="v0.1", help="Model version (v0.1 or v0.2)")
    parser.add_argument("--out", type=str, default="model/model.joblib", help="Path to save trained model")
    parser.add_argument("--metrics", type=str, default="out/metrics.json", help="Path to save metrics")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test dataset size fraction")
    args = parser.parse_args()

    pipeline, rmse, n_train, n_test = train_model(version=args.version, seed=args.seed, test_size=args.test_size)
    save_model(pipeline, args.out)
    save_metrics(rmse, n_train, n_test, version=args.version, path=args.metrics)
    print(f"✅ Training complete | Version: {args.version} | RMSE: {rmse:.4f}")

if __name__ == "__main__":
    main()