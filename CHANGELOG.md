# CHANGELOG

## v0.1 — 2025-10-18
- Model: StandardScaler + LinearRegression (baseline)
- Metrics (held-out test RMSE): 53.8
- Notes: deterministic training (seed=42)
- Artifacts: models/model.joblib, out/metrics.json
- Usage: `python model/train.py --version v0.1 --out models/model.joblib --metrics out/metrics.json --seed 42`

## v0.2 — 2025-10-19
- Model: StandardScaler + Ridge(alpha=1.0)
- Metrics (held-out test RMSE): 49.87 (↓ improved)
- Rationale: Ridge regularization reduces overfitting and improves generalization.
- Artifacts: models/model.joblib, out/metrics.json
- Usage: `python model/train.py --version v0.2 --out models/model.joblib --metrics out/metrics.json --seed 42`