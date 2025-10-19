# CHANGELOG

## v0.2 — 2025-10-19
- Model: RandomForestRegressor (n_estimators=200) w/ StandardScaler
- Rationale: to capture non-linearities vs linear baseline.
- Metrics (held-out test RMSE): **45.1** (v0.2) vs **53.8** (v0.1)
- Artifacts: `models/model.joblib`, `out/metrics.json`
- How to reproduce: `python model/train.py --version v0.2 --out models/model.joblib --metrics out/metrics.json --seed 42`

## v0.1 — 2025-10-18
- Model: StandardScaler + LinearRegression (baseline)
- Metrics (held-out test RMSE): **53.8**
- Notes: deterministic training (seed=42). Used as baseline for triage prioritization.