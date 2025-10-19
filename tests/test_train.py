# tests/test_train.py
import subprocess
import json
import os

def test_quick_train_creates_metrics(tmp_path):
    out_model = tmp_path / "models" / "m.joblib"
    out_metrics = tmp_path / "metrics.json"
    os.makedirs(out_model.parent, exist_ok=True)
    cmd = ["python", "model/train.py", "--out", str(out_model), "--metrics", str(out_metrics), "--version", "v0.1", "--seed", "42"]
    r = subprocess.run(cmd, check=True)
    assert out_model.exists()
    assert out_metrics.exists()
    m = json.loads(out_metrics.read_text())
    assert "rmse" in m