"""Paths and constants shared by training and serving."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "model"
MODEL_PATH = MODEL_DIR / "model.joblib"
METRICS_PATH = MODEL_DIR / "metrics.json"

RANDOM_STATE = 42
LOCAL_TZ = "America/Santiago"  # all coordinates are in Chile; promised_time is in UTC

# Final LightGBM parameters from the randomized search in shopper-time-prediction.ipynb.
MODEL_PARAMS = dict(
    objective="l1",  # matches the MAE evaluation metric
    n_estimators=646,
    learning_rate=0.03,
    num_leaves=56,
    min_child_samples=13,
    subsample=0.9702292921764571,
    subsample_freq=1,
    colsample_bytree=0.8334712691569033,
    reg_alpha=0,
    reg_lambda=5,
    random_state=RANDOM_STATE,
    verbose=-1,
)
