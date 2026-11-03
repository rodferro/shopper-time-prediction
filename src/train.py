"""Train the production model.

1. Build one row per order from the raw CSVs.
2. Fit on 80% of the labeled orders and score the 20% hold-out once (MAE, in minutes).
3. Refit on all labeled orders and save the model to ``model/model.joblib``, with the scores in
   ``model/metrics.json``.

Run from the repository root: ``python -m src.train``.
"""

import json

import joblib
import numpy as np
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

from src.config import DATA_DIR, METRICS_PATH, MODEL_DIR, MODEL_PARAMS, MODEL_PATH, RANDOM_STATE
from src.features import build_features, load_order_rows


def train(X, y, params=MODEL_PARAMS):
    return LGBMRegressor(**params).fit(X, y)


def main():
    rows = load_order_rows(DATA_DIR)
    labeled = rows[rows["total_minutes"].notna()]
    unlabeled = rows[rows["total_minutes"].isna()]
    X, y = build_features(labeled), labeled["total_minutes"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)
    holdout_mae = mean_absolute_error(y_test, train(X_train, y_train).predict(X_test))
    baseline_mae = mean_absolute_error(y_test, np.full(len(y_test), y_train.mean()))
    print(f"hold-out MAE: {holdout_mae:.2f} min (predict-the-mean baseline: {baseline_mae:.2f})")

    model = train(X, y)

    # Sanity check: the final model predicts every unlabeled order.
    predicted = model.predict(build_features(unlabeled))
    assert len(predicted) == len(unlabeled) and np.isfinite(predicted).all() and (predicted > 0).all()

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    metrics = {
        "holdout_mae": round(holdout_mae, 3),
        "baseline_mae": round(baseline_mae, 3),
        "n_train": len(X_train),
        "n_holdout": len(X_test),
        "n_final": len(X),
    }
    METRICS_PATH.write_text(json.dumps(metrics, indent=2) + "\n")
    print(f"saved {MODEL_PATH.relative_to(MODEL_DIR.parent)} (refit on {len(X)} labeled orders)")


if __name__ == "__main__":
    main()
