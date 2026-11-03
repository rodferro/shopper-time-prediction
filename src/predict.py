"""Predict ``total_minutes`` for order rows.

From the repository root: ``python -m src.predict data/sample.csv data/predicted.csv``.
"""

import argparse

import joblib
import pandas as pd

from src.config import MODEL_PATH
from src.features import build_features


def load_model(path=MODEL_PATH):
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; train it first with `python -m src.train`")
    return joblib.load(path)


def predict(model, rows):
    """Return a DataFrame with ``order_id`` and the predicted ``total_minutes``. Raises ``InvalidInput``."""
    return pd.DataFrame({
        "order_id": rows["order_id"].to_numpy(),
        "total_minutes": model.predict(build_features(rows)),
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("input", help="CSV of order rows")
    parser.add_argument("output", help="where to write the predictions CSV")
    args = parser.parse_args()
    predict(load_model(), pd.read_csv(args.input)).to_csv(args.output, index=False)
