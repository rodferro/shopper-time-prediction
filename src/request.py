"""Send a CSV of order rows to the API and save the predictions.

From the repository root: ``python -m src.request [--file data/sample.csv] [--out data/predicted.csv]``.
"""

import argparse

import requests

from src.config import DATA_DIR


def run(file, out, url):
    with open(file, "rb") as f:
        r = requests.post(url, files={"file": f}, timeout=30)
    if not r.ok:
        raise SystemExit(f"request failed ({r.status_code}): {r.text}")
    with open(out, "wb") as f:
        f.write(r.content)
    print(f"saved predictions to {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--file", default=DATA_DIR / "sample.csv")
    parser.add_argument("--out", default=DATA_DIR / "predicted.csv")
    parser.add_argument("--url", default="http://localhost:5000/predict")
    args = parser.parse_args()
    run(args.file, args.out, args.url)
