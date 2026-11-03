"""HTTP API.

- ``GET /health``: ``{"status": "ok"}`` once the model is loaded.
- ``POST /predict``: multipart upload of a CSV of order rows in field ``file``; returns a CSV with
  ``order_id,total_minutes``. Bad input gets a 400 with ``{"error": "..."}``.

Development server: ``python -m src.api``. In Docker it runs under gunicorn.
"""

import pandas as pd
from flask import Flask, jsonify, make_response, request

from src.features import InvalidInput
from src.predict import load_model, predict


def create_app(model=None):
    app = Flask(__name__)
    model = model if model is not None else load_model()

    def bad_request(message):
        return jsonify(error=message), 400

    @app.get("/health")
    def health():
        return jsonify(status="ok")

    @app.post("/predict")
    def do_predict():
        if "file" not in request.files:
            return bad_request("upload a CSV in the 'file' field")
        try:
            rows = pd.read_csv(request.files["file"])
        except (pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeDecodeError) as e:
            return bad_request(f"could not read CSV: {e}")
        try:
            predicted = predict(model, rows)
        except InvalidInput as e:
            return bad_request(str(e))

        resp = make_response(predicted.to_csv(index=False))
        resp.headers["Content-Disposition"] = "attachment; filename=predicted.csv"
        resp.headers["Content-Type"] = "text/csv"
        return resp

    return app


if __name__ == "__main__":
    create_app().run(host="0.0.0.0", port=5000)
