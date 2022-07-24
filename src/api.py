import pandas as pd

from flask import Flask, make_response, request
from predict import predict


app = Flask(__name__)


@app.route("/predict", methods=["POST"])
def do_predict():
    uploaded_file = request.files["file"]
    df = pd.read_csv(uploaded_file)
    predicted = pd.DataFrame({"order_id": df["order_id"], "total_minutes": predict(df)})
    resp = make_response(predicted.to_csv(index=False))
    resp.headers["Content-Disposition"] = "attachment; filename=predicted.csv"
    resp.headers["Content-Type"] = "text/csv"
    return resp


if __name__ == "__main__":
    app.run(host="0.0.0.0")
