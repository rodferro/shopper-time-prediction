import io

import pandas as pd
import pytest

from src.api import create_app


@pytest.fixture
def client(model):
    return create_app(model).test_client()


def post_csv(client, df):
    data = {"file": (io.BytesIO(df.to_csv(index=False).encode()), "orders.csv")}
    return client.post("/predict", data=data, content_type="multipart/form-data")


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200 and r.get_json() == {"status": "ok"}


def test_predict_sample(client, sample):
    r = post_csv(client, sample)
    assert r.status_code == 200
    predicted = pd.read_csv(io.BytesIO(r.data))
    assert list(predicted.columns) == ["order_id", "total_minutes"]
    assert predicted["order_id"].tolist() == sample["order_id"].tolist()
    assert (predicted["total_minutes"] > 0).all()


def test_extra_columns_are_ignored(client, sample):
    r = post_csv(client, sample.assign(total_minutes=60.0, extra="x"))
    assert r.status_code == 200


def test_missing_file(client):
    r = client.post("/predict")
    assert r.status_code == 400 and "file" in r.get_json()["error"]


def test_invalid_rows(client, sample):
    r = post_csv(client, sample.drop(columns="promised_time"))
    assert r.status_code == 400 and "promised_time" in r.get_json()["error"]


def test_empty_csv(client):
    data = {"file": (io.BytesIO(b""), "orders.csv")}
    r = client.post("/predict", data=data, content_type="multipart/form-data")
    assert r.status_code == 400
