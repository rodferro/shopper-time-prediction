"""Feature engineering shared by training and serving.

Training and the API both start from the same *order row* format: one row per order,
already joined with the shopper and store data (see ``INPUT_COLUMNS``). ``load_order_rows``
builds those rows from the raw CSVs, and ``build_features`` turns them into model features,
so training and serving can't drift apart.

The logic mirrors shopper-time-prediction.ipynb, with one difference: the API rows don't carry the store
chain, so ``store_id`` isn't a feature here. With the notebook's parameters this costs
about 0.2 min of CV MAE (16.3 vs 16.1), within fold-to-fold noise.
"""

import numpy as np
import pandas as pd

from src.config import LOCAL_TZ

# lat_x / lng_x: customer location; lat_y / lng_y: store branch location.
INPUT_COLUMNS = [
    "order_id",
    "num_products",
    "kg",
    "un",
    "lat_x",
    "lng_x",
    "promised_time",
    "on_demand",
    "seniority",
    "found_rate",
    "picking_speed",
    "accepted_rate",
    "rating",
    "lat_y",
    "lng_y",
]
REQUIRED_NON_NULL = ["order_id", "lat_x", "lng_x", "lat_y", "lng_y", "promised_time", "on_demand", "seniority"]
NUMERIC_COLUMNS = [
    "num_products", "kg", "un", "lat_x", "lng_x", "lat_y", "lng_y",
    "found_rate", "picking_speed", "accepted_rate", "rating",
]

# Anonymized seniority codes. Their order can't be recovered, so they're an unordered category.
SENIORITY_LEVELS = [
    "41dc7c9e385c4d2b6c1f7836973951bf",
    "50e13ee63f086c2fe84229348bc91b5b",
    "6c90661e6d2c7579f5ce337c3391dbb9",
    "bb29b8d0d196b5db5a5350e5e3ae2b1f",
]

FEATURES = [
    "num_products", "kg", "un", "on_demand",
    "promised_weekday", "promised_hour", "promised_weekhour",
    "seniority", "found_rate", "picking_speed", "accepted_rate", "rating",
    "distance_haversine", "lat_x", "lng_x", "lat_y", "lng_y",
]


class InvalidInput(ValueError):
    """The order rows don't match the expected input format."""


def load_order_rows(data_dir):
    """Join the four raw CSVs into one row per order, in the API input format plus ``total_minutes``."""
    order_products = pd.read_csv(data_dir / "order_products.csv")
    orders = pd.read_csv(data_dir / "orders.csv")
    shoppers = pd.read_csv(data_dir / "shoppers.csv")
    storebranch = pd.read_csv(data_dir / "storebranch.csv")

    size = (order_products
            .groupby(["order_id", "buy_unit"])["quantity"].sum()
            .unstack(fill_value=0)  # 0 if the order has none of that unit
            .rename(columns={"KG": "kg", "UN": "un"}))
    size["num_products"] = order_products.groupby("order_id").size()
    size = size.reset_index()
    size.columns.name = None

    # Left joins: orders without product lines (22 of them) are kept with missing size features.
    rows = (orders
            .rename(columns={"lat": "lat_x", "lng": "lng_x"})
            .merge(size, on="order_id", how="left")
            .merge(shoppers, on="shopper_id", how="left")
            .merge(storebranch.rename(columns={"lat": "lat_y", "lng": "lng_y"}), on="store_branch_id", how="left"))

    assert len(rows) == len(orders) and rows["order_id"].is_unique
    assert rows["seniority"].notna().all() and rows["lat_y"].notna().all()  # every shopper / branch was found
    return rows[INPUT_COLUMNS + ["total_minutes"]]


def validate(rows):
    """Check order rows and return a cleaned copy. Raises ``InvalidInput`` with every problem found."""
    if len(rows) == 0:
        raise InvalidInput("no rows")
    missing = [c for c in INPUT_COLUMNS if c not in rows.columns]
    if missing:
        raise InvalidInput(f"missing columns: {missing}")

    rows = rows[INPUT_COLUMNS].copy()
    problems = []

    nulls = [c for c in REQUIRED_NON_NULL if rows[c].isna().any()]
    if nulls:
        problems.append(f"missing values in {nulls}")

    for col in NUMERIC_COLUMNS:
        values = pd.to_numeric(rows[col], errors="coerce")
        if (values.isna() & rows[col].notna()).any():
            problems.append(f"non-numeric values in {col}")
        rows[col] = values

    promised = pd.to_datetime(rows["promised_time"], errors="coerce", utc=True, format="ISO8601")
    if (promised.isna() & rows["promised_time"].notna()).any():
        problems.append("unparsable promised_time")
    rows["promised_time"] = promised

    on_demand = rows["on_demand"].map(
        lambda v: {"true": 1, "false": 0, "1": 1, "0": 0, "1.0": 1, "0.0": 0}.get(str(v).strip().lower()))
    if (on_demand.isna() & rows["on_demand"].notna()).any():
        problems.append("on_demand must be true/false")
    rows["on_demand"] = on_demand

    unknown = sorted(set(rows["seniority"].dropna()) - set(SENIORITY_LEVELS))
    if unknown:
        problems.append(f"unknown seniority codes: {unknown}")

    if problems:
        raise InvalidInput("; ".join(problems))
    return rows


def haversine(lat1, lng1, lat2, lng2):
    """Great-circle distance in km between two (lat, lng) points given in degrees."""
    lat1, lng1, lat2, lng2 = map(np.radians, (lat1, lng1, lat2, lng2))
    earth_radius_in_km = 6371
    d = np.sin((lat2 - lat1) * 0.5) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lng2 - lng1) * 0.5) ** 2
    return 2 * earth_radius_in_km * np.arcsin(np.sqrt(d))


def build_features(rows):
    """Turn order rows into the model's feature matrix (columns in ``FEATURES`` order). Doesn't modify ``rows``."""
    df = validate(rows)

    # An order with products but none of one unit has 0 of it; orders without products stay missing.
    has_products = df["num_products"].notna()
    for col in ["kg", "un"]:
        df.loc[has_products, col] = df.loc[has_products, col].fillna(0)

    # Local time: without the conversion about 1,300 orders get the wrong weekday.
    promised = df["promised_time"].dt.tz_convert(LOCAL_TZ)
    df["promised_weekday"] = promised.dt.weekday
    df["promised_hour"] = promised.dt.hour
    df["promised_weekhour"] = df["promised_weekday"] * 24 + df["promised_hour"]

    df["seniority"] = pd.Categorical(df["seniority"], categories=SENIORITY_LEVELS)
    df["distance_haversine"] = haversine(df["lat_x"], df["lng_x"], df["lat_y"], df["lng_y"])
    return df[FEATURES]
