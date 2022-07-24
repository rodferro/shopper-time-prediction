import numpy as np
import pandas as pd
import warnings

pd.options.mode.chained_assignment = None


def preprocess(df):
    df[["kg", "un"]].fillna(0, inplace=True)

    df["promised_time"] = pd.to_datetime(df["promised_time"])
    df["promised_weekday"] = df["promised_time"].dt.weekday
    df["promised_weekofyear"] = df["promised_time"].dt.weekofyear
    df["promised_hour"] = df["promised_time"].dt.hour
    df["promised_minute"] = df["promised_time"].dt.minute
    df["promised_weekhour"] = df["promised_weekday"] * 24 + df["promised_hour"]
    df = df.drop("promised_time", axis=1)

    df["on_demand"] = df["on_demand"].astype(int)

    mapping = {
        "6c90661e6d2c7579f5ce337c3391dbb9": 1,
        "50e13ee63f086c2fe84229348bc91b5b": 2,
        "41dc7c9e385c4d2b6c1f7836973951bf": 3,
        "bb29b8d0d196b5db5a5350e5e3ae2b1f": 4,
    }
    df["seniority"] = df["seniority"].map(mapping)

    df["haversine_distance"] = haversine_distance(
        df["lat_x"].values, df["lng_x"].values, df["lat_y"].values, df["lng_y"].values
    )
    df = df.drop(["lat_x", "lng_x", "lat_y", "lng_y"], axis=1)

    return df


def haversine_distance(lat1, lng1, lat2, lng2):
    lat1, lng1, lat2, lng2 = map(np.radians, (lat1, lng1, lat2, lng2))
    earth_radius_in_km = 6371
    lat = lat2 - lat1
    lng = lng2 - lng1
    d = np.sin(lat * 0.5) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(lng * 0.5) ** 2
    return 2 * earth_radius_in_km * np.arcsin(np.sqrt(d))


if __name__ == "__main__":
    train = pd.read_csv("../data/train.csv")
    test = pd.read_csv("../data/test.csv")
    train_preprocessed = preprocess(train)
    test_preprocessed = preprocess(test)
    train_preprocessed.to_csv("../data/train_preprocessed.csv", index=False)
    test_preprocessed.to_csv("../data/test_preprocessed.csv", index=False)
