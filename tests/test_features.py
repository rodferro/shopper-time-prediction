import numpy as np
import pandas as pd
import pytest

from src.features import FEATURES, InvalidInput, build_features, haversine


def test_load_keeps_every_order(rows):
    assert len(rows) == 10_000
    assert rows["total_minutes"].isna().sum() == 2_000
    assert rows["num_products"].isna().sum() == 22  # orders without product lines are kept


def test_build_features_columns_and_no_mutation(sample):
    before = sample.copy()
    X = build_features(sample)
    assert list(X.columns) == FEATURES
    assert len(X) == len(sample)
    pd.testing.assert_frame_equal(sample, before)


def test_promised_time_is_converted_to_local_time(sample):
    # 2019-10-19 01:00 UTC is Friday 22:00 in Santiago (UTC-3).
    row = sample.iloc[[0]].assign(promised_time="2019-10-19 01:00:00+00:00")
    X = build_features(row)
    assert X["promised_weekday"].iloc[0] == 4
    assert X["promised_hour"].iloc[0] == 22
    assert X["promised_weekhour"].iloc[0] == 4 * 24 + 22


def test_missing_unit_quantity_is_zero_only_when_the_order_has_products(sample):
    rows = sample.iloc[[0, 1]].copy()
    rows["kg"] = np.nan
    rows.loc[rows.index[1], "num_products"] = np.nan
    X = build_features(rows)
    assert X["kg"].iloc[0] == 0
    assert np.isnan(X["kg"].iloc[1])


def test_haversine():
    assert haversine(-33.4, -70.6, -33.4, -70.6) == pytest.approx(0)
    assert haversine(0, 0, 1, 0) == pytest.approx(111.19, abs=0.01)


@pytest.mark.parametrize("change, message", [
    (lambda df: df.drop(columns="seniority"), "missing columns"),
    (lambda df: df.assign(seniority="not-a-level"), "unknown seniority"),
    (lambda df: df.assign(promised_time="yesterday"), "promised_time"),
    (lambda df: df.assign(on_demand="maybe"), "on_demand"),
    (lambda df: df.assign(num_products="many"), "non-numeric"),
    (lambda df: df.assign(lat_x=np.nan), "missing values"),
    (lambda df: df.iloc[0:0], "no rows"),
])
def test_invalid_input(sample, change, message):
    with pytest.raises(InvalidInput, match=message):
        build_features(change(sample))
