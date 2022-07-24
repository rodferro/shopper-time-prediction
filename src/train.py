import numpy as np
import pandas as pd
import pickle

from sklearn.metrics import mean_absolute_error
from time import time
from xgboost import XGBRegressor


def train(X_train, y_train):
    reg = XGBRegressor(
        eval_metric="mae",
        subsample=0.7,
        n_estimators=1000,
        max_depth=5,
        learning_rate=0.01,
        colsample_bytree=0.7999999999999999,
        colsample_bylevel=0.8999999999999999,
    )
    reg.fit(X_train, y_train)
    return reg


def score(reg, X_test, y_test):
    return mean_absolute_error(y_test, reg.predict(X_test))


def save(reg):
    pickle.dump(reg, open("../model/model.sav", "wb"))


if __name__ == "__main__":
    df_train = pd.read_csv("../data/train_preprocessed.csv")
    df_test = pd.read_csv("../data/test_preprocessed.csv")

    X_train, y_train = (
        df_train.drop(["order_id", "total_minutes"], axis=1),
        df_train["total_minutes"],
    )
    X_test, y_test = (
        df_test.drop(["order_id", "total_minutes"], axis=1),
        df_test["total_minutes"],
    )

    reg = train(X_train, y_train)

    save(reg)

    print("MAE:", score(reg, X_test, y_test))
