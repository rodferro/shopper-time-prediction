import pandas as pd
import pickle

from preprocess import preprocess

reg = pickle.load(open("../model/model.sav", "rb"))


def predict(X):
    return reg.predict(preprocess(X.drop(["order_id"], axis=1)))


if __name__ == "__main__":
    pass