import requests
import sys

from os import rename

from pandas import read_csv

url = "http://localhost:5000/predict"


def run(file="../data/sample.csv"):
    r = requests.post(url, files={"file": open(file, "rb")})
    with open("../data/predicted.csv", "wb") as f:
        f.write(r.content)


if __name__ == "__main__":
    # TODO: create parameter for file
    run()
