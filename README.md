# Data Science Case Study

Predict how many minutes a shopper needs to complete an order (`total_minutes`). The case statement is [below](#the-case).

- [`case-ml.ipynb`](case-ml.ipynb): the analysis. It covers data preparation, EDA, model selection with cross-validation, error analysis
  and validation checks, and writes [`data/submission.csv`](data/submission.csv) (all 2,000 unlabeled orders).
- [`src/`](src/): the same model packaged for production. It has a training script, a prediction API and a Docker image.

## Results

MAE in minutes on a 20% hold-out set (1,600 orders):

| Model | Hold-out MAE |
|---|---|
| Predict the training mean (baseline) | 27.26 |
| Notebook model (LightGBM, L1 loss, tuned) | 15.69 |
| Production model (same, without `store_id`) | 15.77 |

The production model drops the store-chain feature (`store_id`), because the API input doesn't include it (see [API](#api)).
That costs about 0.2 min of CV MAE, within fold-to-fold noise. Both models are refit on all 8,000 labeled orders.

The notebook explains every decision and the limitations. The main limitation is that all orders come from two days, so
for orders on future days ≈ 18.4–19.7 min MAE is the realistic expectation.

## How to run

### Locally

Requires Python 3.12 (for example `brew install python@3.12`).

```sh
python3.12 -m venv .venv
.venv/bin/pip install -r requirements-dev.txt

.venv/bin/python -m src.train          # trains, prints hold-out MAE, saves model/model.joblib and model/metrics.json
.venv/bin/python -m src.api            # development server on http://localhost:5000
.venv/bin/python -m src.request        # sends data/sample.csv, saves data/predicted.csv
.venv/bin/python -m src.predict data/sample.csv data/predicted.csv   # same, without the server

.venv/bin/pytest                       # tests
.venv/bin/jupyter nbconvert --to notebook --execute --inplace case-ml.ipynb   # re-runs the notebook (≈10 min)
```

Run every command from the repository root.

### Docker

The model is trained while the image builds, so the image is self-contained and the repository holds no model file.

```sh
docker build -t case-ml .
docker run --rm -p 5000:5000 case-ml   # gunicorn on port 5000

curl localhost:5000/health
curl -F file=@data/sample.csv localhost:5000/predict
```

## API

- `GET /health` returns `{"status": "ok"}`.
- `POST /predict` takes a multipart upload of a CSV in the `file` field and returns a CSV with `order_id,total_minutes`.
  Invalid input gets a `400` with `{"error": "..."}`.

The input has one row per order, already joined with the shopper and store data. See [`data/sample.csv`](data/sample.csv):

| Column | Meaning |
|---|---|
| `order_id` | order ID, echoed back |
| `num_products`, `kg`, `un` | number of product lines, total weight of weighed items, total count of unit items |
| `lat_x`, `lng_x` | delivery location |
| `lat_y`, `lng_y` | store branch location |
| `promised_time` | ISO timestamp; treated as UTC if it has no offset |
| `on_demand` | `true` / `false` |
| `seniority`, `found_rate`, `picking_speed`, `accepted_rate`, `rating` | shopper data, as in `shoppers.csv` |

Shopper statistics, `num_products`, `kg` and `un` may be empty. Extra columns are ignored.

## Layout

| Path | Content |
|---|---|
| `src/config.py` | paths, time zone, final model parameters |
| `src/features.py` | joins the raw CSVs into input rows; validation and feature engineering shared by training and serving |
| `src/train.py`, `src/predict.py`, `src/api.py`, `src/request.py` | training, prediction, HTTP API, API client |
| `tests/` | feature and API tests |
| `requirements.txt` / `requirements-dev.txt` | runtime (Docker) / notebook and tests |

---

## The case

Acme Corporation has operations in several cities and countries, delivering thousands of orders every day. In order to deliver these orders on time we depend on good estimations of how much time the shopper needs to complete the order.

You will be creating a machine learning model to make these estimations. As we internally build our machine learning solutions using python, we ask you to do the same. However you are free to use the libraries you are most comfortable with.

## Data

In this repository, we have included data representing the order, shopper and the store branch. 

### File description and data fields
***order_products.csv:***
- order_id: ID of the order
- product_id: ID of the product
- quantity: The quantity ordered of this product
- buy_unit: The unit of the product (KG/UN)

***orders.csv:***
- order_id: ID of the order
- lat: The latitude of the delivery location
- lng: The longitude of the delivery location
- promised_time: The delivery time promised to the user
- on_demand: If true, the order was promised to be delivered in less than X minutes
- shopper_id: ID representing the shopper completed the order.
- store_branch_id: ID of the store branch
- total_minutes: The total minutes it took to complete the order (label)

***shopper.csv***
- shopper_id: ID of the shopper
- seniority: The experience level of the shopper.
- found_rate: Percentage of products found by shopper historical.
- picking_speed: Historical picking speed, products pr minutes.
- accepted_rate: Percentage of orders historically accepted by shopper
- rating: client rating of shopper

***storebranch.csv:***
- store_branch_id: ID of the store branch
- store: ID representing the store
- lat: Latitude of the branch location
- lng: Longitude of the branch location

*All the data has been anonymized*

### Objective

The objective is to predict the `total_minutes` a order takes to complete, where the rows not containing a `total_minutes` value should be set aside as a part of the submission file, containing the `order_id` with the predicted values. 

As we are interested in seeing how you attacked the problem, we also ask you to include your code together with the submission file. The code needs to be well documented, explaining the decisions made. With these explanations, we will be looking at everything from how the data was processed, features used to the completed model and predictions. 

Good luck!