import pandas as pd
import pytest

from src.config import DATA_DIR, MODEL_PARAMS
from src.features import build_features, load_order_rows
from src.train import train


@pytest.fixture(scope="session")
def rows():
    return load_order_rows(DATA_DIR)


@pytest.fixture(scope="session")
def model(rows):
    """A small, fast model with the production setup (fewer trees)."""
    labeled = rows[rows["total_minutes"].notna()]
    return train(build_features(labeled), labeled["total_minutes"], {**MODEL_PARAMS, "n_estimators": 20})


@pytest.fixture
def sample():
    return pd.read_csv(DATA_DIR / "sample.csv")
