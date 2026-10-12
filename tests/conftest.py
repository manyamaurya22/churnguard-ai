"""TEST FIXTURES ONLY.

`make_telco_like` builds a small, schema-compatible table with random values so unit tests can run
without the Kaggle file. It is NOT real data and is never used by the app or for reported results.
"""
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

import config
from src.preprocessing import build_preprocessor


def make_telco_like(n=400, seed=0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    d = {"customerID": [f"T-{i:04d}" for i in range(n)]}
    for col, vals in config.CATEGORY_VALUES.items():
        d[col] = rng.choice(vals, n)
    d["SeniorCitizen"] = rng.integers(0, 2, n)
    d["tenure"] = rng.integers(0, 73, n)
    d["MonthlyCharges"] = rng.uniform(18, 118, n).round(2)
    d["TotalCharges"] = (d["tenure"] * d["MonthlyCharges"]).round(2).astype(object)
    d["TotalCharges"][d["tenure"] == 0] = " "
    p = 0.15 + 0.4 * (d["Contract"] == "Month-to-month") - 0.003 * d["tenure"]
    d["Churn"] = np.where(rng.random(n) < np.clip(p, 0.02, 0.9), "Yes", "No")
    return pd.DataFrame(d)[config.REQUIRED_COLUMNS]


@pytest.fixture
def raw_df():
    return make_telco_like()


@pytest.fixture
def clean_df(raw_df):
    from src.data_loader import clean_dataset
    return clean_dataset(raw_df)


@pytest.fixture
def pipeline(clean_df):
    from src.preprocessing import split_data
    X_tr, _, y_tr, _ = split_data(clean_df)
    pipe = Pipeline([("prep", build_preprocessor()), ("clf", LogisticRegression(max_iter=500))])
    return pipe.fit(X_tr, y_tr)
