"""Preprocessing pipeline + train/test split.

All imputers/scalers/encoders live INSIDE the sklearn Pipeline, so they are fit
on training data only (no leakage) and reused unchanged at prediction time.
"""
from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import config


def build_preprocessor() -> ColumnTransformer:
    numeric = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer(
        [("num", numeric, config.NUMERIC_FEATURES),
         ("cat", categorical, config.CATEGORICAL_FEATURES)],
        verbose_feature_names_out=False,
    )


def split_data(df: pd.DataFrame, test_size: float = config.TEST_SIZE,
               seed: int = config.RANDOM_SEED):
    """Stratified split. Identifier and target are never part of X."""
    X = df[config.FEATURE_COLUMNS]
    y = (df[config.TARGET] == "Yes").astype(int)
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=seed)


def class_balance(y) -> dict:
    counts = pd.Series(y).value_counts()
    n = int(counts.sum())
    neg, pos = int(counts.get(0, 0)), int(counts.get(1, 0))
    return {"n": n, "retained": neg, "churned": pos,
            "churn_rate": pos / n if n else 0.0,
            "imbalance_ratio": (neg / pos) if pos else float("inf")}
