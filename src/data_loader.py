"""Loading, validating and cleaning the IBM Telco Customer Churn CSV."""
from __future__ import annotations

import io
import logging
from pathlib import Path
from typing import IO, Union

import pandas as pd

import config

log = logging.getLogger(__name__)

MISSING_FILE_HELP = (
    "Dataset not found.\n"
    "1. Download 'Telco Customer Churn' from "
    "https://www.kaggle.com/datasets/blastchar/telco-customer-churn\n"
    f"2. Save the CSV as: {config.DATA_FILE}\n"
    "   (or upload it from the sidebar of the app)."
)


class DataLoadError(Exception):
    """Raised when the dataset is missing or does not match the expected schema."""


def read_csv(source: Union[str, Path, IO]) -> pd.DataFrame:
    """Read a CSV and strip whitespace from column names."""
    try:
        df = pd.read_csv(source)
    except Exception as exc:  # unreadable / not a CSV
        raise DataLoadError(f"Could not read CSV: {exc}") from exc
    df.columns = [str(c).strip() for c in df.columns]
    return df


def check_schema(df: pd.DataFrame) -> None:
    missing = [c for c in config.REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise DataLoadError("CSV is missing required Telco columns: " + ", ".join(missing))


def load_raw(source: Union[str, Path, IO, None] = None) -> pd.DataFrame:
    """Load the raw dataset (uncleaned, so quality problems stay visible)."""
    if source is None:
        if not config.DATA_FILE.exists():
            raise DataLoadError(MISSING_FILE_HELP)
        source = config.DATA_FILE
    df = read_csv(source)
    check_schema(df)
    return df


def clean_dataset(raw: pd.DataFrame) -> pd.DataFrame:
    """Return a cleaned copy.

    - strips text whitespace
    - converts numeric columns (blank TotalCharges -> NaN; imputed later INSIDE the
      model pipeline so no information leaks from the test set)
    - removes exact duplicate rows (counted on the Data Quality page)
    - adds `churn_flag` (1 = churned)
    """
    check_schema(raw)
    df = raw.copy()
    for col in df.columns:
        if not pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].astype("string").str.strip()
    for col in config.NUMERIC_FEATURES:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    bad = set(df[config.TARGET].dropna().unique()) - {"Yes", "No"}
    if bad:
        raise DataLoadError(f"Target '{config.TARGET}' has unexpected values: {sorted(bad)}")
    before = len(df)
    df = df.drop_duplicates().dropna(subset=[config.TARGET]).reset_index(drop=True)
    if len(df) < before:
        log.info("Removed %d duplicate/unlabelled rows", before - len(df))
    for col in config.CATEGORICAL_FEATURES + [config.TARGET, config.ID_COL]:
        df[col] = df[col].astype("object")
    df["churn_flag"] = (df[config.TARGET] == "Yes").astype(int)
    return df


def save_uploaded(file_bytes: bytes) -> Path:
    """Validate and save an uploaded dataset to data/. Raises DataLoadError if invalid."""
    df = read_csv(io.BytesIO(file_bytes))
    check_schema(df)
    config.DATA_DIR.mkdir(exist_ok=True)
    config.DATA_FILE.write_bytes(file_bytes)
    return config.DATA_FILE
