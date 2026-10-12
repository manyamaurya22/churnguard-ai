"""Inference: artifact loading (integrity-checked), CSV validation, single/batch prediction."""
from __future__ import annotations

import hashlib
import json
import logging

import joblib
import numpy as np
import pandas as pd

import config

log = logging.getLogger(__name__)


class ModelNotTrainedError(Exception):
    pass


class CSVValidationError(Exception):
    pass


def load_artifacts():
    """Load the trained pipeline from models/ after verifying its SHA-256 hash.

    joblib/pickle files can execute code when loaded, so we only load a file this
    project wrote itself (hash recorded at training time). Never load model files
    from untrusted sources.
    """
    path = config.MODELS_DIR / "best_pipeline.joblib"
    hash_file = config.MODELS_DIR / "best_pipeline.joblib.sha256"
    if not path.exists() or not hash_file.exists():
        raise ModelNotTrainedError("No trained model found. Open 'Model Training & Comparison' "
                                   "and train, or run:  python -m src.train_models")
    if hashlib.sha256(path.read_bytes()).hexdigest() != hash_file.read_text().strip():
        raise ModelNotTrainedError("Model file hash mismatch - refusing to load. Retrain the model.")
    meta = json.loads((config.MODELS_DIR / "metadata.json").read_text())
    return joblib.load(path), meta


def risk_category(p: float, threshold: float = config.DEFAULT_THRESHOLD,
                  medium_fraction: float = 0.5) -> str:
    """High: p >= threshold. Medium: p >= threshold*medium_fraction. Else Low."""
    if p >= threshold:
        return "High"
    if p >= threshold * medium_fraction:
        return "Medium"
    return "Low"


def validate_batch(df: pd.DataFrame):
    """Validate an uploaded CSV. Returns (X, status).

    X      : feature frame (same index as df) with numeric columns coerced
    status : Series, "OK" or a semicolon-separated list of problems for each row.
    Raises CSVValidationError only for file-level problems (empty / missing columns).
    """
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    if df.empty:
        raise CSVValidationError("The uploaded CSV has no rows.")
    missing = [c for c in config.FEATURE_COLUMNS if c not in df.columns]
    if missing:
        raise CSVValidationError("Missing required columns: " + ", ".join(missing))
    clash = [c for c in config.PRED_COLUMNS if c in df.columns]
    if clash:
        raise CSVValidationError("Input already contains output column(s) " + ", ".join(clash)
                                 + ". Rename them before uploading.")
    X = df[config.FEATURE_COLUMNS].copy()
    rules: list[tuple[pd.Series, str]] = []

    for col in config.CATEGORICAL_FEATURES:
        s = X[col].astype("string").str.strip()
        X[col] = s.astype("object")
        rules.append((~s.isin(config.CATEGORY_VALUES[col]), f"{col}: missing/invalid value"))

    for col in config.NUMERIC_FEATURES:
        raw = X[col].astype("string").str.strip()
        num = pd.to_numeric(raw, errors="coerce")
        X[col] = num.astype(float)
        if col == "TotalCharges":  # blank is legitimate (new customers) -> imputed by the pipeline
            rules.append((num.isna() & raw.notna() & (raw != ""), f"{col}: not a number"))
        else:
            rules.append((num.isna(), f"{col}: missing/not a number"))
        rules.append(((num < 0).fillna(False), f"{col}: negative"))
    rules.append((~X["SeniorCitizen"].isin([0, 1]) & X["SeniorCitizen"].notna(),
                  "SeniorCitizen: must be 0 or 1"))

    flags = [(m.to_numpy(dtype=bool), msg) for m, msg in rules]
    status = ["; ".join(msg for m, msg in flags if m[i]) or "OK" for i in range(len(df))]
    return X, pd.Series(status, index=df.index, name="validation_status")


def predict_dataframe(df: pd.DataFrame, pipeline, threshold: float = config.DEFAULT_THRESHOLD) -> pd.DataFrame:
    """Original rows (unchanged, same order) + prediction columns. Invalid rows are kept,
    get empty predictions and a reason in `validation_status`."""
    X, status = validate_batch(df)
    ok = (status == "OK").to_numpy()
    proba = np.full(len(df), np.nan)
    if ok.any():
        proba[ok] = pipeline.predict_proba(X.loc[ok, config.FEATURE_COLUMNS])[:, 1]
    out = df.copy()
    out["churn_probability"] = proba
    out["predicted_churn"] = [("Yes" if p >= threshold else "No") if o else "" for p, o in zip(proba, ok)]
    out["risk_category"] = [risk_category(p, threshold) if o else "" for p, o in zip(proba, ok)]
    out["validation_status"] = status
    return out


def predict_single(record: dict, pipeline, threshold: float = config.DEFAULT_THRESHOLD) -> dict:
    row = pd.DataFrame([record])
    res = predict_dataframe(row, pipeline, threshold).iloc[0]
    if res["validation_status"] != "OK":
        raise CSVValidationError(res["validation_status"])
    return {"probability": float(res["churn_probability"]),
            "predicted_churn": res["predicted_churn"], "risk": res["risk_category"]}
