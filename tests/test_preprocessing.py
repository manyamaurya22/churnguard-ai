import numpy as np
import pytest

import config
from src.data_loader import DataLoadError, clean_dataset
from src.preprocessing import build_preprocessor, class_balance, split_data


def test_clean_handles_blank_total_charges(raw_df):
    df = clean_dataset(raw_df)
    assert df["TotalCharges"].isna().sum() == (raw_df["tenure"] == 0).sum()
    assert set(df["churn_flag"].unique()) <= {0, 1}


def test_missing_column_raises(raw_df):
    with pytest.raises(DataLoadError):
        clean_dataset(raw_df.drop(columns=["Contract"]))


def test_bad_target_raises(raw_df):
    raw_df.loc[0, "Churn"] = "Maybe"
    with pytest.raises(DataLoadError):
        clean_dataset(raw_df)


def test_split_is_stratified_and_leak_free(clean_df):
    X_tr, X_te, y_tr, y_te = split_data(clean_df)
    assert config.TARGET not in X_tr.columns and config.ID_COL not in X_tr.columns
    assert abs(y_tr.mean() - y_te.mean()) < 0.02
    assert set(X_tr.index).isdisjoint(X_te.index)
    # reproducible
    X_tr2, *_ = split_data(clean_df)
    assert list(X_tr.index) == list(X_tr2.index)


def test_preprocessor_outputs_finite_numbers(clean_df):
    X_tr, X_te, _, _ = split_data(clean_df)
    prep = build_preprocessor().fit(X_tr)
    out = prep.transform(X_te)
    assert np.isfinite(out).all()
    assert out.shape[0] == len(X_te)


def test_preprocessor_ignores_unseen_category(clean_df):
    X_tr, X_te, _, _ = split_data(clean_df)
    prep = build_preprocessor().fit(X_tr)
    weird = X_te.head(2).copy()
    weird["Contract"] = "Brand new contract"
    assert np.isfinite(prep.transform(weird)).all()


def test_class_balance(clean_df):
    cb = class_balance(clean_df["churn_flag"])
    assert cb["churned"] + cb["retained"] == cb["n"]
