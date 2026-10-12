import pandas as pd
import pytest

import config
from src.predict import CSVValidationError, predict_dataframe, predict_single, risk_category, validate_batch


def test_missing_columns_rejected(raw_df):
    with pytest.raises(CSVValidationError):
        validate_batch(raw_df.drop(columns=["tenure", "Contract"]))


def test_empty_csv_rejected(raw_df):
    with pytest.raises(CSVValidationError):
        validate_batch(raw_df.iloc[0:0])


def test_output_column_clash_rejected(raw_df):
    raw_df["risk_category"] = "x"
    with pytest.raises(CSVValidationError):
        validate_batch(raw_df)


def test_invalid_rows_flagged_not_dropped(raw_df, pipeline):
    df = raw_df.head(10).copy().astype({"tenure": object, "MonthlyCharges": object})
    df.loc[1, "Contract"] = "Weekly"
    df.loc[2, "tenure"] = "abc"
    df.loc[3, "MonthlyCharges"] = -5
    out = predict_dataframe(df, pipeline)
    assert len(out) == len(df)                                  # nothing silently dropped
    assert out.loc[[1, 2, 3], "validation_status"].ne("OK").all()
    assert out.loc[[1, 2, 3], "churn_probability"].isna().all()
    assert out.loc[[0, 4], "validation_status"].eq("OK").all()
    for col in df.columns:                                      # original data preserved
        assert out[col].astype(str).tolist() == df[col].astype(str).tolist()


def test_probabilities_valid_and_threshold_applied(raw_df, pipeline):
    low = predict_dataframe(raw_df.head(50), pipeline, threshold=0.99)
    high = predict_dataframe(raw_df.head(50), pipeline, threshold=0.01)
    p = low["churn_probability"].dropna()
    assert ((p >= 0) & (p <= 1)).all()
    assert ((high["churn_probability"] >= 0.01) == (high["predicted_churn"] == "Yes")).all()
    assert ((low["churn_probability"] >= 0.99) == (low["predicted_churn"] == "Yes")).all()
    assert (high["predicted_churn"] == "Yes").sum() >= (low["predicted_churn"] == "Yes").sum()


def test_blank_total_charges_is_ok(raw_df, pipeline):
    new = raw_df[raw_df["tenure"] == 0].head(3)
    assert (predict_dataframe(new, pipeline)["validation_status"] == "OK").all()


def test_predict_single_matches_batch(raw_df, pipeline):
    rec = raw_df.iloc[5][config.FEATURE_COLUMNS].to_dict()
    single = predict_single(rec, pipeline)
    batch = predict_dataframe(raw_df.iloc[[5]], pipeline).iloc[0]
    assert single["probability"] == pytest.approx(batch["churn_probability"])


def test_risk_category_bands():
    assert risk_category(0.6, 0.5) == "High"
    assert risk_category(0.3, 0.5) == "Medium"
    assert risk_category(0.1, 0.5) == "Low"
