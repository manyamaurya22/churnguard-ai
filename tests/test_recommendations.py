import pandas as pd
import pytest

from src import recommendations as rec
from src.explainability import explain_customer, global_importance
import config


CUSTOMER = {"customerID": "SECRET-1", "Contract": "Month-to-month", "tenure": 3, "MonthlyCharges": 95.0,
            "InternetService": "Fiber optic", "TechSupport": "No", "OnlineSecurity": "No",
            "PaymentMethod": "Electronic check", "SeniorCitizen": 0}


def test_rules_trigger_and_are_explained():
    out = rec.rule_based(CUSTOMER, 0.8, "High")
    assert len(out["reasons"]) >= 4 and out["actions"][0].startswith("Priority")


def test_payload_strips_identifiers():
    drivers = pd.DataFrame({"feature": ["Contract"], "effect": [0.3]})
    payload = rec.build_safe_payload(CUSTOMER, 0.8, "High", drivers)
    assert "customerID" not in str(payload) and "SECRET" not in str(payload)


def test_payload_validation():
    drivers = pd.DataFrame({"feature": ["Contract"], "effect": [0.3]})
    with pytest.raises(ValueError):
        rec.build_safe_payload(CUSTOMER, 1.5, "High", drivers)
    with pytest.raises(ValueError):
        rec.build_safe_payload(CUSTOMER, 0.5, "Extreme", drivers)


def test_llm_without_key_degrades_gracefully(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    text, err = rec.llm_recommendation({"x": 1})
    assert text is None and "rule-based" in err


def test_explanations_are_real_numbers(clean_df, pipeline):
    X = clean_df[config.FEATURE_COLUMNS]
    imp, method = global_importance(pipeline, X, clean_df["churn_flag"], sample=200)
    assert len(imp) == len(config.FEATURE_COLUMNS) and (imp["importance"] >= 0).all() and method
    exp, _ = explain_customer(pipeline, X.iloc[[0]], X)
    assert exp["effect"].notna().all()
