"""Business recommendations: transparent rules (offline) + optional Claude narrative (server-side)."""
from __future__ import annotations

import logging
import os

import pandas as pd

import config

log = logging.getLogger(__name__)

# Only these keys may ever leave the machine. No customerID, no free text.
SAFE_KEYS = set(config.FEATURE_COLUMNS)


def rule_based(customer: dict, probability: float, risk: str, drivers: pd.DataFrame | None = None) -> dict:
    """Each rule states the condition that triggered it, so it can be audited."""
    c = customer
    reasons, actions = [], []

    def add(reason, action):
        reasons.append(reason)
        actions.append(action)

    if c.get("Contract") == "Month-to-month":
        add("Month-to-month contract (no commitment)", "Offer a discounted 1- or 2-year plan or a loyalty credit.")
    if float(c.get("tenure", 99)) <= 12:
        add(f"Short tenure ({c.get('tenure')} months)", "Run an onboarding check-in and first-year engagement programme.")
    if c.get("InternetService") == "Fiber optic" and c.get("TechSupport") == "No":
        add("Fiber customer without tech support", "Bundle free/discounted tech support for 3-6 months.")
    if c.get("InternetService") in ("Fiber optic", "DSL") and c.get("OnlineSecurity") == "No":
        add("No online security add-on", "Offer a security add-on trial to raise product stickiness.")
    if c.get("PaymentMethod") == "Electronic check":
        add("Pays by electronic check", "Incentivise automatic payment (card/bank) with a small bill credit.")
    if float(c.get("MonthlyCharges", 0)) >= 80:
        add(f"High monthly charge ({float(c['MonthlyCharges']):.2f})", "Review the plan for right-sizing or a value bundle before they shop around.")
    if c.get("SeniorCitizen") in (1, "1") and c.get("TechSupport") == "No":
        add("Senior customer without tech support", "Proactive assisted-support outreach.")
    if not actions:
        actions.append("No specific risk rule triggered; keep standard engagement.")
    if risk == "High":
        actions.insert(0, "Priority: contact within days via the retention team with a personalised offer.")
    elif risk == "Medium":
        actions.insert(0, "Add to a nurture campaign and monitor.")
    top = ""
    if drivers is not None and len(drivers):
        up = drivers[drivers["effect"] > 0].head(3)["feature"].tolist()
        if up:
            top = "Top model drivers raising risk: " + ", ".join(up) + "."
    summary = f"Churn probability {probability:.1%} ({risk} risk). " + top
    return {"summary": summary.strip(), "reasons": reasons, "actions": actions, "mode": "rule-based (offline)"}


def build_safe_payload(customer: dict, probability: float, risk: str, drivers: pd.DataFrame) -> dict:
    """Validate + anonymise. Raises ValueError on invalid input."""
    if not 0.0 <= float(probability) <= 1.0:
        raise ValueError("probability must be between 0 and 1")
    if risk not in ("Low", "Medium", "High"):
        raise ValueError("risk must be Low, Medium or High")
    profile = {k: (v.item() if hasattr(v, "item") else v) for k, v in customer.items() if k in SAFE_KEYS}
    drv = []
    for _, r in drivers.head(6).iterrows():
        drv.append({"feature": str(r["feature"]), "effect": round(float(r["effect"]), 4)})
    return {"profile": profile, "churn_probability": round(float(probability), 4),
            "risk": risk, "top_drivers": drv}


SYSTEM_PROMPT = (
    "You are a telecom retention analyst. You get an anonymised customer profile, a model's churn "
    "probability and the model's real feature effects (positive = raises churn risk). Write a brief "
    "business note: (1) why the customer may be at risk, (2) 2-3 retention actions, (3) what segment "
    "this customer represents. Use only the supplied data, do not invent statistics, and say that the "
    "model output is probabilistic. Max 180 words.")


def llm_recommendation(payload: dict, api_key: str | None = None, model: str | None = None):
    """Returns (text | None, error | None). Never raises."""
    key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
    if not key:
        return None, "No ANTHROPIC_API_KEY configured - using offline rule-based recommendations."
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=key, timeout=30.0, max_retries=2)
        msg = client.messages.create(
            model=model or os.environ.get("ANTHROPIC_MODEL", config.ANTHROPIC_MODEL_DEFAULT),
            max_tokens=400, system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": str(payload)}])
        text = "".join(b.text for b in msg.content if getattr(b, "type", "") == "text").strip()
        return (text, None) if text else (None, "The API returned an empty response.")
    except Exception as exc:  # network, auth, rate limit, missing package ...
        log.warning("Claude API call failed: %s", type(exc).__name__)
        return None, f"AI narrative unavailable ({type(exc).__name__}). Showing rule-based advice instead."


def priority_segments(pred_df: pd.DataFrame, columns=("Contract", "InternetService", "PaymentMethod"),
                      min_size: int = 30) -> pd.DataFrame:
    """Segments ranked by number of high-risk customers (needs `risk_category`, `MonthlyCharges`)."""
    rows = []
    valid = pred_df[pred_df["risk_category"] != ""]
    for col in columns:
        for val, g in valid.groupby(col):
            if len(g) < min_size:
                continue
            hi = g[g["risk_category"] == "High"]
            rows.append({"segment": f"{col} = {val}", "customers": len(g), "high_risk": len(hi),
                         "high_risk_share": len(hi) / len(g), "avg_probability": g["churn_probability"].mean(),
                         "monthly_revenue_high_risk": hi["MonthlyCharges"].sum()})
    return pd.DataFrame(rows).sort_values("monthly_revenue_high_risk", ascending=False).reset_index(drop=True) \
        if rows else pd.DataFrame(columns=["segment", "customers", "high_risk", "high_risk_share",
                                           "avg_probability", "monthly_revenue_high_risk"])
