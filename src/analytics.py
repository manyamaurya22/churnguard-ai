"""Data quality, KPIs, segmentation, correlation and SQL analytics (all computed from the data)."""
from __future__ import annotations

import pandas as pd

import config
from src import database

TENURE_BINS = [-1, 12, 24, 48, 72, 10_000]
TENURE_LABELS = ["0-12 mo", "13-24 mo", "25-48 mo", "49-72 mo", "72+ mo"]

# Read-only example queries shown in the EDA page (run against SQLite).
SQL_QUERIES = {
    "Churn rate by contract": (
        "SELECT Contract, COUNT(*) AS customers, "
        "ROUND(100.0*SUM(Churn='Yes')/COUNT(*),2) AS churn_pct "
        "FROM customers GROUP BY Contract ORDER BY churn_pct DESC"),
    "Monthly revenue lost to churn, by payment method": (
        "SELECT PaymentMethod, ROUND(SUM(CASE WHEN Churn='Yes' THEN MonthlyCharges END),2) "
        "AS monthly_revenue_churned, COUNT(*) AS customers "
        "FROM customers GROUP BY PaymentMethod ORDER BY monthly_revenue_churned DESC"),
    "Churn by internet service": (
        "SELECT InternetService, COUNT(*) AS customers, "
        "ROUND(100.0*SUM(Churn='Yes')/COUNT(*),2) AS churn_pct, "
        "ROUND(AVG(MonthlyCharges),2) AS avg_monthly FROM customers GROUP BY InternetService"),
}


def data_quality_report(raw: pd.DataFrame) -> dict:
    """Quality report on the RAW data (before cleaning)."""
    blank = raw.apply(lambda s: int(s.astype("string").str.strip().eq("").sum()))
    missing = raw.isna().sum() + blank
    table = pd.DataFrame({"dtype": raw.dtypes.astype(str), "missing_or_blank": missing,
                          "missing_pct": (missing / max(len(raw), 1) * 100).round(2),
                          "unique": raw.nunique()})
    tc = pd.to_numeric(raw["TotalCharges"], errors="coerce")
    return {
        "rows": len(raw), "columns": raw.shape[1],
        "exact_duplicate_rows": int(raw.duplicated().sum()),
        "duplicate_customer_ids": int(raw[config.ID_COL].duplicated().sum()),
        "total_missing_cells": int(missing.sum()),
        "table": table.reset_index(names="column"),
        "unparseable_total_charges": raw.loc[tc.isna(), :],
    }


def kpis(df: pd.DataFrame) -> dict:
    n = len(df)
    churned = int(df["churn_flag"].sum())
    return {
        "customers": n, "churned": churned,
        "churn_rate": churned / n if n else 0.0,
        "retention_rate": 1 - churned / n if n else 0.0,
        "avg_monthly_charges": float(df["MonthlyCharges"].mean()) if n else 0.0,
        "monthly_revenue": float(df["MonthlyCharges"].sum()),
        "monthly_revenue_at_risk": float(df.loc[df["churn_flag"] == 1, "MonthlyCharges"].sum()),
        "avg_tenure": float(df["tenure"].mean()) if n else 0.0,
    }


def add_segments(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["tenure_group"] = pd.cut(out["tenure"], bins=TENURE_BINS, labels=TENURE_LABELS).astype(str)
    return out


def segment_table(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Customers, churn rate and revenue by segment."""
    g = df.groupby(col, observed=True).agg(
        customers=("churn_flag", "size"), churned=("churn_flag", "sum"),
        churn_rate=("churn_flag", "mean"), avg_monthly=("MonthlyCharges", "mean"),
        monthly_revenue=("MonthlyCharges", "sum")).reset_index()
    return g.sort_values("churn_rate", ascending=False)


def correlation_matrix(df: pd.DataFrame) -> pd.DataFrame:
    cols = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges", "churn_flag"]
    return df[cols].corr()


def auto_insights(df: pd.DataFrame) -> list[str]:
    """Plain-language insights, every number computed from the loaded data."""
    d = add_segments(df)
    k = kpis(d)
    out = [f"Overall churn rate is {k['churn_rate']:.1%} ({k['churned']:,} of {k['customers']:,} customers)."]
    for col, label in [("Contract", "contract type"), ("tenure_group", "tenure band"),
                       ("PaymentMethod", "payment method"), ("InternetService", "internet service")]:
        t = segment_table(d, col)
        t = t[t["customers"] >= 30]
        if len(t) >= 2:
            hi, lo = t.iloc[0], t.iloc[-1]
            out.append(f"By {label}, churn is highest for '{hi[col]}' ({hi['churn_rate']:.1%}) "
                       f"and lowest for '{lo[col]}' ({lo['churn_rate']:.1%}).")
    ch = d[d.churn_flag == 1]["MonthlyCharges"].mean()
    st = d[d.churn_flag == 0]["MonthlyCharges"].mean()
    out.append(f"Churned customers pay {ch:.2f} per month on average vs {st:.2f} for retained customers.")
    if k["monthly_revenue"]:
        out.append(f"Monthly recurring revenue tied to churned customers: {k['monthly_revenue_at_risk']:,.0f} "
                   f"({k['monthly_revenue_at_risk'] / k['monthly_revenue']:.1%} of total).")
    return out


def run_sql(name: str, db_path=None) -> pd.DataFrame:
    return database.run_query(SQL_QUERIES[name], db_path=db_path)
