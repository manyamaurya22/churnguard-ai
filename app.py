"""ChurnGuard AI - Streamlit UI. All logic lives in src/; this file only draws pages."""
from __future__ import annotations

import logging
import os

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import config
from src import analytics, database, explainability as xai, predict, recommendations as rec
from src import train_models as tm
from src.data_loader import DataLoadError, MISSING_FILE_HELP, clean_dataset, load_raw, save_uploaded
from src.preprocessing import split_data

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
st.set_page_config(page_title="ChurnGuard AI", page_icon="🛡️", layout="wide")

PAGES = ["Executive Overview", "Exploratory Data Analysis", "Data Quality", "Model Training & Comparison",
         "Single Customer Prediction", "Batch CSV Prediction", "Explainable AI",
         "Business Recommendations", "Prediction History", "Project Documentation"]
RISK_COLORS = {"Low": "#2ecc71", "Medium": "#f5a623", "High": "#e5484d"}


# ------------------------------------------------------------------ theme
def inject_css(dark: bool) -> None:
    bg, card, text, sub, border = (("#0e1117", "#161b26", "#e8ecf3", "#9aa4b5", "#263044") if dark
                                   else ("#f6f8fc", "#ffffff", "#1b2433", "#5b6678", "#dfe5ef"))
    st.markdown(f"""<style>
    .stApp {{ background:{bg}; color:{text}; }}
    [data-testid="stSidebar"] {{ background:{card}; border-right:1px solid {border}; }}
    h1,h2,h3,p,label,span,div {{ font-family:'Inter','Segoe UI',system-ui,sans-serif; }}
    .kpi {{ background:{card}; border:1px solid {border}; border-radius:14px; padding:16px 18px;
            animation:fade .5s ease both; transition:transform .2s, box-shadow .2s; }}
    .kpi:hover {{ transform:translateY(-3px); box-shadow:0 8px 24px rgba(0,0,0,.18); }}
    .kpi .l {{ color:{sub}; font-size:.78rem; text-transform:uppercase; letter-spacing:.06em; }}
    .kpi .v {{ font-size:1.7rem; font-weight:700; margin-top:2px; color:{text}; }}
    .kpi .s {{ color:{sub}; font-size:.78rem; }}
    .pill {{ display:inline-block; padding:4px 14px; border-radius:999px; color:#fff; font-weight:600; }}
    .note {{ background:{card}; border-left:4px solid #4c8bf5; padding:10px 14px; border-radius:8px; }}
    @keyframes fade {{ from {{opacity:0; transform:translateY(8px);}} to {{opacity:1; transform:none;}} }}
    </style>""", unsafe_allow_html=True)


def kpi_row(items: list[tuple]) -> None:
    cols = st.columns(len(items))
    for col, (label, value, sub) in zip(cols, items):
        col.markdown(f'<div class="kpi"><div class="l">{label}</div><div class="v">{value}</div>'
                     f'<div class="s">{sub}</div></div>', unsafe_allow_html=True)


def style(fig, height=380):
    fig.update_layout(template="plotly_dark" if st.session_state.get("dark", True) else "plotly_white",
                      height=height, margin=dict(l=10, r=10, t=50, b=10),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    return fig


# ------------------------------------------------------------------ data / model loading
@st.cache_data(show_spinner="Loading dataset...")
def get_data(_stamp: float):
    raw = load_raw()
    df = clean_dataset(raw)
    database.load_customers(df)
    return raw, df


@st.cache_resource(show_spinner=False)
def get_model(_stamp: float):
    return predict.load_artifacts()


def model_stamp() -> float:
    p = config.MODELS_DIR / "best_pipeline.joblib"
    return p.stat().st_mtime if p.exists() else 0.0


def require_data():
    stamp = config.DATA_FILE.stat().st_mtime if config.DATA_FILE.exists() else 0.0
    try:
        return get_data(stamp)
    except DataLoadError as exc:
        st.error("Dataset problem")
        st.code(str(exc))
        st.info("Use the **Upload dataset** box in the sidebar, or place the file in the `data/` folder.")
        st.stop()


def require_model():
    try:
        return get_model(model_stamp())
    except predict.ModelNotTrainedError as exc:
        st.warning(str(exc))
        st.stop()


def api_key() -> str | None:
    try:
        if "ANTHROPIC_API_KEY" in st.secrets:
            return st.secrets["ANTHROPIC_API_KEY"]
    except Exception:
        pass
    return os.environ.get("ANTHROPIC_API_KEY") or None


# ------------------------------------------------------------------ pages
def page_overview():
    raw, df = require_data()
    d = analytics.add_segments(df)
    k = analytics.kpis(d)
    st.title("🛡️ ChurnGuard AI")
    st.caption("Customer churn prediction & business intelligence - every number below is computed from your dataset.")
    kpi_row([("Customers", f"{k['customers']:,}", "in dataset"),
             ("Churn rate", f"{k['churn_rate']:.1%}", f"{k['churned']:,} churned"),
             ("Retention", f"{k['retention_rate']:.1%}", "customers kept"),
             ("Avg monthly charge", f"{k['avg_monthly_charges']:.2f}", "per customer"),
             ("Monthly revenue at risk", f"{k['monthly_revenue_at_risk']:,.0f}", "from churned customers")])
    st.write("")
    c1, c2 = st.columns(2)
    pie = px.pie(d, names="Churn", hole=0.55, title="Churn split",
                 color="Churn", color_discrete_map={"Yes": "#e5484d", "No": "#4c8bf5"})
    c1.plotly_chart(style(pie), use_container_width=True)
    seg = analytics.segment_table(d, "Contract")
    bar = px.bar(seg, x="Contract", y="churn_rate", text=seg["churn_rate"].map("{:.1%}".format),
                 title="Churn rate by contract")
    bar.update_yaxes(tickformat=".0%")
    c2.plotly_chart(style(bar), use_container_width=True)
    st.subheader("Key insights (auto-generated from the data)")
    for line in analytics.auto_insights(df):
        st.markdown(f"- {line}")
    ev = tm.load_evaluation()
    if ev:
        b = ev["leaderboard"].set_index("model").loc[ev["best_model"]]
        st.markdown(f'<div class="note">Deployed model: <b>{ev["best_model"]}</b> - test ROC-AUC '
                    f'{b["roc_auc"]:.3f}, recall {b["recall"]:.3f} at threshold {b["threshold"]}.</div>',
                    unsafe_allow_html=True)
    else:
        st.info("No model trained yet - go to **Model Training & Comparison**.")


def page_eda():
    _, df = require_data()
    d = analytics.add_segments(df)
    st.title("Exploratory Data Analysis")
    with st.expander("Filters", expanded=True):
        c = st.columns(4)
        contract = c[0].multiselect("Contract", sorted(d["Contract"].unique()))
        net = c[1].multiselect("Internet service", sorted(d["InternetService"].unique()))
        pay = c[2].multiselect("Payment method", sorted(d["PaymentMethod"].unique()))
        lo, hi = int(d["tenure"].min()), int(d["tenure"].max())
        ten = c[3].slider("Tenure (months)", lo, hi, (lo, hi))
    f = d[d["tenure"].between(*ten)]
    for col, sel in [("Contract", contract), ("InternetService", net), ("PaymentMethod", pay)]:
        if sel:
            f = f[f[col].isin(sel)]
    if f.empty:
        st.warning("No customers match these filters.")
        return
    k = analytics.kpis(f)
    kpi_row([("Customers", f"{k['customers']:,}", "after filters"), ("Churn rate", f"{k['churn_rate']:.1%}", ""),
             ("Avg tenure", f"{k['avg_tenure']:.1f}", "months"),
             ("Avg monthly", f"{k['avg_monthly_charges']:.2f}", ""),
             ("Revenue at risk", f"{k['monthly_revenue_at_risk']:,.0f}", "monthly")])
    st.write("")
    seg_col = st.selectbox("Segment by", ["Contract", "tenure_group", "PaymentMethod", "InternetService",
                                          "gender", "SeniorCitizen", "PaperlessBilling", "TechSupport"])
    t = analytics.segment_table(f, seg_col)
    c1, c2 = st.columns(2)
    fig = px.bar(t, x=seg_col, y="churn_rate", hover_data=["customers", "avg_monthly"], title=f"Churn rate by {seg_col}")
    fig.update_yaxes(tickformat=".0%")
    c1.plotly_chart(style(fig), use_container_width=True)
    c2.plotly_chart(style(px.histogram(f, x="tenure", color="Churn", barmode="overlay", opacity=0.65,
                                       title="Tenure distribution", color_discrete_map={"Yes": "#e5484d", "No": "#4c8bf5"})),
                    use_container_width=True)
    c3, c4 = st.columns(2)
    c3.plotly_chart(style(px.box(f, x="Churn", y="MonthlyCharges", color="Churn", title="Monthly charges vs churn",
                                 color_discrete_map={"Yes": "#e5484d", "No": "#4c8bf5"})), use_container_width=True)
    corr = analytics.correlation_matrix(f)
    c4.plotly_chart(style(px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
                                    title="Correlation (numeric features + churn)")), use_container_width=True)
    st.subheader("Segment table")
    st.dataframe(t.style.format({"churn_rate": "{:.1%}", "avg_monthly": "{:.2f}", "monthly_revenue": "{:,.0f}"}),
                 use_container_width=True)
    st.subheader("Feature distribution")
    feat = st.selectbox("Feature", config.FEATURE_COLUMNS)
    if feat in config.NUMERIC_FEATURES:
        st.plotly_chart(style(px.histogram(f, x=feat, color="Churn", marginal="box", barmode="overlay", opacity=0.65)),
                        use_container_width=True)
    else:
        st.plotly_chart(style(px.histogram(f, x=feat, color="Churn", barmode="group")), use_container_width=True)
    st.subheader("SQL explorer (SQLite)")
    q = st.selectbox("Query", list(analytics.SQL_QUERIES))
    st.code(analytics.SQL_QUERIES[q], language="sql")
    st.dataframe(analytics.run_sql(q), use_container_width=True)


def page_quality():
    raw, df = require_data()
    q = analytics.data_quality_report(raw)
    st.title("Data Quality")
    kpi_row([("Rows", f"{q['rows']:,}", ""), ("Columns", q["columns"], ""),
             ("Exact duplicate rows", q["exact_duplicate_rows"], "removed on cleaning"),
             ("Duplicate customer IDs", q["duplicate_customer_ids"], ""),
             ("Missing / blank cells", q["total_missing_cells"], "")])
    st.subheader("Column report")
    st.dataframe(q["table"], use_container_width=True)
    bad = q["unparseable_total_charges"]
    if len(bad):
        st.subheader(f"TotalCharges not numeric ({len(bad)} rows)")
        st.caption("Blank TotalCharges are kept and imputed with the training median inside the model pipeline "
                   "(fit on training data only, so no leakage).")
        st.dataframe(bad[["customerID", "tenure", "MonthlyCharges", "TotalCharges", "Churn"]], use_container_width=True)
    cb = analytics.kpis(df)
    st.subheader("Class balance")
    st.write(f"Churned: **{cb['churned']:,}** ({cb['churn_rate']:.1%}) vs retained **{cb['customers'] - cb['churned']:,}**. "
             "The classes are imbalanced, so accuracy alone is misleading; use recall, F1 and ROC-AUC, and consider class weighting.")


def page_training():
    _, df = require_data()
    st.title("Model Training & Comparison")
    with st.expander("Training options", expanded=True):
        c = st.columns(4)
        cw = c[0].checkbox("Class weighting", True, help="Counter the churn/no-churn imbalance")
        tune = c[1].checkbox("Hyperparameter search", True, help="RandomizedSearchCV, scored by ROC-AUC")
        n_iter = c[2].slider("Search iterations", 2, 20, 8)
        tf_ok = tm.tensorflow_available()
        keras_on = c[3].checkbox("Keras neural network", False, disabled=not tf_ok)
        if not tf_ok:
            st.caption("🧠 " + tm.TF_INSTALL_HELP)
        if not tm.XGB_AVAILABLE:
            st.caption("XGBoost is not installed (`pip install xgboost`) - skipped.")
    if st.button("🚀 Train / retrain models", type="primary"):
        box = st.status("Training...", expanded=True)
        try:
            tm.train_all(df, use_class_weight=cw, tune=tune, n_iter=n_iter, include_keras=keras_on,
                         progress=lambda m: box.write(m))
            box.update(label="Training complete", state="complete")
            get_model.clear()
        except Exception as exc:
            box.update(label="Training failed", state="error")
            st.exception(exc)
    ev = tm.load_evaluation()
    if not ev:
        st.info("Train the models to see the leaderboard.")
        return
    if ev.get("keras_message"):
        st.warning(ev["keras_message"])
    board = ev["leaderboard"]
    st.subheader("Leaderboard (held-out test set)")
    show = board.drop(columns=["confusion_matrix"])
    st.dataframe(show.style.format({c: "{:.3f}" for c in show.columns if show[c].dtype.kind == "f"}),
                 use_container_width=True)
    st.markdown(f'<div class="note"><b>Selection rule:</b> deployed model = highest mean {config.CV_FOLDS}-fold CV ROC-AUC '
                f'on the training split among classical models → <b>{ev["best_model"]}</b>. The test set is only used for reporting. '
                f'Class weighting: {ev["used_class_weight"]}, tuning: {ev["tuned"]}, seed: {config.RANDOM_SEED}.</div>',
                unsafe_allow_html=True)
    st.write("")
    names = list(ev["curves"])
    c1, c2 = st.columns(2)
    roc = go.Figure([go.Scatter(x=ev["curves"][n]["fpr"], y=ev["curves"][n]["tpr"], name=n, mode="lines") for n in names])
    roc.add_shape(type="line", x0=0, y0=0, x1=1, y1=1, line=dict(dash="dash", color="gray"))
    roc.update_layout(title="ROC curves", xaxis_title="False positive rate", yaxis_title="True positive rate")
    c1.plotly_chart(style(roc), use_container_width=True)
    pr = go.Figure([go.Scatter(x=ev["curves"][n]["recall"], y=ev["curves"][n]["precision"], name=n, mode="lines") for n in names])
    pr.update_layout(title="Precision-Recall curves", xaxis_title="Recall", yaxis_title="Precision")
    c2.plotly_chart(style(pr), use_container_width=True)
    st.subheader("Threshold explorer")
    pick = st.selectbox("Model", names, index=names.index(ev["best_model"]))
    thr = st.slider("Classification threshold", 0.05, 0.95, config.DEFAULT_THRESHOLD, 0.05)
    m = tm.evaluate(ev["y_test"], ev["test_proba"][pick], thr)
    kpi_row([("Accuracy", f"{m['accuracy']:.3f}", ""), ("Precision", f"{m['precision']:.3f}", ""),
             ("Recall", f"{m['recall']:.3f}", ""), ("F1", f"{m['f1']:.3f}", ""), ("ROC-AUC", f"{m['roc_auc']:.3f}", "threshold-free")])
    cm = np.array(m["confusion_matrix"])
    fig = px.imshow(cm, text_auto=True, x=["Pred: stay", "Pred: churn"], y=["Actual: stay", "Actual: churn"],
                    color_continuous_scale="Blues", title=f"Confusion matrix - {pick} @ {thr:.2f}")
    st.plotly_chart(style(fig, 340), use_container_width=True)
    st.subheader("Does deep learning help on tabular data?")
    st.write("Usually only a little. For small/medium tabular data like this, tree ensembles and regularised linear models "
             "are strong, faster, easier to explain and need less tuning. Neural nets shine with images, text, audio or very "
             "large datasets with complex interactions. Compare the Keras row (if trained) with the classical rows above - "
             "same test data, same metrics - and let the numbers decide.")


def customer_form() -> dict:
    c = st.columns(3)
    rec_ = {}
    rec_["gender"] = c[0].selectbox("Gender", config.CATEGORY_VALUES["gender"])
    rec_["SeniorCitizen"] = int(c[1].selectbox("Senior citizen", [0, 1], format_func=lambda v: "Yes" if v else "No"))
    rec_["Partner"] = c[2].selectbox("Partner", config.CATEGORY_VALUES["Partner"])
    rec_["Dependents"] = c[0].selectbox("Dependents", config.CATEGORY_VALUES["Dependents"], index=1)
    rec_["tenure"] = c[1].number_input("Tenure (months)", 0, 120, 12)
    rec_["PhoneService"] = c[2].selectbox("Phone service", config.CATEGORY_VALUES["PhoneService"])
    for i, col in enumerate(["MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
                             "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"]):
        rec_[col] = c[i % 3].selectbox(col, config.CATEGORY_VALUES[col], index=1 if col != "InternetService" else 1)
    rec_["Contract"] = c[2].selectbox("Contract", config.CATEGORY_VALUES["Contract"])
    rec_["PaperlessBilling"] = c[0].selectbox("Paperless billing", config.CATEGORY_VALUES["PaperlessBilling"])
    rec_["PaymentMethod"] = c[1].selectbox("Payment method", config.CATEGORY_VALUES["PaymentMethod"])
    rec_["MonthlyCharges"] = c[2].number_input("Monthly charges", 0.0, 500.0, 70.0, 0.05)
    rec_["TotalCharges"] = c[0].number_input("Total charges", 0.0, 100000.0,
                                             float(round(rec_["tenure"] * rec_["MonthlyCharges"], 2)), 0.05)
    return rec_


def page_single():
    _, df = require_data()
    pipe, meta = require_model()
    st.title("Single Customer Prediction")
    thr = st.slider("Risk threshold (High risk if probability ≥ threshold)", 0.05, 0.95, config.DEFAULT_THRESHOLD, 0.05)
    with st.form("single_form"):
        record = customer_form()
        go_btn = st.form_submit_button("Predict churn", type="primary")
    if go_btn:
        try:
            res = predict.predict_single(record, pipe, thr)
        except predict.CSVValidationError as exc:
            st.error(f"Invalid input: {exc}")
            return
        X = df[config.FEATURE_COLUMNS]
        drivers, method = xai.explain_customer(pipe, pd.DataFrame([record])[config.FEATURE_COLUMNS], X)
        st.session_state["single_result"] = dict(record=record, res=res, drivers=drivers, method=method, thr=thr)
        database.log_prediction("single", res["probability"], thr, res["risk"], meta["best_model"], inputs=record)
    s = st.session_state.get("single_result")
    if not s:
        return
    res, drivers = s["res"], s["drivers"]
    c1, c2 = st.columns([1, 2])
    gauge = go.Figure(go.Indicator(mode="gauge+number", value=res["probability"] * 100, number={"suffix": "%"},
                                   title={"text": "Churn probability"},
                                   gauge={"axis": {"range": [0, 100]}, "bar": {"color": RISK_COLORS[res["risk"]]},
                                          "threshold": {"line": {"color": "white", "width": 3}, "value": s["thr"] * 100}}))
    c1.plotly_chart(style(gauge, 300), use_container_width=True)
    c1.markdown(f'Risk category: <span class="pill" style="background:{RISK_COLORS[res["risk"]]}">{res["risk"]}</span>',
                unsafe_allow_html=True)
    top = drivers.head(10).iloc[::-1]
    bar = go.Figure(go.Bar(x=top["effect"], y=[f"{f} = {v}" for f, v in zip(top["feature"], top["value"])], orientation="h",
                           marker_color=["#e5484d" if e > 0 else "#2ecc71" for e in top["effect"]]))
    bar.update_layout(title="What drives this prediction (red ↑ risk, green ↓ risk)")
    c2.plotly_chart(style(bar, 380), use_container_width=True)
    st.caption(f"Explanation method: {s['method']}")
    advice = rec.rule_based(s["record"], res["probability"], res["risk"], drivers)
    st.subheader("Suggested retention actions")
    st.write(advice["summary"])
    for r_, a_ in zip(advice["reasons"], advice["actions"][-len(advice["reasons"]):] if advice["reasons"] else []):
        st.markdown(f"- **{r_}** → {a_}")
    if not advice["reasons"]:
        st.markdown("- " + advice["actions"][-1])
    if advice["actions"] and advice["reasons"] and len(advice["actions"]) > len(advice["reasons"]):
        st.info(advice["actions"][0])
    if st.button("✨ Generate AI narrative (Claude, optional)"):
        try:
            payload = rec.build_safe_payload(s["record"], res["probability"], res["risk"], drivers)
            with st.spinner("Asking Claude (anonymised data only)..."):
                text, err = rec.llm_recommendation(payload, api_key())
            st.write(text) if text else st.warning(err)
        except ValueError as exc:
            st.error(str(exc))


def page_batch():
    pipe, meta = require_model()
    st.title("Batch CSV Prediction")
    st.caption("Required columns: " + ", ".join(config.FEATURE_COLUMNS) + " (customerID and other columns are kept as-is).")
    st.download_button("Download blank template (headers only)",
                       pd.DataFrame(columns=[config.ID_COL] + config.FEATURE_COLUMNS).to_csv(index=False),
                       "churnguard_template.csv", "text/csv")
    thr = st.slider("Risk threshold", 0.05, 0.95, config.DEFAULT_THRESHOLD, 0.05, key="bthr")
    up = st.file_uploader("Upload CSV", type=["csv"])
    if not up:
        return
    try:
        df = pd.read_csv(up)
        out = predict.predict_dataframe(df, pipe, thr)
    except (predict.CSVValidationError, pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeDecodeError) as exc:
        st.error(f"Could not process this file: {exc}")
        return
    ok = out["validation_status"] == "OK"
    valid = out[ok]
    kpi_row([("Rows uploaded", f"{len(out):,}", ""), ("Predicted", f"{ok.sum():,}", "valid rows"),
             ("Invalid rows", f"{(~ok).sum():,}", "kept, flagged below"),
             ("High risk", f"{(valid['risk_category'] == 'High').sum():,}", f"{(valid['risk_category'] == 'High').mean():.1%}" if len(valid) else ""),
             ("Avg probability", f"{valid['churn_probability'].mean():.1%}" if len(valid) else "n/a", "")])
    if (~ok).any():
        st.warning(f"{(~ok).sum()} row(s) failed validation and were NOT predicted (they remain in the output).")
        st.dataframe(out.loc[~ok, ["validation_status"]], use_container_width=True)
    if len(valid):
        dist = valid["risk_category"].value_counts().reindex(["Low", "Medium", "High"]).fillna(0).reset_index()
        dist.columns = ["risk", "customers"]
        c1, c2 = st.columns(2)
        c1.plotly_chart(style(px.bar(dist, x="risk", y="customers", color="risk", color_discrete_map=RISK_COLORS,
                                     title="Risk distribution")), use_container_width=True)
        c2.plotly_chart(style(px.histogram(valid, x="churn_probability", nbins=30, title="Probability distribution")),
                        use_container_width=True)
    st.dataframe(out, use_container_width=True)
    st.download_button("⬇️ Download results CSV", out.to_csv(index=False), "churnguard_predictions.csv", "text/csv")
    if len(valid) and st.checkbox("Save these predictions to history"):
        ref = valid[config.ID_COL] if config.ID_COL in valid else pd.Series([""] * len(valid), index=valid.index)
        database.log_predictions_bulk([("batch", str(r), p, thr, c, meta["best_model"]) for r, p, c in
                                       zip(ref, valid["churn_probability"], valid["risk_category"])])
        st.success("Saved.")


def page_xai():
    _, df = require_data()
    pipe, meta = require_model()
    st.title("Explainable AI")
    X_tr, X_te, y_tr, y_te = split_data(df)
    imp, method = xai.global_importance(pipe, X_te, y_te)
    if method.startswith("FALLBACK"):
        st.warning(method + " - install `shap` for SHAP explanations.")
    else:
        st.caption(f"Method: {method}, computed on a sample of the held-out test set.")
    top = imp.head(15).iloc[::-1]
    st.plotly_chart(style(px.bar(top, x="importance", y="feature", orientation="h", title="Global feature importance"), 480),
                    use_container_width=True)
    summ = xai.shap_values_by_feature(pipe, X_te)
    if summ:
        Xs, g, unit = summ
        order = g.abs().mean().sort_values(ascending=False).head(10).index
        rows = [pd.DataFrame({"feature": f, "shap": g[f], "customer": Xs.index}) for f in order]
        st.plotly_chart(style(px.strip(pd.concat(rows), x="shap", y="feature", title=f"SHAP value per customer ({unit}); right = raises churn risk"), 460),
                        use_container_width=True)
    st.subheader("Explain one customer from the test set")
    idx = st.selectbox("Customer (row id)", list(X_te.index[:300]))
    row = X_te.loc[[idx]]
    p = float(pipe.predict_proba(row)[:, 1][0])
    drivers, m2 = xai.explain_customer(pipe, row, X_tr)
    st.write(f"Predicted churn probability: **{p:.1%}**  |  actual outcome: **{'Churned' if y_te.loc[idx] else 'Stayed'}**")
    show = drivers.head(10)
    st.plotly_chart(style(px.bar(show.iloc[::-1], x="effect", y="feature", orientation="h", color=(show.iloc[::-1]["effect"] > 0),
                                 color_discrete_map={True: "#e5484d", False: "#2ecc71"}, title="Factors increasing (red) / decreasing (green) risk")),
                    use_container_width=True)
    st.caption(f"Method: {m2}")
    st.dataframe(show, use_container_width=True)


def page_recs():
    _, df = require_data()
    pipe, meta = require_model()
    st.title("Business Recommendations")
    thr = st.slider("High-risk threshold", 0.05, 0.95, config.DEFAULT_THRESHOLD, 0.05, key="rthr")
    _, X_te, _, _ = split_data(df)
    test = df.loc[X_te.index]
    pred = predict.predict_dataframe(test[config.FEATURE_COLUMNS], pipe, thr)
    pred["MonthlyCharges"] = test["MonthlyCharges"].values
    seg = rec.priority_segments(pred)
    st.caption("Computed on the held-out test split only, so the model has not seen these customers during training.")
    if seg.empty:
        st.info("Not enough customers per segment.")
        return
    st.dataframe(seg.style.format({"high_risk_share": "{:.1%}", "avg_probability": "{:.1%}",
                                   "monthly_revenue_high_risk": "{:,.0f}"}), use_container_width=True)
    top = seg.iloc[0]
    st.success(f"Largest revenue exposure: **{top['segment']}** - {int(top['high_risk'])} high-risk customers, "
               f"{top['monthly_revenue_high_risk']:,.0f} monthly revenue.")
    st.subheader("Playbook (rule-based, auditable)")
    st.markdown("""
| Trigger | Action |
|---|---|
| Month-to-month contract | Discounted 12/24-month plan or loyalty credit |
| Tenure ≤ 12 months | Onboarding check-ins, first-year engagement |
| Fiber without tech support | Bundle tech support for 3-6 months |
| Electronic-check payer | Autopay incentive |
| Monthly charge ≥ 80 | Plan right-sizing / value bundle |
""")
    if st.button("✨ AI summary of top segments (Claude, optional)"):
        agg = {"segments": seg.head(5)[["segment", "customers", "high_risk", "high_risk_share"]].round(3).to_dict("records")}
        text, err = rec.llm_recommendation(agg, api_key())  # aggregates only, no customer-level data
        st.write(text) if text else st.warning(err)


def page_history():
    st.title("Prediction History")
    h = database.get_history()
    if h.empty:
        st.info("No predictions saved yet. Single predictions are saved automatically.")
        return
    c = st.columns(3)
    src = c[0].multiselect("Source", sorted(h["source"].unique()))
    risk = c[1].multiselect("Risk", ["Low", "Medium", "High"])
    if src:
        h = h[h["source"].isin(src)]
    if risk:
        h = h[h["risk_category"].isin(risk)]
    st.dataframe(h, use_container_width=True)
    st.download_button("Download history CSV", h.to_csv(index=False), "prediction_history.csv", "text/csv")
    if st.button("Delete all history"):
        database.clear_history()
        st.rerun()


def page_docs():
    st.title("Project Documentation")
    st.markdown("""
### What it does
Predicts telecom customer churn, explains *why*, and suggests retention actions.

### Architecture
`app.py` (UI) → `src/predict.py` (inference) → `models/best_pipeline.joblib` (preprocessing + model in ONE pipeline).
`src/train_models.py` trains/tunes/compares; `src/explainability.py` (SHAP); `src/recommendations.py` (rules + optional Claude);
`src/database.py` (SQLite); `src/analytics.py` (KPIs/SQL); `src/data_loader.py` (CSV checks).

### Why a single Pipeline?
Imputers, scalers and encoders are *fit on training data only* and saved with the model, so predictions use exactly the same
preprocessing (no leakage, no train/serve mismatch).

### Metric choice
Accuracy misleads on imbalanced data. Models are selected by cross-validated **ROC-AUC** on training data; the **threshold** is a
business knob (lower threshold = catch more churners, more false alarms).

### Privacy
customerID and any non-feature columns are never sent to the Claude API; only whitelisted features, probability and feature effects.
The API key is read from environment/Streamlit secrets. Model files are hash-checked before loading.

### Limitations
One historical dataset; correlation ≠ causation; no cost data for choosing the economically optimal threshold;
probabilities are not calibrated unless you add calibration; Keras is optional.

### Interview cheat-sheet
- *Why recall matters?* Missing a churner costs more than a wasted offer.
- *Why SHAP?* Consistent, additive per-customer attributions.
- *Why not accuracy?* ~73% of customers don't churn, so "always no" scores high.
- *Leakage?* Split first; all fitting inside the pipeline on train only.
""")


# ------------------------------------------------------------------ router
def main():
    with st.sidebar:
        st.markdown("## 🛡️ ChurnGuard AI")
        page = st.radio("Navigate", PAGES, label_visibility="collapsed")
        st.session_state["dark"] = st.toggle("Dark mode", value=st.session_state.get("dark", True))
        st.divider()
        up = st.file_uploader("Upload dataset (Telco CSV)", type=["csv"])
        if up is not None and st.button("Use this dataset"):
            try:
                save_uploaded(up.getvalue())
                get_data.clear()
                st.success("Dataset saved.")
            except DataLoadError as exc:
                st.error(str(exc))
        st.caption("Dataset: " + ("✅ found" if config.DATA_FILE.exists() else "❌ missing"))
        st.caption("Model: " + ("✅ trained" if model_stamp() else "❌ not trained"))
        st.caption("Claude API: " + ("🟢 key set" if api_key() else "⚪ offline mode (rule-based)"))
    inject_css(st.session_state["dark"])
    if not config.DATA_FILE.exists() and page in PAGES[:4] + PAGES[4:5] + ["Explainable AI", "Business Recommendations"]:
        st.error("Dataset not found")
        st.code(MISSING_FILE_HELP)
        if page != "Project Documentation":
            return
    {"Executive Overview": page_overview, "Exploratory Data Analysis": page_eda, "Data Quality": page_quality,
     "Model Training & Comparison": page_training, "Single Customer Prediction": page_single,
     "Batch CSV Prediction": page_batch, "Explainable AI": page_xai, "Business Recommendations": page_recs,
     "Prediction History": page_history, "Project Documentation": page_docs}[page]()


main()
