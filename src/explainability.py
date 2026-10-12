"""Explainable AI: SHAP (preferred) with real, clearly-labelled fallbacks. Nothing is fabricated."""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

import config

log = logging.getLogger(__name__)

try:
    import shap
    SHAP_AVAILABLE = True
except Exception:  # pragma: no cover
    SHAP_AVAILABLE = False


def _origin(encoded_name: str) -> str:
    """Map an encoded column (e.g. 'Contract_Two year') back to the original feature."""
    if encoded_name in config.NUMERIC_FEATURES:
        return encoded_name
    for col in config.CATEGORICAL_FEATURES:
        if encoded_name.startswith(col + "_"):
            return col
    return encoded_name


def _positive_class(sv) -> np.ndarray:
    if isinstance(sv, list):
        return np.asarray(sv[-1])
    sv = np.asarray(sv)
    return sv[:, :, -1] if sv.ndim == 3 else sv


def _explainer(pipeline, background_t):
    clf = pipeline.named_steps["clf"]
    name = type(clf).__name__
    if name in ("RandomForestClassifier", "DecisionTreeClassifier", "XGBClassifier"):
        return shap.TreeExplainer(clf), ("log-odds" if name == "XGBClassifier" else "probability")
    if name == "LogisticRegression":
        return shap.LinearExplainer(clf, background_t), "log-odds"
    raise ValueError(f"No SHAP explainer configured for {name}")


def _shap_by_feature(pipeline, X: pd.DataFrame, background: pd.DataFrame):
    prep = pipeline.named_steps["prep"]
    Xt, Bt = prep.transform(X), prep.transform(background)
    names = list(prep.get_feature_names_out())
    explainer, unit = _explainer(pipeline, Bt)
    sv = _positive_class(explainer.shap_values(Xt))
    df = pd.DataFrame(sv, columns=names)
    grouped = df.T.groupby(_origin).sum().T  # sum one-hot columns per original feature
    return grouped, unit


def global_importance(pipeline, X: pd.DataFrame, y=None, sample: int = 500,
                      seed: int = config.RANDOM_SEED):
    """Returns (DataFrame[feature, importance], method_label)."""
    Xs = X.sample(min(sample, len(X)), random_state=seed)
    if SHAP_AVAILABLE:
        try:
            g, unit = _shap_by_feature(pipeline, Xs, Xs)
            imp = g.abs().mean().sort_values(ascending=False)
            return (imp.rename("importance").rename_axis("feature").reset_index(),
                    f"SHAP (mean |SHAP value|, {unit} units)")
        except Exception as exc:
            log.warning("SHAP failed (%s); using permutation importance", exc)
    if y is None:
        raise RuntimeError("SHAP unavailable and no labels supplied for permutation importance.")
    ys = y.loc[Xs.index]
    r = permutation_importance(pipeline, Xs, ys, scoring="roc_auc", n_repeats=5,
                               random_state=seed, n_jobs=1)
    imp = pd.Series(r.importances_mean, index=Xs.columns).sort_values(ascending=False)
    return (imp.rename("importance").rename_axis("feature").reset_index(),
            "FALLBACK: permutation importance (drop in ROC-AUC when a feature is shuffled)")


def shap_values_by_feature(pipeline, X: pd.DataFrame, sample: int = 300, seed: int = config.RANDOM_SEED):
    """Per-row SHAP values by original feature for the beeswarm-style summary. None if SHAP unavailable."""
    if not SHAP_AVAILABLE:
        return None
    Xs = X.sample(min(sample, len(X)), random_state=seed)
    try:
        g, unit = _shap_by_feature(pipeline, Xs, Xs)
        g.index = Xs.index
        return Xs, g, unit
    except Exception as exc:
        log.warning("SHAP summary failed: %s", exc)
        return None


def explain_customer(pipeline, row: pd.DataFrame, background: pd.DataFrame):
    """Local explanation for one customer (1-row DataFrame).
    Returns (DataFrame[feature, value, effect], method_label); positive effect = raises churn risk."""
    values = row.iloc[0]
    if SHAP_AVAILABLE:
        try:
            bg = background.sample(min(200, len(background)), random_state=config.RANDOM_SEED)
            g, unit = _shap_by_feature(pipeline, row, bg)
            out = pd.DataFrame({"feature": g.columns, "effect": g.iloc[0].to_numpy()})
            out["value"] = [str(values[f]) for f in out["feature"]]
            return (out.reindex(out["effect"].abs().sort_values(ascending=False).index)
                    .reset_index(drop=True), f"SHAP ({unit} units)")
        except Exception as exc:
            log.warning("SHAP local explanation failed (%s); using occlusion fallback", exc)
    # Fallback: replace each feature by a typical value and measure the real change in probability.
    base = float(pipeline.predict_proba(row)[:, 1][0])
    effects = []
    for col in config.FEATURE_COLUMNS:
        typical = (background[col].median() if col in config.NUMERIC_FEATURES
                   else background[col].mode().iloc[0])
        alt = row.copy()
        alt[col] = typical
        effects.append((col, str(values[col]), base - float(pipeline.predict_proba(alt)[:, 1][0])))
    out = pd.DataFrame(effects, columns=["feature", "value", "effect"])
    return (out.reindex(out["effect"].abs().sort_values(ascending=False).index).reset_index(drop=True),
            "FALLBACK: occlusion (probability change vs. replacing the feature with a typical value)")
