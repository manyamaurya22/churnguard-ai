"""Train, tune, compare and save models.  Run:  python -m src.train_models [--no-tune] [--keras]

Model selection rule (stated up-front, no peeking at the test set):
  the deployable model is the classical-ML pipeline with the highest mean
  cross-validated ROC-AUC on the TRAINING split. ROC-AUC is threshold-free and
  suits imbalanced data; the decision threshold is tuned separately by the
  business (cost of a missed churner vs. cost of a retention offer).
The held-out test set is used only for reporting, for every model equally.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import platform
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix,
                             f1_score, precision_recall_curve, precision_score,
                             recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

import config
from src.data_loader import clean_dataset, load_raw
from src.preprocessing import build_preprocessor, class_balance, split_data

log = logging.getLogger(__name__)

try:
    from xgboost import XGBClassifier
    XGB_AVAILABLE = True
except Exception:  # pragma: no cover
    XGB_AVAILABLE = False


def tensorflow_available() -> bool:
    try:
        import tensorflow  # noqa: F401
        return True
    except Exception:
        return False


TF_INSTALL_HELP = ("TensorFlow is not installed. Optional install:  pip install tensorflow  "
                   "(Python 3.9-3.12). The rest of the app works without it.")


def evaluate(y_true, proba, threshold: float = config.DEFAULT_THRESHOLD) -> dict:
    y_true = np.asarray(y_true)
    proba = np.asarray(proba)
    pred = (proba >= threshold).astype(int)
    return {
        "accuracy": accuracy_score(y_true, pred),
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred, zero_division=0),
        "f1": f1_score(y_true, pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, proba),
        "pr_auc": average_precision_score(y_true, proba),
        "confusion_matrix": confusion_matrix(y_true, pred, labels=[0, 1]).tolist(),
        "threshold": threshold,
    }


def _candidates(use_class_weight: bool, y_train) -> dict:
    """name -> (estimator, param distributions for RandomizedSearchCV)."""
    seed = config.RANDOM_SEED
    cw = "balanced" if use_class_weight else None
    c = {
        "Logistic Regression": (
            LogisticRegression(max_iter=2000, class_weight=cw, random_state=seed),
            {"clf__C": [0.01, 0.1, 1, 10, 100]}),
        "Decision Tree": (
            DecisionTreeClassifier(class_weight=cw, random_state=seed),
            {"clf__max_depth": [3, 4, 5, 6, 8], "clf__min_samples_leaf": [5, 10, 20, 50]}),
        "Random Forest": (
            RandomForestClassifier(class_weight=cw, random_state=seed, n_jobs=-1),
            {"clf__n_estimators": [200, 300], "clf__max_depth": [6, 8, 12, None],
             "clf__min_samples_leaf": [2, 5, 10]}),
    }
    if XGB_AVAILABLE:
        spw = (float((np.asarray(y_train) == 0).sum()) / max((np.asarray(y_train) == 1).sum(), 1)
               if use_class_weight else 1.0)
        c["XGBoost"] = (
            XGBClassifier(eval_metric="logloss", random_state=seed, n_jobs=1,
                          scale_pos_weight=spw),
            {"clf__n_estimators": [100, 200, 300], "clf__max_depth": [2, 3, 4],
             "clf__learning_rate": [0.03, 0.05, 0.1], "clf__subsample": [0.8, 1.0]})
    return c


def train_keras(X_train, y_train, X_test, y_test, use_class_weight: bool = True) -> dict:
    """Optional neural network on the same split. Its own preprocessor is fit on train only."""
    if not tensorflow_available():
        raise RuntimeError(TF_INSTALL_HELP)
    import tensorflow as tf
    from tensorflow import keras
    from sklearn.utils.class_weight import compute_class_weight

    tf.keras.utils.set_random_seed(config.RANDOM_SEED)
    prep = build_preprocessor()
    Xt = prep.fit_transform(X_train)
    Xe = prep.transform(X_test)
    reg = keras.regularizers.l2(1e-3)
    model = keras.Sequential([
        keras.layers.Input(shape=(Xt.shape[1],)),
        keras.layers.Dense(64, activation="relu", kernel_regularizer=reg),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(32, activation="relu", kernel_regularizer=reg),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(1, activation="sigmoid"),
    ])
    model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="binary_crossentropy",
                  metrics=[keras.metrics.AUC(name="auc")])
    cw = None
    if use_class_weight:
        w = compute_class_weight("balanced", classes=np.array([0, 1]), y=np.asarray(y_train))
        cw = {0: float(w[0]), 1: float(w[1])}
    stop = keras.callbacks.EarlyStopping(monitor="val_auc", mode="max", patience=10,
                                         restore_best_weights=True)
    hist = model.fit(Xt, np.asarray(y_train), validation_split=0.2, epochs=150, batch_size=64,
                     class_weight=cw, callbacks=[stop], verbose=0)
    proba = model.predict(Xe, verbose=0).ravel()
    res = evaluate(y_test, proba)
    res["epochs_run"] = len(hist.history["loss"])
    res["best_val_auc"] = float(max(hist.history["val_auc"]))
    res["proba"] = proba
    return res


def _sha256(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def train_all(df: pd.DataFrame, use_class_weight: bool = True, tune: bool = True,
              n_iter: int = 8, include_keras: bool = False, save: bool = True,
              progress=None) -> dict:
    """Full training run. `df` must be the cleaned dataset. Returns a results dict."""
    say = progress or (lambda msg: log.info(msg))
    X_train, X_test, y_train, y_test = split_data(df)
    cv = StratifiedKFold(config.CV_FOLDS, shuffle=True, random_state=config.RANDOM_SEED)
    fitted, rows, test_proba, curves, details = {}, [], {}, {}, {}

    for name, (est, grid) in _candidates(use_class_weight, y_train).items():
        say(f"Training {name} ...")
        pipe = Pipeline([("prep", build_preprocessor()), ("clf", est)])
        if tune:
            search = RandomizedSearchCV(pipe, grid, n_iter=min(n_iter, int(np.prod([len(v) for v in grid.values()]))),
                                        scoring="roc_auc", cv=cv, random_state=config.RANDOM_SEED, n_jobs=1)
            search.fit(X_train, y_train)
            pipe, cv_auc = search.best_estimator_, float(search.best_score_)
            details[name] = {k.replace("clf__", ""): v for k, v in search.best_params_.items()}
        else:
            cv_auc = float(cross_val_score(pipe, X_train, y_train, cv=cv, scoring="roc_auc").mean())
            pipe.fit(X_train, y_train)
            details[name] = {}
        proba = pipe.predict_proba(X_test)[:, 1]
        m = evaluate(y_test, proba)
        fpr, tpr, _ = roc_curve(y_test, proba)
        pr, rc, _ = precision_recall_curve(y_test, proba)
        curves[name] = {"fpr": fpr.tolist(), "tpr": tpr.tolist(),
                        "precision": pr.tolist(), "recall": rc.tolist()}
        fitted[name], test_proba[name] = pipe, proba
        rows.append({"model": name, "type": "classical ML", "cv_roc_auc": cv_auc, **m})

    keras_msg = None
    if include_keras:
        try:
            say("Training Keras neural network ...")
            k = train_keras(X_train, y_train, X_test, y_test, use_class_weight)
            proba = k.pop("proba")
            fpr, tpr, _ = roc_curve(y_test, proba)
            pr, rc, _ = precision_recall_curve(y_test, proba)
            curves["Keras NN"] = {"fpr": fpr.tolist(), "tpr": tpr.tolist(),
                                  "precision": pr.tolist(), "recall": rc.tolist()}
            test_proba["Keras NN"] = proba
            rows.append({"model": "Keras NN", "type": "deep learning", "cv_roc_auc": np.nan,
                         **{kk: v for kk, v in k.items() if kk in
                            ("accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc",
                             "confusion_matrix", "threshold")}})
            details["Keras NN"] = {"epochs_run": k["epochs_run"], "best_val_auc": k["best_val_auc"]}
        except Exception as exc:
            keras_msg = str(exc)
            log.warning("Keras skipped: %s", exc)

    board = pd.DataFrame(rows)
    classical = board[board["type"] == "classical ML"].sort_values("cv_roc_auc", ascending=False)
    best_name = classical.iloc[0]["model"]
    board = board.sort_values("roc_auc", ascending=False).reset_index(drop=True)

    result = {
        "leaderboard": board, "best_model": best_name, "keras_message": keras_msg,
        "details": details, "curves": curves, "y_test": np.asarray(y_test).tolist(),
        "test_proba": {k: np.asarray(v).tolist() for k, v in test_proba.items()},
        "class_balance_train": class_balance(y_train), "class_balance_test": class_balance(y_test),
        "used_class_weight": use_class_weight, "tuned": tune,
    }
    if save:
        config.MODELS_DIR.mkdir(exist_ok=True)
        path = config.MODELS_DIR / "best_pipeline.joblib"
        joblib.dump(fitted[best_name], path)
        (config.MODELS_DIR / "best_pipeline.joblib.sha256").write_text(_sha256(path))
        meta = {
            "best_model": best_name, "selection_metric": "mean CV ROC-AUC on training split",
            "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "seed": config.RANDOM_SEED, "n_train": len(X_train), "n_test": len(X_test),
            "features": config.FEATURE_COLUMNS, "best_params": details[best_name],
            "versions": {"python": platform.python_version(), "sklearn": sklearn.__version__,
                         "pandas": pd.__version__, "numpy": np.__version__},
        }
        (config.MODELS_DIR / "metadata.json").write_text(json.dumps(meta, indent=2, default=str))
        evaluation = {k: v for k, v in result.items() if k != "leaderboard"}
        evaluation["leaderboard"] = board.to_dict(orient="records")
        (config.MODELS_DIR / "evaluation.json").write_text(json.dumps(evaluation, default=str))
        say(f"Saved best model: {best_name}")
    return result


def load_evaluation() -> dict | None:
    p = config.MODELS_DIR / "evaluation.json"
    if not p.exists():
        return None
    ev = json.loads(p.read_text())
    ev["leaderboard"] = pd.DataFrame(ev["leaderboard"])
    return ev


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description="Train ChurnGuard models")
    ap.add_argument("--no-tune", action="store_true", help="skip hyperparameter search (faster)")
    ap.add_argument("--no-class-weight", action="store_true")
    ap.add_argument("--keras", action="store_true", help="also train the optional neural network")
    args = ap.parse_args()
    df = clean_dataset(load_raw())
    res = train_all(df, use_class_weight=not args.no_class_weight, tune=not args.no_tune,
                    include_keras=args.keras)
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(res["leaderboard"].drop(columns=["confusion_matrix"]).round(4).to_string(index=False))
    print("Best (by CV ROC-AUC):", res["best_model"])


if __name__ == "__main__":
    main()
