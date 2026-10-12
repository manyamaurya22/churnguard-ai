# 🛡️ ChurnGuard AI — Intelligent Customer Churn Prediction & Business Intelligence Platform

An end-to-end data science project: predict which telecom customers are likely to churn, **explain why**, and suggest
**retention actions**. Everything shown in the app is computed from the real IBM Telco dataset — no hard-coded predictions,
no invented metrics.

> 📸 Screenshots: `docs/overview.png`, `docs/model-comparison.png`, `docs/single-prediction.png`, `docs/shap.png` *(add yours here)*

## Features
| Area | What you get |
|---|---|
| Analytics | data-quality report, duplicate detection, KPIs, tenure/contract/payment segmentation, correlations, interactive filters, SQLite + SQL explorer |
| ML | Logistic Regression, Decision Tree, Random Forest, XGBoost (if installed); stratified split, CV + RandomizedSearchCV, class weighting, leaderboard, ROC & PR curves, confusion matrix, adjustable threshold |
| Deep learning | optional Keras MLP (dropout, L2, early stopping) compared on the *same* test set; clear install hint if TensorFlow is missing |
| Explainability | SHAP global + per-customer; labelled permutation / occlusion fallbacks if SHAP is missing |
| Recommendations | transparent rule engine (offline) + optional Claude narrative (server-side, anonymised input) |
| App | 10 Streamlit pages, dark/light theme, KPI cards, Plotly charts, prediction history |

## Architecture
```
app.py (UI only)
 ├─ src/data_loader.py     read + validate + clean CSV
 ├─ src/preprocessing.py   ColumnTransformer (impute/scale/one-hot) + stratified split
 ├─ src/train_models.py    train, tune, compare, save (joblib + SHA-256) + optional Keras
 ├─ src/predict.py         integrity-checked load, CSV validation, single/batch inference
 ├─ src/explainability.py  SHAP / fallbacks
 ├─ src/recommendations.py rules + optional Claude API
 ├─ src/analytics.py       KPIs, segments, SQL queries
 └─ src/database.py        SQLite: customers + prediction history
config.py  models/  data/  tests/  notebooks/
```
Preprocessing and the classifier live in **one sklearn Pipeline**, so training and inference share exactly the same transforms
and nothing is fitted on test data.

## Quick start (Windows + VS Code)
1. Install Python 3.10–3.12 and VS Code (with the Python extension).
2. Download the dataset from <https://www.kaggle.com/datasets/blastchar/telco-customer-churn> and save it as
   `data/WA_Fn-UseC_-Telco-Customer-Churn.csv` (or upload it in the app sidebar).
3. In the VS Code terminal (PowerShell) from the project folder:
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1          # if blocked: Set-ExecutionPolicy -Scope Process Bypass
pip install -r requirements.txt
copy .env.example .env              # optional: add ANTHROPIC_API_KEY
python -m pytest -q                 # run unit tests
python -m src.train_models          # optional: train from the CLI (or train in the app)
streamlit run app.py                # ← one-command launch
```
macOS/Linux: `source .venv/bin/activate` instead of the activate script, `cp` instead of `copy`.
Optional deep learning: `pip install tensorflow`, then tick "Keras neural network" on the training page.

## How models are chosen (no peeking)
The deployed model is the classical pipeline with the highest mean **cross-validated ROC-AUC on the training split**.
ROC-AUC is threshold-free and robust to imbalance. The **test set is used only for reporting**, identically for every model.
The decision threshold is a business setting: lowering it catches more churners at the cost of more retention offers.

## Privacy & security
* API key only via `ANTHROPIC_API_KEY` env var / `.env` / Streamlit secrets (`.streamlit/secrets.toml`) — all git-ignored.
* Only whitelisted, anonymised features + probability + feature effects go to Claude; **customerID is never sent**.
* Without a key the app runs fully offline with rule-based advice.
* `joblib` files can run code when loaded: the app loads only `models/best_pipeline.joblib` and verifies its SHA-256 (written at
  training time) first. Never load model files from untrusted sources.
* SQL uses parameterised queries; uploaded CSVs are validated row-by-row.

## Tests
`python -m pytest -q` — preprocessing, leakage-free split, CSV validation, batch prediction (invalid rows kept & flagged),
recommendation anonymisation and explanation sanity. Tests use a small **synthetic, schema-compatible fixture** (clearly marked
test-only in `tests/conftest.py`); it is never used for app results.

## Reproducibility
Seed `42` (config.py) for split, CV, models; versions are saved in `models/metadata.json`. Same data + same package versions
⇒ same split and results.

## Limitations
* One historical, single-company dataset; results may not transfer to other telcos or periods.
* Correlation is not causation — retention actions are hypotheses to A/B test.
* Probabilities are not explicitly calibrated; the default 0.5 threshold is not cost-optimal.
* No cost/benefit data, so the economically best threshold is left to the user.
* SHAP values are model explanations, not causal effects.

## Resume-ready description
**ChurnGuard AI** — Built an end-to-end customer churn prediction platform on the IBM Telco dataset: automated data-quality
checks, EDA and SQL analytics (Pandas, SQLite, Plotly); leakage-safe scikit-learn pipelines comparing Logistic Regression, Decision
Tree, Random Forest and XGBoost with cross-validation, tuning and threshold analysis; optional Keras baseline; SHAP explanations;
rule-based and Claude-powered retention recommendations with anonymised inputs; and a 10-page Streamlit app with batch scoring,
prediction history and unit tests. *(Add your own measured results from your run, e.g. your leaderboard ROC-AUC.)*

## Interview prep — plain-language walkthrough
* **Pipeline** = a conveyor belt: fill blanks → scale numbers → one-hot encode text → model. Saved as one object.
* **Stratified split** keeps the churn ratio equal in train and test.
* **Why not accuracy?** Most customers stay, so "predict no churn" already looks accurate. Use recall/precision/F1/ROC-AUC.
* **Class weighting** makes mistakes on the rare class cost more.
* **Cross-validation** averages performance over several train/validation splits to pick models fairly.
* **ROC-AUC** = chance a random churner gets a higher score than a random loyal customer.
* **SHAP** splits a prediction into per-feature contributions (positive = pushes toward churn).
* **Data leakage** = information from test data influencing training; avoided by fitting everything inside the pipeline on train only.
* **Deep learning on tabular data** rarely beats tree ensembles here; it is included as an honest comparison.
