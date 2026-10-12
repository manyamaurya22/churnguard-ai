"""Central configuration: paths, seeds, schema. Change values here, not in the code."""
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
DB_PATH = DATA_DIR / "churnguard.db"
DATA_FILE = DATA_DIR / "WA_Fn-UseC_-Telco-Customer-Churn.csv"

RANDOM_SEED = 42
TEST_SIZE = 0.2
CV_FOLDS = 3
DEFAULT_THRESHOLD = 0.5

TARGET = "Churn"
ID_COL = "customerID"

# SeniorCitizen is stored as 0/1 in the dataset; it is treated as a numeric feature.
NUMERIC_FEATURES = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]
CATEGORICAL_FEATURES = [
    "gender", "Partner", "Dependents", "PhoneService", "MultipleLines",
    "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
    "PaperlessBilling", "PaymentMethod",
]
FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES
REQUIRED_COLUMNS = [ID_COL] + FEATURE_COLUMNS + [TARGET]

_YES_NO = ["Yes", "No"]
_YES_NO_NET = ["Yes", "No", "No internet service"]
# Allowed values per categorical column (from the public dataset documentation).
CATEGORY_VALUES = {
    "gender": ["Female", "Male"],
    "Partner": _YES_NO, "Dependents": _YES_NO, "PhoneService": _YES_NO,
    "PaperlessBilling": _YES_NO,
    "MultipleLines": ["Yes", "No", "No phone service"],
    "InternetService": ["DSL", "Fiber optic", "No"],
    "OnlineSecurity": _YES_NO_NET, "OnlineBackup": _YES_NO_NET,
    "DeviceProtection": _YES_NO_NET, "TechSupport": _YES_NO_NET,
    "StreamingTV": _YES_NO_NET, "StreamingMovies": _YES_NO_NET,
    "Contract": ["Month-to-month", "One year", "Two year"],
    "PaymentMethod": ["Electronic check", "Mailed check",
                      "Bank transfer (automatic)", "Credit card (automatic)"],
}

PRED_COLUMNS = ["churn_probability", "predicted_churn", "risk_category", "validation_status"]

# LLM (optional). Key is read from env var ANTHROPIC_API_KEY or Streamlit secrets.
ANTHROPIC_MODEL_DEFAULT = "claude-sonnet-5-5"
