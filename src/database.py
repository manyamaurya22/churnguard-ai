"""SQLite layer: customer records + prediction history. Uses parameterised SQL only."""
from __future__ import annotations

import json
import logging
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone

import pandas as pd

import config

log = logging.getLogger(__name__)


@contextmanager
def connect(db_path=None):
    path = db_path or config.DB_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db(db_path=None) -> None:
    with connect(db_path) as c:
        c.execute("""CREATE TABLE IF NOT EXISTS predictions(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT NOT NULL, source TEXT NOT NULL, customer_ref TEXT,
            probability REAL NOT NULL, threshold REAL NOT NULL,
            risk_category TEXT NOT NULL, model_name TEXT, inputs_json TEXT)""")


def load_customers(df: pd.DataFrame, db_path=None) -> int:
    """(Re)load customer records into the `customers` table."""
    out = df.drop(columns=["churn_flag"], errors="ignore")
    with connect(db_path) as c:
        out.to_sql("customers", c, if_exists="replace", index=False)
    return len(out)


def run_query(sql: str, params: tuple = (), db_path=None) -> pd.DataFrame:
    with connect(db_path) as c:
        return pd.read_sql_query(sql, c, params=params)


def log_prediction(source: str, probability: float, threshold: float, risk: str,
                   model_name: str = "", customer_ref: str = "", inputs: dict | None = None,
                   db_path=None) -> None:
    init_db(db_path)
    with connect(db_path) as c:
        c.execute(
            "INSERT INTO predictions(created_at,source,customer_ref,probability,threshold,"
            "risk_category,model_name,inputs_json) VALUES(?,?,?,?,?,?,?,?)",
            (datetime.now(timezone.utc).isoformat(timespec="seconds"), source, customer_ref,
             float(probability), float(threshold), risk, model_name,
             json.dumps(inputs, default=str) if inputs else None))


def log_predictions_bulk(rows: list[tuple], db_path=None) -> None:
    """rows: (source, customer_ref, probability, threshold, risk, model_name)."""
    init_db(db_path)
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    with connect(db_path) as c:
        c.executemany(
            "INSERT INTO predictions(created_at,source,customer_ref,probability,threshold,"
            "risk_category,model_name) VALUES(?,?,?,?,?,?,?)",
            [(now, r[0], r[1], float(r[2]), float(r[3]), r[4], r[5]) for r in rows])


def get_history(limit: int = 1000, db_path=None) -> pd.DataFrame:
    init_db(db_path)
    return run_query("SELECT id,created_at,source,customer_ref,probability,threshold,"
                     "risk_category,model_name FROM predictions ORDER BY id DESC LIMIT ?",
                     (int(limit),), db_path)


def clear_history(db_path=None) -> None:
    init_db(db_path)
    with connect(db_path) as c:
        c.execute("DELETE FROM predictions")
