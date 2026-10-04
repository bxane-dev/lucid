import sqlite3
from datetime import datetime, timezone
from .config import DB_PATH


def init_db() -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                source_dataset TEXT,
                source_recording TEXT,
                ground_truth TEXT,
                prediction TEXT,
                confidence REAL,
                model_version TEXT
            )
            """
        )
        conn.commit()


def log_prediction(*, source_dataset, source_recording, ground_truth, prediction, confidence, model_version) -> None:
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            INSERT INTO predictions
            (created_at, source_dataset, source_recording, ground_truth, prediction, confidence, model_version)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (datetime.now(timezone.utc).isoformat(), source_dataset, source_recording, ground_truth, prediction, confidence, model_version),
        )
        conn.commit()
