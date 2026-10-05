import sqlite3
from datetime import datetime, timezone

from .config import DB_PATH


COLUMNS = {
    "state": "TEXT",
    "state_confidence": "REAL",
    "provenance_sha256": "TEXT",
}


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
                model_version TEXT,
                state TEXT,
                state_confidence REAL,
                provenance_sha256 TEXT
            )
            """
        )

        existing = {
            row[1]
            for row in conn.execute("PRAGMA table_info(predictions)")
        }
        for name, sql_type in COLUMNS.items():
            if name not in existing:
                conn.execute(
                    f"ALTER TABLE predictions ADD COLUMN {name} {sql_type}"
                )
        conn.commit()


def log_prediction(
    *,
    source_dataset,
    source_recording,
    ground_truth,
    prediction,
    confidence,
    model_version,
    state=None,
    state_confidence=None,
    provenance_sha256=None,
) -> None:
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            INSERT INTO predictions
            (
                created_at,
                source_dataset,
                source_recording,
                ground_truth,
                prediction,
                confidence,
                model_version,
                state,
                state_confidence,
                provenance_sha256
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                datetime.now(timezone.utc).isoformat(),
                source_dataset,
                source_recording,
                ground_truth,
                prediction,
                confidence,
                model_version,
                state,
                state_confidence,
                provenance_sha256,
            ),
        )
        conn.commit()


def recent_predictions(limit: int = 100) -> list[dict]:
    limit = max(1, min(int(limit), 1000))
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """
            SELECT
                id,
                created_at,
                source_dataset,
                source_recording,
                ground_truth,
                prediction,
                confidence,
                model_version,
                state,
                state_confidence,
                provenance_sha256
            FROM predictions
            ORDER BY id DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
    return [dict(row) for row in rows]
