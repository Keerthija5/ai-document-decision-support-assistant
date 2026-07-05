from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3


@dataclass
class FeedbackRecord:
    document_id: str
    question: str
    helpful: bool
    correction: str = ""
    evaluation: dict | None = None


class FeedbackStore:
    def __init__(self, database_path: str | Path):
        self.database_path = Path(database_path)
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self._create_table()

    def add(self, feedback: FeedbackRecord) -> int:
        with sqlite3.connect(self.database_path) as connection:
            cursor = connection.execute(
                """
                INSERT INTO feedback
                    (created_at, document_id, question, helpful, correction, evaluation_json)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    datetime.now(timezone.utc).isoformat(),
                    feedback.document_id,
                    feedback.question,
                    int(feedback.helpful),
                    feedback.correction.strip(),
                    json.dumps(feedback.evaluation or {}, ensure_ascii=True),
                ),
            )
            return int(cursor.lastrowid)

    def summary(self) -> dict:
        with sqlite3.connect(self.database_path) as connection:
            total, helpful = connection.execute(
                "SELECT COUNT(*), COALESCE(SUM(helpful), 0) FROM feedback"
            ).fetchone()
        return {
            "total_feedback": int(total),
            "helpful_feedback": int(helpful),
            "not_helpful_feedback": int(total - helpful),
        }

    def _create_table(self) -> None:
        with sqlite3.connect(self.database_path) as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS feedback (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at TEXT NOT NULL,
                    document_id TEXT NOT NULL,
                    question TEXT NOT NULL,
                    helpful INTEGER NOT NULL,
                    correction TEXT NOT NULL,
                    evaluation_json TEXT NOT NULL
                )
                """
            )
