from __future__ import annotations

import json
import logging
import time


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for field in (
            "document_id",
            "document_name",
            "word_count",
            "chunk_count",
            "query_latency_ms",
            "retrieved_count",
            "human_review_required",
        ):
            value = getattr(record, field, None)
            if value is not None:
                payload[field] = value
        return json.dumps(payload, ensure_ascii=True)


def configure_logging() -> None:
    root = logging.getLogger()
    if any(getattr(handler, "_rag_json_handler", False) for handler in root.handlers):
        return
    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())
    handler._rag_json_handler = True  # type: ignore[attr-defined]
    root.addHandler(handler)
    root.setLevel(logging.INFO)


class Timer:
    def __enter__(self) -> "Timer":
        self.started = time.perf_counter()
        self.elapsed_ms = 0.0
        return self

    def __exit__(self, *_args: object) -> None:
        self.elapsed_ms = (time.perf_counter() - self.started) * 1000
