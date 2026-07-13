from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    chunk_size: int = 150
    chunk_overlap: int = 30
    default_top_k: int = 5
    maximum_top_k: int = 8
    minimum_retrieval_score: float = 0.03
    maximum_file_size_mb: int = 10
    minimum_document_words: int = 20
    feedback_database: Path = Path(".app_cache/feedback.db")
    storage_backend: str = "local"
    archive_directory: Path = Path(".app_cache/archive")
    archive_query_text: bool = False
    s3_bucket: str = ""
    s3_prefix: str = "document-intelligence"

    @property
    def maximum_file_size_bytes(self) -> int:
        return self.maximum_file_size_mb * 1024 * 1024


def load_settings() -> Settings:
    return Settings(
        chunk_size=int(os.getenv("RAG_CHUNK_SIZE", "150")),
        chunk_overlap=int(os.getenv("RAG_CHUNK_OVERLAP", "30")),
        default_top_k=int(os.getenv("RAG_TOP_K", "5")),
        maximum_top_k=int(os.getenv("RAG_MAX_TOP_K", "8")),
        minimum_retrieval_score=float(os.getenv("RAG_MIN_SCORE", "0.03")),
        maximum_file_size_mb=int(os.getenv("RAG_MAX_FILE_MB", "10")),
        minimum_document_words=int(os.getenv("RAG_MIN_DOCUMENT_WORDS", "20")),
        feedback_database=Path(os.getenv("RAG_FEEDBACK_DB", ".app_cache/feedback.db")),
        storage_backend=os.getenv("RAG_STORAGE_BACKEND", "local").lower(),
        archive_directory=Path(os.getenv("RAG_ARCHIVE_DIR", ".app_cache/archive")),
        archive_query_text=os.getenv("RAG_ARCHIVE_QUERY_TEXT", "false").lower() == "true",
        s3_bucket=os.getenv("RAG_S3_BUCKET", ""),
        s3_prefix=os.getenv("RAG_S3_PREFIX", "document-intelligence").strip("/"),
    )


SETTINGS = load_settings()
