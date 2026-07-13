from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

from src.config import Settings


@dataclass(frozen=True)
class StorageResult:
    backend: str
    location: str


class ArchiveStore:
    """Stores non-sensitive processing records for review and reproducibility."""

    def __init__(self, settings: Settings):
        self.settings = settings

    def status(self) -> dict[str, str | bool]:
        return {
            "backend": self.settings.storage_backend,
            "local_directory": str(self.settings.archive_directory),
            "s3_bucket_configured": bool(self.settings.s3_bucket),
            "stores_raw_document_text": False,
            "stores_full_query_text": self.settings.archive_query_text,
        }

    def save_json(self, folder: str, record_id: str, payload: dict[str, Any]) -> StorageResult:
        safe_payload = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "storage_note": (
                "This archive stores metadata, evaluation results, and review traces. "
                "It does not store raw uploaded document text."
            ),
            **payload,
        }
        filename = f"{_safe_name(record_id)}.json"
        if self.settings.storage_backend == "s3":
            return self._save_s3(folder, filename, safe_payload)
        return self._save_local(folder, filename, safe_payload)

    def _save_local(self, folder: str, filename: str, payload: dict[str, Any]) -> StorageResult:
        target_dir = self.settings.archive_directory / _safe_name(folder)
        target_dir.mkdir(parents=True, exist_ok=True)
        target_path = target_dir / filename
        target_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        return StorageResult(backend="local", location=str(target_path))

    def _save_s3(self, folder: str, filename: str, payload: dict[str, Any]) -> StorageResult:
        if not self.settings.s3_bucket:
            raise RuntimeError("RAG_S3_BUCKET is required when RAG_STORAGE_BACKEND=s3.")
        try:
            import boto3
        except ImportError as exc:
            raise RuntimeError(
                "S3 archive support requires boto3. Install optional cloud dependencies first."
            ) from exc

        key_parts = [
            self.settings.s3_prefix,
            _safe_name(folder),
            filename,
        ]
        key = "/".join(part for part in key_parts if part)
        boto3.client("s3").put_object(
            Bucket=self.settings.s3_bucket,
            Key=key,
            Body=json.dumps(payload, indent=2, sort_keys=True).encode("utf-8"),
            ContentType="application/json",
        )
        return StorageResult(backend="s3", location=f"s3://{self.settings.s3_bucket}/{key}")


def _safe_name(value: str) -> str:
    cleaned = "".join(character if character.isalnum() or character in {"-", "_"} else "-" for character in value)
    return cleaned.strip("-") or "record"
