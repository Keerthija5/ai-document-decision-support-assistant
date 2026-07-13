from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from src.config import Settings
from src.feedback_store import FeedbackRecord, FeedbackStore
from src.service import DocumentService


SAMPLE_TEXT = """
Visual Quality Inspection

The system requires labelled product images and metadata from the production line.
The main risks are poor image quality, inconsistent labels, class imbalance, and
changes in lighting conditions. Start with a controlled pilot using one product
family. Evaluate the model using accuracy, F1-score, and false negative rate.
"""


class ServiceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.service = DocumentService(Settings(minimum_document_words=10))
        self.document = self.service.add_text("quality.txt", SAMPLE_TEXT)

    def test_adds_document_with_chunks(self) -> None:
        self.assertGreater(self.document.metadata()["chunk_count"], 0)
        self.assertIn(self.document.document_id, self.service.documents)

    def test_returns_grounded_answer(self) -> None:
        result = self.service.query(self.document.document_id, "What are the main risks?")
        self.assertTrue(result.sources)
        self.assertIn("risk", result.answer.lower())

    def test_refuses_unrelated_question(self) -> None:
        result = self.service.query(
            self.document.document_id,
            "Which hospital purchased the software?",
        )
        self.assertFalse(result.sources)
        self.assertIn("not enough evidence", result.answer.lower())

    def test_feedback_store_summarises_records(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = FeedbackStore(Path(directory) / "feedback.db")
            store.add(
                FeedbackRecord(
                    document_id=self.document.document_id,
                    question="What are the risks?",
                    helpful=True,
                )
            )
            self.assertEqual(store.summary()["helpful_feedback"], 1)

    def test_archives_metadata_without_raw_document_text(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            service = DocumentService(
                Settings(
                    minimum_document_words=10,
                    archive_directory=Path(directory) / "archive",
                )
            )
            document = service.add_text("quality.txt", SAMPLE_TEXT)
            result = service.query(document.document_id, "What are the main risks?")

            document_archive = Path(document.archive_location)
            query_archive = Path(result.archive_location)
            self.assertTrue(document_archive.exists())
            self.assertTrue(query_archive.exists())
            self.assertNotIn("labelled product images", document_archive.read_text(encoding="utf-8"))
            self.assertNotIn("labelled product images", query_archive.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
