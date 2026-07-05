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


if __name__ == "__main__":
    unittest.main()
