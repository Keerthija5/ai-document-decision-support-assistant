from __future__ import annotations

import unittest

from src.config import Settings
from src.validation import (
    InputValidationError,
    assess_question_support,
    validate_document_input,
    validate_extracted_text,
    validate_question,
    validate_top_k,
)


class ValidationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.settings = Settings(maximum_file_size_mb=1, minimum_document_words=5)

    def test_accepts_supported_text_document(self) -> None:
        validate_document_input("notes.txt", b"enough content", self.settings)

    def test_rejects_unsupported_extension(self) -> None:
        with self.assertRaises(InputValidationError) as context:
            validate_document_input("notes.docx", b"content", self.settings)
        self.assertEqual(context.exception.code, "unsupported_file_type")

    def test_rejects_short_extracted_text(self) -> None:
        with self.assertRaises(InputValidationError) as context:
            validate_extracted_text("only three words", self.settings)
        self.assertEqual(context.exception.code, "insufficient_readable_text")

    def test_normalises_question_whitespace(self) -> None:
        self.assertEqual(validate_question("  What   are the risks? "), "What are the risks?")

    def test_rejects_invalid_top_k(self) -> None:
        with self.assertRaises(InputValidationError):
            validate_top_k(99, self.settings)

    def test_question_support_rejects_missing_specific_concepts(self) -> None:
        result = assess_question_support(
            "Which cloud provider hosts the supplier documents?",
            "Supplier documents contain quotations and engineering assumptions.",
        )

        self.assertFalse(result.supported)
        self.assertIn("cloud", result.missing_terms)
        self.assertIn("provider", result.missing_terms)

    def test_question_support_accepts_grounded_question(self) -> None:
        result = assess_question_support(
            "Which outputs should the assistant generate?",
            "The assistant should generate a structured summary and list of risks.",
        )

        self.assertTrue(result.supported)


if __name__ == "__main__":
    unittest.main()
