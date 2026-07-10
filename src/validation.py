from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

from src.config import Settings


class InputValidationError(ValueError):
    def __init__(self, message: str, code: str):
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class QuestionSupport:
    supported: bool
    matched_terms: tuple[str, ...]
    missing_terms: tuple[str, ...]
    coverage: float


_QUESTION_STOP_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "be",
    "by",
    "do",
    "does",
    "for",
    "from",
    "how",
    "in",
    "is",
    "it",
    "main",
    "of",
    "on",
    "or",
    "the",
    "to",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
}


def validate_document_input(filename: str, content: bytes, settings: Settings) -> None:
    if not filename.strip():
        raise InputValidationError("A document filename is required.", "missing_filename")
    suffix = Path(filename).suffix.lower()
    if suffix not in {".pdf", ".txt", ".md", ".docx", ".png", ".jpg", ".jpeg"}:
        raise InputValidationError(
            "Unsupported file type. Upload a PDF, Word document, image, TXT, or Markdown document.",
            "unsupported_file_type",
        )
    if not content:
        raise InputValidationError("The uploaded document is empty.", "empty_document")
    if len(content) > settings.maximum_file_size_bytes:
        raise InputValidationError(
            f"The file exceeds the {settings.maximum_file_size_mb} MB limit.",
            "file_too_large",
        )


def validate_extracted_text(text: str, settings: Settings) -> None:
    word_count = len(text.split())
    if word_count < settings.minimum_document_words:
        raise InputValidationError(
            f"Only {word_count} readable words were extracted; at least "
            f"{settings.minimum_document_words} are required.",
            "insufficient_readable_text",
        )


def validate_question(question: str) -> str:
    cleaned = " ".join(question.split())
    if not cleaned:
        raise InputValidationError("A question is required.", "empty_question")
    if len(cleaned) < 4:
        raise InputValidationError("The question is too short.", "question_too_short")
    if len(cleaned) > 1000:
        raise InputValidationError("The question exceeds 1,000 characters.", "question_too_long")
    return cleaned


def validate_top_k(top_k: int, settings: Settings) -> int:
    if not 1 <= top_k <= settings.maximum_top_k:
        raise InputValidationError(
            f"top_k must be between 1 and {settings.maximum_top_k}.",
            "invalid_top_k",
        )
    return top_k


def assess_question_support(
    question: str,
    document_text: str,
    minimum_coverage: float = 0.5,
) -> QuestionSupport:
    """Check whether the document contains the meaningful concepts in a question."""
    question_terms = {
        _normalise_term(term)
        for term in re.findall(r"[A-Za-z]+", question.lower())
        if term not in _QUESTION_STOP_WORDS
    }
    question_terms.discard("")
    if not question_terms:
        return QuestionSupport(True, (), (), 1.0)

    document_terms = {
        _normalise_term(term)
        for term in re.findall(r"[A-Za-z]+", document_text.lower())
    }
    matched = tuple(sorted(question_terms & document_terms))
    missing = tuple(sorted(question_terms - document_terms))
    coverage = len(matched) / len(question_terms)
    return QuestionSupport(
        supported=coverage >= minimum_coverage,
        matched_terms=matched,
        missing_terms=missing,
        coverage=round(coverage, 4),
    )


def _normalise_term(term: str) -> str:
    irregular = {
        "categories": "category",
        "classes": "class",
        "data": "data",
        "risks": "risk",
    }
    if term in irregular:
        return irregular[term]
    if len(term) > 5 and term.endswith("ing"):
        term = term[:-3]
    elif len(term) > 4 and term.endswith("ied"):
        term = f"{term[:-3]}y"
    elif len(term) > 4 and term.endswith("ed"):
        term = term[:-2]
    elif len(term) > 4 and term.endswith("s"):
        term = term[:-1]
    return term
