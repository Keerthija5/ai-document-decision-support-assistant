from __future__ import annotations

from src.rag_assistant import answer_question
from src.retriever import RetrievedChunk


def test_definition_and_types_are_structured() -> None:
    source = RetrievedChunk(
        document_name="friction.pdf",
        chunk_id=2,
        score=0.8,
        text=(
            "[Page 14] Friction is resistance during motion between contact partners. "
            "Types of friction based on the lubrication state include boundary friction, "
            "mixed friction, and fluid friction."
        ),
    )

    result = answer_question(
        "Definition of friction and types of friction",
        [source],
    )

    assert "Definition" in result.answer
    assert "Types mentioned in the document" in result.answer
    assert "- Boundary friction" in result.answer
    assert "- Mixed friction" in result.answer
    assert "- Fluid friction" in result.answer


def test_missing_explicit_definition_is_disclosed() -> None:
    source = RetrievedChunk(
        document_name="friction.pdf",
        chunk_id=2,
        score=0.8,
        text="Types of friction include boundary friction and fluid friction.",
    )

    result = answer_question(
        "Definition of friction and types of friction",
        [source],
    )

    assert "do not contain a clear one-sentence definition" in result.answer
    assert "- Boundary friction" in result.answer


def test_type_headings_after_repeated_section_labels_are_extracted() -> None:
    source = RetrievedChunk(
        document_name="wear.pdf",
        chunk_id=5,
        score=0.8,
        text=(
            "Types of wear Adhesion: Adhesion occurs on a sliding surface. "
            "Types of wear Abrasion: Typical mechanism for hard-soft contact. "
            "Types of wear Tribochemical wear: It can be caused by corrosion."
        ),
    )

    result = answer_question(
        "Definition of wear and types of wear",
        [source],
    )

    assert "- Adhesion wear" in result.answer
    assert "- Abrasion wear" in result.answer
    assert "- Tribochemical wear" in result.answer
