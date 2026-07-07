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


def test_subtype_explanation_is_not_used_as_main_definition() -> None:
    source = RetrievedChunk(
        document_name="wear.pdf",
        chunk_id=5,
        score=0.8,
        text=(
            "Types of wear Tribochemical wear: The most common tribochemical wear is caused by corrosion. "
            "Types of wear Adhesion: Adhesion on the sliding surface of a journal bearing. "
            "Types of wear Abrasion: Typical mechanism for hard-soft contact."
        ),
    )

    result = answer_question("Define wear and list types of wear", [source])

    assert "do not contain a clear one-sentence definition" in result.answer
    assert "Definition\nTribochemical wear" not in result.answer
    assert "- Tribochemical wear" in result.answer
    assert "- Adhesion wear" in result.answer
    assert "- Abrasion wear" in result.answer


def test_broken_type_fragments_are_filtered_out() -> None:
    source = RetrievedChunk(
        document_name="wear.pdf",
        chunk_id=7,
        score=0.8,
        text=(
            "Wear Types of wear Tribochemical wear: corrosion damage. "
            "Fatigue wear can occur under repeated loading. "
            "Abrasive wear happens with hard particles. "
            "For wear analysis this wear is discussed on wear surfaces."
        ),
    )

    result = answer_question("Define wear and list types of wear", [source])

    assert "- Tribochemical wear" in result.answer
    assert "- Fatigue wear" in result.answer
    assert "- Abrasive wear" in result.answer
    assert "- For wear" not in result.answer
    assert "- Ing wear" not in result.answer
    assert "- This wear" not in result.answer
    assert "- On wear" not in result.answer


def test_natural_define_and_types_wording_uses_structured_answer() -> None:
    source = RetrievedChunk(
        document_name="wear.pdf",
        chunk_id=5,
        score=0.8,
        text=(
            "Wear is the progressive loss of material from contacting surfaces. "
            "Types of wear Adhesion: Material transfers between surfaces. "
            "Types of wear Abrasion: A hard surface removes material."
        ),
    )

    result = answer_question("Define wear and types", [source])

    assert "Definition" in result.answer
    assert "Types mentioned in the document" in result.answer
    assert "- Adhesion wear" in result.answer
    assert "- Abrasion wear" in result.answer


def test_what_is_topic_and_its_types_uses_structured_answer() -> None:
    source = RetrievedChunk(
        document_name="friction.pdf",
        chunk_id=2,
        score=0.8,
        text=(
            "Friction is resistance to relative motion between surfaces. "
            "Types of friction include static friction and sliding friction."
        ),
    )

    result = answer_question("What is friction and what are its types?", [source])

    assert "Definition" in result.answer
    assert "- Static friction" in result.answer
    assert "- Sliding friction" in result.answer


def test_advantages_and_disadvantages_are_grouped_for_study() -> None:
    source = RetrievedChunk(
        document_name="simulation.pdf",
        chunk_id=3,
        score=0.8,
        text=(
            "An advantage of simulation is safe experimentation without changing the real system. "
            "A disadvantage is that model development can be costly and time-consuming."
        ),
    )

    result = answer_question(
        "Explain the advantages and disadvantages of simulation",
        [source],
    )

    assert "Advantages" in result.answer
    assert "Disadvantages" in result.answer
    assert "safe experimentation" in result.answer
    assert "costly" in result.answer


def test_causes_and_effects_are_grouped_for_study() -> None:
    source = RetrievedChunk(
        document_name="wear.pdf",
        chunk_id=4,
        score=0.8,
        text=(
            "Abrasive wear is caused by hard particles between surfaces. "
            "It leads to material loss and surface damage."
        ),
    )

    result = answer_question("What causes abrasive wear and what are its effects?", [source])

    assert "Causes" in result.answer
    assert "Effects" in result.answer
    assert "hard particles" in result.answer
    assert "surface damage" in result.answer


def test_classification_wording_resolves_the_topic_subject() -> None:
    source = RetrievedChunk(
        document_name="wear.pdf",
        chunk_id=3,
        score=0.8,
        text=(
            "Wear is the progressive loss of material from contacting surfaces. "
            "Types of wear Adhesion: Material transfers between surfaces. "
            "Types of wear Abrasion: A hard surface removes material."
        ),
    )

    result = answer_question(
        "Give the definition and classification of wear",
        [source],
    )

    assert "Wear is the progressive loss of material" in result.answer
    assert "- Adhesion wear" in result.answer
    assert "- Abrasion wear" in result.answer
    assert "- Material wear" not in result.answer
    assert "- Loss wear" not in result.answer
