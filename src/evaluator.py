from __future__ import annotations

from dataclasses import dataclass, asdict
import re

from src.retriever import RetrievedChunk


@dataclass
class EvaluationResult:
    relevance: int
    completeness: int
    grounding: int
    consistency: int
    hallucination_risk: str
    human_review_required: bool
    notes: list[str]

    def to_dict(self) -> dict:
        return asdict(self)


def evaluate_answer(question: str, answer: str, sources: list[RetrievedChunk], missing_information: list[str]) -> EvaluationResult:
    relevance = _overlap_score(question, answer)
    completeness = max(20, 100 - len(missing_information) * 20)
    grounding = _grounding_score(answer, sources)
    consistency = 90 if len(answer.split()) > 12 and answer.endswith((".", "!", "?")) else 65

    avg = (relevance + completeness + grounding + consistency) / 4
    if grounding < 35 or avg < 45:
        risk = "High"
    elif grounding < 60 or avg < 70:
        risk = "Medium"
    else:
        risk = "Low"

    notes = []
    if not sources:
        notes.append("No supporting source chunks were used.")
    if missing_information:
        notes.extend([f"Missing information: {item}" for item in missing_information])
    if grounding < 60:
        notes.append("Answer grounding is partial; source context should be reviewed.")
    if relevance < 50:
        notes.append("Answer may not be strongly aligned with the question.")

    return EvaluationResult(
        relevance=relevance,
        completeness=completeness,
        grounding=grounding,
        consistency=consistency,
        hallucination_risk=risk,
        human_review_required=risk != "Low" or bool(missing_information),
        notes=notes or ["Answer appears sufficiently supported for a first review."],
    )


def _overlap_score(left: str, right: str) -> int:
    left_terms = _expand_terms(_terms(left))
    right_terms = _expand_terms(_terms(right))
    if not left_terms:
        return 0
    overlap = len(left_terms & right_terms) / len(left_terms)
    return int(min(100, max(0, round(overlap * 100))))


def _grounding_score(answer: str, sources: list[RetrievedChunk]) -> int:
    if not sources:
        return 0
    source_text = " ".join(source.text for source in sources)
    answer_terms = _terms(answer)
    source_terms = _terms(source_text)
    if not answer_terms:
        return 0
    overlap = len(answer_terms & source_terms) / len(answer_terms)
    retrieval_strength = min(1.0, sum(source.score for source in sources[:3]))
    score = (overlap * 0.75 + retrieval_strength * 0.25) * 100
    return int(min(100, max(0, round(score))))


def _terms(text: str) -> set[str]:
    stop_words = {"the", "and", "for", "with", "from", "that", "this", "into", "using", "are", "was", "were", "has"}
    return {
        term.lower()
        for term in re.findall(r"[A-Za-z][A-Za-z\-]+", text)
        if len(term) > 2 and term.lower() not in stop_words
    }


def _expand_terms(terms: set[str]) -> set[str]:
    synonyms = {
        "risk": {"risk", "risks", "challenge", "limitation", "issue", "failure"},
        "risks": {"risk", "risks", "challenge", "limitation", "issue", "failure"},
        "recommended": {"recommend", "recommendation", "recommended", "next", "step", "action"},
        "recommend": {"recommend", "recommendation", "recommended", "next", "step", "action"},
        "steps": {"step", "steps", "action", "actions", "recommendation"},
        "data": {"data", "dataset", "input", "source", "metadata"},
        "evaluate": {"evaluate", "evaluation", "metric", "kpi", "test", "baseline"},
    }
    expanded = set(terms)
    for term in list(terms):
        expanded.update(synonyms.get(term, set()))
    return expanded
