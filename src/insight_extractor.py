from __future__ import annotations

from dataclasses import dataclass, asdict
import re


@dataclass
class DecisionInsights:
    summary: str
    requirements: list[str]
    risks: list[str]
    recommendations: list[str]
    action_items: list[str]
    missing_information: list[str]

    def to_dict(self) -> dict:
        return asdict(self)


REQUIREMENT_TERMS = ("must", "should", "require", "needs", "expected", "goal", "objective")
RISK_TERMS = ("risk", "challenge", "limitation", "issue", "missing", "uncertain", "bias", "leakage")
ACTION_TERMS = ("implement", "evaluate", "compare", "prepare", "document", "integrate", "test", "define")


def extract_insights(text: str, max_items: int = 6) -> DecisionInsights:
    sentences = _sentences(text)
    summary = " ".join(sentences[:3]) if sentences else "No summary available."
    requirements = _filter_sentences(sentences, REQUIREMENT_TERMS, max_items)
    risks = _filter_sentences(sentences, RISK_TERMS, max_items)
    action_items = _filter_sentences(sentences, ACTION_TERMS, max_items)
    recommendations = _build_recommendations(requirements, risks, action_items)
    missing_information = _infer_missing_information(text)
    return DecisionInsights(
        summary=summary,
        requirements=requirements,
        risks=risks,
        recommendations=recommendations,
        action_items=action_items,
        missing_information=missing_information,
    )


def build_decision_matrix(insights: DecisionInsights) -> list[dict]:
    matrix = []
    primary_recommendations = insights.recommendations or ["Review the document manually and define the next decision step."]
    for idx, recommendation in enumerate(primary_recommendations, start=1):
        risk = insights.risks[idx - 1] if idx - 1 < len(insights.risks) else "No explicit risk found in the source document."
        required_data = insights.requirements[idx - 1] if idx - 1 < len(insights.requirements) else "Required data not clearly specified."
        matrix.append(
            {
                "option": f"Decision option {idx}",
                "recommendation": recommendation,
                "benefit": "Supports a more structured and evidence-based decision.",
                "risk": risk,
                "required_data": required_data,
                "implementation_effort": _estimate_effort(recommendation, risk),
            }
        )
    return matrix


def readiness_score(insights: DecisionInsights) -> dict:
    score = 50
    score += min(len(insights.requirements) * 6, 18)
    score += min(len(insights.action_items) * 5, 15)
    score += min(len(insights.recommendations) * 5, 10)
    score -= min(len(insights.risks) * 4, 16)
    score -= min(len(insights.missing_information) * 6, 18)
    score = max(0, min(100, score))
    if score >= 75:
        label = "High readiness"
        reason = "the document includes enough requirements, actions, and decision signals for a first pilot discussion"
    elif score >= 50:
        label = "Medium readiness"
        reason = "the document has useful information, but some risks, metrics, owners, or data details still need clarification"
    else:
        label = "Low readiness"
        reason = "the document is missing important information needed for reliable decision support"
    return {"score": score, "label": label, "reason": reason}


def _sentences(text: str) -> list[str]:
    return [sentence.strip() for sentence in re.split(r"(?<=[.!?])\s+", text) if len(sentence.split()) >= 6]


def _filter_sentences(sentences: list[str], terms: tuple[str, ...], max_items: int) -> list[str]:
    matches = []
    for sentence in sentences:
        lower = sentence.lower()
        if any(term in lower for term in terms):
            matches.append(sentence)
    return matches[:max_items]


def _build_recommendations(requirements: list[str], risks: list[str], action_items: list[str]) -> list[str]:
    recommendations = []
    if requirements:
        recommendations.append("Validate the most important requirements with stakeholders before implementation.")
    if risks:
        recommendations.append("Review identified risks and define mitigation actions before making a decision.")
    if action_items:
        recommendations.append("Convert extracted action items into an implementation checklist with owners and deadlines.")
    if not recommendations:
        recommendations.append("Collect more specific requirements, risks, and KPIs before proceeding.")
    return recommendations


def _infer_missing_information(text: str) -> list[str]:
    checks = {
        "clear KPIs or success metrics": ("kpi", "metric", "measure", "accuracy", "score"),
        "implementation owner or stakeholder": ("owner", "stakeholder", "team", "responsible"),
        "timeline or deadline": ("deadline", "timeline", "date", "month", "week"),
        "data source description": ("data source", "dataset", "input data", "source"),
    }
    lower = text.lower()
    return [label for label, terms in checks.items() if not any(term in lower for term in terms)]


def _estimate_effort(recommendation: str, risk: str) -> str:
    joined = f"{recommendation} {risk}".lower()
    if any(term in joined for term in ("integration", "deploy", "production", "automation")):
        return "High"
    if any(term in joined for term in ("validate", "compare", "review", "test")):
        return "Medium"
    return "Low"
