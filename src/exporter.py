from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path

import pandas as pd

from src.evaluator import EvaluationResult
from src.insight_extractor import DecisionInsights


def timestamped_name(prefix: str, extension: str) -> str:
    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    return f"{prefix}_{stamp}.{extension}"


def build_export_payload(
    insights: DecisionInsights,
    decision_matrix: list[dict],
    evaluation: EvaluationResult | None,
    question: str | None,
    answer: str | None,
) -> dict:
    return {
        "question": question,
        "answer": answer,
        "insights": insights.to_dict(),
        "decision_matrix": decision_matrix,
        "evaluation": evaluation.to_dict() if evaluation else None,
    }


def payload_to_json(payload: dict) -> str:
    return json.dumps(payload, indent=2, ensure_ascii=False)


def matrix_to_csv(decision_matrix: list[dict]) -> str:
    return pd.DataFrame(decision_matrix).to_csv(index=False)


def payload_to_markdown(payload: dict) -> str:
    insights = payload["insights"]
    lines = [
        "# AI Decision Support Report",
        "",
        "## Question",
        payload.get("question") or "No question provided.",
        "",
        "## Answer",
        payload.get("answer") or "No answer generated.",
        "",
        "## Summary",
        insights.get("summary", ""),
        "",
        "## Requirements",
        *_bullet_lines(insights.get("requirements", [])),
        "",
        "## Risks",
        *_bullet_lines(insights.get("risks", [])),
        "",
        "## Recommendations",
        *_bullet_lines(insights.get("recommendations", [])),
        "",
        "## Missing Information",
        *_bullet_lines(insights.get("missing_information", [])),
    ]
    evaluation = payload.get("evaluation")
    if evaluation:
        lines.extend(
            [
                "",
                "## Evaluation",
                f"- Relevance: {evaluation['relevance']}/100",
                f"- Completeness: {evaluation['completeness']}/100",
                f"- Grounding: {evaluation['grounding']}/100",
                f"- Consistency: {evaluation['consistency']}/100",
                f"- Hallucination risk: {evaluation['hallucination_risk']}",
                f"- Human review required: {evaluation['human_review_required']}",
            ]
        )
    return "\n".join(lines)


def save_text(path: str | Path, content: str) -> None:
    Path(path).write_text(content, encoding="utf-8")


def _bullet_lines(items: list[str]) -> list[str]:
    return [f"- {item}" for item in items] if items else ["- Not identified in the source document."]

