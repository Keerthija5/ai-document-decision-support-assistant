from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
from statistics import mean

from src.logging_config import configure_logging
from src.service import DocumentService


PROJECT_ROOT = Path(__file__).parent
DEFAULT_DATASET = PROJECT_ROOT / "evaluation_data/golden_questions.json"
DEFAULT_DOCUMENT_DIR = PROJECT_ROOT / "data/sample_documents"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs/evaluation"


def normalise(text: str) -> str:
    return " ".join(text.lower().replace("-", " ").split())


def keyword_coverage(expected_keywords: list[str], text: str) -> float:
    if not expected_keywords:
        return 1.0
    cleaned = normalise(text)
    matched = sum(normalise(keyword) in cleaned for keyword in expected_keywords)
    return matched / len(expected_keywords)


def evaluate_case(service: DocumentService, document_id: str, case: dict) -> dict:
    result = service.query(document_id, case["question"])
    source_text = " ".join(source["text"] for source in result.sources)
    answerable = bool(case["answerable"])
    expected_keywords = case["expected_keywords"]
    retrieval_hit = keyword_coverage(expected_keywords, source_text) > 0 if answerable else not result.sources
    answer_coverage = keyword_coverage(expected_keywords, result.answer) if answerable else 1.0
    correct_refusal = (
        not answerable
        and not result.sources
        and "not enough evidence" in result.answer.lower()
    )
    passed = (
        retrieval_hit and answer_coverage >= 0.5
        if answerable
        else correct_refusal
    )
    return {
        "id": case["id"],
        "document": case["document"],
        "question": case["question"],
        "answerable": answerable,
        "retrieval_hit": retrieval_hit,
        "answer_keyword_coverage": round(answer_coverage, 4),
        "correct_refusal": correct_refusal,
        "grounding_score": result.evaluation["grounding"],
        "human_review_required": result.evaluation["human_review_required"],
        "source_count": len(result.sources),
        "latency_ms": result.query_latency_ms,
        "passed": passed,
        "answer": result.answer,
    }


def run(dataset_path: Path, document_dir: Path, output_dir: Path) -> dict:
    configure_logging()
    cases = json.loads(dataset_path.read_text(encoding="utf-8"))
    service = DocumentService()
    document_ids = {}
    for filename in sorted({case["document"] for case in cases}):
        path = document_dir / filename
        record = service.add_text(filename, path.read_text(encoding="utf-8"))
        document_ids[filename] = record.document_id

    rows = [
        evaluate_case(service, document_ids[case["document"]], case)
        for case in cases
    ]
    answerable_rows = [row for row in rows if row["answerable"]]
    unanswerable_rows = [row for row in rows if not row["answerable"]]
    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset": str(dataset_path),
        "total_questions": len(rows),
        "answerable_questions": len(answerable_rows),
        "unanswerable_questions": len(unanswerable_rows),
        "overall_pass_rate": round(mean(row["passed"] for row in rows), 4),
        "retrieval_hit_rate": round(mean(row["retrieval_hit"] for row in answerable_rows), 4),
        "mean_answer_keyword_coverage": round(
            mean(row["answer_keyword_coverage"] for row in answerable_rows), 4
        ),
        "correct_refusal_rate": round(
            mean(row["correct_refusal"] for row in unanswerable_rows), 4
        ),
        "mean_grounding_score": round(mean(row["grounding_score"] for row in rows), 2),
        "mean_latency_ms": round(mean(row["latency_ms"] for row in rows), 2),
        "failed_question_ids": [row["id"] for row in rows if not row["passed"]],
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "evaluation_results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary_path = output_dir / "evaluation_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the local document assistant.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--documents", type=Path, default=DEFAULT_DOCUMENT_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    summary = run(args.dataset, args.documents, args.output)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
