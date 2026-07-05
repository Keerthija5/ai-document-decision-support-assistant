from __future__ import annotations

from fastapi.testclient import TestClient

from api import app


client = TestClient(app)


def test_health_endpoint() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_text_document_and_query_workflow() -> None:
    document_response = client.post(
        "/documents/text",
        json={
            "name": "quality.txt",
            "text": (
                "The visual inspection pilot requires labelled images and metadata. "
                "The main risks are poor image quality, inconsistent labels, class "
                "imbalance, and lighting changes. Evaluation should use accuracy, "
                "F1-score, confusion matrix, and false negative rate."
            ),
        },
    )
    assert document_response.status_code == 201
    document_id = document_response.json()["document_id"]
    query_response = client.post(
        "/query",
        json={
            "document_id": document_id,
            "question": "What are the main risks?",
            "top_k": 3,
        },
    )
    assert query_response.status_code == 200
    assert query_response.json()["sources"]
