from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from src.config import SETTINGS
from src.feedback_store import FeedbackRecord, FeedbackStore
from src.logging_config import configure_logging
from src.service import DocumentService
from src.validation import InputValidationError


configure_logging()
service = DocumentService()
feedback_store = FeedbackStore(SETTINGS.feedback_database)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    yield


app = FastAPI(
    title="Document Intelligence and Evaluation API",
    description=(
        "Local API for document ingestion, source-grounded TF-IDF retrieval, "
        "answer evaluation, structured insights, and user feedback."
    ),
    version="2.0.0",
    lifespan=lifespan,
)


class TextDocumentRequest(BaseModel):
    name: str = Field(default="pasted_document.txt", min_length=1, max_length=200)
    text: str = Field(min_length=1)


class QueryRequest(BaseModel):
    document_id: str = Field(min_length=1)
    question: str = Field(min_length=1, max_length=1000)
    top_k: int = Field(default=SETTINGS.default_top_k, ge=1, le=SETTINGS.maximum_top_k)


class FeedbackRequest(BaseModel):
    document_id: str = Field(min_length=1)
    question: str = Field(min_length=1, max_length=1000)
    helpful: bool
    correction: str = Field(default="", max_length=3000)
    evaluation: Optional[dict] = None


@app.exception_handler(InputValidationError)
async def input_validation_handler(_request, exc: InputValidationError):
    status = 404 if exc.code == "document_not_found" else 422
    return JSONResponse(
        status_code=status,
        content={"error": exc.code, "detail": str(exc)},
    )


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "version": app.version,
        "retrieval": "tfidf",
        "documents_in_memory": len(service.documents),
    }


@app.post("/documents/text", status_code=201)
def add_text_document(request: TextDocumentRequest) -> dict:
    return service.add_text(request.name, request.text).metadata()


@app.post("/documents/upload", status_code=201)
async def upload_document(file: UploadFile = File(...)) -> dict:
    content = await file.read()
    return service.add_upload(file.filename or "", content).metadata()


@app.get("/documents/{document_id}")
def get_document(document_id: str) -> dict:
    return service.get_document(document_id).metadata()


@app.get("/documents/{document_id}/insights")
def get_document_insights(document_id: str) -> dict:
    document = service.get_document(document_id)
    return {
        "document": document.metadata(),
        "insights": document.insights.to_dict(),
        "decision_matrix": document.decision_matrix,
    }


@app.post("/query")
def query_document(request: QueryRequest) -> dict:
    return service.query(
        request.document_id,
        request.question,
        request.top_k,
    ).__dict__


@app.post("/evaluate")
def evaluate_query(request: QueryRequest) -> dict:
    result = service.query(request.document_id, request.question, request.top_k)
    return {
        "document": result.document,
        "question": result.question,
        "evaluation": result.evaluation,
        "query_latency_ms": result.query_latency_ms,
        "source_count": len(result.sources),
    }


@app.post("/feedback", status_code=201)
def add_feedback(request: FeedbackRequest) -> dict:
    service.get_document(request.document_id)
    feedback_id = feedback_store.add(
        FeedbackRecord(
            document_id=request.document_id,
            question=request.question,
            helpful=request.helpful,
            correction=request.correction,
            evaluation=request.evaluation,
        )
    )
    return {"feedback_id": feedback_id, "status": "recorded"}


@app.get("/feedback/summary")
def feedback_summary() -> dict:
    return feedback_store.summary()
