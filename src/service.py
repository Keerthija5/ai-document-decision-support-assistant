from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
from io import BytesIO
import logging

from src.config import SETTINGS, Settings
from src.document_loader import LoadedDocument, load_uploaded_file, normalise_text
from src.evaluator import EvaluationResult, evaluate_answer
from src.insight_extractor import DecisionInsights, build_decision_matrix, extract_insights
from src.logging_config import Timer
from src.rag_assistant import GroundedAnswer, answer_question
from src.retriever import RetrievedChunk, TfidfRetriever
from src.storage import ArchiveStore
from src.text_chunker import TextChunk, chunk_text
from src.validation import (
    InputValidationError,
    assess_question_support,
    validate_document_input,
    validate_extracted_text,
    validate_question,
    validate_top_k,
)


logger = logging.getLogger(__name__)


@dataclass
class DocumentRecord:
    document_id: str
    name: str
    source_type: str
    text: str
    chunks: list[TextChunk]
    retriever: TfidfRetriever
    insights: DecisionInsights
    decision_matrix: list[dict]
    archive_location: str = ""

    def metadata(self) -> dict:
        return {
            "document_id": self.document_id,
            "name": self.name,
            "source_type": self.source_type,
            "word_count": len(self.text.split()),
            "chunk_count": len(self.chunks),
            "archive_location": self.archive_location,
        }


@dataclass
class QueryResult:
    document: dict
    question: str
    answer: str
    sources: list[dict]
    missing_information: list[str]
    question_support: dict
    evaluation: dict
    query_latency_ms: float
    archive_location: str = ""


class DocumentService:
    def __init__(self, settings: Settings = SETTINGS):
        self.settings = settings
        self.documents: dict[str, DocumentRecord] = {}
        self.archive_store = ArchiveStore(settings)

    def add_upload(self, filename: str, content: bytes) -> DocumentRecord:
        validate_document_input(filename, content, self.settings)
        document = load_uploaded_file(BytesIO(content), filename)
        return self._add_document(document, content)

    def add_text(self, name: str, text: str) -> DocumentRecord:
        content = text.encode("utf-8")
        validate_document_input(name, content, self.settings)
        document = LoadedDocument(
            name=name,
            text=normalise_text(text),
            source_type="text",
        )
        return self._add_document(document, content)

    def get_document(self, document_id: str) -> DocumentRecord:
        try:
            return self.documents[document_id]
        except KeyError as exc:
            raise InputValidationError(
                f"Document '{document_id}' was not found in this API session.",
                "document_not_found",
            ) from exc

    def query(self, document_id: str, question: str, top_k: int | None = None) -> QueryResult:
        document = self.get_document(document_id)
        cleaned_question = validate_question(question)
        selected_top_k = validate_top_k(
            top_k if top_k is not None else self.settings.default_top_k,
            self.settings,
        )

        with Timer() as timer:
            retrieved = document.retriever.search(cleaned_question, top_k=selected_top_k)
            support = assess_question_support(cleaned_question, document.text)
            if not support.supported:
                retrieved = []
            grounded_answer = answer_question(
                cleaned_question,
                retrieved,
                min_score=self.settings.minimum_retrieval_score,
            )
            evaluation = evaluate_answer(
                cleaned_question,
                grounded_answer.answer,
                grounded_answer.sources,
                grounded_answer.missing_information,
            )
            archive_location = self._archive_query(
                document,
                cleaned_question,
                grounded_answer,
                support,
                evaluation,
                round(timer.elapsed_ms, 2),
            )

        logger.info(
            "query_completed",
            extra={
                "document_id": document_id,
                "document_name": document.name,
                "query_latency_ms": round(timer.elapsed_ms, 2),
                "retrieved_count": len(grounded_answer.sources),
                "question_support_coverage": support.coverage,
                "human_review_required": evaluation.human_review_required,
            },
        )
        return QueryResult(
            document=document.metadata(),
            question=cleaned_question,
            answer=grounded_answer.answer,
            sources=[asdict(source) for source in grounded_answer.sources],
            missing_information=grounded_answer.missing_information,
            question_support=asdict(support),
            evaluation=evaluation.to_dict(),
            query_latency_ms=round(timer.elapsed_ms, 2),
            archive_location=archive_location,
        )

    def _add_document(self, document: LoadedDocument, content: bytes) -> DocumentRecord:
        validate_extracted_text(document.text, self.settings)
        document_id = hashlib.sha256(document.name.encode("utf-8") + content).hexdigest()[:16]
        chunks = chunk_text(
            document.name,
            document.text,
            max_words=self.settings.chunk_size,
            overlap_words=self.settings.chunk_overlap,
        )
        record = DocumentRecord(
            document_id=document_id,
            name=document.name,
            source_type=document.source_type,
            text=document.text,
            chunks=chunks,
            retriever=TfidfRetriever(chunks),
            insights=extract_insights(document.text),
            decision_matrix=[],
        )
        record.decision_matrix = build_decision_matrix(record.insights)
        record.archive_location = self._archive_document(record)
        self.documents[document_id] = record
        logger.info(
            "document_processed",
            extra={
                "document_id": document_id,
                "document_name": document.name,
                "word_count": len(document.text.split()),
                "chunk_count": len(chunks),
            },
        )
        return record

    def _archive_document(self, record: DocumentRecord) -> str:
        payload = {
            "record_type": "document_metadata",
            "document": {
                "document_id": record.document_id,
                "name": record.name,
                "source_type": record.source_type,
                "word_count": len(record.text.split()),
                "chunk_count": len(record.chunks),
            },
            "insight_summary": {
                "requirements": len(record.insights.requirements),
                "risks": len(record.insights.risks),
                "recommendations": len(record.insights.recommendations),
                "missing_information": len(record.insights.missing_information),
            },
        }
        try:
            result = self.archive_store.save_json("documents", record.document_id, payload)
            return result.location
        except Exception as exc:
            logger.warning(
                "document_archive_failed",
                extra={"document_id": record.document_id, "reason": str(exc)},
            )
            return ""

    def _archive_query(
        self,
        document: DocumentRecord,
        question: str,
        grounded_answer: GroundedAnswer,
        support,
        evaluation: EvaluationResult,
        latency_ms: float,
    ) -> str:
        question_hash = hashlib.sha256(question.encode("utf-8")).hexdigest()[:16]
        query_record_id = f"{document.document_id}-{question_hash}"
        payload = {
            "record_type": "query_evaluation",
            "document": document.metadata(),
            "question_hash": question_hash,
            "question": question if self.settings.archive_query_text else "",
            "source_count": len(grounded_answer.sources),
            "source_trace": [
                {
                    "document_name": source.document_name,
                    "chunk_id": source.chunk_id,
                    "score": round(source.score, 4),
                }
                for source in grounded_answer.sources
            ],
            "missing_information": grounded_answer.missing_information,
            "question_support": asdict(support),
            "evaluation": evaluation.to_dict(),
            "query_latency_ms": latency_ms,
        }
        try:
            result = self.archive_store.save_json("queries", query_record_id, payload)
            return result.location
        except Exception as exc:
            logger.warning(
                "query_archive_failed",
                extra={"document_id": document.document_id, "reason": str(exc)},
            )
            return ""
