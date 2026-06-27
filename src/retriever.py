from __future__ import annotations

from dataclasses import dataclass
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.text_chunker import TextChunk


@dataclass
class RetrievedChunk:
    document_name: str
    chunk_id: int
    text: str
    score: float


class TfidfRetriever:
    def __init__(self, chunks: list[TextChunk]):
        if not chunks:
            raise ValueError("Cannot build retriever without document chunks.")
        self.chunks = chunks
        self.vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
        self.matrix = self.vectorizer.fit_transform([chunk.text for chunk in chunks])

    def search(self, query: str, top_k: int = 5) -> list[RetrievedChunk]:
        if not query.strip():
            return []
        query = _normalise_query(query)
        query_vec = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vec, self.matrix).ravel()
        ranked_ids = scores.argsort()[::-1][:top_k]
        results: list[RetrievedChunk] = []
        for idx in ranked_ids:
            chunk = self.chunks[int(idx)]
            results.append(
                RetrievedChunk(
                    document_name=chunk.document_name,
                    chunk_id=chunk.chunk_id,
                    text=chunk.text,
                    score=float(scores[int(idx)]),
                )
            )
        return results


def _normalise_query(query: str) -> str:
    lower = re.sub(r"\s+", " ", query.lower()).strip()
    replacements = {
        "omnett": "omnet++",
        "omnet ": "omnet++ ",
        "simulatation": "simulation",
        "simulaton": "simulation",
        "summery": "summary",
    }
    for old, new in replacements.items():
        lower = lower.replace(old, new)
    expansions = {
        "how to": "workflow steps procedure build execute",
        "omnet++": "omnet++ omnet discrete event simulator simulation framework ned msg ini c++ modules gates links messages",
        "simulation": "simulation modeling model experiment run execute output results",
    }
    extra_terms = [extra for trigger, extra in expansions.items() if trigger in lower]
    return f"{lower} {' '.join(extra_terms)}".strip()
