from __future__ import annotations

from dataclasses import dataclass
import re


@dataclass
class TextChunk:
    document_name: str
    chunk_id: int
    text: str


def chunk_text(document_name: str, text: str, max_words: int = 150, overlap_words: int = 30) -> list[TextChunk]:
    words = re.findall(r"\S+", text)
    if not words:
        return []
    if max_words <= overlap_words:
        raise ValueError("max_words must be larger than overlap_words.")

    chunks: list[TextChunk] = []
    start = 0
    chunk_id = 1
    step = max_words - overlap_words
    while start < len(words):
        end = min(start + max_words, len(words))
        chunk_words = words[start:end]
        chunks.append(TextChunk(document_name=document_name, chunk_id=chunk_id, text=" ".join(chunk_words)))
        if end == len(words):
            break
        start += step
        chunk_id += 1
    return chunks

