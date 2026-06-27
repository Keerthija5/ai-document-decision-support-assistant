from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO


@dataclass
class LoadedDocument:
    name: str
    text: str
    source_type: str


def _read_pdf(file_obj: BinaryIO) -> str:
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise RuntimeError("PDF support requires pypdf. Install requirements.txt first.") from exc

    reader = PdfReader(file_obj)
    pages = []
    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        if text.strip():
            pages.append(f"[Page {page_number}]\n{text}")
    return "\n\n".join(pages)


def _read_text(file_obj: BinaryIO) -> str:
    raw = file_obj.read()
    if isinstance(raw, str):
        return raw
    return raw.decode("utf-8", errors="ignore")


def load_uploaded_file(file_obj: BinaryIO, filename: str) -> LoadedDocument:
    suffix = Path(filename).suffix.lower()
    if suffix == ".pdf":
        text = _read_pdf(file_obj)
        source_type = "pdf"
    elif suffix in {".txt", ".md"}:
        text = _read_text(file_obj)
        source_type = "text"
    else:
        raise ValueError("Unsupported file type. Please upload a PDF, TXT, or MD file.")

    return LoadedDocument(name=filename, text=normalise_text(text), source_type=source_type)


def load_sample_document(path: str | Path) -> LoadedDocument:
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    return LoadedDocument(name=path.name, text=normalise_text(text), source_type=path.suffix.lstrip("."))


def normalise_text(text: str) -> str:
    lines = [line.strip() for line in text.replace("\r", "\n").split("\n")]
    cleaned = []
    blank_seen = False
    for line in lines:
        if not line:
            if not blank_seen:
                cleaned.append("")
            blank_seen = True
            continue
        cleaned.append(line)
        blank_seen = False
    return "\n".join(cleaned).strip()

