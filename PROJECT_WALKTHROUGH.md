# Project Walkthrough

This document explains the project in interview-friendly language.

## 1. Problem

Students and teams often have long PDFs, lecture notes, research papers, or use-case documents. Reading the whole document manually takes time, and normal PDF search only finds exact words. This app helps users upload a document, understand what it contains, ask questions, inspect source text, and export a structured report.

## 2. Core Idea

The project is a local document intelligence assistant. It does not blindly answer from memory. It first extracts text from the uploaded document, splits the text into chunks, retrieves relevant chunks for the user question, and then generates an answer using those chunks and mode-specific logic.

## 3. Main Pipeline

```text
Document upload
    -> Text extraction
    -> Text cleaning
    -> Chunking
    -> Retrieval index
    -> Question intent classification
    -> Answer generation
    -> Evaluation
    -> Source traceability and export
```

## 4. Important Concepts

### Text Extraction

PDFs are not always easy to read programmatically. The app uses `pypdf` to extract text from each page. It also stores page markers like `[Page 1]` so the app can show page-level extracted text.

### PDF Preview

PDF text extraction is not the same as visual reading. Some diagrams may be images. To help the user compare the original PDF with extracted text, the app renders PDF pages as images using `pypdfium2`.

### Chunking

Long documents are split into smaller overlapping chunks. This makes retrieval more focused. Overlap helps avoid losing context at chunk boundaries.

### Retrieval

Version 1 uses TF-IDF retrieval from scikit-learn. TF-IDF finds chunks that share important terms with the question. It is explainable and works locally, but it is less flexible than embedding-based semantic search.

### Intent Classification

The app first checks what kind of question the user is asking:

- brief summary
- whole summary
- study topics
- page navigation
- normal source-grounded Q&A

This prevents broad questions like “Give me summary of the whole PDF” from failing just because no single chunk matches perfectly.

### Analysis Modes

Modes guide answer style:

- Study Notes: explain for learning and exam revision.
- Industrial AI / Quality: focus on risks, KPIs, data, validation, and pilot readiness.
- Research Paper: focus on problem, method, dataset, result, and limitation.
- Business Decision: focus on stakeholders, dependencies, options, and next actions.

### Evaluation

The app gives basic quality scores:

- relevance
- completeness
- grounding
- consistency
- hallucination risk

These scores are not perfect truth. They are first-level indicators that help decide whether the answer needs manual review.

## 5. What Makes It Practical

- It is a working Streamlit app, not just a notebook.
- It supports PDF preview and extracted text inspection.
- It stores recent documents locally.
- It has multiple answer modes.
- It includes source traceability.
- It can export structured reports.
- It explains when diagram or source extraction may need manual review.

## 6. How To Explain In An Interview

Short version:

> I built a document intelligence assistant that lets users upload PDFs or text documents, extracts and chunks the content, retrieves relevant source context, and generates intent-aware answers. I added PDF preview, a local study library, whole-document summaries, source traceability, answer evaluation, and exportable reports.

More technical version:

> The app uses pypdf for text extraction, pypdfium2 for PDF page preview, TF-IDF retrieval with scikit-learn, and modular Python components for chunking, retrieval, answer generation, insight extraction, evaluation, and export. I also implemented a question-intent router so broad questions like whole-document summaries are handled differently from specific source-grounded questions.

## 7. Known Limitations

- TF-IDF retrieval can miss semantically related chunks if wording differs.
- Rule-based answers are explainable but less flexible than LLM-generated answers.
- PDF diagrams may not be fully captured by text extraction.
- Mode-specific logic can be improved further for industrial, research, and business documents.

## 8. Next Version Ideas

- Add semantic embeddings and vector search.
- Add optional LLM generation with citations.
- Add multi-document subject libraries.
- Add flashcards and quiz mode.
- Add stronger industrial-quality and research-paper templates.
- Deploy the app so it can be used from a phone or laptop.
