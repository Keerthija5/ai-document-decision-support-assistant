# AI-Powered Document Intelligence & Decision Support Assistant

A Streamlit-based document assistant for PDF/TXT/Markdown files. It combines document upload, PDF page preview, extracted-text navigation, source-grounded question answering, study summaries, decision-support insights, answer evaluation, and exportable reports.

This project is designed as a practical AI workflow rather than a notebook-only demo. Version 1 focuses on a reliable local prototype with explainable retrieval and transparent evaluation.

## What It Does

- Upload PDF, TXT, or Markdown documents.
- Preview uploaded PDF pages as rendered images inside the app.
- View extracted text page by page for copying and verification.
- Keep a local study library of recent documents.
- Ask document-grounded questions using retrieval over text chunks.
- Route questions by intent, such as brief summary, whole summary, study topics, page navigation, risks, recommendations, and evaluation.
- Use analysis modes for Study Notes, Industrial AI / Quality, Research Paper, and Business Decision workflows.
- Generate structured decision insights, missing-information checks, and a decision matrix.
- Evaluate generated answers for relevance, completeness, grounding, consistency, and hallucination risk.
- Export reports as JSON, CSV, or Markdown.

## Why This Project

Many simple RAG projects stop at “chat with a PDF.” This project adds practical layers that are useful in study and workplace settings:

- source traceability through retrieved chunks
- document overview, PDF page preview, and page-level extracted text
- intent-aware answer behaviour
- study summaries and exam-oriented explanations
- decision-support outputs for business or industrial use cases
- answer evaluation and human-review flags
- exportable reports

## Analysis Modes

The app includes four modes. The mode guides the answer style, while the user can still type any question.

- **Study Notes:** summaries, whole-document explanations, important topics, beginner explanations, page navigation, and exam preparation.
- **Industrial AI / Quality:** risks, data requirements, KPIs, pilot readiness, validation, and quality-review points.
- **Research Paper:** problem, method, dataset, experiments, results, limitations, and future work.
- **Business Decision:** stakeholders, inputs/outputs, benefits, risks, dependencies, and next actions.

## Workflow

```text
Upload document
      |
Extract text from PDF/TXT/MD
      |
Create overlapping text chunks
      |
Build TF-IDF retrieval index
      |
Classify user question intent
      |
Retrieve relevant source chunks
      |
Generate mode-aware answer
      |
Evaluate answer quality
      |
Show sources, dashboard, and exports
```

## Tech Stack

- Python
- Streamlit
- scikit-learn TF-IDF retrieval
- Pandas
- NumPy
- pypdf for text extraction
- pypdfium2 for PDF page preview

The first version is intentionally local and explainable. It does not require a paid LLM API. Future versions can add embeddings, vector databases, and API-based LLM generation.

## Project Structure

```text
app.py
requirements.txt
.streamlit/config.toml
data/sample_documents/
src/
  document_loader.py
  text_chunker.py
  retriever.py
  rag_assistant.py
  insight_extractor.py
  evaluator.py
  exporter.py
```

Runtime folders such as `.app_cache/`, `static/pdf_cache/`, and `outputs/` are ignored by Git because they may contain uploaded private documents or generated files.

## Run Locally

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Example Questions

Study:

- Give me a brief summary of the whole PDF.
- Give me the whole summary of the PDF.
- What are the important topics I have to cover?
- Explain this topic like I am a beginner.
- Which page contains this topic?

Industrial AI / Quality:

- What are the key risks and recommended next steps?
- What data is required for this use case?
- How should this prototype be evaluated?
- Which information is missing before a pilot?

Research:

- What is the problem, method, result, and limitation?
- What datasets or experiments are mentioned?
- Which claims need stronger evidence?

Business Decision:

- What decision is being supported?
- What are the benefits, risks, and dependencies?
- What should be validated before implementation?

## Evaluation Logic

The app includes a transparent evaluation layer:

- **Relevance:** overlap between the question and answer.
- **Completeness:** checks whether expected information is missing.
- **Grounding:** overlap between the answer and retrieved source chunks.
- **Consistency:** basic output-quality check.
- **Hallucination risk:** derived from grounding and average quality.

This is not a replacement for expert review. It is a first-level quality signal that helps decide when human review is needed.

## Current Limitations

- Retrieval uses TF-IDF, not dense semantic embeddings.
- Answer generation is rule-based and template-guided in Version 1.
- PDF extraction depends on the quality of the source PDF.
- Image-only diagrams may not be fully understood; the app shows a diagram/manual-review note where relevant.
- Domain-specific modes can be improved further with stronger templates and evaluation logic.

## Future Improvements

- Add embedding-based retrieval with FAISS or Chroma.
- Add optional LLM generation with prompt templates and citations.
- Add multi-document search across a full subject folder.
- Add flashcards, quiz generation, and exam mode.
- Add better research-paper and industrial-quality templates.
- Add user accounts or deployment for phone/laptop access.
- Export polished PDF reports.

## Resume Bullet

Built a Streamlit-based document intelligence assistant that processes PDF/TXT/Markdown files, extracts and chunks document text, retrieves source-grounded context, and generates intent-aware answers across study, research, quality, and business-decision modes. Added PDF page preview, local study library, whole-document summaries, page-level extracted text navigation, answer evaluation, hallucination-risk indicators, and exportable JSON/CSV/Markdown reports.
