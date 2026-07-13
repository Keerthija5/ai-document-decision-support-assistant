# AI-Powered Document Intelligence & Decision Support Assistant

I started this project because I wanted a better way to work through long lecture PDFs. Finding a sentence was not enough: I wanted to ask a question, see the source behind the answer, and know when the document did not contain enough evidence.

The project grew into a local document-analysis prototype for study material and workplace-style documents. It now has a Streamlit interface, a reusable Python service, a REST API, answer-quality checks, local feedback collection, a privacy-aware archive for metadata and evaluation traces, a regression benchmark, tests, Docker support, and CI.

After sharing the app with a few classmates, I noticed a real problem: many lecture PDFs are not clean text files. They often contain screenshots, scanned pages, diagrams, or slides where the important definition is inside an image. I added DOCX and image upload support, plus OCR fallback for screenshot-heavy PDFs, so the assistant can handle more of the study material students actually use.

I also added a small archive layer because I wanted the project to be reviewable after a session. The archive saves document metadata, retrieval traces, and evaluation results, but not the raw uploaded document text. That keeps the default setup safer for lecture notes or personal files while still giving enough information to debug and improve the system.

The current version deliberately uses transparent TF-IDF retrieval and extractive, template-guided answers. It does not call a hosted LLM API, so documents stay local and the answer path is easier to inspect.

## The Problem It Tries to Solve

Students often have several lecture files and limited revision time. A normal search can find matching words, while a general chatbot may answer confidently without showing whether the answer came from the uploaded notes.

This assistant is designed around three practical questions:

1. Can I find the relevant part of a document quickly?
2. Can I verify the answer against its source?
3. Will the system refuse a question when the document does not support it?

The same workflow can also be used to inspect AI use-case proposals, research notes, and business decision documents.

## What It Does

- Uploads PDF, DOCX, image, TXT, and Markdown documents with type, size, and readable-text validation.
- Uses OCR fallback for screenshots, scanned pages, and image-heavy PDF pages when normal text extraction is not enough.
- Previews PDF pages and extracted page text.
- Keeps a small local library of recently processed documents.
- Splits text into overlapping chunks and retrieves evidence with TF-IDF.
- Checks whether meaningful concepts from the question are supported by the document.
- Refuses unsupported questions instead of returning the nearest unrelated paragraph.
- Handles common study questions such as definitions, types/classifications, advantages, disadvantages, causes, effects, summaries, and exam-focused topic lists.
- Provides Study Notes, Industrial AI / Quality, Research Paper, and Business Decision modes.
- Extracts risks, requirements, recommendations, action items, and missing information.
- Evaluates relevance, completeness, grounding, consistency, and review risk.
- Stores helpful / not-helpful feedback and optional corrections in local SQLite.
- Stores privacy-aware document metadata and query-evaluation traces in a local archive.
- Supports optional S3-compatible archiving for metadata and evaluation records when cloud storage is explicitly configured.
- Exports JSON, CSV, and Markdown reports.
- Exposes ingestion, query, evaluation, insight, and feedback endpoints through FastAPI.

## Screenshots

### Upload and Study Library
![Upload and Study Library](assets/screenshots/upload-and-study-library.png)

### PDF Preview and Extracted Text
![PDF Preview and Extracted Text](assets/screenshots/pdf-preview-and-extracted-text.png)

### Source-Grounded Answer
![Source-Grounded Answer](assets/screenshots/source-grounded-answer.png)

### Unsupported-Question Refusal
![Unsupported-Question Refusal](assets/screenshots/unsupported-question-refusal.png)

### Evaluation Dashboard
![Evaluation Dashboard](assets/screenshots/evaluation-dashboard.png)

### Local User Feedback
![Local User Feedback](assets/screenshots/local-user-feedback-confirmation.png)

### FastAPI Documentation
![FastAPI Swagger Documentation](assets/screenshots/fastapi-swagger-documentation.png)

## How It Works

```text
Document
   |
Validation and text extraction
   |
Overlapping text chunks
   |
TF-IDF retrieval index
   |
Question support check
   |-------------------- unsupported -> refusal + review flag
   |
Intent-aware extractive answer
   |
Grounding and quality evaluation
   |
Sources, feedback, insights, archive records, and exports
```

The Streamlit app and API share the same configuration and core processing modules. Operational logs include IDs, counts, latency, and review status, but not document contents.

## Local Archive and Optional S3 Storage

By default, the assistant writes small JSON archive records under `.app_cache/archive/`. I added this so I can inspect what happened after a query without keeping a full private copy of the uploaded document.

The archive stores:

- document ID, file name, source type, word count, and chunk count
- insight counts such as number of risks, requirements, recommendations, and missing-information items
- query evaluation metrics, retrieval scores, source chunk IDs, latency, and review status

The archive does not store raw uploaded document text. Full question text is also disabled by default; only a question hash is stored. If I need full local debugging, I can turn it on with `RAG_ARCHIVE_QUERY_TEXT=true`.

For a cloud-style setup, the same metadata archive can be sent to S3:

```bash
pip install -r requirements-cloud.txt
export RAG_STORAGE_BACKEND=s3
export RAG_S3_BUCKET=your-bucket-name
export RAG_S3_PREFIX=document-intelligence
uvicorn api:app --reload
```

I kept this optional because S3 is useful for review records and exported results, but it should not be required for a classmate who only wants to upload a lecture PDF locally.

## Controlled Evaluation

I created a small golden dataset with 21 questions across three included sample documents:

- 18 answerable questions
- 3 intentionally unanswerable questions
- industrial visual inspection, cost engineering, and clinical documentation examples

The current regression run achieved:

| Check | Result |
|---|---:|
| Overall pass rate | 21/21 |
| Retrieval hit rate | 18/18 |
| Correct refusal rate | 3/3 |
| Mean expected-keyword coverage | 100% |
| Mean grounding score | 67.33/100 |

These numbers are useful for catching regressions in the included examples. They are not a claim of 100% accuracy on unseen documents or real users. The dataset and recorded result are available in `evaluation_data/`.

I also added regression tests after real feedback from classmates, especially around study questions like definitions and type lists. This helped catch cases where messy PDF extraction could produce broken labels or unsupported answers.

Run the benchmark with:

```bash
python run_evaluation.py
```

## REST API

Start the API:

```bash
uvicorn api:app --reload
```

Open `http://127.0.0.1:8000/docs` for the interactive endpoint documentation.

Main endpoints:

| Method | Endpoint | Purpose |
|---|---|---|
| `GET` | `/health` | Service and retrieval status |
| `POST` | `/documents/text` | Ingest pasted text |
| `POST` | `/documents/upload` | Upload PDF, DOCX, image, TXT, or Markdown |
| `GET` | `/documents/{id}/insights` | Structured decision insights |
| `POST` | `/query` | Answer with sources and support evidence |
| `POST` | `/evaluate` | Return answer-quality metrics |
| `POST` | `/feedback` | Store a local rating or correction |
| `GET` | `/feedback/summary` | Summarise collected feedback |

Documents are held in memory for the current API process. This keeps the prototype simple, but it means document IDs do not survive an API restart.

The health endpoint also reports the active archive backend and whether S3 is configured.

## Run the Streamlit App

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Tests

```bash
pytest -q
```

The tests cover input validation, grounded queries, unsupported-question refusal, feedback storage, the evaluation dataset, and the API workflow. GitHub Actions runs compilation and tests after pushes and pull requests.

## Docker

The included Dockerfile runs the API:

```bash
docker build -t document-intelligence-api .
docker run --rm -p 8000:8000 document-intelligence-api
```

## Project Structure

```text
app.py                         Streamlit interface
api.py                         FastAPI endpoints
run_evaluation.py              Golden-set benchmark runner
evaluation_data/               Questions and recorded benchmark result
data/sample_documents/         Non-sensitive evaluation documents
tests/                         Unit and API tests
src/
  config.py                    Environment-based settings
  document_loader.py           PDF/DOCX/image/TXT/Markdown extraction and OCR fallback
  text_chunker.py              Overlapping chunk creation
  retriever.py                 TF-IDF retrieval
  rag_assistant.py             Intent-aware extractive answers
  service.py                   Reusable document workflow
  validation.py                Input and question-support checks
  evaluator.py                 Answer-quality checks
  insight_extractor.py         Structured decision fields
  feedback_store.py            Local SQLite feedback
  storage.py                   Local/S3 metadata archive for review traces
  logging_config.py            Privacy-aware operational logging
  exporter.py                  JSON/CSV/Markdown reports
```

## Planned User Study

The next step is a small study with 5–10 students using non-sensitive lecture PDFs. I want to measure task time, source retrieval, unsupported-question refusal, helpfulness, corrections, and confidence after source verification.

The complete plan and five-question survey are in [USER_STUDY.md](USER_STUDY.md). Results will only be added after the study is actually completed.

## Privacy

Uploaded documents, previews, feedback, and exports remain local in the default setup. Runtime files and databases are excluded from Git. More detail is available in [PRIVACY.md](PRIVACY.md).

## Current Limitations

- TF-IDF does not understand meaning as deeply as embedding-based retrieval.
- Answers are extractive and template-guided, not generated by an LLM.
- The support check is lexical and can miss paraphrases.
- OCR can read many screenshot-based slides and scanned pages, but complex charts, diagrams, handwriting, and low-quality images may still need manual review.
- The benchmark is small and based on included sample documents.
- The API uses in-memory document storage and is not a production deployment.

## Next Steps

- Run the planned student study and report measured results.
- Add dense retrieval and compare it against the TF-IDF baseline.
- Add optional LLM generation with citations and strict fallback behavior.
- Test a larger, more varied golden dataset.
- Expand the metadata archive into a small review dashboard for feedback and failure analysis.
