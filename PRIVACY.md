# Privacy Notes

This project runs locally by default. I designed it this way because uploaded study material, research notes, and workplace documents may contain information that should not be sent to an external service.

## What stays local

- Uploaded documents and extracted text
- Generated document previews
- Questions, optional corrections, and answer ratings
- The SQLite feedback database
- Exported reports

The application does not call a paid or hosted LLM API in the current version. Runtime documents, previews, databases, and exports are excluded from Git through `.gitignore`.

## Logging

The service records operational metadata such as document ID, filename, word count, chunk count, latency, and review status. It does not write document contents or full questions to application logs.

## Feedback

Feedback is stored in `.app_cache/feedback.db`. A rating may include the question and an optional correction, so users should avoid entering personal, confidential, or patient information. The database can be deleted locally when the feedback is no longer required.

## Limits

This is a student prototype, not a certified system for medical, legal, financial, or confidential production data. Anyone evaluating it with real users should use non-sensitive or anonymised documents and obtain consent before collecting feedback.
