# Project handoff

Last updated: 18 November 2026, after repository and uploaded-data cleanup.

## Latest user request

Leave the provider/quota issue for now. Clean the workspace so only the project
folder remains, including removal of PDFs and databases. Do not resume quota
feature work unless requested.

## Current state

- The workspace contains only `RAG/`; useful reference materials are in
  `project_docs/` (`Metrics.png`, `MultiQuery_vector.png`, and the handbook DOCX).
- Removed uploaded PDFs, Chroma/SQLite databases, extraction caches, evaluation
  datasets/results/history, temporary QA scripts/screenshots, pytest output and
  bytecode caches. A fresh document upload and preparation are required.
- Removed the retired Streamlit interface: `app.py`, `pages/`, `.streamlit/`,
  `run_streamlit.py`, and `requirements-streamlit.txt`.
- Preserved React/TypeScript, FastAPI, Python RAG/OCR/evaluation services, tests,
  deployment configuration, `.git`, `.env`, `.venv`, and frontend dependencies.
- The app is stopped. Runtime storage directories recreate automatically when
  the application is used. Do not start it solely to verify this clean data state.
- Existing uncommitted changes are preserved; do not reset the repository.

## Run and configuration

From `RAG/`:

```powershell
.\.venv\Scripts\python.exe -B run_app.py
```

Open the printed URL (default `http://127.0.0.1:8501`). Local model initialization
can take time; startup makes no inference requests. Upload and prepare a PDF,
then use **Generate questions & evaluate** in the Evaluation view.

Rebuild after frontend changes:

```powershell
cd frontend
npm run build
cd ..
```

Keys remain only in `RAG/.env`. Never print them or copy them into frontend code,
Git, logs, screenshots or handoff notes. The current generator/drafter is Google
`gemini-3.5-flash-lite`; verification and Ragas judging use OpenRouter
`inclusionai/ling-3.2-flash:free`, with `EVALUATION_CONCURRENCY=1`.

## Evaluation behavior and provider issue

The system generates source-linked questions and references, validates quoted
source evidence, and uses an independent judge to verify references before
scoring. Manual dataset import is optional. Full comparison evaluates Keyword,
Vector, Hybrid and Reranking against identical cases.

Report faithfulness, answer relevancy, context precision, context recall and
factual correctness, plus retrieval/citation diagnostics, timing and explicit
valid/failed/skipped counts. Never fabricate missing scores or infer retrieval
quality from provider errors. Synthetic references are not human ground truth.

The user's mostly empty answer-quality comparison was traced to OpenRouter's
free-route request quota: 110 judge requests, 16 responses and 94 HTTP 429
rejections. The quota work was canceled before implementation. Large Ragas runs
can require multiple calls per metric, question and strategy. Earlier Google
Gemma verification also encountered provider 503/504 errors.

Existing useful fixes remain: bounded timeouts/retries and JSON validation,
separate judge HTTP-client lifetimes, provider-key redaction, and page-citation
parsing for `p.`/`pp.`. Previous live reports and correction backups were cleared
as requested; historical timings in README describe earlier checks.

## Verification and remaining work

Post-cleanup verification: all 174 backend tests passed in 69.08 seconds
(five existing third-party warnings); the React production build passed.
Temporary test data was removed afterward. No live model requests were needed.
`git diff --check` passed. The app remains stopped with no uploaded PDFs or
runtime databases in the project.

Public authentication/accounts, durable in-flight job recovery, Linux container
execution, representative complex scan/chart validation and judge calibration
remain future work. Browser cookies currently identify local workspaces;
keep the localhost deployment default. Zero hosting subscription is the user's
constraint; API usage is separate.
