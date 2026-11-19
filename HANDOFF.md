# Project handoff

Updated: 20 November 2026.

## Current state

The Gemini migration is complete. The user then requested removing unused and
redundant code and organizing all application Python under one backend folder.
Deployment planning remains deferred; this update is source cleanup and packaging.
The maintained application is React/TypeScript with FastAPI; Streamlit is retired.

The selected preset is Google `gemini-3.1-flash-lite` for question drafting,
answers, source verification and Ragas scoring, with minimal thinking and
concurrency one. Existing credentials and local data are retained. A live
one-question Quality preview on the handbook completed all five answer-quality
metrics with no failed samples after retrying a temporary answer HTTP 503.
This is a small workflow check, not judge calibration or a general accuracy claim.
The same model drafts, verifies and scores; reports label that limitation.

## Start and validate

From this directory:

```powershell
.\.venv\Scripts\python.exe -B -m backend
```

Open http://127.0.0.1:8501. Startup warms local embeddings and model clients
without making model inference requests. Restart Python after changing `.env`.

```powershell
.\.venv\Scripts\python.exe -B -m pytest tests -q -p no:cacheprovider
cd frontend
npm run build
```

Final backend validation passed **292 tests** in 46.66 seconds with five existing
dependency/deprecation warnings. The production React build passed. The frontend
build needed sandbox escalation because esbuild could not resolve the existing
Vite configuration through restricted parent directories. Regression tests are
offline and isolate provider caches; they do not establish live model quality.
Earlier desktop/mobile checks covered authentication, ownership, recovery and
multi-PDF flows. No live OIDC or Linux container validation is claimed.

The refactored app restarted successfully at http://127.0.0.1:8501. Health,
React serving, both selected Gemini models, all three saved documents, all five
saved reports and existing index chunks were checked without inference requests.
The OCR module self-test passed for its synthetic clean English image/PDF scan.

## Backend organization

Application Python lives in `backend/`; HTTP/auth/catalog services are in
`backend/api/`. Tests remain in `tests/` and React in `frontend/`. The launcher,
evaluation CLI and OCR CLI use `python -m backend`,
`python -m backend.advanced_evaluation` and `python -m backend.ocr_doctor`.
Docker and README use these commands; old root script/import paths are removed.

`backend/files.py` defines the repository root and shared atomic JSON read/write
helpers. Embedding/provider caches and default evaluation output retain their
original locations. API ingestion explicitly uses the supplied workspace root
for indexes and extraction caches, including single-PDF preparation. A regression
test uploads, prepares and reuses a real native PDF from a different working
directory with local test embeddings.

Removed unused ingestion demo/wrappers, the uncalled AI-summary ingestion option,
the Gemma-to-Google alias, unused evaluation-plan code and redundant imports.
Chunk conversion is now named `chunks_to_documents`; tests exercise actual index
and generation services directly. Duplicate JSON writers/readers are consolidated
and failed writes retain completed artifacts. Application Python is 83 lines
smaller overall. Existing provider, OCR, account, recovery and multi-PDF features
remain supported. Dependency pins are retained because the apparent unused
packages are required by Ragas/LangChain or supported provider adapters.

## Provider behavior

- Gemini 3.1 Flash-Lite uses the standard Google text API, not the Live API.
  The runtime configures minimal reasoning and suppresses thought output.
- Drafting retries structured HTTP 500/502/503/504 errors within its original
  deadline. `SYNTHETIC_MAX_RETRIES` defaults to one and is bounded to 0–2;
  `SYNTHETIC_RETRY_DELAY_SECONDS` defaults to one and is bounded to 0.1–5.
- Answers use the same transient status policy, with
  `GENERATION_MAX_RETRIES` and `GENERATION_RETRY_DELAY_SECONDS`. Retries happen
  only before answer text is emitted and receive the remaining request timeout.
  Partial streamed answers are retained as incomplete rather than replayed.
- Authentication, HTTP 429 quota errors, malformed references and truncated
  output are not treated as transient answer failures. Provider status codes
  are extracted from structured errors and their causes, not error prose.
- Google/OpenRouter judges validate complete JSON, share a bounded retry/repair
  budget, cache completed judging steps and retain measured provider usage.
  OpenRouter requires a fixed `:free` ID, zero-price routing and its own key.
- Ollama supports drafting, answers, verification and scoring with no cloud
  model credential. Each evaluation owns and closes its local transports.
  Decoding schemas omit string-length bounds that Ollama's grammar compiler
  can reject; full Pydantic validation and exact quote checks remain enforced.
  `gemma3:270m` and `gemma3:1b` receive text/tables without image inputs. The
  270M trial passed simple Q&A but failed source-quote drafting; model capacity
  still matters for successful evaluation. Imported source-checked questions
  can test retrieval, answers and judging without automatic drafting.

## Preserve data and credentials

Keep `.env`, installed `.venv`/frontend dependencies, the production frontend
build, original PDFs/assets, indexes, extraction caches, generated/imported
questions and evaluation reports. Runtime storage is in `docs/`, `db/`, `.cache/`,
`evaluation_datasets/` and `evaluation_runs/`; these are ignored by Git.
Never print or publish keys, put them in React/VITE variables or bake them into
images. Preserve the parent workspace's original CSV/manifest exports and
`RESULTS_REVIEW.md`, plus the useful `project_docs/` reference assets.
Remove only confirmed disposable test directories, helpers and obsolete logs.

## Implemented application and release boundary

Automatic questions require exact indexed source evidence and a passing source
verdict before scoring. Reports expose sample counts, failures/skips, evidence,
provider limits, latency and uncertainty; unavailable scores are not zero.
One-to-three-PDF workflows support separate or combined preparation/evaluation,
original-file/page provenance, source coverage and reports per selected scope.

SQLite stores owned resources, durable job settings/progress and explicit
restart/retry recovery. Duplicate retries share a replacement job; completed
artifacts and valid caches are reused. One application process is supported.

OIDC login uses authorization code/PKCE, one-use browser-bound state and nonce,
verified issuer/audience/subject/signatures and revocable hashed sessions.
Private API reads/writes and progress require the verified owner. Browser caches
and remembered jobs reset on account changes. `APP_AUTH_MODE=local` remains the
localhost development default; existing local resources are not assigned to the
first OIDC account. Invalid OIDC configuration stops startup.

The Dockerfile/Compose configuration keeps localhost binding, persistent local
stores, non-root execution, the pinned embedding revision and a health check.
Linux execution remains unverified. Follow `PROJECT_PLAN.md` after deployment
planning resumes: live OIDC/TLS registration, worker coordination and account
usage/spend limits, reviewed scan/table/chart/judge evidence, container and
backup/restore validation. Hosting subscription cost is intended to stay zero;
model usage, electricity and connectivity are separate costs.
