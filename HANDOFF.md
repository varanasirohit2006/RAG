# Project handoff

Updated: 19 November 2026.

## Current work

The user requested removal of unnecessary project files and publication of the
pending project changes to GitHub. Disposable browser QA output is removed;
historical handoff/roadmap notes are condensed. Preserve application sources,
tests, deployment configuration, reference documents and existing local data.
The maintained app is React/TypeScript with FastAPI. Streamlit is retired.

## Start and validate

From this directory:

```powershell
.\.venv\Scripts\python.exe -B run_app.py
```

The default URL is http://127.0.0.1:8501. Startup warms local search dependencies
without making model inference requests. The main server was stopped at the last
health check; temporary browser fixture servers were stopped after validation.

```powershell
.\.venv\Scripts\python.exe -B -m pytest tests -q -p no:cacheprovider
cd frontend
npm run build
```

Cleanup/publication verification passed all **249 tests** in 108.31 seconds
with five existing dependency warnings; the production React build passed.
The earlier final authentication suite passed 32 tests. Offline Edge desktop/mobile
flows passed, including login/logout, account isolation, expiry, saved-work
recovery and multi-PDF evaluation. No paid model calls ran for those checks.
Windows sandbox restrictions may require approval for the test temporary
directory and frontend dependency resolution. Existing dependency warnings
are documented by the test output; offline tests do not calibrate model quality.

## Configuration and data

- Keep the backend `.env` and existing credentials private. Never print secrets,
  put them in React/VITE variables, stage them in Git or bake them into images.
- The current user-selected generator, question drafter, source verifier and judge
  use Google `gemini-3.1-flash-lite`. Same-model verification is deliberately
  labelled as non-independent. Evaluation concurrency is two; individual judge
  requests have 90-second timeouts and complete metrics have 360-second deadlines.
- Existing PDFs, indexes, extraction caches, datasets and results are in `docs/`,
  `db/`, `.cache/`, `evaluation_datasets/` and `evaluation_runs/`. Preserve them.
  These directories, `.env`, installed dependencies and build output are ignored.
- The parent workspace also contains the user's original CSV/manifest exports
  and `RESULTS_REVIEW.md`. Preserve those user files outside the Git repository.
  The review verified provenance and aggregate arithmetic, but the run was partial:
  only four of 24 samples and 31 of 120 answer-quality scores completed. Retrieval
  covered all six questions per strategy; incomplete quality scores cannot fairly
  rank strategies. Provider 429/503 errors do not establish missing billing.
- `project_docs/` contains the original metric images and evaluation handbook;
  these are useful project references. Keep `.venv` and frontend dependencies
  available for local development. Keep the production frontend build for startup.

## Implemented behavior

Automatic evaluation drafts source-linked questions, validates exact supporting
quotes and checks references before scoring through the production RAG pipeline.
Reports retain real per-sample evidence, valid/failed/skipped counts, provider
errors and latency. Missing scores remain unavailable. No automatic overall
winner is invented. Structured judging steps and completed answers are cached;
strategy scheduling alternates to reduce provider-quota bias.

One-to-three-PDF upload supports separate preparation or one combined workspace.
Batch evaluation saves a dataset/report per original or one combined report.
Combined chunks preserve original-file boundaries, page provenance and source
coverage; source links show combined and original pages. Batch inputs persist
and use the explicit restart-recovery path. Ownership checks cover all files.

The durable SQLite catalog records job settings, progress and completed results.
Startup/shutdown mark interrupted work for explicit retry. Duplicate retries share
one replacement job; completed artifacts and valid caches are reused. Cancelled
jobs remain cancelled. Recovery currently supports one application process;
unfinished provider requests may need to run again and use credits.

OIDC authorization-code login uses PKCE S256, browser-bound one-use state, nonce
and verified issuer/subject identities. RS256/ES256 signatures, audience,
authorized party and timestamps are checked; trusted JWKS keys refresh on
rotation. Opaque session cookies are HttpOnly; only their hashes are stored.
Sessions persist across restart, expire and revoke on logout. Progress streams
stop after expiry/revocation. Accounts retain their IDs across name/email changes.
Private documents, answers, datasets, jobs, retries, assets and exports require
the verified owner. React caches and remembered jobs reset on account changes.

`APP_AUTH_MODE=local` remains the default. OIDC requires provider registration,
`APP_BASE_URL`, `OIDC_ISSUER`, `OIDC_CLIENT_ID` and a suitable token authentication
method/secret. Register the exact `APP_BASE_URL/api/auth/callback` redirect URI.
See README and `.env.example`. Invalid OIDC configuration stops startup.
Public URLs require HTTPS; loopback HTTP is supported for development.
Existing local resources are not automatically assigned to the first account.
No live identity-provider configuration was supplied, and `.env` was not changed.

## Remaining release work

Follow `PROJECT_PLAN.md`: live OIDC/TLS setup, shared-worker coordination,
per-account usage/spend limits, Linux container execution and dependency locking,
load/backup/deletion validation, and reviewed scan/chart/judge calibration.
The hosting constraint is zero subscription cost; API usage is separate. Keep
localhost deployment defaults until the public release requirements are met.
