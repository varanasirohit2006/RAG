# RAG project roadmap

Updated: 20 November 2026. See README for operation and HANDOFF for continuation.

## Agreed scope

- React/TypeScript with a FastAPI/Python backend for English PDF document Q&A,
  source inspection and automatic RAG evaluation.
- Native text, scanned pages, tables and charts, with extraction/grounding limits
  visible rather than overstated accuracy claims.
- Private accounts, owned documents/results, durable jobs and reproducible reports.
- One computer/server initially; zero hosting subscription cost. Model API usage,
  electricity, storage and connectivity are separate costs.
- The current user-selected model is Google `gemini-3.1-flash-lite` for drafting,
  answers, source checking and judging. Same-model checks are labelled as
  non-independent; configuration supports a separately selected judge.

## Implemented

| Area | Current behavior |
| --- | --- |
| Interface | React navigation, document library, cited answers, source explorer, retrieval lab, evaluation setup and results/export views. Streamlit is retired. |
| Extraction | Native text and tables first; bundled CPU RapidOCR on scans; PDFium visual evidence; bounded pages, upload size and rendering resources. OCR diagnostic/transcription scoring is available. |
| Retrieval | Stable chunk IDs, persistent extraction/index reuse, keyword/vector/hybrid retrieval, actual query fusion and optional bounded Cohere reranking. |
| Evaluation | Automatic source-linked questions and references, exact quote validation, production-path answers, five answer-quality metrics, retrieval/citation diagnostics and explicit unavailable/failure counts. |
| Provider handling | Google, fixed free OpenRouter routes and optional Ollama generation/verification/scoring; bounded retries, remaining-time answer retries before streaming text, schema validation, separate request/metric deadlines and reusable judging-step caches. |
| Multiple PDFs | Upload up to three originals, separate or combined preparation/evaluation, per-original reports or one combined report, original boundaries/page links and source coverage. |
| Recovery | Durable job settings/progress, explicit retry after restart, deduplicated replacement jobs, completed-artifact recovery and valid stage-cache reuse. One process is supported. |
| Accounts | Configurable OIDC login/logout with PKCE and signature/claim validation, issuer/subject identities, durable expiring/revocable sessions and private API ownership. Browser caches/jobs reset on account changes. |
| Packaging | React/Python Linux Dockerfile, localhost-bound Compose, persistent volumes, non-root execution, pinned embedding revision and API health check. Linux execution remains unverified. |

Final regression checks and the production React build are recorded in HANDOFF.
Earlier checks also covered desktop/mobile browser flows. A live one-question
Gemini 3.1 Flash-Lite Quality preview completed all five answer-quality metrics
after retrying a temporary HTTP 503 during answer generation. This proves a
small workflow can complete; representative evidence and judge calibration
remain release work. Deployment planning is deferred at the user's request.
Existing local documents/results and credentials are retained and ignored by Git.

## 1. Finish account deployment setup

Register a managed OpenID Connect client and configure issuer, client ID, token
authentication and the exact callback URI. Serve React and the API at one HTTPS
origin; retain loopback HTTP only for development. Local browser-workspace mode
must not become a public anonymous fallback. Confirm login/logout, expiry,
returning-user ownership and access denial using the real configured provider.
Existing local resources remain local unless deliberately migrated.

Acceptance: two signed-in users cannot discover or retrieve each other's
PDFs, assets, answers, datasets, jobs, retries, exports or cached results.
Accounts survive browser/server restarts. Invalid configuration prevents access.

## 2. Coordinate shared workers and account usage

Replace single-process job assumptions with durable scheduling, worker leases,
idempotent stage commits and ownership checks in workers. PostgreSQL/pgvector
and Redis are candidates for shared deployment; preserve original PDFs/assets
on backed-up storage and keep PostgreSQL as the source of job/result truth.
Apply transaction-scoped identity and row-level security when migrating.
Keep keyword BM25 and any PostgreSQL full-text strategy separately labelled.

Add per-account request/upload/evaluation quotas and explicit usage/spend limits,
with a reviewable estimate before large paid evaluations. Retain cancellation,
provider-blocked states and cache reuse without replaying completed paid stages.

Acceptance: worker interruption/retry does not lose completed work or duplicate
committed operations; accounts stay isolated; ordinary Q&A remains responsive
while background work runs. Cache invalidation and ownership survive migration.

## 3. Validate representative evidence and judges

Build a reviewed corpus of approximately 100–200 diverse questions across held-out
PDFs. Include paraphrases, multi-page/cross-PDF cases, numbers/tables/charts,
distractors and genuinely unanswerable cases. Review alternate relevant passages
and reference completeness. Keep tuning and held-out test documents separate.

Measure OCR character/word error rate against verified transcriptions, table/cell
and chart-value errors against reviewed values, and answer/citation grounding
against the exact evidence provided to generation. Calibrate an independently
configured judge or human audit and inspect disagreements/repeated-score variation.
Use paired strategy comparisons on the same completed questions.

Acceptance: every reference has source evidence; reports expose missing scores,
synthetic-label limits and uncertainty; failures are retained as failures.
Six synthetic questions or a few faithfulness scores cannot establish general
system accuracy. Never invent scores or an unsupported best-strategy claim.

## 4. Verify containers, dependencies and operations

Run native/scanned/mixed/table/chart fixtures inside the Linux image as well as
Windows. Validate the supported Python 3.12/3.13 environments and lock transitive
dependencies after clean builds. Test cold/warm ingestion and Q&A, representative
concurrency, peak memory, provider failures, restart/retry and cancellation.

Exercise database migrations, backups/restoration, rollback, document/account
deletion and retention. Add CI for meaningful schema, ownership, extraction,
recovery and container checks. Keep live/paid provider tests explicitly opt-in.
Set measured latency/memory/quality targets for the declared document sizes and
hardware rather than promising generic response times.

Acceptance: a clean machine starts from documented Compose commands; workflows
and restored data work after restart; no public-release claim precedes these checks.

## 5. Publish a public endpoint only after readiness

Select and validate a zero-subscription hosting/tunnel arrangement on the existing
computer, including domain/TLS, persistence, resource limits, identity configuration
and an API spend limit. Expose only the HTTPS entry point and keep data services
internal. Record the deployed revision and health checks; provide a rollback path.
A GitHub source push alone does not deploy the app or create paid infrastructure.

## Reference documentation

- [Ragas metrics](https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/)
- [DeepEval G-Eval](https://deepeval.com/docs/metrics-introduction)
- [Unstructured extraction strategies](https://docs.unstructured.io/open-source/core-functionality/partitioning)
- [Tesseract data and installation](https://tesseract-ocr.github.io/tessdoc/Installation.html)
- [OpenID Connect token validation](https://openid.net/specs/openid-connect-core-1_0.html#IDTokenValidation)
- [PyJWT claim validation](https://pyjwt.readthedocs.io/en/stable/usage.html)
