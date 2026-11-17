# Plan for a reliable public RAG evaluation app

Prepared 16 November 2026 after reviewing all six Python files, the README,
dependency manifest, local OCR tooling, installed evaluation library, and the
two screenshots supplied in the workspace (now in `project_docs/`).

## Current cleanup checkpoint — 18 November 2026

React/TypeScript and FastAPI are the maintained interface. The workspace now
contains only `RAG/`. Uploaded PDFs, databases/indexes, extraction caches,
evaluation datasets/history and temporary QA artifacts have been cleared.
The retired Streamlit interface and launcher have been removed. Source, tests,
configuration, `.env`, installed dependencies and useful reference documents
remain. Upload and prepare a new PDF before evaluating. The app is stopped.

The user deferred API quota improvements and requested this cleanup instead.
The configured generator is Gemini 3.5 Flash Lite; the independent judge is
OpenRouter `inclusionai/ling-3.2-flash:free` at concurrency one. Earlier missing
quality scores were traced to the free-route request quota. Provider failures
must remain visible. Earlier implementation notes and measurements below are
historical; follow `HANDOFF.md` for the current state.

## Agreed scope

- A public web app with individual accounts and private documents/results.
- English PDFs containing native text, scanned pages, tables, and charts.
- Accurate evaluation is a core product feature.
- Next frontend: React with TypeScript. Complete repository cleanup first,
  then migrate the interface while preserving the Python RAG services.
- Fix local Windows OCR and provide a reproducible Linux container setup.
- Keep deployment simple: Docker Compose on one server for the first release.
- Answer generation uses `gemini-3.5-flash-lite` following the latest request;
  evaluation retains `gemma-4-31b-it`. Their configuration remains separate,
  with reviewed references and human calibration needed to assess score quality.
- Zero hosting subscription: run on the existing computer. API usage is
  separate. A free public service with accounts has not yet been established.
- Establish the plan first. This document does not mean the fixes are implemented.

## Implementation progress

The latest evaluation requirement is automatic question generation and scoring;
manual question import is optional. The React/API flow generates source-linked
cases, independently verifies references, and compares Keyword, Vector, Hybrid
and Reranking using the four screenshot metrics plus factual correctness and
additional diagnostics. Missing metrics, failure counts, synthetic limitations
and provenance remain explicit. Multiple-agent review fixed quote validation,
duplicate generation, saved-set provenance and citations to omitted context.

The latest full deterministic suite passed 135 tests. Build and browser checks
passed. A one-question live check received Google 503 model-demand errors during
drafting/verification and produced no quality scores. Current live Gemma judging
remains unverified. See HANDOFF.md for provider-choice follow-up and next work.

The following OCR work was implemented and verified on the local Windows
Python 3.13 environment after the initial audit:

- Native-text-first PDF extraction, native ruled-table HTML, visual page evidence,
  and bundled CPU RapidOCR 3.10.0 for scans. Tesseract/Poppler are optional.
- Bounded rendering, file/page limits, lazy shared model loading, explicit
  high-resolution mode, and an OCR diagnostic/reference-scoring command.
- Content/configuration-based upload caching, safe content-based filenames,
  session index reuse, and stable vector IDs to avoid repeated upsert duplicates.
- Removed eager evaluation initialization from ordinary dashboard startup and
  replaced the simulated query-expansion trace with actual execution stages.
- Nineteen extraction/vector tests passed, including a mixed native/scanned PDF,
  preservation of visual evidence through chunking, and real Chroma reingestion
  without duplicate records.
- The known clean English image and scanned-PDF self-test returned zero character
  and word errors. This is fixture validation, not a real-document quality claim.
- Streamlit's application test started the dashboard without runtime exceptions.

Original audit findings below describe the pre-change state. At the OCR
checkpoint, evaluation integrity, account isolation, background jobs, and public
deployment remained open; the subsequent performance work is described below.
The local test virtual environment reuses existing system packages; clean locked
environment/container validation is still part of the release work.

The subsequent performance work adds persistent extraction/index reuse, bounded
generation with streaming, optional bounded Cohere reranking, and cached
retrieval. PDF preparation, answers, and evaluation run through session-owned
background jobs with progress and cancellation. Serial cloud summarization is
disabled during ordinary indexing; raw text/table evidence and image assets are
retained. Evaluation now imports source-linked datasets, uses a separately
configured Gemma judge, records per-case outcomes, and distinguishes a limited preview,
full comparison, and retrieval-only measurements. Account isolation and public
deployment remain release requirements.

The supplied PDF for performance diagnosis is a 101-page RAG evaluation handbook,
with native text, one table, and no scans/images. Its earlier fixed benchmark
references to thermal efficiency and unrelated sections are not valid references
for this handbook. A chunking bug reconstructed a full original page for each
split slice, duplicating indexed evidence. The final pipeline uses the sliced
chunk text and preserves original element metadata for assets and provenance.
The rebuilt index contains 156 records and 155 distinct passage texts.

Measured local document preparation after the chunking fix: the one-time index
rebuild took 53.655 seconds plus 1.949 seconds to load cached extraction. A repeat
run took 0.008 seconds for cached extraction and 0.182 seconds for index reuse.
The earlier uncached PDF extraction took 9.454 seconds. These measurements do
not include model answer generation and do not establish a general latency SLA.

Final validation: 74 regression tests passed in 22.07 seconds, including OCR,
chunk coverage, persistent index reuse, streaming/deadline handling, evaluation
cache invalidation after embedding changes, metric validation, and cancellation. All 21 application/page/
test Python files passed syntax parsing, and `git diff --check` found no whitespace
errors. The local environment still emitted three third-party dependency and
deprecation warnings; the clean Linux container remains unverified.
Streamlit file watching is disabled in project configuration and the container
command, avoiding repeated traversal of the large Transformers package tree.

Final browser checks confirmed question typing during preparation is retained
without issuing a generation request, mismatched evaluation datasets disable Run,
valid datasets enable Run, and a one-question retrieval evaluation across three
local strategies completed in 6.08 seconds with no failed/skipped samples. The
390-by-844 viewport had no horizontal overflow or console/page exceptions.
Browser checks used Playwright and DOM assertions; screenshots were not visually
inspected in this environment.

Two bounded live Gemma generation requests returned provider 503 failures: the
single-question preview finished as failed in 11.525 seconds, and the UI surfaced
its failure in 66.181 seconds. The preview's local retrieval took 0.037 seconds
and Cohere reranking 1.910 seconds. No Gemma judge requests occurred and no valid
answer-quality scores were produced. These are failure timings, not successful
answer latency. No further live calls or preview replay followed those failures.
Deterministic tests verified completed-result replay makes no new model calls.

## What the audit established

Latest model/speed verification: generation was switched to
`gemini-3.5-flash-lite`, with minimal reasoning and its supported sampling
configuration; Gemma remains the evaluation judge. A standalone production
answer took 7.149 seconds and correctly named the three measurement layers with
page 2 citations. Browser startup exposed a pandas native-import error and an
index-loading stall. The new `run_app.py` initializes native dependencies, the
local search runtime, and configured provider clients before accepting sessions.
The final browser answer took 19.844 seconds, compared with 119.355 seconds
before client initialization moved to the launcher. Document preparation took
3.589 seconds. Startup itself took 31.23 + 3.44 seconds without any model or
reranking requests. These are individual timings, not a general latency SLA;
source correctness was reviewed for one question, and no judge evaluation ran.
Detailed measurements are in `test-results/gemini-flash-lite-ui-speed.json`.
After these changes, all 75 regression tests passed in 17.54 seconds; all 22
application/page/test Python files passed syntax parsing. The three existing
third-party warnings remain visible. Docker's launcher was updated to use
`run_app.py`; Linux image build/runtime validation remains pending.

| Finding | Evidence | Consequence |
| --- | --- | --- |
| Invalid evaluation value in the supplied image | `project_docs/Metrics.png`: reranking answer relevancy is `5.001232` | This value cannot be accepted as a valid relevancy score or used to rank strategies. |
| Retrieval summary does not match displayed results | `project_docs/MultiQuery_vector.png`: displayed labels cover eight unique chunks, summary says nine; displayed score mean is approximately `0.795887`, summary says `0.7912` | Recompute summaries from stored records; screenshots alone cannot establish which stage introduced the discrepancy. |
| Screenshots refer to different entry points | Images run `run_evaluation.py` and `vector_search_multi_query.py`; these files are absent here | Obtain original result records and scripts where available before claiming to reproduce those runs. |
| Benchmark is tied to one document | `advanced_evaluation.py:150`: six fixed questions reference particular sections and `rag_evaluation_100_pages.pdf` | They are not valid ground truth for arbitrary uploaded PDFs. |
| Very small benchmark | Six questions per strategy, including two unanswerable questions | Averages are unstable; report answerable and unanswerable cases separately. |
| Evaluation differs from the user-facing pipeline | Evaluator generates from `page_content`; main app sends original text, tables, and images | Current results do not measure the exact answers users receive. |
| Generator and judge use the same configured model | `advanced_evaluation.py:73`; the separate `EVALUATION_MODEL` constant is unused | Configure an independent judge and calibrate it against reviewed examples. |
| Costly work repeats during questions | `app.py:92` rebuilds summaries/index; `rag_pipeline.py:139` creates the embedding model again | Slow questions, repeated API spending, and possible duplicate chunks. |
| Document identity is based on filename | `app.py:95`; ingestion cache is based on a file path | Same-name uploads can collide or reuse stale data. |
| Shared storage and results | Shared `docs/`, selectable global `db/` directories, one metrics CSV | Users could access each other's data; concurrent runs can overwrite reports. |
| Multi-query results do not influence one answer path | Hybrid page calculates `fused_results` but reranks `hybrid_results` | The displayed fusion step is not the strategy actually producing that answer. |
| Main query expansion trace is simulated | `app.py:43` | Execution traces must describe operations that actually ran. |
| OCR executable is not discoverable | No Tesseract command, executable in either default Program Files path, or matching standard uninstall entry was found | Missing or undiscovered Tesseract is a likely local blocker; inspect the actual OCR error during implementation. |
| OCR path is Windows-specific | `rag_pipeline.py:29` hardcodes `C:\Program Files\Tesseract-OCR\tesseract.exe` | Container OCR will fail without platform-aware configuration. |
| Poppler is available | `pdfinfo` and `pdftoppm` report version `25.07.0` | Do not reinstall working Poppler without evidence of a problem. |
| Environment is not reproducible | Unpinned dependencies; `langchain_ollama` is imported but not declared; no project virtual environment found | A clean installation can differ from the existing global environment. |
| Startup does unnecessary initialization | `app.py` imports evaluation; evaluation constructs models at import time | Regular app startup can trigger model loading or credential errors. |
| Failures can look successful | Broad exception handling, global warning suppression, unpropagated thread failures, means that skip missing values | Surface failed/degraded runs and retain failure counts. |
| No release infrastructure found | No tests, CI, container files, authentication, or migrations in this checkout | These must be added before a public release. |

All six Python files pass AST syntax parsing. The earlier malformed `python -c`
command was a diagnostic-command quoting problem, not a syntax error in these
source files. This audit did not run a live OCR extraction or paid evaluation.
No PDFs or raw evaluation CSV/JSON records were supplied with the screenshots.
Docker's client is installed, but its engine was not reachable in this session;
container execution remains unverified. Global Python is `3.13.2`, Ragas is
`0.4.3`, and the installed LangChain evaluation wrappers contain deprecation
warnings that the application currently suppresses.

## Cleanup checkpoint before React migration

Removed 157 generated or superseded files: pytest temporary results, bytecode
and pytest caches, and five obsolete diagnostic reports. Removed unused direct
dependencies `langchain-cohere`, `plotly`, and `tabulate`; the live reranker still
uses the retained `cohere` SDK. Added frontend build/dependency output and secret
file exclusions to Git and Docker ignore rules.

The source documents, supplied screenshots and datasets, `.env`, virtual
environment, extraction/model caches, prepared indexes, evaluation history,
latest benchmark evidence and all test source files remain available. The
current Streamlit app, its advanced retrieval pages, launcher and configuration
remain useful migration references and keep the application runnable.

Cleanup verification: all 75 existing tests passed in 16.88 seconds with no
model API calls. The three existing third-party warnings remain visible.

## Recommended first-release design

Build a React/TypeScript interface with a FastAPI adapter around the existing
Python service layer. Port document upload, progress/cancellation, cited answers,
chunk exploration, evaluation dataset validation, metrics/history and exports.
Initialize shared models and provider clients during API startup, carrying over
the proven launcher behavior. Use JSON response adapters for native Python
objects/DataFrames and SSE for job progress and answer text. Keep the current
Streamlit interface as a reference until the replacement passes those checks.

Deploy five Compose services: HTTPS proxy, web app, worker, PostgreSQL with
pgvector, and Redis for queued work. Use PostgreSQL as the durable source for
accounts, documents, chunks, jobs, evaluation samples, and results. Store
original PDFs and extracted images on a backed-up persistent volume, with
opaque identifiers and ownership records. Redis must not be the sole record
of whether a job exists or completed.

Use managed OpenID Connect login with the React client and verify access tokens
in FastAPI; map verified issuer/subject identities to local user records.
Apply authorization in the service layer for every document, query, download,
job, cache, and evaluation operation. Add PostgreSQL row-level security as a
second boundary, with transaction-scoped identity so pooled connections never
inherit another user's identity. Workers receive the owning user and document
IDs and enforce the same rules.

Retain existing Chroma data as a migration source and local benchmark baseline.
Move production vectors to pgvector to simplify persistence and enforce access
alongside metadata. Preserve BM25 as a named benchmark strategy. Any PostgreSQL
full-text strategy must have its own label: full-text search is not automatically
equivalent to BM25. Select the production retrieval default using measured
quality, latency, memory, and cost on the representative corpus.

Initial release assumptions: one server, CPU inference for embeddings/OCR,
external model APIs, document-level private workspaces, bounded background
concurrency, and no high-availability guarantee. Revisit the UI and hosting
architecture after measured load establishes a need.

## Implementation order and acceptance checks

### 1. Make evaluation records trustworthy

- Preserve any original CSV/JSON and per-question traces; mark the screenshot
  values as unverified until linked to the corresponding run.
- Introduce validated dataset and result schemas. Store run ID, owner,
  document SHA-256, corpus/index version, question ID, reference answer,
  evidence pages/chunks, answerability, actual retrieved evidence, response,
  strategy, model versions, rubric version, token usage, timing, and errors.
- Check finite values and the documented bounds for each metric. Keep raw
  cosine scores distinct from bounded quality scores. Reject invalid records
  from ranking, retain their error details, and never clamp them to hide bugs.
- Recompute means/counts from the same records displayed in the UI. Deduplicate
  by stable document/chunk IDs, not by text or a display label.
- Report completed, failed, skipped, and not-applicable counts for every metric.
  Never use `mean()` alone to turn a partial run into an apparent success.
- Remove automatic claims such as "reranking significantly boosts precision"
  unless the current valid data supports them. Replace the unqualified overall
  average with separate quality gates and a quality/latency/cost comparison.

Acceptance: an out-of-range score cannot become a best-strategy recommendation;
displayed totals and aggregates reconcile exactly with exportable records;
failed or partial runs are clearly labelled.

### 2. Repair OCR and stabilize startup

- Create a project virtual environment and lock compatible dependencies after
  checking available wheels. Prefer Python 3.12 for the Linux image initially;
  support local Python 3.13 only after the complete extraction stack is verified.
- Split app, extraction, and optional evaluation dependencies. Remove unused
  imports and construct models lazily rather than during module import.
- Add an OCR diagnostic command that checks Tesseract, English trained data,
  Poppler, configuration paths, permissions, layout models, and a tiny known
  English scan. Errors must name the failed component and proposed remedy.
- Install/configure Tesseract if the diagnostic confirms it is missing. Discover
  the executable from configuration, PATH, and supported Windows locations.
  Configure both OCR wrappers consistently. Keep Poppler's working installation.
- Install Tesseract, English language data, Poppler, and required Linux libraries
  explicitly in the worker image. Pre-download and verify layout/embedding
  artifacts so first users do not trigger downloads unexpectedly.
- Route native-text pages to direct extraction, scans to OCR, and pages requiring
  table/image layout to high-resolution extraction. Preserve reading order,
  provenance, and duplicate removal when combining page results.
- Use bounded resolution, rotation/deskew and quality checks where appropriate.
  A failed layout pass must surface a degraded result; it must not silently
  pretend all tables and charts were extracted correctly.

Acceptance: native, scanned, mixed, table, and chart fixtures run on Windows and
inside Linux containers; a missing tool yields an actionable error; ordinary
text PDFs avoid unnecessary high-resolution processing.

### 3. Establish a defensible evaluation baseline

Use current supported Ragas APIs as the primary evaluation framework. Its
existing integration makes this an efficient starting point. DeepEval G-Eval
can provide a secondary, independently configured audit on selected examples.
Neither framework is universally best: dataset validity, source evidence,
judge calibration, and identical production execution are decisive.

| Layer | Measures | Evidence required |
| --- | --- | --- |
| OCR | Character/word error rate; missing text; reading-order checks | Human-verified transcriptions for representative scan fixtures |
| Tables/charts | Cell/value accuracy, structural errors, numeric/chart QA | Reviewed cells, labels, units, chart values, and tolerances |
| Retrieval | Recall@k, precision@k, MRR, nDCG@k | Reviewed relevant source spans, with graded relevance for nDCG |
| Response | Faithfulness, factual correctness, relevance, citation support and coverage | Actual production response and exact evidence sent to its generator |
| Unanswerable cases | Correct abstention, false-answer rate, over-abstention | Explicit answerability labels, scored separately from answerable cases |
| Multimodal response | Visual grounding and relevance | Actual image/table evidence and a judge with verified vision support |
| Operations | Ingestion duration, p50/p95 retrieval and answer latency, tokens, cost, peak memory, failures | Per-stage measured traces, separated into cold and warm runs |

- Replace the hardcoded six questions with importable document-linked datasets.
  Start with a small reviewed fixture set for regression checks and work toward
  100-200 reviewed questions spanning multiple representative PDFs.
- Synthetic question generation may draft cases, but generated references must
  be checked against the original source. Keep document-level tuning and held-out
  test splits separate; do not tune against final test results.
- Include single-fact, numeric/table, chart, multi-page, paraphrase, distractor,
  and genuinely unanswerable cases. Do not treat a refusal sentence as ordinary
  factual ground truth for context recall.
- Share the same retrieval, context assembly, prompt, and generation services
  between the UI and evaluator. Retain exact context including multimodal assets.
- Configure generator and judge independently; validate structured outputs,
  permissions, availability, and model modalities. Calibrate judges using human
  labels and repeat a sample to estimate variability.
- Compare keyword, vector, hybrid, reranked hybrid, and actual multi-query fusion
  on the same corpus and questions. Record retrieval k, final context budgets,
  prompts, and reranker changes so quality improvements can be attributed.
- Preserve per-case outcomes, confidence intervals, failures, and run provenance.
  Use paired comparisons for strategy changes. With only six questions, describe
  results as exploratory rather than evidence of a general winner.

Acceptance: every reference is linked to source evidence; every scored answer
uses the production path; independent audits reconcile disagreements; reports
make uncertainty and missing labels explicit. Do not invent baseline scores.

### 4. Optimize ingestion and questions against that baseline

- Use owner-scoped SHA-256 document identity and a pipeline configuration hash.
  Make ingestion idempotent with stable chunk IDs and explicit index revisions.
- Run extraction, summaries, embeddings, and index construction once per revision.
  A question must never rebuild the document index.
- Reuse embedding models and API clients. Batch embeddings and use bounded
  concurrency for summarization, retrieval expansion, and evaluation.
- Cache document extraction and retrievers using owner, corpus revision, model,
  and strategy settings. Invalidate on replacement/deletion; limit cache memory.
- Store original images as assets, not repeated base64 blobs in vector metadata.
  Preserve source document, page, element/span, language, and image MIME type.
- Preserve useful original text if summarization fails; do not index only a
  truncated fallback. Use summaries to aid retrieval while retaining raw evidence.
- Retrieve lexical and semantic candidates, fuse by stable IDs, and actually
  feed fused results into reranking/generation. Include the original query in
  multi-query retrieval; enable expansion only where its measured benefit pays
  for its extra latency and API cost.
- Evaluate embedding and reranker alternatives against the baseline before
  changing defaults. Larger models are not automatically faster or better.
- Bound context by tokens and relevant image budgets; require page citations
  and evidence-based abstention. Distinguish provider failures from model answers.
- Stream answers where the provider supports it; show genuine execution progress.
  Apply timeouts, retry only transient errors, and avoid rerunning completed paid
  stages after later failures. Add job cancellation and spend limits.

Acceptance: repeated ingestion creates no duplicates; repeated questions incur
no extraction/index rebuild; recovery reuses completed stages; candidate changes
improve measured latency/cost without unacceptable held-out quality regressions.
Set numeric quality thresholds and latency targets after measuring the baseline
on the intended server and documenting document sizes and concurrency.

### 5. Add public accounts and reliable product flows

- Implement sign-in/out, personal document library, ingestion progress, cited Q&A,
  dataset import/review, evaluation history, and authorized result exports.
- Enforce ownership on server-side operations and caches. Remove the shared raw
  vector-store picker and global results file from user flows.
- Validate PDF content as well as extension; use safe storage IDs, upload/page
  limits, and bounded extraction resources. Sanitize extracted table HTML before
  rendering. Treat document text as untrusted evidence in prompts.
- Apply per-account query/upload/evaluation quotas and concurrency limits. Require
  dataset validation and a cost estimate before starting a large evaluation.
- Provide durable job states, cancellation, retry, recovery, and comprehensible
  errors. Account/document deletion must clean related assets, index entries,
  results, and caches according to a documented retention policy.
- Keep secrets in runtime configuration and redact logs. Add migrations, structured
  logs, request/job IDs, readiness checks, and model/tool preflight checks.

Acceptance: two users cannot discover or retrieve each other's documents, jobs,
results, downloads, or cached answers; ordinary interactions stay responsive
while OCR/evaluation runs; restart does not lose completed documents/results.

### 6. Package, validate, and deploy

- Add Dockerfiles, `.dockerignore`, Compose configuration, `.env.example`, locked
  dependency files, migrations, persistent volumes, and an operational README.
- Limit exposed ports to the HTTPS proxy; keep PostgreSQL and Redis internal.
  Run application containers without root and define memory/CPU/concurrency limits.
- Add CI for meaningful core tests, dataset/result validation, authentication and
  ownership integration tests, extraction fixtures, and dependency/container builds.
  Paid model tests are opt-in; local fakes cannot prove real model quality.
- Exercise end-to-end login, upload, extraction, cited Q&A, dataset review,
  evaluation/export, deletion, and failure recovery in a browser.
- Load-test a declared mix of concurrent questions and background jobs. Verify
  restart recovery, worker retry behavior, database migrations, backup restoration,
  health checks, and rollback before launch.
- Prepare deployment on a selected VPS with domain/TLS and runtime credentials.
  Record the actual deployed revision and health checks when hosting is configured.

Acceptance: a clean machine can start the app from documented Compose commands;
required checks pass; representative workflows work after restart; backups restore
successfully; no public-release claim is made before these checks complete.

## React migration implemented

The main launcher now serves a React + TypeScript build and FastAPI in one warmed
Python process. Upload/preparation, streamed cited answers, source pagination,
reviewed dataset validation, all evaluation profiles, history/exports, and the
multi-query retrieval lab use the existing Python services. SQLite persists the
local catalog without adding a hosting subscription or database service.

Browser workspace ownership and same-origin submission checks are implemented.
Public account authentication, durable job recovery, shared-worker coordination,
and public hosting remain later stages. The Streamlit fallback was subsequently
removed during cleanup. The container now has a React build stage and
an API health check; Linux image execution remains a release check.

## Cost and practical limits

The current user constraint is **zero hosting subscription cost**, with model API
charges considered separately. The provided Compose setup runs on the existing
computer, persists local data, and exposes only localhost by default. It does not
create a paid server or publish a public endpoint. Existing electricity, storage,
connectivity, and the computer remaining online still determine availability.

Do not promise reliable public compute on an unspecified free hosting tier.
External publication requires a selected hosting/tunnel arrangement, account
isolation, TLS, and validation of that provider's resource/persistence limits.
Heavy high-resolution OCR may require more memory or a dedicated worker later.

Report usage by operation: PDF pages processed, summary/answer tokens, embedding
usage where applicable, reranker requests, and judge tokens. Evaluation may cost
more than ordinary Q&A because each strategy/question creates several judge
tasks. Cache by dataset/run configuration, use inexpensive deterministic metrics
first, and run costly judges deliberately. Do not estimate total monthly API
spend without measured usage and current provider prices.

## Inputs still needed during implementation

The current plan can proceed without these, but validating the reported run
requires the original evaluation CSV/JSON and, preferably, per-question results,
the source PDF, dataset, and scripts corresponding to the screenshots. Public
hosting ultimately needs a selected server, domain, identity configuration,
model credentials, and a spend limit. Secrets must be configured outside source
control and must not be pasted into logs or reports.

## Documentation used

- [Ragas metrics](https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/)
- [DeepEval metrics and G-Eval](https://deepeval.com/docs/metrics-introduction)
- [Unstructured partitioning strategies](https://docs.unstructured.io/open-source/core-functionality/partitioning)
- [Tesseract installation and language data](https://tesseract-ocr.github.io/tessdoc/Installation.html)

These public documentation pages were retrieved during the audit. The proposed
framework choice is a recommendation based on fit for this project, not a claim
that one vendor has objectively the best evaluation for every task.
