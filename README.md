# RAG Workspace

A React + TypeScript workspace for **automatic RAG evaluation**, backed by
FastAPI and Python. Upload an English PDF, let the system generate and verify
its own test questions, then compare retrieval strategies and answer quality.
Manual questions are not required to run a benchmark.

The app also provides cited document Q&A, source inspection and a retrieval lab.
Native text and tables use local extraction; scanned pages use bundled RapidOCR
without installing Tesseract or Poppler.

## Get started

Requirements: Python 3.12 or 3.13, Node.js 22+, and provider credentials for
question generation and quality judging. The Linux container uses Python 3.12.

From the project folder on Windows:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
cd frontend
npm ci
npm run build
cd ..
```

On macOS/Linux, activate the environment with `source .venv/bin/activate`.

Create `.env` from `.env.example` **only if it does not already exist**, then
configure your credentials. Preserve existing API keys when changing settings.

```dotenv
GOOGLE_API_KEY=your_google_key
GENERATION_PROVIDER=google
GENERATION_MODEL=gemini-3.1-flash-lite
SYNTHETIC_GENERATION_PROVIDER=google
SYNTHETIC_GENERATION_MODEL=gemini-3.1-flash-lite
SYNTHETIC_GENERATION_CONCURRENCY=1

EVALUATION_PROVIDER=google
EVALUATION_MODEL=gemini-3.1-flash-lite
EVALUATION_REASONING_EFFORT=minimal
EVALUATION_CONCURRENCY=1
EVALUATION_METRIC_TIMEOUT=360
SYNTHETIC_ALLOW_SAME_MODEL_VERIFICATION=true
```

The current preset uses standard Google Gemini 3.1 Flash-Lite for question drafting,
answers, source-reference checks and Ragas scoring. Start with one question and
**Quality preview**; each question needs multiple provider requests. Check your
Google AI Studio project's available quota before increasing the run size.
Same-model source checking is explicitly labelled and is not
independent verification. Set a different judge and disable the same-model flag
when an independent comparison is needed. Cohere reranking is optional and uses
`COHERE_API_KEY` with the settings in `.env.example`.

Optional OpenRouter generation/judging still requires fixed `:free` model IDs,
zero-price routing and an explicitly configured `OPENROUTER_API_KEY`.

Ollama can also handle drafting, answers, verification and scoring without a
cloud model key. Set `GENERATION_PROVIDER`, `SYNTHETIC_GENERATION_PROVIDER` and
`EVALUATION_PROVIDER` to `ollama`; set all three model settings and
`LOCAL_GENERATION_MODEL` to the installed model ID. `OLLAMA_BASE_URL` defaults to
`http://localhost:11434`. See the complete overrides in `.env.example`.
Small text-only models such as `gemma3:270m` can answer basic questions while
failing exact-quote drafting or structured judging. In that case, imported
source-checked questions let you test retrieval, answers and scoring separately.
Ollama decoding omits string-length bounds that its grammar compiler can reject;
the full schema and exact source quotes are still validated before scoring.

Start the app:

```powershell
.\.venv\Scripts\python.exe -B run_app.py
```

Open the printed URL, normally **http://127.0.0.1:8501**. The launcher warms local
search dependencies before accepting requests; startup can take time and makes
no inference requests. Use `--port 8517` if the default port is occupied.

API keys belong only in the backend `.env`. Never add them to React, `VITE_*`
variables, Git or screenshots. Restart Python after changing backend settings.

## Pages and workflow

The sidebar separates the main evaluation workflow from exploration tools.
The document selector in the header sets the document used by each page.
Navigation uses browser URLs such as `#/evaluation`, so refresh and browser
Back/Forward preserve the current page.

| Page                      | What to do here                                                               |
| ------------------------- | ----------------------------------------------------------------------------- |
| **Documents**       | Upload PDFs, search your library, prepare files and select a document.        |
| **Evaluation**      | Choose automatic or saved questions, configure the scope, and start a run.    |
| **Results**         | Select a run, compare metrics, inspect its test questions and export records. |
| **Ask document**    | Ask an individual question and read its cited answer and supporting evidence. |
| **Source explorer** | Browse paginated original text, tables and visual evidence.                   |
| **Retrieval lab**   | Test keyword/vector/hybrid search, fusion and optional model-assisted steps.  |

To run an automatic evaluation:

1. Upload a PDF on **Documents** and wait for preparation to finish.
2. Open **Evaluation** and select **Generate automatically**.
3. Choose 1–50 questions and an evaluation scope, then **Generate & evaluate**.
4. Open **Results** to compare strategies and inspect source-backed references.

Upload one to three PDFs in a single file selection or drag-and-drop. **Keep PDFs
separate** prepares each original in one bounded background job; **Combine into
one PDF workspace** creates a combined PDF while retaining the originals.

On Evaluation, **PDF scope** offers the current workspace, **Each PDF separately**,
or **Combined PDFs**. Choose two or three original PDFs for the multi-PDF options.
Separate mode applies the question count to each PDF and saves a dataset/report
per document. Combined mode applies the count to the whole collection and permits
questions spanning original PDFs. Its passages retain original-file provenance,
and source links open both combined page numbers and the corresponding original
PDF pages. Generated-reference coverage and any unrepresented original PDF remain
visible in the manifest/warnings. Batch errors stay associated with their PDF.

The system samples indexed passages across the document, drafts questions and
references, validates supporting quotes against stable chunks, and source-checks
each reference before evaluating RAG answers. Unsupported or duplicate
cases are rejected. If fewer cases pass than requested, the report shows the
shortfall. If no cases pass, scoring does not start.

Completed question sets and evaluation work are cached by document, models and
configuration. **Models & cache settings** allows deliberate recomputation.
Progress, cancellation and provider errors remain visible while a run executes.

## Restart recovery

Job settings, progress and completed results persist in the local SQLite catalog.
After a server restart, **Saved work** lists your interrupted operations, even
if the browser has lost its tab storage. Select **Resume work** to retry using
the saved question, retrieval options or evaluation scope. **Open progress**
reconnects to work already running.

Interrupted work is never started automatically. Evaluation retries reuse valid
completed question sets, answers and metric caches, including when the original
run requested recomputation. Repeated clicks return the same replacement job.
Completed answers/reports committed just before shutdown are restored without
making new model requests. Unfinished provider requests may need to run again
and use API credits; incomplete streamed text remains labelled as incomplete.

Explicitly cancelled jobs stay cancelled. Older records without saved operation
settings remain readable; start those operations again from their page. Recovery
currently supports one application process on the local computer. Coordination
between multiple workers remains future work.

## Account sign-in

`APP_AUTH_MODE=local` keeps the existing local browser workspace. To use private
accounts, register an OpenID Connect application with your identity provider and
set these backend environment variables:

```dotenv
APP_AUTH_MODE=oidc
APP_BASE_URL=https://rag.example.com
OIDC_ISSUER=https://your-provider.example.com
OIDC_CLIENT_ID=your_registered_client_id
OIDC_CLIENT_SECRET=your_client_secret
```

Register the exact redirect URI `APP_BASE_URL/api/auth/callback` with the provider,
using the actual origin in place of `APP_BASE_URL`, then restart Python. The issuer
must match discovery metadata exactly. HTTPS is required for public URLs; HTTP
on localhost/127.0.0.1 is supported for development. Configure the reverse proxy
to serve React and `/api` from that one origin. This adds no hosting service.

The default provider configuration uses authorization-code login, PKCE S256,
RS256 signatures and `client_secret_basic` token authentication. Providers can
instead use `OIDC_SIGNING_ALGORITHM=ES256` or
`OIDC_TOKEN_AUTH_METHOD=client_secret_post`. A registered public PKCE client can
use `OIDC_TOKEN_AUTH_METHOD=none` without a client secret. Configure the matching
settings on the provider. Discovery and JWKS must be reachable by the backend.

The browser receives an HttpOnly cookie; identity/access tokens and client secrets
remain off the frontend. Sessions persist across restarts, expire after eight
hours by default (`AUTH_SESSION_SECONDS`, 300–86400), and are revoked on sign-out.
Sign-out ends this app's session; the provider may retain its own login session.
Accounts are keyed by verified issuer and subject, so changing a display name or
email retains the same library. Document, answer, dataset, job and export routes
require the signed-in owner. Browser caches and remembered jobs reset when
switching accounts. The server also stops progress streams after session expiry
or revocation. Missing/invalid OIDC configuration prevents startup rather than
opening an anonymous workspace.

Existing local documents and results stay on disk and remain assigned to their
local workspace. OIDC login does not automatically claim them. Use local mode to
access existing work; upload the original PDFs into a new account when needed.
Do not switch a public deployment back to local mode. Account support is tested
with signed offline provider fixtures; live provider setup, TLS deployment,
shared workers and production backup/load checks remain release work.

## Evaluation scopes

| Scope                     | Work performed                                                                                                                                          |
| ------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Full comparison** | Every selected question across Keyword, Vector, Hybrid and Reranking, with answer-quality and retrieval metrics.                                        |
| **Quality preview** | A deterministic subset through the production retrieval/generation pipeline; five cases by default, configurable with `EVALUATION_PREVIEW_QUESTIONS`. |
| **Retrieval only**  | Local Keyword, Vector and Hybrid evidence selection, without RAG answers or quality judging.                                                            |

Automatic drafting and source verification use model APIs even for retrieval-only
runs. Reusing a saved dataset for retrieval-only scoring makes no model requests.
Full comparisons can require many judge calls per question and strategy; free
provider quotas can be exhausted before all metrics finish. Start with a small
preview to confirm model availability. HTTP 429/503/504 errors are provider
failures, not evidence of poor retrieval or answer quality.

Automatic drafting retries temporary HTTP 500/502/503/504 failures once by default,
using exponential backoff within the original request deadline. Configure
`SYNTHETIC_MAX_RETRIES` (0–2) and `SYNTHETIC_RETRY_DELAY_SECONDS` (0.1–5).
Cancellation remains available during the delay; HTTP 429 and authentication
errors do not trigger these retries. A different model can also encounter provider
availability or quota errors.

Answer generation also retries these temporary errors once before emitting any
answer text. `GENERATION_MAX_RETRIES` (0–2) and
`GENERATION_RETRY_DELAY_SECONDS` (0.1–5) control retries. The backoff and retry
share `MODEL_TIMEOUT_SECONDS`; the retry client receives only the remaining
request time. Partial streamed answers are retained as incomplete and are not
replayed, and authentication/quota errors are not retried.

Missing reranker configuration and reranker fallback remain explicit in reports.
No paid judge is silently substituted for a configured free OpenRouter model.
Only fixed `:free` model IDs with zero-price routing are accepted on that path.

## Reading results

**Results** displays one selected run at a time, with separate report sections
for answer quality, retrieval, citations/diagnostics, latency, test questions and
run configuration. Exports include summary CSV, sample JSONL and the run manifest.

| Answer-quality metric         | What it measures                                                     |
| ----------------------------- | -------------------------------------------------------------------- |
| **Faithfulness**        | Support for answer claims in the evidence supplied to the generator. |
| **Answer relevancy**    | How closely the response addresses the test question.                |
| **Context precision**   | How much of the selected evidence is useful for answering.           |
| **Context recall**      | Coverage of reference information by the selected evidence.          |
| **Factual correctness** | Claim agreement with the source-verified reference, using F1.        |

Additional metrics include precision@k, recall@k, MRR, nDCG@k, hit rate and MAP@k;
structural citation checks; exact match/token F1; applicable abstention and visual
checks; stage latency percentiles; and failure counts.

Every metric displays its valid sample count and any failures or skipped/N/A
cases. Confidence intervals can be expanded where available. **An unavailable
score is not zero.** Invalid values are excluded, and no fabricated overall
score or automatic winner is reported.

Provider-blocked cases have their own label. A visible provider notice gives the
recorded rejection and reset time, when supplied. Expand **Why metric attempts
failed** to inspect each affected strategy, question, metric and cause. Strategy
scheduling alternates across the comparison; complete multi-request Ragas metrics
have a separate deadline from individual provider requests. Completed structured
judging steps are cached so later failures do not force those steps to repeat.

Generated references are automatically checked, not human-reviewed ground truth.
They cover extracted text and table text; chart-only and unanswerable cases are
not generated automatically. References and relevance labels can be incomplete,
so source-supported extra claims or alternate valid passages may reduce some
scores. Inspect the questions and calibrate against reviewed cases before making
broad accuracy claims. Citation structure and wording overlap do not prove
semantic support.

To reuse or import questions, choose **Use saved questions** on Evaluation.
JSON/CSV cases must match the PDF SHA-256 and contain a question, reference answer,
answerability and evidence pages or relevant chunk IDs. Download the document’s
import template from that page; inspect and export test sets from Results.

CLI evaluation is also available:

```powershell
.\.venv\Scripts\python.exe -B advanced_evaluation.py --db "db/chroma_your_index" --dataset "questions.json" --profile full
```

## OCR and performance

PDF extraction reads native text/tables with pdfplumber, renders visual evidence
with PDFium, and uses CPU RapidOCR on scans. Its ONNX models ship with the wheel.
Scanned tables remain OCR text and images; cell reconstruction and complex chart
understanding require separate validation. Optional Tesseract/Unstructured layout
extraction is documented by `requirements-hires.txt` and the settings in
`.env.example`.

Check OCR without making model API calls:

```powershell
.\.venv\Scripts\python.exe -B ocr_doctor.py --self-test
```

Use `--pdf`, `--reference` and `--output` to measure character/word error rates
against a verified transcription. OCR confidence is not an accuracy measurement.

Document extraction, embeddings and indexes are reused across questions. Model
clients are warmed once; answers stream and evaluation uses bounded concurrency,
timeouts and reusable caches. Source, report and exploration views load on demand.
Document size, model availability and provider quotas still affect response time.

## Development and checks

Run the Python launcher on port 8501, then start Vite in a second terminal:

```powershell
cd frontend
npm run dev
```

Open the Vite URL. `/api` requests proxy to Python; frontend changes reload
immediately. Rebuild with `npm run build` before using the production launcher.

```powershell
python -m pip install -r requirements-dev.txt
python -B -m pytest tests -q -p no:cacheprovider
cd frontend
npm run build
```

Backend regression tests are offline; live provider availability and judge
calibration need separate checks. The restart-recovery milestone includes tests
for server shutdown, schema migration, ownership, concurrent retries and completed
artifact recovery, plus desktop/mobile browser checks using offline fixtures.
Live identity-provider setup, coordination between workers and Linux container
execution remain release work.

## Container deployment

Docker Desktop with Linux containers can run the app on your existing computer
without a hosting subscription. Model usage, electricity and connectivity are
separate costs.

```powershell
docker compose up --build
```

Open **http://localhost:8501**. Compose binds to localhost and persists documents,
indexes, extraction caches and evaluation results in named volumes. The image
builds React, serves it with FastAPI, preloads the pinned local embedding model,
runs as a non-root user and checks `/api/health`. `.env` and local uploaded data
are excluded from the image. Initial package/model downloads can take time.

The default cookie identifies a **local workspace**. Keep the localhost binding.
Public deployment requires OIDC account configuration, shared worker/storage
coordination, TLS and a hosting option validated for this workload.

## Project layout

```text
frontend/src/       React pages, navigation and shared UI components
api/                FastAPI routes, ownership, jobs and catalog persistence
*_runtime.py        OCR, indexing, retrieval, generation and judge services
advanced_evaluation.py       Ragas scoring and report generation
evaluation_generation.py    Source-verified automatic question generation
tests/              Backend regression tests
project_docs/       Original metric images and evaluation handbook DOCX
HANDOFF.md          Current state and continuation notes
PROJECT_PLAN.md     Roadmap and release acceptance checks
```

Runtime data is created as needed in `docs/`, `db/`, `.cache/`,
`evaluation_datasets/` and `evaluation_runs/`; it is excluded from Git. Existing
local documents and evaluation history are retained. Disposable QA output and
generated build metadata can be removed without deleting saved work. The retired
Streamlit interface has been removed.
