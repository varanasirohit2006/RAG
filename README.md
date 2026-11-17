# Multimodal RAG Workspace

React + TypeScript provides the document workspace; FastAPI adapts the existing
Python OCR, search, generation and Ragas evaluation services. Upload PDFs, ask
streamed cited questions, browse source passages, and compare retrieval strategies
using automatically generated, source-verified questions or imported datasets.

The workspace contains only this project folder. Uploaded PDFs, databases,
indexes, extraction caches, evaluation history and temporary QA artifacts were
cleared on 18 November 2026. Upload and prepare a PDF to start a new evaluation.
Runtime storage is created automatically when needed. `.env`, installed
dependencies, source, tests and deployment files are preserved. Reference images
and the original handbook DOCX are in `project_docs/`.

## Response time and progress

Ingestion indexes original OCR text and native table content directly instead of
sending every table/image chunk through a serial cloud summarization call.
Extracted documents, local embedding models, lexical retrievers, and completed
indexes are reused. Pipeline/configuration revisions are recorded in index
manifests, and questions use the existing index.

Generation receives bounded relevant original evidence, including selected
images. Tokens stream as they arrive, and stage progress separates extraction,
indexing, retrieval, reranking, generation, and evaluation. Google requests have
a deadline and one SDK attempt; reranking has a shorter deadline. Evaluation
uses bounded concurrency and reusable answer/metric caches.

Gemma verification and judging honor `EVALUATION_TIMEOUT` (90 seconds by
default), including retry backoff and JSON repair. `EVALUATION_MAX_RETRIES=1`
allows at most two attempts shared between transient HTTP 500/502/503/504 errors
and malformed JSON; the SDK makes one attempt per call. Authentication/access
errors fail immediately. Automatic-generation diagnostics record HTTP codes,
local deadline failures and actual verifier requests/retries. Unverified
references are never scored.

### Optional free OpenRouter judge

Store the OpenRouter key only in the project-root `.env`, alongside the existing
Google key. To explicitly choose the free judge for both source verification and
Ragas metrics, set these variables and restart the Python app:

```dotenv
OPENROUTER_API_KEY=your_key_here
EVALUATION_PROVIDER=openrouter
EVALUATION_MODEL=inclusionai/ling-3.2-flash:free
EVALUATION_CONCURRENCY=1
```

Gemini 3.5 Flash Lite still drafts questions and generates answers with your
Google credits. OpenRouter judging uses a dedicated key and endpoint, a fixed
`:free` model ID, and zero-price provider constraints. Paid model IDs and random
routers are rejected. Schema validation, source verification, deadlines, retry
budgets, failure counts and the existing Ragas metrics remain in place. The same
model through different provider aliases cannot verify its own question drafts.

Start with one question in Preview. Free routes have request/quota limits, and a
full Ragas comparison can require multiple calls per metric/question/strategy.
HTTP 429 is displayed as a rate/quota failure rather than retried repeatedly.
Model names and package revisions are recorded with runs; catalog availability
alone does not establish verdict accuracy or response speed.

The public OpenRouter catalog on 18 November 2026 lists
`inclusionai/ling-3.2-flash:free` (Ant Digital) and
`nvidia/nemotron-3.5-super-vl:free` (NVIDIA) at zero prompt/completion price, with
structured-output support. They are candidates for a small source-verification
and known-reference accuracy check. The free Gemma 4 31B route currently uses
Google AI Studio, so it may share the direct Google service's availability issue.
Confirm current model availability in <https://openrouter.ai/api/v1/models>.

The configured Ling route completed a real one-question automatic Preview in
265.362 seconds: 1.8074 seconds for the RAG answer and 186.1558 seconds for 13
structured judge calls. All five applicable quality metrics had valid results
and no failures. These are single-run timings, not a general speed/accuracy
guarantee. NVIDIA passed two small source checks but truncated a faithfulness
verdict during a larger five-metric comparison; Ling remains the selected judge.

An earlier live report identified a citation parser bug: `p.`/`pp.` labels were
ignored. That parser is fixed. The earlier report and its correction artifacts
were removed with evaluation history during cleanup.
Factual correctness remains reference-based F1; a sparse synthetic reference can
penalize additional source-supported claims. Review question/reference completeness
before using an automatic benchmark to make accuracy claims.

The first upload still needs extraction and embedding work. Full evaluation still
performs real model calls. Provider rate limits, document size, network latency,
and judge configuration affect runtime; these changes do not establish a fixed
response-time or accuracy guarantee.

Defaults in `.env.example` include a 60-second generation deadline, 1024 output
tokens, 12000 context characters, three visual assets with maximum dimension
1280, and batches of 32 embeddings. Adjust output/context limits deliberately
for more complex questions. Generation and evaluation have separate reasoning
settings. The chosen Gemma model does not receive Gemini-specific thinking
parameters; that optional setting only applies when explicitly choosing a
supported Gemini 2.5 Flash model.

The supplied PDF had 101 native-text pages, no scans or embedded images, and one
ruled table. Its native inspection took about 16 seconds; lexical retrieval took
less than a second. The old index contained 54 duplicate document contents among
156 records. These measurements identify local work and index redundancy; precise
attribution of a past 25-minute cloud response requires per-stage request logs.

The final chunking fix retains distinct split passages: the rebuilt index has
156 records and 155 distinct texts. A repeat preparation run took 0.008 seconds
for cached extraction and 0.182 seconds for index reuse; the one-time rebuild
took 53.655 seconds. These timings exclude model answer generation.

## PDF extraction and OCR

The default extraction pipeline requires **no Tesseract or Poppler installation**.
It reads native PDF text/tables with pdfplumber, renders visual evidence with
PDFium, and uses CPU RapidOCR for scanned pages. RapidOCR 3.10.0 ships the ONNX
models used here, so OCR works locally without API keys or model downloads during
extraction. The engine is loaded once and reused.

Native ruled tables are extracted as escaped HTML. Scanned tables remain OCR text
and page images; the default engine does not reconstruct their cell structure.
Page previews preserve embedded images and detected vector graphics for the
multimodal answer pipeline. Complex reading order and borderless/scanned tables
still need validation on representative documents.

## Setup

1. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   # Windows
   .\.venv\Scripts\Activate.ps1
   # macOS/Linux
   source .venv/bin/activate
   ```
2. Install backend dependencies:
   ```bash
   python -m pip install -r requirements.txt
   ```
   Build the frontend once with Node.js 22 or newer:
   ```powershell
   cd frontend
   npm ci
   npm run build
   cd ..
   ```
3. Create `.env` from `.env.example` if it does not exist. Preserve existing keys
   when changing settings:
   ```env
   GOOGLE_API_KEY=your_gemini_api_key
   COHERE_API_KEY=your_cohere_api_key
   GENERATION_PROVIDER=google
   GENERATION_MODEL=gemini-3.5-flash-lite
   EVALUATION_PROVIDER=google
   EVALUATION_MODEL=gemma-4-31b-it
   ```

   OCR defaults work without changes. Generation and judge evaluation require
   valid provider credentials; document extraction does not. Cohere is optional
   for production reranking. Keep keys in `.env` and restart after changes.

To check OCR alone with a smaller dependency install:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-ocr.txt
.\.venv\Scripts\python.exe ocr_doctor.py --self-test
```

To extract your own PDF and measure OCR against a verified transcription:

```powershell
.\.venv\Scripts\python.exe ocr_doctor.py --pdf "report.pdf" --reference "verified-text.txt" --output "test-results/report-ocr.json"
```

The reference must correspond to the full extracted text in page order. Character
and word error rates measure differences from that reference; OCR confidence does
not measure accuracy. Error rates may exceed 1 when there are many insertions.
The built-in self-test uses a clean synthetic English scan and does not establish
accuracy for arbitrary uploaded PDFs.

Use `OCR_DPI`, `OCR_THREADS`, `OCR_MIN_CONFIDENCE`, `OCR_MAX_PIXELS`,
`PDF_MAX_PAGES`, and `PDF_MAX_FILE_MB` in `.env` to tune quality/resource limits.
Defaults are 200 DPI, two inference threads, 0.5 minimum line confidence,
12 million rendered pixels, 300 pages, and 50 MB per PDF.

## Optional Tesseract / high-resolution layout extraction

For the older Unstructured layout pipeline, install `requirements-hires.txt`,
Tesseract with English trained data, and Poppler. Then configure:

```env
PDF_EXTRACTION_MODE=hi_res
OCR_BACKEND=tesseract
# Only needed when the installed executable is not discoverable:
TESSERACT_CMD=C:\Program Files\Tesseract-OCR\tesseract.exe
```

Tesseract can also be used for scan text with `PDF_EXTRACTION_MODE=auto` and
`OCR_BACKEND=tesseract`. High-resolution layout mode may download its separate
layout/table models and has additional system dependencies. It is an explicit
option rather than a silent fallback.

## Model Configuration

Answer generation uses `gemini-3.5-flash-lite` with minimal reasoning, following
the latest request. The local judge is configured as OpenRouter
`inclusionai/ling-3.2-flash:free`; Google `gemma-4-31b-it` remains supported.
Generation and judge settings remain separately configurable. The Flash Lite integration omits custom
sampling parameters that this model does not support.

A live production answer on the supplied PDF completed in 7.149 seconds,
including 4.685 seconds of generation, 0.309 seconds of local retrieval and
2.155 seconds of Cohere reranking. The answer named all three measurement layers
correctly and cited page 2. A fresh Python process took another 33.481 seconds to
load local libraries and the saved embedding index before this request. This is
one source-checked case, not a general latency or accuracy benchmark.

The final browser test with the launcher displayed the correct answer in 19.844
seconds, versus 119.355 seconds before provider-client pre-initialization. PDF
preparation took 3.589 seconds. Its measured backend answer time was 12.308
seconds, including 10.368 seconds of generation. Launcher startup took 31.23
seconds for local search initialization plus 3.44 seconds for provider clients;
it submitted no inference/reranking requests during startup. These are individual
observations on one reviewed question; network and model latency may vary.

The earlier Gemma browser request received a provider HTTP 503 after 66.181 seconds
including retrieval/reranking. A separate one-case preview received HTTP 503
after 11.525 seconds, before judging. Those runs established no successful Gemma
answer or quality scores. Gemma judge availability remains unverified. The local
retrieval profile completed its one-case, three-strategy
browser check in 6.08 seconds. OCR uses its own local models.

Optional local generation uses `GENERATION_PROVIDER=ollama`,
`LOCAL_GENERATION_MODEL`, and `OLLAMA_BASE_URL`; its service/model must be
installed separately. Local-model speed and quality require their own validation.

## Running the Application

- **React workspace and API:**
  ```bash
  python -B run_app.py
  ```
  Open `http://127.0.0.1:8501`. Use `--port 8517` if another app already uses 8501.
- **Frontend development:** run the Python launcher on port 8501, then in a
  second terminal run `cd frontend` and `npm run dev`. Vite proxies `/api` to the
  backend; open the Vite URL. Frontend edits reload automatically; restart Python
  for backend or `.env` changes. Rebuild React before using the production launcher.
- **CLI Evaluation with a reviewed dataset:**
  ```bash
  python advanced_evaluation.py --db "db/chroma_your_index" --dataset "questions.json" --profile preview
  ```

The launcher loads the local search model, native dependencies, and configured
provider clients once, before
accepting browser sessions. This avoids the concurrent dependency initialization
that stalled document preparation on the tested Windows environment. Wait for
the printed app URL before opening the browser; fresh startup may take about
30 seconds. Restart the launcher after changing code or `.env` settings.

## Automatic evaluation

Evaluation is the default workspace view. Upload and prepare a PDF, choose the
number of questions (1–50), and select **Generate questions & evaluate**. No
question entry or dataset import is required. The default full comparison tests
keyword, vector, hybrid, and reranked retrieval against the same generated cases.

The system samples indexed passages across the document, drafts questions and
reference answers with the configured Gemini generator, checks supporting quotes
against stable source chunks, and verifies each reference with the separately
configured independent judge. Unsupported or duplicate drafts are rejected. If fewer
cases pass verification than requested, the report states the shortfall; if none
pass, evaluation does not start. Generated cases and their evidence are available
for inspection and JSON download. Completed question sets and evaluation results
are cached by source, models, and configuration.

Synthetic references are automatically checked, not human-reviewed ground truth.
They cover extracted text and table text. Chart-only and unanswerable questions
are not generated automatically; optional source-labelled imported cases can
cover those scenarios. Cited chunk labels are not exhaustive relevance judgments,
so alternate valid passages can lower retrieval scores. Calibration against a
reviewed benchmark remains useful, but is not a prerequisite for automatic runs.

Reports include the four metrics in `project_docs/Metrics.png`: **faithfulness, answer
relevancy, context precision, and context recall**, plus factual correctness.
Retrieval reports include precision@k, recall@k, MRR, nDCG@k, hit rate and MAP@k.
Structural citation checks, lexical diagnostics, stage timings, latency
percentiles, and failure counts are reported separately. Each quality score has
valid sample counts, missing/failed counts and confidence intervals when enough
samples exist. Structural citation checks do not prove semantic citation support.
Invalid values such as the screenshot's answer relevancy `5.001232` are excluded.
There is no fabricated overall score or automatic declaration of a winner.

Automatic drafting and verification use API credits for every profile. The
retrieval-only scoring phase is local; reusing an existing dataset for retrieval
evaluation makes no model requests. Missing reranker configuration and reranker
fallbacks remain explicit. Provider failures do not become fabricated scores.

Earlier automatic-evaluation validation passed 135 tests; the React build passed.
Playwright/Edge verified desktop/mobile automatic submission, generated question
inspection, metric tables, missing/failed scores, and the real provider-failure
view. Successful report rendering also used explicitly labelled isolated fixtures;
those scores were never saved as real evaluation results. A bounded one-question
live run received Google 503 model-demand errors during drafting/verification and
produced no verified dataset or quality scores. Subsequent OpenRouter live
validation is described above; those live artifacts have since been cleared.

A follow-up check on 17 November confirmed that the existing Google key can
retrieve metadata for `gemma-4-31b-it`, which advertises `generateContent`.
After fixing the hidden 45-second timeout cap and adding bounded retries, a real
handbook source-verification request still received HTTP 503/high demand on both
attempts, finishing in 20.567 seconds. Model visibility and credentials were
confirmed; successful inference and end-to-end automatic scores remain unverified.

## Optional imported datasets and profiles

Evaluation uses a dataset linked to the uploaded PDF, generated automatically or
optionally imported. The previous six hardcoded questions are not reused for
unrelated documents. For an existing reviewed benchmark, import JSON or CSV
through the app or CLI. JSON structure:

```json
{
  "document_hash": "replace_with_the_PDF_SHA256",
  "questions": [
    {
      "id": "q1",
      "question": "Your question about this document",
      "reference_answer": "An answer verified against the original PDF",
      "answerable": true,
      "reference_pages": [1],
      "relevant_chunk_ids": []
    },
    {
      "id": "q2",
      "question": "A question the document cannot answer",
      "reference_answer": "The document does not provide this information.",
      "answerable": false,
      "reference_pages": [],
      "relevant_chunk_ids": []
    }
  ]
}
```

Replace the hash with the full SHA-256 of the PDF. Answerable cases need verified
evidence pages or relevant chunk IDs. CSV uses corresponding column names,
repeats the document hash, and stores lists as JSON or semicolon-separated values.

| Profile | Work performed | Interpretation |
| --- | --- | --- |
| `preview` | Up to five selected questions through production retrieval, judged with the four screenshot metrics plus factual correctness | A smaller labelled preview, not a replacement for full evaluation |
| `full` | All questions across keyword, vector, hybrid, and reranked retrieval, with five configured Ragas metrics | Complete comparison for this dataset and configuration |
| `retrieval` | Local keyword/vector/hybrid retrieval and evidence-label metrics; no answer generation, judge, or API reranker | Measures retrieval only, not answer faithfulness/correctness |

Visual evidence and unanswerable questions use applicable metrics. Missing labels,
provider failures, skipped metrics, and valid sample counts are explicit. Lexical
token overlap is not presented as factual correctness. Independently configured
judges and human review remain necessary to assess evaluation quality.

```powershell
.\.venv\Scripts\python.exe advanced_evaluation.py --db "db/chroma_your_index" --dataset "questions.json" --profile full
.\.venv\Scripts\python.exe advanced_evaluation.py --db "db/chroma_your_index" --dataset "questions.json" --profile retrieval
```

Per-question outputs, manifests, timings, score denominators, and reusable caches
live in `evaluation_runs/`. Use `--force` for a deliberate new run rather than
reusing completed cached work. Cancellation is cooperative; an in-flight provider
request may last until its deadline.

## Run with no hosting subscription

Run on your existing computer, directly or with Docker Desktop using Linux
containers. This adds no server subscription. Model API usage is separate, and
your computer still needs to remain powered and connected.

```powershell
docker compose up --build
```

Open `http://localhost:8501`. Compose binds the service to localhost and preserves
documents, indexes, extraction caches, and evaluation results in named volumes.
The image runs as a non-root user, installs CPU PyTorch, and preloads the pinned
MiniLM embedding model during build. RapidOCR's models ship in its wheel. Secrets
and existing documents/indexes are excluded from the image.

The container builds React in a Node stage, serves its static files and API from
one Python process, and checks `/api/health`. Production does not run a frontend
development server. Rebuild after changes to packaged code.

The preloaded embedding revision is fixed in `Dockerfile`. Rebuild the image and
provide matching model assets if changing its embedding model/revision; the
default container disables Hugging Face downloads at runtime.

The initial image build downloads packages and model assets and may take time.
The default image uses native/RapidOCR extraction; optional Tesseract high-res
mode requires extending its system dependencies.

Browser workspaces use an HTTP-only same-origin cookie and ownership checks.
They are local workspaces, not authenticated public accounts. Keep the localhost
binding for this version. Public accounts are not implemented yet. Publishing requires an access/auth
decision and a hosting option with appropriate durable storage. Free hosted
platforms may impose sleep, persistence, or resource limits; a zero-price public
production service has not been established. Completed artifacts persist in
volumes; in-memory jobs are not guaranteed to resume after restart.

## Validation and project status

After the 18 November cleanup, all 174 backend tests and the React production
build passed. Temporary test data was removed; the app remains stopped with
no uploaded PDFs or runtime databases.

```bash
python -m pip install -r requirements-dev.txt
python -m pytest tests -q
python ocr_doctor.py --self-test --output test-results/ocr-smoke.json
docker compose config --quiet
```

Browser tests use Playwright; install its browser assets with
`python -m playwright install chromium` when needed. Paid provider checks are
separate from deterministic offline tests. Direct runtime package versions are
pinned to the locally inspected environment; complete cross-platform transitive
locking and Linux image execution still need release validation.

Public accounts and deployment hardening remain tracked in
[PROJECT_PLAN.md](PROJECT_PLAN.md).

## React architecture and data compatibility

React and FastAPI are the maintained interface. The retired Streamlit interface,
its pages, launcher and optional dependency manifest were removed. The Python
RAG/OCR/evaluation services and their tests remain. Frontend dependency/build
output is excluded from Git and Docker. Historical measurements in this README
describe earlier checks; their PDF/index/result artifacts were cleared during
cleanup and are not bundled benchmarks.

`frontend/src` contains the typed React views. Evaluation, retrieval and source
views load on demand; SWR deduplicates reads and SSE streams job progress and
partial answers. If streaming disconnects, the browser falls back to polling.
Navigation and browser refresh retain job IDs in session storage.

`api/main.py` owns HTTP validation and document ownership; `api/runtime.py`
adapts bounded background jobs and native results into browser-safe JSON.
`api/storage.py` persists catalog metadata, reviewed datasets, answer history,
and run references in `db/web_catalog.sqlite3`. PDF files, extraction caches,
stable chunk IDs, Chroma indexes, and evaluation artifacts retain their existing
formats. The first local workspace adopts legacy hash-named PDFs and matching
run history. Reuploading an existing document reuses its catalog entry and index.

Keep `.env` in the Python project root. React has no API-key input or secret
build variables. Model credentials are consumed only by Python. Asset/download
routes require workspace ownership and never accept arbitrary filesystem paths.
The queue allows four outstanding jobs globally and two per workspace, with two
worker threads. Cancellation is cooperative. Completed data survives restarts;
in-flight jobs do not resume automatically. Clearing the browser cookie creates
a new workspace, so retain it to access your existing library.

The retrieval lab supports keyword/vector/weighted hybrid search, manual or
generated query variations, reciprocal rank fusion, optional reranking, and an
optional grounded answer. Default lab search and the retrieval-only evaluation
profile make no model API calls. Preview/full evaluation preserves the configured
judge and actual score denominators, missing metrics, warnings, and exports.

Migration validation: 84 backend tests passed; the production React build and
Compose configuration passed. Playwright with installed Edge checked desktop
(1440×1000) and mobile (390×844), document reuse, source pagination, local
retrieval, wrong-PDF dataset rejection, evaluation/downloads, cancellation, and
answer history after browser refresh and server restart. No unexpected browser
errors remained; the deliberately invalid dataset returned the expected 422.

On the reviewed handbook question, the React answer became visible in 7.598
seconds (7.410 seconds in the backend, including 6.754 seconds of generation)
and correctly named the three layers with page 2 evidence. A fresh one-question
retrieval-only evaluation completed visibly in 1.210 seconds; cached evaluation
took 0.526 seconds. A small synthetic text/table/chart PDF prepared in 0.557
seconds and a clean English scan in 2.006 seconds. These are individual smoke
checks, not general speed or OCR-accuracy guarantees. Full paid Gemma judging
was not repeated during migration. Docker Desktop's engine was stopped, so the
updated Linux image has not been built or run locally.
