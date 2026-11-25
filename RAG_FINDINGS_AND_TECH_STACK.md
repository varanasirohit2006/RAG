# RAG Evaluation Findings, Model Performance & Tech Stack

**Project:** `rag_eval` (`RAG Workspace`)  
**Evaluation Run Analyzed:** `c0962c8b903029be` (`testresults/`)  
**Active Model Tested:** `qwen3-vl:2b-instruct` (via local Ollama)  
**Document:** `RAG_Evaluation_Handbook.docx` (PDF, 101 pages, 145 chunks)

---

## 1. Executive Summary

The latest 5-question full evaluation run (`testresults/`) using local **`ollama / qwen3-vl:2b-instruct`** demonstrates that the RAG pipeline **retrieves the right evidence reliably and generates factually accurate answers** when using **Keyword (BM25)**, **Hybrid (RRF)**, or **Reranking (Cohere)**:

- **Completion:** **16/20 samples fully completed**, **4 partial** (all on one multi-part question), and **0 skipped** (eliminating the HTTP 429 quota failures seen in earlier cloud runs).
- **Retrieval:** **Keyword**, **Hybrid**, and **Reranking** achieved **100% Recall@3 (`1.000`)** across all 5 questions; **Vector** alone achieved **80% Recall@3 (`0.800`)**, missing 1 question.
- **Answer Quality:** **Hybrid** (`0.964` Factual Correctness, `1.000` Faithfulness, `0.873` Answer Relevancy) and **Reranking** (`0.963` Factual Correctness, `1.000` Faithfulness, `0.821` Answer Relevancy) performed best overall.
- **Key Limitations:**
  1. **Citation formatting is weak** (`0.40–0.60` page validity, `0.00–0.40` source validity), which is common for a compact 2B-parameter model.
  2. **Local inference & judging latency is high** (`6,890.5s` / ~115 minutes total for 20 samples).
  3. **Same-model & small-N caveat:** `qwen3-vl:2b-instruct` drafted, answered, and judged all 5 synthetic questions, so quality scores should be treated as a functional local validation rather than an independently calibrated benchmark.

---

## 2. Model & RAG Performance (`testresults/` — Run `c0962c8b903029be`)

### 2.1. Run Configuration
- **Profile:** `full` (5 synthetic questions × 4 retrieval strategies = 20 samples)
- **Retrieval Cutoffs:** `retrieval_k = 10`, `metric_k = 3`
- **Generator & Judge:** `ollama` / `qwen3-vl:2b-instruct`
- **Embeddings:** `sentence-transformers/all-MiniLM-L6-v2` (local CPU)
- **Reranker:** `cohere` / `rerank-english-v3.0`
- **Total Wall-Clock Duration:** `6,890.54s` (~114.8 minutes, including `12` metric cache hits)

### 2.2. Summary Metrics by Strategy

| Strategy | Recall@3 | Precision@3* | MRR | nDCG@3 | Faithfulness | Answer Relevancy | Context Precision | Context Recall | Factual Correctness | Token F1 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Keyword (BM25)** | **1.000** | 0.333 | **1.000** | **1.000** | **1.000** | 0.858 | 1.000 | 1.000 | 0.928 | 0.441 |
| **Vector (Dense)** | 0.800 | 0.267 | 0.800 | 0.800 | 0.938 | 0.816 | 1.000 | 1.000 | 0.800 | 0.385 |
| **Hybrid (RRF)** | **1.000** | 0.333 | 0.900 | 0.926 | **1.000** | **0.873** | 1.000 | 1.000 | **0.964** | **0.459** |
| **Reranking (Cohere)** | **1.000** | 0.333 | **1.000** | **1.000** | **1.000** | 0.821 | 1.000 | 1.000 | 0.963 | 0.451 |

*\*Note on Precision@3: Each of the 5 questions has 1 labeled relevant chunk, so at `k = 3`, the maximum mathematically possible Precision@3 is `1/3 = 0.333`. Keyword, Hybrid, and Reranking all hit 100% of that ceiling.*

### 2.3. Citation Diagnostics & Latency

| Strategy | Citation Page Validity | Citation Page Coverage | Citation Source Validity | Retrieval (s) | Generation (s) | Judge (s) | Avg Sample (s) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Keyword (BM25)** | 0.500 | 0.400 | 0.200 | **0.008s** | 42.36s | 365.79s | 408.18s |
| **Vector (Dense)** | 0.400 | 0.400 | 0.000 | 0.133s | **21.68s** | **198.83s** | **220.67s** |
| **Hybrid (RRF)** | **0.600** | 0.400 | 0.200 | 0.134s | 33.65s | 312.09s | 345.90s |
| **Reranking (Cohere)** | 0.400 | 0.400 | **0.400** | 1.121s | 39.95s | 361.95s | 403.05s |

### 2.4. Detailed Findings
1. **Keyword, Hybrid, and Reranking vs. Vector:**
   - **Vector** missed question `auto-fcb97174a5d0df80` (*"What are the three measurement layers used in RAG evaluation and how are they defined?"*), resulting in `0.0` Recall@3 and `0.0` MRR for that question.
   - **Keyword**, **Hybrid**, and **Reranking** all retrieved the correct chunk for every question (`Recall@3 = 1.000`), with **Keyword** and **Reranking** placing it at rank #1 every time (`MRR = 1.000`).
2. **Partial Metric Failures (`auto-0b382df713f9ba4a`):**
   - All 4 `partial` samples occurred on the same long, multi-part question (*ethical obligations of annotators*). Each strategy failed 1 Ragas judge metric:
     - Keyword: `context_recall` failed
     - Vector: `faithfulness` failed
     - Hybrid: `faithfulness` failed
     - Reranking: `factual_correctness` failed
3. **Citation Compliance:**
   - `qwen3-vl:2b-instruct` frequently omitted or misformatted bracketed page/source references (scoring `0.0` on citations across all strategies for `auto-62c7d1db41983934` and `auto-0b382df713f9ba4a`).
4. **Comparison with Earlier Cloud Run (`0d440ada`):**
   - Earlier Gemini 3.1 Flash-Lite run (`0d440ada`) suffered **14 skipped samples** (HTTP 429 quota exhaustion) and **0 matched questions** across all 4 strategies.
   - Moving to local Ollama (`c0962c8b903029be`) eliminated rate-limit skips (`0` skipped) and allowed a complete 5-question comparison across all 4 strategies.

---

## 3. Complete Project Tech Stack

### 3.1. Backend & API Layer
- **Language / Runtime:** Python 3.12 / 3.13
- **Web Framework:** **FastAPI** (`0.137.2`) + **Uvicorn** (`0.38.0`) with **Server-Sent Events (SSE)** for real-time job progress and token streaming
- **Validation & Settings:** **Pydantic** (`2.13.4`), **python-dotenv** (`1.2.2`), **python-multipart** (`0.0.20`)
- **Authentication & Security:** **PyJWT[crypto]** (`2.15.1`) supporting both local cookie workspaces (`rag_workspace`) and OIDC PKCE (`RS256` / `ES256`)
- **Persistence:** **SQLite** (`web_catalog.sqlite3` for documents, sessions, and durable restart-recoverable jobs) + atomic JSON/JSONL/CSV file storage (`evaluation_runs/`, `evaluation_datasets/`, `.cache/`)

### 3.2. Document Extraction, OCR & Chunking
- **Native PDF Parsing:** **pdfplumber** (`>=0.11.7,<0.12`) for text and structured table extraction
- **PDF Rendering & Merging:** **pypdfium2** (`>=5,<6`) for page rendering and multi-PDF batch stitching
- **Local OCR:** **RapidOCR** (`3.10.0`) + **onnxruntime** (`>=1.20,<2`) for CPU-based scan OCR without external Tesseract/Poppler dependencies
- **Document Chunking:** **Unstructured** (`>=0.18,<0.23`, `chunk_by_title`) + **Pillow** (`13.0.0`) for visual evidence resizing/storage

### 3.3. Embeddings, Vector Store & Retrieval
- **Embeddings:** **sentence-transformers** (`5.5.0`, `all-MiniLM-L6-v2`) + **PyTorch** (CPU wheel) + **langchain-huggingface** (`1.2.2`)
- **Vector Database:** **ChromaDB** via **langchain-chroma** (`1.1.0`)
- **Lexical Search:** **rank-bm25** (`0.2.2`)
- **Hybrid Fusion & Reranking:** Reciprocal Rank Fusion (RRF) + **Cohere** (`5.21.1`, `rerank-english-v3.0`)

### 3.4. LLM Orchestration & Providers
- **Framework:** **LangChain** (`1.3.1`), **langchain-core** (`1.6.6`), **langchain-community** (`0.4.1`), **langchain-classic** (`1.0.7`)
- **Structured Output & Retries:** **instructor** (`1.15.1`), **tenacity** (`9.1.2`), **httpx** (`0.28.1`)
- **Supported Model Providers:**
  - **Local Ollama:** **langchain-ollama** (`1.1.0`) — currently configured with `qwen3-vl:2b-instruct`
  - **Google Gemini:** **google-genai** (`2.23.0`), **langchain-google-genai** (`4.4.0`)
  - **OpenAI / OpenAI-Compatible Proxies / OpenRouter:** **openai** (`2.36.0`), **langchain-openai** (`1.2.1`)

### 3.5. Evaluation & Benchmarking
- **LLM-as-a-Judge & Metrics:** **Ragas** (`0.4.3`), **datasets** (`4.8.5`), **pandas** (`2.2.3`)
- **Retrieval Metrics:** Recall@k, Precision@k, MRR, nDCG@k, Hit Rate@k, MAP@k
- **Generation & Diagnostic Metrics:** Faithfulness, Answer Relevancy, Context Precision, Context Recall, Factual Correctness, Normalized Exact Match, Token F1, Citation Page/Source Validity & Coverage, Stage Latencies, 95% Bootstrap Confidence Intervals

### 3.6. Frontend
- **UI Framework:** **React** (`19.2.0`) + **React DOM** (`19.2.0`)
- **Language & Bundler:** **TypeScript** (`~5.9.3`) + **Vite** (`6.4.6`) + `@vitejs/plugin-react` (`4.7.0`)
- **Data Fetching & Caching:** **SWR** (`2.3.6`) + custom `useJob` SSE/polling hook
- **Rendering & Sanitization:** **react-markdown** (`10.1.0`) + **DOMPurify** (`3.3.0`)

### 3.7. Testing & Deployment
- **Testing:** **pytest** (`>=8,<10`) for offline backend regression tests (339 tests), **Playwright** (`1.65.2`) for browser checks, `tsc -b` + `vite build` for frontend verification
- **Containerization:** **Docker** (`Dockerfile` with `node:22-alpine` multi-stage build + `python:3.12-slim`) & **Docker Compose** (`compose.yaml`)

---

## 4. Recommendations
1. **Default Retrieval Strategy:** Use **Hybrid** or **Reranking** for Q&A and production workloads — both achieved `1.000` Recall@3 and `~0.96` Factual Correctness, whereas standalone Vector missed 20% of test questions.
2. **Improve Citations & Judge Reliability:** `qwen3-vl:2b-instruct` works well for local Q&A, but if strict bracketed citations and faster/more reliable judging are needed, configure a larger model (e.g., `deepseek-r1:14b` or an independent cloud judge) for `EVALUATION_MODEL`.
3. **Scale Test Coverage:** Expand beyond 5 synthetic questions (e.g., 20–50 questions including unanswerable and human-reviewed cases) before making formal accuracy comparisons.
