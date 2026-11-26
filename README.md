# RAG Workspace & Automated Evaluation

A full-stack **Retrieval-Augmented Generation (RAG)** and **automated evaluation** platform built with **FastAPI**, **LangChain**, **ChromaDB**, **Ragas**, and **React + TypeScript**.

Upload native or scanned PDFs, compare four retrieval strategies side-by-side (**Keyword BM25**, **Dense Vector**, **Hybrid RRF**, and **Cohere Reranking**), ask cited multimodal follow-up questions with streaming responses, and run automated source-verified LLM-as-a-judge benchmarks.

---

## Key Features

- **Multimodal PDF Ingestion & Local OCR**: Extracts native text and HTML tables with `pdfplumber` and handles scanned/visual pages locally via `pypdfium2` + ONNX-based `RapidOCR` (no external Tesseract or Poppler binaries required).
- **4 Retrieval Strategies + Query Expansion**:
  - **Keyword** (`BM25Retriever`)
  - **Vector** (`ChromaDB` cosine similarity with `all-MiniLM-L6-v2`)
  - **Hybrid** (`Reciprocal Rank Fusion`, `k=60`)
  - **Reranking** (`Cohere rerank-english-v3.0`)
  - Plus structured **Multi-Query Retrieval** (`QueryVariations`) and **History-Aware Query Rewriting**.
- **Flexible Model Providers**: Supports fully local inference via **Ollama** (`qwen3-vl:2b-instruct`), **Google Gemini**, and **OpenAI-compatible** / **OpenRouter** endpoints.
- **Automated Evaluation Pipeline**: Drafts source-checked questions directly from document chunks, computes IR metrics (`Recall@k`, `Precision@k`, `MRR`, `nDCG@k`, `MAP@k`, `Hit Rate`), runs **Ragas** LLM-as-a-judge metrics (`Faithfulness`, `Answer Relevancy`, `Context Precision/Recall`, `Factual Correctness`), and checks bracketed page/source citations.
- **Interactive Dashboard**: Real-time **Server-Sent Events (SSE)** job progress and token streaming, citation source viewer, retrieval lab, and run comparison analytics.

---

## Tech Stack

| Layer | Technologies |
| :--- | :--- |
| **Frontend** | React `19.2`, TypeScript `5.9`, Vite `6.4`, SWR `2.3`, `react-markdown`, `DOMPurify` |
| **Backend & API** | Python `3.12`/`3.13`, FastAPI `0.137`, Uvicorn `0.38`, Pydantic `2.13`, PyJWT (Local cookie & OIDC PKCE), SQLite (`web_catalog.sqlite3`) |
| **PDF & OCR** | `pdfplumber`, `pypdfium2`, `RapidOCR` (`3.10.0`) + `onnxruntime`, `Unstructured` (`chunk_by_title`), `Pillow` |
| **Embeddings & Retrieval** | `sentence-transformers` (`all-MiniLM-L6-v2`), PyTorch (CPU), `ChromaDB` (`langchain-chroma`), `rank-bm25`, `Cohere` (`rerank-english-v3.0`) |
| **LLM & Evaluation** | `LangChain`, `langchain-ollama`, `google-genai`, `openai`, `instructor`, `Ragas` (`0.4.3`), `datasets`, `pandas`, `pytest` |

---

## Benchmark Findings (`qwen3-vl:2b-instruct` via Local Ollama)

Evaluated on **5 synthetic questions** across all **4 retrieval strategies** (`20` samples, `retrieval_k = 10`, `metric_k = 3`) over the 101-page `RAG_Evaluation_Handbook` PDF (`145` chunks), comparing local Ollama (`qwen3-vl:2b-instruct`) against earlier cloud runs:

| Strategy | Recall@3 | Precision@3\* | MRR | nDCG@3 | Faithfulness | Answer Rel. | Factual Corr. | Citation Page Val. | Avg Sample Time |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Keyword (BM25)** | **1.000** | 0.333 | **1.000** | **1.000** | **1.000** | 0.858 | 0.928 | 0.500 | 408.2s |
| **Vector (Dense)** | 0.800 | 0.267 | 0.800 | 0.800 | 0.938 | 0.816 | 0.800 | 0.400 | **220.7s** |
| **Hybrid (RRF)** | **1.000** | 0.333 | 0.900 | 0.926 | **1.000** | **0.873** | **0.964** | **0.600** | 345.9s |
| **Reranking (Cohere)** | **1.000** | 0.333 | **1.000** | **1.000** | **1.000** | 0.821 | 0.963 | 0.400 | 403.1s |

*\*Each question has 1 labeled relevant chunk, so `1/3 = 0.333` is the mathematical Precision@3 ceiling at `k = 3`. Context Precision and Context Recall scored `1.000` across all scored samples.*

### Key Takeaways & Limitations
- **Hybrid & Reranking outperform Vector alone**: Dense vector missed 20% of questions (`Recall@3 = 0.800`), whereas **Keyword**, **Hybrid**, and **Reranking** retrieved 100% of gold chunks (`Recall@3 = 1.000`). **Hybrid** (`0.964` Factual Correctness) and **Reranking** (`0.963`) achieved the highest overall answer accuracy.
- **Local Reliability vs. Cloud Quotas**: Moving to local Ollama eliminated the HTTP 429 quota skips seen in earlier cloud runs (`0` skipped vs. `14` skipped previously), finishing `16/20` samples cleanly and `4/20` partial (each missing 1 judge metric on the longest multi-part question).
- **Trade-offs with a Compact 2B Model**: `qwen3-vl:2b-instruct` delivers strong local retrieval/QA accuracy but exhibits weak bracketed citation compliance (`0.40–0.60` page validity) and slower local judge throughput (~115 min for 20 full samples). Using an independent, larger judge model (e.g., `deepseek-r1:14b` or a cloud judge) and expanding beyond 5 synthetic questions is recommended for formal calibration.

---

## Project Structure

```text
RAG/
├── backend/
│   ├── __main__.py               # App launcher (python -B -m backend)
│   ├── document_extraction.py    # PDF text/table extraction, RapidOCR, multi-PDF batching
│   ├── rag_pipeline.py           # Title-based chunking, local embeddings, ChromaDB store
│   ├── retrieval_runtime.py      # Vector, BM25, Hybrid RRF, Multi-Query, Cohere Reranker
│   ├── generation_runtime.py     # Multimodal generation & history-aware query rewriting
│   ├── evaluation_data.py        # Dataset loading, IR metrics & quote evidence verification
│   ├── evaluation_generation.py  # Synthetic Q&A drafting & source verification
│   ├── evaluation_judge.py       # Ragas LLM-as-a-judge scoring
│   ├── evaluation_diagnostics.py # Metric failure diagnostics & summaries
│   ├── advanced_evaluation.py    # Multi-strategy evaluation runner & CLI
│   ├── processing_jobs.py        # Background job runner
│   ├── provider_limits.py        # Provider rate-limit handling & preflight checks
│   ├── files.py                  # Repo paths & atomic JSON persistence
│   └── api/                      # FastAPI routes, auth, SQLite catalog & SSE jobs
├── frontend/                     # React 19 + TypeScript + Vite + SWR UI
├── tests/                        # Offline pytest regression suite
├── requirements.txt              # Consolidated Python dependencies
└── .env.example                  # Environment configuration template
```

---

## Quick Start (Windows / PowerShell)

```powershell
# 1. Create & activate virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt

# 2. Build frontend (if not already built)
cd frontend
npm ci
npm run build
cd ..

# 3. Copy .env.example to .env, configure your provider (Ollama / Gemini / OpenAI), and launch
.\.venv\Scripts\python.exe -B -m backend
```

Open **http://127.0.0.1:8501** in your browser.

---

## Running Verification

```powershell
.\.venv\Scripts\python.exe -B -m pytest tests -q -p no:cacheprovider
cd frontend
npm run build
```
