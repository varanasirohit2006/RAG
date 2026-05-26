# Multimodal RAG & Evaluation Dashboard

Welcome to the **Multimodal Retrieval-Augmented Generation (RAG) Dashboard**! This project is a complete, production-grade implementation of a RAG pipeline designed to analyze complex, real-world PDF documents containing a mix of raw text, structural tables, and embedded images. 

It is built with **Streamlit** for the frontend, **LangChain** and **ChromaDB** for core orchestration and indexing, and evaluates retrieval performance using **Ragas** metrics across four strategies: Keyword, Vector, Hybrid (Ensemble), and Cross-Encoder Reranking.

---

## 🚀 Key Features

* **Multimodal Extraction:** Partitions PDFs into clean text elements, base64-encoded image structures, and HTML tables using `unstructured`.
* **AI-Enhanced Summarization:** Generates searchable descriptions for charts, tables, and image contents using Google's `gemma-4-31b-it` model.
* **Hybrid & Reranked Retrieval:** 
  * *Vector:* Semantic similarity searches.
  * *Keyword:* Exact text queries via BM25.
  * *Hybrid:* Reciprocal Rank Fusion (RRF) ensemble.
  * *Reranking:* Cohere Cross-Encoder re-prioritization.
* **LinkedIn-Ready Analytics Dashboard:** Displays comparative Ragas metrics (Faithfulness, Answer Relevancy, Context Precision, and Context Recall) that adapt dynamically to dark and light modes.

---

## 🛠️ Prerequisites & System Requirements

To extract text from images and scan high-resolution PDFs, you must install **Tesseract OCR** on your system.

### 1. Installing Tesseract OCR

#### On Windows (Recommended)
1. Download the Windows installer from [UB Mannheim Tesseract Installer](https://github.com/UB-Mannheim/tesseract/wiki).
2. Run the `.exe` installer. 
3. **Important:** Note the installation directory. By default, it installs to:
   `C:\Program Files\Tesseract-OCR\tesseract.exe`
4. The pipeline is pre-configured to look for Tesseract at this exact path. If you install it elsewhere, update line 29 of `rag_pipeline.py`:
   ```python
   unstructured_pytesseract.pytesseract.tesseract_cmd = r'C:\Path\To\Your\tesseract.exe'
   ```
5. Add Tesseract to your System PATH:
   * Search for "Edit the system environment variables" in your Windows search bar.
   * Click **Environment Variables...**.
   * Under "System variables", select **Path** and click **Edit...**.
   * Click **New** and add `C:\Program Files\Tesseract-OCR`.
   * Click OK to save.

#### On macOS
Install Tesseract using Homebrew:
```bash
brew install tesseract
```

#### On Linux (Ubuntu/Debian)
Install Tesseract using apt:
```bash
sudo apt-get update
sudo apt-get install -y tesseract-ocr libtesseract-dev
```

---

## 📦 Project Setup

Follow these steps to set up the project locally:

### Step 1: Create a Virtual Environment
It is highly recommended to isolate your dependencies using a virtual environment:
```bash
# Create the environment
python -m venv venv

# Activate on Windows (PowerShell)
.\venv\Scripts\Activate.ps1

# Activate on macOS/Linux
source venv/bin/activate
```

### Step 2: Install Python Dependencies
Install all required libraries using the provided requirements file:
```bash
pip install -r requirements.txt
```

### Step 3: Configure Environment Variables
Create a file named `.env` in the root folder (or modify the existing one) and add your API keys & model settings:
```env
GOOGLE_API_KEY=your_gemini_api_key
COHERE_API_KEY=your_cohere_api_key

# (Optional) Model configuration - Defaults to gemma-4-31b-it if not set
GENERATION_MODEL=gemma-4-31b-it
```

*Note: The `.env` file is excluded from git tracking in `.gitignore` to protect your API keys.*

---

## 🔄 Switching LLM Models

The project is fully modular and supports switching the LLM dynamically. You do not need to modify any Python files; simply change the model name in your `.env` file:
```env
# Switch to Gemini 1.5 Flash for high-speed execution
GENERATION_MODEL=gemini-1.5-flash
```

### Recommended Models:
* **`gemma-4-31b-it` (Default):** Excellent open-source reasoning model.
* **`gemini-1.5-flash`:** Ultra-fast, highly cost-efficient, and excellent at parsing multi-modal content (like images/tables).
* **`gemini-1.5-pro`:** Top-tier reasoning and accuracy, ideal for very large or complex technical documents.

*Note: Changing this environment variable updates the model globally across document summarization, RAG queries, multi-query generation, and all Ragas evaluation judges.*

---

## 💻 Running the Application

### 1. Launching the Streamlit UI
This runs the full dashboard app where you can upload documents, query them, inspect chunk types, and visualize Ragas evaluation scores:
```bash
streamlit run app.py
```

### 2. Running the RAG Evaluation Suite via CLI
If you want to run the full metrics pipeline manually in your terminal, run:
```bash
python advanced_evaluation.py
```
This saves the results to `rag_performance_metrics.csv` which is then visualized in the Streamlit UI dashboard.

---

## 📋 Dependency Checklist

The following essential libraries are installed as part of the setup:

* **RAG & LangChain Ecosystem:**
  * `langchain`: Core RAG orchestration.
  * `langchain-community`: Community retrievers (like BM25).
  * `langchain-classic`: Ensemble search wrapper.
  * `langchain_chroma`: Native vector store bindings.
  * `langchain-google-genai`: Wrapper for the Gemini / Gemma API.
  * `langchain-cohere`: Cohere Reranking models.
* **Document Extraction:**
  * `unstructured[all-docs]`: Multi-modal document partitioner.
  * `pytesseract` & `unstructured.pytesseract`: Connects python scripts to the system's Tesseract installation.
  * `Pillow`: Image decoding and manipulation.
* **Evaluation:**
  * `ragas`: RAG performance metrics calculation framework.
  * `datasets` & `pandas`: Internal data representation.
* **UI & Graphics:**
  * `streamlit`: Main dashboard framework.
  * `plotly`: Dynamic bar charts for metric comparisons.

---

## 💡 Troubleshooting & Common Issues

* **"Tesseract is not found or not in path"**
  Ensure you installed Tesseract OCR and that the file path specified in `rag_pipeline.py` (line 29) matches your local installation executable.
* **"Multiple candidates is not enabled for this model"**
  Gemini APIs do not support generating multiple completion options (`n > 1`) in a single query. The project resolves this by initializing Ragas with `LangchainLLMWrapper(..., bypass_n=True)`. Do not remove this flag, otherwise the evaluation will crash.
* **Streamlit Dark Mode Contrast Problems**
  The dashboard styles are fully responsive and adapt to Streamlit themes using CSS variables (`var(--text-color)`, etc.). If text seems hard to read, refresh your Streamlit tab to force the browser to sync theme settings.
