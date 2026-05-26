# Multimodal RAG & Evaluation Dashboard

This project is a Streamlit dashboard that implements a Multimodal Retrieval-Augmented Generation (RAG) pipeline to ingest complex PDFs (containing text, tables, and images) and evaluate different retrieval strategies using Ragas.

## Prerequisites

To extract elements from PDFs, you need to install Tesseract OCR:

- **Windows:** Download the installer and install to the default path `C:\Program Files\Tesseract-OCR\tesseract.exe`.
- **macOS:** Run `brew install tesseract`.
- **Linux:** Run `sudo apt-get install tesseract-ocr`.

## Setup

1. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   # Windows
   .\venv\Scripts\Activate.ps1
   # macOS/Linux
   source venv/bin/activate
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Configure environment variables in `.env`:
   ```env
   GOOGLE_API_KEY=your_gemini_api_key
   COHERE_API_KEY=your_cohere_api_key
   GENERATION_MODEL=gemma-4-31b-it
   ```

## Model Configuration

You can change the generation model globally by updating the `GENERATION_MODEL` variable in your `.env` file. Recommended models include:
- `gemma-4-31b-it` (Default)
- `gemini-1.5-flash` (Fastest, recommended for multimodal data)
- `gemini-1.5-pro` (Highest accuracy)

## Running the Application

- **Streamlit Dashboard:**
  ```bash
  streamlit run app.py
  ```
- **CLI Ragas Evaluation:**
  ```bash
  python advanced_evaluation.py
  ```
