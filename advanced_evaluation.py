import os
import logging

# Suppress noisy transformers and general warnings
logging.getLogger("transformers").setLevel(logging.ERROR)
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import asyncio
from dotenv import load_dotenv
from datasets import Dataset

# LangChain & Chroma
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_cohere import CohereRerank
from langchain_core.messages import HumanMessage
from langchain_core.documents import Document

# Ragas Evaluation
from ragas import evaluate, RunConfig
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

load_dotenv()

# --- CONFIGURATION ---
def get_db_path():
    db_root = "db"
    if not os.path.exists(db_root): return None
    dirs = [d for d in os.listdir(db_root) if os.path.isdir(os.path.join(db_root, d))]
    return os.path.join(db_root, dirs[0]) if dirs else None

DB_PATH = get_db_path()
GENERATION_MODEL = os.getenv("GENERATION_MODEL", "gemma-4-31b-it")
EVALUATION_MODEL = "deepseek-r1:14b"

# --- INITIALIZATION ---
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# 1. Generation Model (Gemma) - Used for answer generation
gen_llm = ChatGoogleGenerativeAI(model=GENERATION_MODEL, temperature=0)

# 2. Evaluation Wrapper to handle Gemma's multimodal/thinking output for Ragas
class GemmaTextWrapper(ChatGoogleGenerativeAI):
    def invoke(self, input, config=None, **kwargs):
        response = super().invoke(input, config, **kwargs)
        if isinstance(response.content, list):
            text_parts = [item["text"] for item in response.content if isinstance(item, dict) and "text" in item]
            text_parts += [item for item in response.content if isinstance(item, str)]
            response.content = "".join(text_parts)
        return response
    
    async def ainvoke(self, input, config=None, **kwargs):
        response = await super().ainvoke(input, config, **kwargs)
        if isinstance(response.content, list):
            text_parts = [item["text"] for item in response.content if isinstance(item, dict) and "text" in item]
            text_parts += [item for item in response.content if isinstance(item, str)]
            response.content = "".join(text_parts)
        return response

eval_wrapper_llm = GemmaTextWrapper(model=GENERATION_MODEL, temperature=0)

# Wrappers for Ragas (using Cloud API for evaluation for speed)
ragas_llm = LangchainLLMWrapper(eval_wrapper_llm, bypass_n=True)
ragas_embeddings = LangchainEmbeddingsWrapper(embedding_model)

def get_retrievers(db_path=None):
    target_db_path = db_path or DB_PATH
    if not target_db_path or not os.path.exists(target_db_path):
        raise ValueError(f"Database not found at {target_db_path}. Please upload a document in the UI first.")
        
    db = Chroma(persist_directory=target_db_path, embedding_function=embedding_model)
    all_docs_data = db.get()
    
    # Extract documents from the data returned by db.get()
    docs = []
    if 'documents' in all_docs_data and all_docs_data['documents']:
        for i in range(len(all_docs_data['documents'])):
            doc = Document(
                page_content=all_docs_data['documents'][i],
                metadata=all_docs_data['metadatas'][i] if all_docs_data['metadatas'] else {}
            )
            docs.append(doc)
    
    if not docs:
        raise ValueError("No documents found in the vector store. Evaluation cannot proceed.")
        
    vector = db.as_retriever(search_kwargs={"k": 5})
    bm25 = BM25Retriever.from_documents(docs)
    bm25.k = 5
    hybrid = EnsembleRetriever(retrievers=[vector, bm25], weights=[0.5, 0.5])
    
    return vector, bm25, hybrid, docs

from tenacity import retry, stop_after_attempt, wait_exponential

@retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=4, max=60))
async def get_rag_response(query, retriever, reranker=None):
    # Step 1: Retrieval
    docs = retriever.invoke(query)
    
    # Step 2: Optional Reranking
    if reranker:
        docs = reranker.compress_documents(docs, query)
    
    # Step 3: Generation
    context = "\n\n".join([d.page_content for d in docs])
    prompt = f"Answer the question based ONLY on the context.\n\nContext:\n{context}\n\nQuestion: {query}"
    response = await gen_llm.ainvoke([HumanMessage(content=prompt)])
    
    # Robustly extract text content from potential multimodal response
    content = response.content
    answer_text = ""
    if isinstance(content, list):
        text_parts = []
        for item in content:
            if isinstance(item, dict) and "text" in item:
                text_parts.append(item["text"])
            elif isinstance(item, str):
                text_parts.append(item)
        answer_text = "".join(text_parts)
    else:
        answer_text = str(content)
        
    return {
        "answer": answer_text,
        "contexts": [d.page_content for d in docs]
    }

async def run_evaluation(db_path=None):
    print(f"Starting Advanced RAG Evaluation", flush=True)
    print(f"  -> Generation: {GENERATION_MODEL}", flush=True)
    print(f"  -> Evaluation (Judge): {GENERATION_MODEL} (via Cloud API)", flush=True)
    
    vector_r, keyword_r, hybrid_r, all_docs = get_retrievers(db_path)
    
    # Define a high-quality Evaluation Set
    eval_questions = [
        {
            "question": "What is the baseline efficiency evaluated under optimal thermal conditions specifically for the system described in Section 4: Natural Language Processing?",
            "ground_truth": "The baseline efficiency of this system is evaluated at 94.2% under optimal thermal conditions within Section 4."
        },
        {
            "question": "According to Section 7: Cloud Infrastructure Scalability, what is the minimum requirement for data redundancy factors within the distributed transaction ledger framework?",
            "ground_truth": "Data redundancy factors must never drop below 3.0 within the distributed transaction ledger framework under Section 7."
        },
        {
            "question": "In Section 3: Autonomous Vehicle Navigation, what exact percentage of computational overhead is reduced by algorithm theta-prime compared to traditional matrix inversion?",
            "ground_truth": "Algorithm theta-prime reduces computational overhead by exactly 14.7% compared to traditional matrix inversion within Section 3."
        },
        {
            "question": "Under Section 11: Microservices Design Patterns, what specific handshake counts are mandated by Protocol delta-7 before data transit?",
            "ground_truth": "Protocol delta-7 requires a minimum of three distinct cryptographic handshakes before data transit under Section 11."
        },
        {
            "question": "What is the recommended room temperature for hosting the node clusters mentioned in rag_evaluation_100_pages.pdf?",
            "ground_truth": "I cannot answer this question because the document does not mention the recommended room temperature for hosting the node clusters."
        },
        {
            "question": "Which specific cloud service provider (like AWS or Azure) is recommended for scaling the infrastructure in rag_evaluation_100_pages.pdf?",
            "ground_truth": "I cannot answer this question because the document does not mention any specific cloud service providers like AWS or Azure."
        }
    ]

    reranker = None
    try:
        reranker = CohereRerank(model="rerank-english-v3.0", top_n=3)
    except:
        print("Cohere Reranker not available. Skipping Reranking stage.", flush=True)

    strategies = ["Keyword", "Vector", "Hybrid"]
    if reranker: strategies.append("Reranking")

    full_eval_data = {"question": [], "answer": [], "contexts": [], "ground_truth": [], "strategy_label": []}

    for strategy in strategies:
        print(f"\n[STRATEGY: {strategy}] Starting answer generation...", flush=True)
        
        for i, item in enumerate(eval_questions):
            query = item["question"]
            print(f"  -> Processing Question {i+1}/{len(eval_questions)}: {query[:50]}...", flush=True)
            
            if strategy == "Keyword":
                resp = await get_rag_response(query, keyword_r)
            elif strategy == "Vector":
                resp = await get_rag_response(query, vector_r)
            elif strategy == "Hybrid":
                resp = await get_rag_response(query, hybrid_r)
            else: # Reranking
                print(f"     (Applying Cohere Reranker...)", flush=True)
                resp = await get_rag_response(query, hybrid_r, reranker)
            
            full_eval_data["question"].append(query)
            full_eval_data["answer"].append(resp["answer"])
            full_eval_data["contexts"].append(resp["contexts"])
            full_eval_data["ground_truth"].append(item["ground_truth"])
            full_eval_data["strategy_label"].append(strategy)
        
        print(f"[STRATEGY: {strategy}] Done.", flush=True)

    # Convert to Dataset and evaluate ONCE for all strategies
    print(f"--- Running Ragas Evaluation on {len(full_eval_data['question'])} samples (96 total metric jobs) ---", flush=True)
    dataset = Dataset.from_dict({k: v for k, v in full_eval_data.items() if k != "strategy_label"})
    
    # Configure Ragas for heavy/slow processing
    run_config = RunConfig(timeout=3600, max_workers=4)
    
    result = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        llm=ragas_llm,
        embeddings=ragas_embeddings,
        run_config=run_config,
        # Ensure only one candidate is requested (Gemma doesn't support multi-candidates)
        # This fixes the 'Multiple candidates is not enabled for this model' error
    )
    
    res_df = result.to_pandas()
    res_df["strategy"] = full_eval_data["strategy_label"]

    # Summarize
    summary = res_df.groupby("strategy")[[
        "faithfulness", "answer_relevancy", "context_precision", "context_recall"
    ]].mean()

    print("\nFINAL NUMERICAL PERFORMANCE COMPARISON", flush=True)
    print("=" * 60, flush=True)
    print(summary.to_markdown(), flush=True)
    print("=" * 60, flush=True)
    
    # Save to CSV for visualization later
    summary.to_csv("rag_performance_metrics.csv")
    print("Numerical evidence saved to 'rag_performance_metrics.csv'", flush=True)
    return summary

if __name__ == "__main__":
    asyncio.run(run_evaluation())
