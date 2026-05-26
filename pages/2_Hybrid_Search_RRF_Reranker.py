import streamlit as st
import os
from collections import defaultdict
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_cohere import CohereRerank
from langchain_core.messages import HumanMessage, SystemMessage
from dotenv import load_dotenv
from pydantic import BaseModel
from typing import List

load_dotenv()

# --- Configuration & Styling ---
st.set_page_config(
    page_title="Hybrid Search + RRF + Reranker",
    page_icon=None,
    layout="wide"
)

st.title("Hybrid Search + RRF + Reranker")
st.markdown("Advanced RAG pipeline combining semantic search, keyword search, rank fusion, and reranking.")

# --- Cached Resources ---
@st.cache_resource
def get_models():
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    model_name = os.getenv("GENERATION_MODEL", "gemma-4-31b-it")
    llm = ChatGoogleGenerativeAI(model=model_name, temperature=0)
    return embedding_model, llm

@st.cache_resource
def initialize_retrievers(db_path, _embedding_model, vector_weight, bm25_weight):
    if not os.path.exists(db_path):
        return None, None, None
    
    try:
        db = Chroma(
            persist_directory=db_path,
            embedding_function=_embedding_model,
            collection_metadata={"hnsw:space": "cosine"}
        )
        
        all_docs_data = db.get()
        from langchain_core.documents import Document
        docs_for_bm25 = [Document(page_content=t, metadata=m) for t, m in zip(all_docs_data['documents'], all_docs_data['metadatas'])]
        
        if not docs_for_bm25:
            return None, None, None
            
        bm25 = BM25Retriever.from_documents(docs_for_bm25)
        bm25.k = 10
        vector = db.as_retriever(search_kwargs={"k": 10})
        
        hybrid = EnsembleRetriever(
            retrievers=[vector, bm25],
            weights=[vector_weight, bm25_weight]
        )
        return hybrid, vector, len(docs_for_bm25)
    except Exception as e:
        st.error(f"Initialization error: {e}")
        return None, None, None

# Setup
persistent_directory = "db/chroma_db"
embedding_model, llm = get_models()

# Pydantic model for multi-query
class QueryVariations(BaseModel):
    queries: List[str]

# --- Helper Functions ---
def reciprocal_rank_fusion(chunk_lists, k=60):
    rrf_scores = defaultdict(float)
    all_unique_chunks = {}
    for query_idx, chunks in enumerate(chunk_lists, 1):
        for position, chunk in enumerate(chunks, 1):
            chunk_content = chunk.page_content
            all_unique_chunks[chunk_content] = chunk
            position_score = 1 / (k + position)
            rrf_scores[chunk_content] += position_score
    return sorted(
        [(all_unique_chunks[chunk_content], score) for chunk_content, score in rrf_scores.items()],
        key=lambda x: x[1],
        reverse=True
    )

# --- Sidebar ---
with st.sidebar:
    st.markdown("### Search Settings")
    
    # List available databases
    db_root = "db"
    available_dbs = []
    if os.path.exists(db_root):
        available_dbs = [d for d in os.listdir(db_root) if os.path.isdir(os.path.join(db_root, d))]
    
    if available_dbs:
        selected_db = st.selectbox("Select Vector Store", available_dbs)
        db_path = os.path.join(db_root, selected_db)
    else:
        st.error("No vector stores found in 'db/' folder. Please upload a document in the main app first.")
        db_path = persistent_directory

    vector_weight = st.slider("Vector Weight", 0.0, 1.0, 0.7)
    bm25_weight = round(1.0 - vector_weight, 1)
    st.write(f"BM25 Weight: {bm25_weight}")
    
    st.markdown("---")
    st.markdown("### Reranker Settings")
    use_reranker = st.checkbox("Use Cohere Reranker", value=True)
    top_n_rerank = st.slider("Top N Reranked Docs", 1, 10, 5)

# Initialize Retrievers (Cached)
hybrid_retriever, vector_retriever, doc_count = initialize_retrievers(db_path, embedding_model, vector_weight, bm25_weight)

if hybrid_retriever:
    st.success(f"Connected to: **{db_path}** ({doc_count} docs)")
else:
    st.warning("Retriever initialization failed or vector store not found.")

# --- Main Logic ---
query = st.text_input("Enter your query:", placeholder="e.g., Tesla financial performance and production updates")

if st.button("Run Advanced Retrieval") and hybrid_retriever:
    if query:
        # Step 1: Multi-Query
        with st.expander("1. Query Variations", expanded=False):
            with st.spinner("Generating variations..."):
                llm_with_tools = llm.with_structured_output(QueryVariations)
                mq_prompt = f"Generate 3 different variations of this query: {query}"
                mq_response = llm_with_tools.invoke(mq_prompt)
                variations = mq_response.queries
                for v in variations:
                    st.write(f"- {v}")
        
        # Step 2: Hybrid Search & RRF
        st.markdown("### 2. Hybrid Search & RRF")
        with st.spinner("Performing hybrid search..."):
            hybrid_results = hybrid_retriever.invoke(query)
            
            if 'variations' in locals():
                st.markdown("#### Manual RRF")
                chunk_lists = [vector_retriever.invoke(v) for v in variations]
                fused_results = reciprocal_rank_fusion(chunk_lists)
                st.write(f"Fused {len(fused_results)} unique chunks using RRF.")
                if st.checkbox("Show RRF Results"):
                    for i, (doc, score) in enumerate(fused_results[:5], 1):
                        st.write(f"**Rank {i} (Score: {score:.4f}):** {doc.page_content[:200]}...")

            st.markdown("#### Ensemble Hybrid Results")
            if st.checkbox("Show Ensemble Raw Results"):
                for i, doc in enumerate(hybrid_results[:5], 1):
                    st.write(f"**Rank {i}:** {doc.page_content[:200]}...")

        # Step 3: Reranking
        if use_reranker:
            st.markdown("### 3. Cohere Reranking")
            with st.spinner("Reranking results..."):
                try:
                    reranker = CohereRerank(model="rerank-english-v3.0", top_n=top_n_rerank)
                    final_docs = reranker.compress_documents(hybrid_results, query)
                    for i, doc in enumerate(final_docs, 1):
                        with st.expander(f"🏆 Reranked Rank {i}"):
                            st.write(doc.page_content)
                except Exception as e:
                    st.error(f"Reranking failed: {e}")
                    final_docs = hybrid_results[:top_n_rerank]
        else:
            final_docs = hybrid_results[:top_n_rerank]

        # Step 4: Final Answer
        st.markdown("### 4. Final RAG Response")
        with st.spinner("Generating final answer..."):
            context = "\n\n".join([f"- {doc.page_content}" for doc in final_docs])
            combined_input = f"Based on the following documents, answer: {query}\n\nDocuments:\n{context}"
            messages = [SystemMessage(content="You are a helpful assistant."), HumanMessage(content=combined_input)]
            result = llm.invoke(messages)
            
            # Robustly extract text content
            content = result.content
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
            st.write(answer_text)
    else:
        st.error("Please enter a query.")
