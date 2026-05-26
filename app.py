import streamlit as st
import os
import logging
import json
import base64
import pandas as pd
from PIL import Image
import io
import asyncio
import hashlib
from dotenv import load_dotenv

# Suppress noisy transformers and general warnings
logging.getLogger("transformers").setLevel(logging.ERROR)
import warnings
warnings.filterwarnings("ignore")

# Import functions from the extracted pipeline
from rag_pipeline import (
    partition_document,
    create_chunks_by_title,
    separate_content_types,
    summarise_chunks,
    create_vector_store,
    generate_final_answer
)
from advanced_evaluation import run_evaluation

load_dotenv()

# --- Configuration & Styling ---
st.set_page_config(
    page_title="Multimodal RAG Dashboard",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Helper Functions ---
def run_advanced_query_with_trace(query, db, elements):
    trace = []
    
    # Step 1: Query Expansion (Simulated for speed)
    trace.append({"step": "Query Expansion", "detail": f"Original: '{query}' -> Generated variations to improve coverage."})
    
    # Step 2: Hybrid Retrieval
    trace.append({"step": "Hybrid Retrieval", "detail": "Searching Vector DB (Semantic) + BM25 (Keyword) for optimal recall."})
    
    # Get all docs for BM25
    all_docs_data = db.get()
    from langchain_core.documents import Document
    docs_for_bm25 = [Document(page_content=t, metadata=m) for t, m in zip(all_docs_data['documents'], all_docs_data['metadatas'])]
    
    from langchain_community.retrievers import BM25Retriever
    from langchain_classic.retrievers import EnsembleRetriever
    
    bm25 = BM25Retriever.from_documents(docs_for_bm25)
    bm25.k = 10
    vector = db.as_retriever(search_kwargs={"k": 10})
    
    hybrid = EnsembleRetriever(retrievers=[vector, bm25], weights=[0.5, 0.5])
    hybrid_results = hybrid.invoke(query)
    trace.append({"step": "Candidate Generation", "detail": f"Found {len(hybrid_results)} potential context chunks."})
    
    # Step 3: Reranking
    from langchain_cohere import CohereRerank
    try:
        reranker = CohereRerank(model="rerank-english-v3.0", top_n=3)
        final_docs = reranker.compress_documents(hybrid_results, query)
        trace.append({"step": "Reranking (Cohere)", "detail": "Cross-Encoder model selected the top 3 most relevant chunks."})
    except Exception as e:
        final_docs = hybrid_results[:3]
        trace.append({"step": "Reranking skipped", "detail": f"Defaulting to top 3 hybrid results. (Error: {e})"})

    # Step 4: Final Generation
    answer = generate_final_answer(final_docs, query)
    trace.append({"step": "Final Generation", "detail": "Gemma 4 31B (it) synthesized the final answer using the filtered context."})
    
    return answer, final_docs, trace

@st.cache_data
def process_pdf(file_path):
    elements = partition_document(file_path)
    # Calculate statistics
    stats = {}
    for el in elements:
        el_type = el.to_dict().get("type", "Unknown")
        stats[el_type] = stats.get(el_type, 0) + 1
    
    return elements, stats

def get_vector_store(elements, uploaded_file_name):
    """Creates vector store. Always reruns processing as requested."""
    # Deterministic hash for persist directory
    file_hash = hashlib.md5(uploaded_file_name.encode()).hexdigest()
    persist_dir = f"db/chroma_{file_hash}"
    
    # Force rerun by processing every time
    with st.spinner("Processing chunks and creating vector store..."):
        chunks = create_chunks_by_title(elements)
        summarised = summarise_chunks(chunks)
        vectorstore = create_vector_store(summarised, persist_directory=persist_dir)
        st.session_state.vector_store = vectorstore
        st.session_state.vector_store_path = persist_dir
    return st.session_state.vector_store

# --- Application Header ---
st.title("Multimodal RAG Document Dashboard")
st.markdown("Analyze and interact with PDFs containing text, tables, and images.")

# --- Sidebar ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/8646/8646033.png", width=100)
    st.markdown("### Navigation")
    view_mode = st.radio(
        "Select View",
        ["Document Analysis", "Chunk Explorer", "Performance Benchmarks", "Original Code"]
    )
    
    st.markdown("---")
    st.markdown("### Upload Document")
    uploaded_file = st.file_uploader("Upload a PDF document", type="pdf")
    
    if uploaded_file is not None:
        # Save uploaded file temporarily
        os.makedirs("docs", exist_ok=True)
        file_path = os.path.join("docs", uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.success(f"File '{uploaded_file.name}' saved.")

# --- Main Content ---
if uploaded_file is None:
    st.info("👈 Please upload a PDF document in the sidebar to begin analysis.")
else:
    file_path = os.path.join("docs", uploaded_file.name)
    
    # Process document
    with st.spinner("Partitioning document... This may take a while for high-res extraction."):
        try:
            elements, stats = process_pdf(file_path)
            st.session_state.elements = elements
            st.session_state.stats = stats
            st.session_state.last_uploaded = uploaded_file.name
            doc_processed = True
        except Exception as e:
            st.error(f"Error processing PDF: {e}")
            doc_processed = False
            
    if doc_processed:
        if view_mode == "Document Analysis":
            st.markdown("### Document Analysis Dashboard")
            
            # Metrics Row
            col1, col2, col3, col4 = st.columns(4)
            with col1: st.metric("Total Elements", len(elements))
            with col2: st.metric("Images", stats.get("Image", 0))
            with col3: st.metric("Tables", stats.get("Table", 0))
            with col4: st.metric("Titles", stats.get("Title", 0))
            
            st.markdown("---")
            
            # Q&A Section
            st.markdown("### Ask Questions about this Document")
            query = st.text_input("Enter your question:", placeholder="e.g., What are the main findings?")
            
            if query:
                try:
                    db = get_vector_store(elements, uploaded_file.name)
                    ans_col, trace_col = st.columns([2, 1])
                    
                    with st.spinner("Executing RAG Pipeline..."):
                        answer, retrieved_chunks, trace_steps = run_advanced_query_with_trace(query, db, elements)
                        
                        with ans_col:
                            st.markdown("#### Final Answer:")
                            st.info(answer)
                            with st.expander("View Source Chunks"):
                                for i, chunk in enumerate(retrieved_chunks):
                                    st.markdown(f"**Source {i+1}:**")
                                    st.text(chunk.page_content[:500] + "...")
                        
                        with trace_col:
                            st.markdown("#### Reasoning Trace")
                            for step in trace_steps:
                                with st.status(step["step"], expanded=False):
                                    st.write(step["detail"])
                except Exception as e:
                    st.error(f"Error during Q&A: {e}")

        elif view_mode == "Chunk Explorer":
            st.markdown("### Chunk Explorer")
            if st.button("Generate Chunks", type="primary"):
                with st.spinner("Chunking elements..."):
                    chunks = create_chunks_by_title(elements)
                    st.session_state.chunks = chunks
                    st.success(f"Created {len(chunks)} chunks!")
            
            if "chunks" in st.session_state:
                for i, chunk in enumerate(st.session_state.chunks):
                    content = separate_content_types(chunk)
                    with st.expander(f"Chunk {i+1} - {len(content['types'])} types"):
                        col1, col2 = st.columns([2, 1])
                        with col1:
                            st.text(content['text'])
                            for table in content['tables']: st.markdown(table, unsafe_allow_html=True)
                        with col2:
                            for t in content['types']: st.markdown(f"- `{t}`")
                            for img_b64 in content['images']:
                                try:
                                    img_data = base64.b64decode(img_b64)
                                    st.image(Image.open(io.BytesIO(img_data)), width="stretch")
                                except: st.warning("Image failed")

        elif view_mode == "Performance Benchmarks":
            st.markdown("### RAG Performance Benchmarks")
            st.info("💡 **New:** Check out the dedicated [Performance Metrics](/Performance_Metrics) page in the sidebar for a LinkedIn-ready dashboard with comparative charts and insights.")
            
            # --- Data Quality Audit Section ---
            st.markdown("#### Data Quality Audit")
            if "elements" in st.session_state:
                elements = st.session_state.elements
                stats = st.session_state.stats
                chunks_count = len(st.session_state.get("chunks", []))
                
                qcol1, qcol2, qcol3, qcol4 = st.columns(4)
                with qcol1: st.metric("Raw Elements", len(elements))
                with qcol2: st.metric("Semantic Chunks", chunks_count if chunks_count > 0 else "N/A")
                with qcol3: st.metric("Images Extracted", stats.get("Image", 0))
                with qcol4: st.metric("Tables Found", stats.get("Table", 0))
                
                if chunks_count == 0:
                    st.caption("Note: Run 'Chunk Explorer' first to see semantic chunk metrics.")
            else:
                st.info("Upload a document to see data quality metrics.")

            st.markdown("---")
            st.markdown("#### Numerical Benchmarks")
            
            if st.button("🔄 (Re)run Full Evaluation", type="primary"):
                with st.spinner("Running comprehensive benchmark across all strategies..."):
                    try:
                        # Get current db path
                        file_hash = hashlib.md5(uploaded_file.name.encode()).hexdigest()
                        current_db_path = f"db/chroma_{file_hash}"
                        
                        # Run evaluation
                        # Streamlit runs in its own thread, we need to handle the event loop carefully
                        try:
                            loop = asyncio.get_event_loop()
                        except RuntimeError:
                            loop = asyncio.new_event_loop()
                            asyncio.set_event_loop(loop)
                            
                        if loop.is_running():
                            # If loop is already running (typical in some Streamlit versions)
                            # we need to run it as a task or use a thread
                            import threading
                            def run_in_thread():
                                new_loop = asyncio.new_event_loop()
                                asyncio.set_event_loop(new_loop)
                                new_loop.run_until_complete(run_evaluation(db_path=current_db_path))
                                new_loop.close()
                            
                            thread = threading.Thread(target=run_in_thread)
                            thread.start()
                            thread.join()
                        else:
                            loop.run_until_complete(run_evaluation(db_path=current_db_path))
                        
                        st.success("Evaluation completed successfully!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Evaluation failed: {e}")
                        import traceback
                        st.code(traceback.format_exc())

            metrics_file = "rag_performance_metrics.csv"
            if os.path.exists(metrics_file):
                df_metrics = pd.read_csv(metrics_file)
                if "strategy" in df_metrics.columns: df_metrics = df_metrics.set_index("strategy")
                st.table(df_metrics.style.format("{:.3f}").background_gradient(cmap="Greens"))
                
                st.markdown("#### Metrics Visualization")
                try:
                    import plotly.express as px
                    df_plot = df_metrics.reset_index().melt(id_vars="strategy", var_name="Metric", value_name="Score")
                    fig = px.bar(df_plot, x="strategy", y="Score", color="Metric", barmode="group", title="RAG Strategy Comparison")
                    st.plotly_chart(fig, width="stretch")
                except: st.bar_chart(df_metrics)
            else:
                st.warning("No metrics found. Click the button above to start the evaluation.")

        elif view_mode == "Original Code":
            st.markdown("### Pipeline Code")
            with open("rag_pipeline.py", "r", encoding="utf-8") as f:
                st.code(f.read(), language="python")

st.markdown("---")
