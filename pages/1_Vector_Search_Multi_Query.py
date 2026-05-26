import streamlit as st
import os
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
from pydantic import BaseModel
from typing import List

load_dotenv()

# --- Configuration & Styling ---
st.set_page_config(
    page_title="Vector Search + Multi-Query",
    page_icon=None,
    layout="wide"
)

st.title("Vector Search + Multi-Query Retrieval")
st.markdown("This page generates multiple query variations to improve retrieval coverage.")

# --- Cached Resources ---
@st.cache_resource
def get_models():
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    model_name = os.getenv("GENERATION_MODEL", "gemma-4-31b-it")
    llm = ChatGoogleGenerativeAI(model=model_name, temperature=0)
    return embedding_model, llm

@st.cache_resource
def load_vector_db(db_path, _embedding_model):
    if os.path.exists(db_path):
        return Chroma(
            persist_directory=db_path,
            embedding_function=_embedding_model,
            collection_metadata={"hnsw:space": "cosine"}
        )
    return None

# Setup
persistent_directory = "db/chroma_db"
embedding_model, llm = get_models()

# Pydantic model for structured output
class QueryVariations(BaseModel):
    queries: List[str]

# --- Sidebar ---
with st.sidebar:
    st.markdown("### Settings")
    
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

    num_variations = st.slider("Number of Query Variations", 1, 5, 3)
    k_value = st.slider("Retriever K", 1, 10, 5)

# Initialize DB if it exists
db = load_vector_db(db_path, embedding_model)
if db:
    st.success(f"Connected to: **{db_path}**")
else:
    st.warning(f"Vector store not found at {db_path}.")

# --- Main Logic ---
query = st.text_input("Enter your query:", placeholder="e.g., How does Tesla make money?")

if st.button("Run Multi-Query Retrieval") and db:
    if query:
        # Step 1: Generate Multiple Query Variations
        st.markdown("### 1. Generating Query Variations")
        with st.spinner("Generating variations..."):
            llm_with_tools = llm.with_structured_output(QueryVariations)
            prompt = f"""Generate {num_variations} different variations of this query that would help retrieve relevant documents:

Original query: {query}

Return {num_variations} alternative queries that rephrase or approach the same question from different angles."""

            response = llm_with_tools.invoke(prompt)
            query_variations = response.queries

            for i, variation in enumerate(query_variations, 1):
                st.write(f"**{i}.** {variation}")

        st.divider()

        # Step 2: Search with Each Query Variation
        st.markdown("### 2. Multi-Query Retrieval Results")
        retriever = db.as_retriever(search_kwargs={"k": k_value})
        
        cols = st.columns(len(query_variations))
        
        for i, variation in enumerate(query_variations):
            with cols[i]:
                st.markdown(f"**Query {i+1}:**\n*{variation}*")
                with st.spinner(f"Searching..."):
                    docs = retriever.invoke(variation)
                    st.write(f"Retrieved {len(docs)} documents")
                    for j, doc in enumerate(docs, 1):
                        with st.expander(f"Doc {j}"):
                            st.write(doc.page_content)
    else:
        st.error("Please enter a query.")
elif not db:
    st.info("Waiting for vector store connection...")
