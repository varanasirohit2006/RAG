"""
Multimodal RAG Pipeline
========================
This module contains the complete pipeline for:
1. Document partitioning (PDF extraction)
2. Chunking by title
3. Content type separation (text, tables, images)
4. AI-enhanced summarization
5. Vector store creation (ChromaDB)
6. RAG query and answer generation
"""

import json
import os
from typing import List
import unstructured_pytesseract
from unstructured.partition.pdf import partition_pdf
from unstructured.chunking.title import chunk_by_title
from langchain_core.documents import Document
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv

load_dotenv()

# Configure Tesseract
unstructured_pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def partition_document(file_path: str):
    """Extract elements from PDF using unstructured"""
    print(f"Partitioning document: {file_path}")
    elements = partition_pdf(
        filename=file_path,
        strategy="hi_res",
        infer_table_structure=True,
        extract_image_block_types=["Image"],
        extract_image_block_to_payload=True
    )
    print(f"Extracted {len(elements)} elements")
    return elements

def create_chunks_by_title(elements):
    """Create semantically meaningful chunks"""
    print("Creating chunks by title...")
    chunks = chunk_by_title(
        elements=elements,
        combine_text_under_n_chars=200,
        max_characters=2000,
        new_after_n_chars=1500,
    )
    print(f"Created {len(chunks)} chunks")
    return chunks

def separate_content_types(chunk):
    """Break a chunk into its component types (text, tables, images)"""
    if hasattr(chunk, 'metadata') and hasattr(chunk.metadata, 'orig_elements'):
        elements = chunk.metadata.orig_elements
    else:
        elements = [chunk]
    
    text_parts, tables, images, element_types = [], [], [], set()
    
    for element in elements:
        element_dict = element.to_dict()
        element_type = element_dict["type"]
        element_types.add(element_type)
        
        if element_type == "Table":
            html = element_dict["metadata"].get("text_as_html", "")
            if html: tables.append(html)
            text_parts.append(element_dict.get("text", ""))
        elif element_type == "Image":
            img_base64 = element_dict["metadata"].get("image_base64", "")
            if img_base64: images.append(img_base64)
            text_parts.append(element_dict.get("text", ""))
        else:
            text_parts.append(element_dict.get("text", ""))
    
    return {"text": "\n\n".join(text_parts), "tables": tables, "images": images, "types": element_types}

def create_ai_enhanced_summary(text: str, tables: list, images: list) -> str:
    """Use AI to create searchable descriptions of multimodal content"""
    try:
        model_name = os.getenv("GENERATION_MODEL", "gemma-4-31b-it")
        llm = ChatGoogleGenerativeAI(model=model_name, temperature=0)
        message_content = [{"type": "text", "text": f"Analyze this document section.\n\nTEXT:\n{text}\n\nTables: {len(tables)}"}]
        
        for i, table_html in enumerate(tables):
            message_content.append({"type": "text", "text": f"Table {i+1} HTML:\n{table_html}"})
        
        max_images = 3
        for img_base64 in images[:max_images]:
            message_content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"}})
        
        if len(images) > max_images:
            message_content[0]["text"] += f"\nNote: {len(images) - max_images} additional images omitted."

        response = llm.invoke([HumanMessage(content=message_content)])
        
        # Robustly extract text content
        content = response.content
        if isinstance(content, list):
            text_parts = []
            for item in content:
                if isinstance(item, dict) and "text" in item:
                    text_parts.append(item["text"])
                elif isinstance(item, str):
                    text_parts.append(item)
            return "".join(text_parts)
        return str(content)
    except Exception as e:
        print(f"AI summary failed: {e}")
        return f"{text[:300]}... [Tables: {len(tables)}, Images: {len(images)}]"

def summarise_chunks(chunks):
    """Process all chunks with AI Summaries"""
    print("Processing chunks with AI Summaries...")
    langchain_documents = []
    for i, chunk in enumerate(chunks):
        print(f"   Processing chunk {i+1}/{len(chunks)}")
        content_data = separate_content_types(chunk)
        if content_data['tables'] or content_data['images']:
            enhanced_content = create_ai_enhanced_summary(content_data['text'], content_data['tables'], content_data['images'])
        else:
            enhanced_content = content_data['text']
        
        doc = Document(
            page_content=enhanced_content,
            metadata={"original_content": json.dumps({"raw_text": content_data['text'], "tables_html": content_data['tables'], "images_base64": content_data['images']})}
        )
        langchain_documents.append(doc)
    return langchain_documents

def create_vector_store(documents, persist_directory="db/chroma_db"):
    """Create and persist ChromaDB vector store"""
    print(f"Creating vector store at {persist_directory}...")
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embedding_model,
        persist_directory=persist_directory, 
        collection_metadata={"hnsw:space": "cosine"}
    )
    return vectorstore

def run_complete_ingestion_pipeline(pdf_path: str):
    """Run the complete RAG ingestion pipeline"""
    elements = partition_document(pdf_path)
    chunks = create_chunks_by_title(elements)
    summarised = summarise_chunks(chunks)
    return create_vector_store(summarised)

def generate_final_answer(chunks, query):
    """Generate final answer using multimodal content"""
    try:
        model_name = os.getenv("GENERATION_MODEL", "gemma-4-31b-it")
        llm = ChatGoogleGenerativeAI(model=model_name, temperature=0)
        prompt_text = f"Based on the following documents, answer: {query}\n\nCONTENT:\n"
        
        for i, chunk in enumerate(chunks):
            prompt_text += f"--- Doc {i+1} ---\n"
            if "original_content" in chunk.metadata:
                data = json.loads(chunk.metadata["original_content"])
                prompt_text += f"TEXT: {data.get('raw_text', '')}\n"
                for j, table in enumerate(data.get('tables_html', [])):
                    prompt_text += f"Table {j+1}: {table}\n"
        
        message_content = [{"type": "text", "text": prompt_text}]
        max_total_images, images_added = 5, 0
        for chunk in chunks:
            if images_added >= max_total_images: break
            if "original_content" in chunk.metadata:
                imgs = json.loads(chunk.metadata["original_content"]).get("images_base64", [])
                for img in imgs:
                    if images_added < max_total_images:
                        message_content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img}"}})
                        images_added += 1
        
        response = llm.invoke([HumanMessage(content=message_content)])
        
        # Robustly extract text content
        content = response.content
        if isinstance(content, list):
            text_parts = []
            for item in content:
                if isinstance(item, dict) and "text" in item:
                    text_parts.append(item["text"])
                elif isinstance(item, str):
                    text_parts.append(item)
            return "".join(text_parts)
        return str(content)
    except Exception as e:
        print(f"Answer failed: {e}")
        return "Error generating answer."

if __name__ == "__main__":
    # Example usage
    file_path = "./docs/sample.pdf"
    if os.path.exists(file_path):
        db = run_complete_ingestion_pipeline(file_path)
        query = "What is the main topic?"
        retrieved = db.as_retriever(search_kwargs={"k": 3}).invoke(query)
        print(generate_final_answer(retrieved, query))
