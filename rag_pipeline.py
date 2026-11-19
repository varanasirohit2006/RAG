"""Reusable document ingestion, grounded generation, and retrieval services."""

from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.documents import Document
from unstructured.chunking.title import chunk_by_title

from document_extraction import ExtractionSettings, extract_document
from generation_runtime import (
    GenerationError,
    generate_final_answer,
    get_generation_model,
    prepare_generation_context,
)
from index_runtime import (
    PIPELINE_VERSION,
    build_vector_store,
    elements_fingerprint,
    embedding_name,
    embedding_revision,
    index_is_ready,
    index_lock,
    open_vector_store,
    read_index_manifest,
)
from retrieval_runtime import get_retrievers, get_reranker, retrieve_documents

load_dotenv()


@lru_cache(maxsize=1)
def get_embedding_model():
    from langchain_huggingface import HuggingFaceEmbeddings

    batch_size = int(os.getenv("EMBEDDING_BATCH_SIZE", "32"))
    if not 1 <= batch_size <= 512:
        raise ValueError("EMBEDDING_BATCH_SIZE must be between 1 and 512.")
    model_kwargs = {"device": os.getenv("EMBEDDING_DEVICE", "cpu")}
    local_path = _local_embedding_snapshot()
    if local_path:
        model_kwargs["local_files_only"] = True
    elif embedding_revision():
        model_kwargs["revision"] = embedding_revision()
    return HuggingFaceEmbeddings(
        model_name=str(local_path) if local_path else embedding_name(),
        cache_folder=str(_embedding_cache_directory()),
        model_kwargs=model_kwargs,
        encode_kwargs={"batch_size": batch_size, "normalize_embeddings": True},
    )


def _embedding_cache_directory():
    if os.getenv("HF_HUB_CACHE"):
        return Path(os.getenv("HF_HUB_CACHE"))
    if os.getenv("HF_HOME"):
        return Path(os.getenv("HF_HOME")) / "hub"
    return Path(__file__).resolve().parent / ".cache" / "huggingface" / "hub"


def _local_embedding_snapshot():
    configured = os.getenv("EMBEDDING_MODEL_PATH")
    if configured:
        path = Path(configured)
        if not (path / "modules.json").is_file():
            raise ValueError("EMBEDDING_MODEL_PATH must point to a downloaded SentenceTransformer model.")
        return path
    revision = embedding_revision()
    candidates = [_embedding_cache_directory(), Path.home() / ".cache" / "huggingface" / "hub"]
    model_folder = "models--" + embedding_name().replace("/", "--")
    for cache in candidates:
        root = cache / model_folder
        if revision:
            snapshots = [root / "snapshots" / revision]
        else:
            try:
                current = (root / "refs" / "main").read_text(encoding="utf-8").strip()
                snapshots = [root / "snapshots" / current]
            except OSError:
                try:
                    snapshots = list((root / "snapshots").iterdir())
                except OSError:
                    snapshots = []
        for path in snapshots:
            if (path / "modules.json").is_file() and (path / "config.json").is_file() and any(
                (path / weight).is_file() for weight in ("model.safetensors", "pytorch_model.bin")
            ):
                return path
    return None


def partition_document(file_path: str, *, extraction_mode=None, ocr_backend=None):
    return extract_document(
        file_path,
        settings=ExtractionSettings.from_env(mode=extraction_mode, ocr_backend=ocr_backend),
    )


def create_chunks_by_title(elements):
    return chunk_by_title(
        elements=elements, combine_text_under_n_chars=200,
        max_characters=2000, new_after_n_chars=1500,
    )


def separate_content_types(chunk):
    """Preserve original text/table/image evidence rather than generated substitutes."""
    originals = getattr(getattr(chunk, "metadata", None), "orig_elements", None)
    elements = originals or [chunk]
    text_parts, tables, images, types = [], [], [], set()
    for element in elements:
        data = element.to_dict()
        element_type = data["type"]
        metadata = data.get("metadata", {})
        types.add(element_type)
        text = data.get("text", "")
        if text:
            text_parts.append(text)
        if element_type == "Table" and metadata.get("text_as_html"):
            tables.append(metadata["text_as_html"])
        if element_type == "Image" and metadata.get("image_base64"):
            images.append(metadata["image_base64"])
    # CompositeElement/TableChunk text is already a bounded slice. Reconstructing
    # full original-element text here repeats the entire page for every slice.
    text = chunk.text if originals and hasattr(chunk, "text") else "\n\n".join(text_parts)
    return {"text": text, "tables": tables, "images": images, "types": types}


def create_ai_enhanced_summary(text: str, tables: list, images: list, *, provider=None):
    """Optional expensive enhancement; ordinary ingestion does not call this."""
    document = Document(page_content=text, metadata={
        "original_content": json.dumps({
            "raw_text": text, "tables_html": tables, "images_base64": images,
        }),
    })
    return generate_final_answer(
        [document], "Describe this document section for retrieval. Preserve its key facts, numbers, table labels, and image content.",
        provider=provider,
    )


def summarise_chunks(chunks, *, use_ai_summaries=False, provider=None, progress_callback=None):
    """Prepare searchable raw evidence with no model calls unless explicitly requested."""
    documents = []
    total = len(chunks)
    for index, chunk in enumerate(chunks):
        data = separate_content_types(chunk)
        originals = getattr(getattr(chunk, "metadata", None), "orig_elements", None) or [chunk]
        pages, sources = set(), []
        for element in originals:
            metadata = getattr(element, "metadata", None)
            if metadata and metadata.page_number is not None:
                pages.add(metadata.page_number)
            if metadata and metadata.filename and metadata.filename not in sources:
                sources.append(metadata.filename)
        text = data["text"]
        if use_ai_summaries and (data["tables"] or data["images"]):
            # Raw evidence remains available for the answer even when summaries are enabled.
            text = create_ai_enhanced_summary(text, data["tables"], data["images"], provider=provider)
        if not text.strip():
            if data["tables"]:
                text = "\n".join(data["tables"])
            elif data["images"]:
                text = "Visual document evidence on pages " + ", ".join(str(page) for page in sorted(pages))
            else:
                continue
        documents.append(Document(page_content=text, metadata={
            "source": ", ".join(sources) or "document",
            "pages": json.dumps(sorted(pages)),
            "content_types": ",".join(sorted(data["types"])),
            "original_content": json.dumps({
                "raw_text": data["text"], "tables_html": data["tables"], "images_base64": data["images"],
            }),
        }))
        if progress_callback:
            progress_callback(index + 1, total, "Preparing searchable text and source evidence...")
    return documents


def create_vector_store(documents, persist_directory="db/chroma_db", *, progress_callback=None):
    """Batch upserts and reuse existing record IDs instead of embedding them again."""
    return build_vector_store(
        documents, persist_directory, get_embedding_model(), progress_callback=progress_callback,
    )


def load_vector_store(db_path):
    return open_vector_store(db_path, get_embedding_model())


def get_or_create_vector_store(elements, persist_directory, *, progress_callback=None,
                               document_hash=None, source_documents=None):
    """Reopen a verified on-disk index across sessions and application restarts."""
    with index_lock(persist_directory):
        fingerprint = elements_fingerprint(elements)
        if source_documents:
            import hashlib
            fingerprint = hashlib.sha256((fingerprint + json.dumps(source_documents, sort_keys=True)).encode()).hexdigest()
        if index_is_ready(persist_directory, fingerprint, document_hash):
            db = load_vector_store(persist_directory)
            manifest = read_index_manifest(persist_directory)
            if db._collection.count() == manifest["count"]:
                if progress_callback:
                    progress_callback(1, 1, f"Reused saved index: {manifest['count']} chunks.")
                return db
        if progress_callback:
            progress_callback(0, 1, "Chunking document once...")
        if source_documents:
            chunks = []
            assigned = 0
            for source in source_documents:
                selected = [element for element in elements if element.metadata.page_number is not None
                            and source['page_start'] <= element.metadata.page_number < source['page_start'] + source['page_count']]
                assigned += len(selected)
                chunks.extend(create_chunks_by_title(selected))
            if assigned != len(elements):
                raise ValueError('Combined PDF evidence has missing page provenance; its original files cannot be mapped safely.')
        else:
            chunks = create_chunks_by_title(elements)
        documents = summarise_chunks(chunks, progress_callback=progress_callback)
        if source_documents:
            from pdf_batches import annotate_chunks
            annotate_chunks(documents, source_documents)
        database = build_vector_store(
            documents, persist_directory, get_embedding_model(), progress_callback=progress_callback,
            replace=True, source_fingerprint=fingerprint, document_hash=document_hash,
        )
        if source_documents:
            from index_runtime import write_index_manifest
            manifest = read_index_manifest(persist_directory)
            write_index_manifest(Path(persist_directory), {**manifest, 'source_documents': source_documents, 'corpus_type': 'combined_pdfs'})
        return database


def run_complete_ingestion_pipeline(pdf_path: str, *, persist_directory="db/chroma_db",
                                    progress_callback=None, document_hash=None):
    elements = partition_document(pdf_path)
    return get_or_create_vector_store(
        elements, persist_directory, progress_callback=progress_callback, document_hash=document_hash,
    )


if __name__ == "__main__":
    file_path = "./docs/sample.pdf"
    if os.path.exists(file_path):
        db = run_complete_ingestion_pipeline(file_path)
        query = "What is the main topic?"
        documents, _ = retrieve_documents(db, query)
        print(generate_final_answer(documents, query))
