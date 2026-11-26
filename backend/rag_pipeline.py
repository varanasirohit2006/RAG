"""Ingestion pipeline: chunking, local embeddings, and ChromaDB vector store (rag-for-beginners style)."""

from __future__ import annotations

import base64
import hashlib
import json
import os
import threading
import time
import uuid
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.documents import Document
from unstructured.chunking.title import chunk_by_title

from backend.files import PROJECT_ROOT, read_json, write_json
from backend.generation_runtime import document_payload

load_dotenv()

PIPELINE_VERSION = "fast-v3"
MANIFEST_NAME = "index_manifest.json"
_index_locks = {}
_locks_guard = threading.Lock()


def embedding_name():
    return os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")


def embedding_revision():
    configured = os.getenv("EMBEDDING_REVISION")
    if configured:
        return configured
    return "1110a243fdf4706b3f48f1d95db1a4f5529b4d41" if embedding_name() == "sentence-transformers/all-MiniLM-L6-v2" else ""


def _embedding_cache_directory():
    if os.getenv("HF_HUB_CACHE"):
        return Path(os.getenv("HF_HUB_CACHE"))
    if os.getenv("HF_HOME"):
        return Path(os.getenv("HF_HOME")) / "hub"
    return PROJECT_ROOT / ".cache" / "huggingface" / "hub"


def index_lock(directory):
    key = str(Path(directory).resolve())
    with _locks_guard:
        return _index_locks.setdefault(key, threading.RLock())


def read_index_manifest(directory):
    value = read_json(Path(directory) / MANIFEST_NAME)
    return value if isinstance(value, dict) else None


def write_index_manifest(directory, manifest):
    write_json(Path(directory) / MANIFEST_NAME, manifest)


def elements_fingerprint(elements):
    digest = hashlib.sha256()
    for el in elements:
        digest.update(json.dumps(el.to_dict(), sort_keys=True, default=str).encode() + b"\n")
    return digest.hexdigest()


def index_is_ready(directory, source_fingerprint=None, document_hash=None):
    manifest = read_index_manifest(directory)
    if not manifest:
        return False
    return (
        manifest.get("schema_version") == 2
        and manifest.get("pipeline_version") == PIPELINE_VERSION
        and manifest.get("embedding_model") == embedding_name()
        and manifest.get("embedding_revision", "") == embedding_revision()
        and (source_fingerprint is None or manifest.get("source_fingerprint") == source_fingerprint)
        and (document_hash is None or manifest.get("document_hash") == document_hash)
        and manifest.get("count", 0) > 0
    )


@lru_cache(maxsize=1)
def get_embedding_model():
    from langchain_huggingface import HuggingFaceEmbeddings

    batch_size = int(os.getenv("EMBEDDING_BATCH_SIZE", "32"))
    model_kwargs = {"device": os.getenv("EMBEDDING_DEVICE", "cpu")}
    if embedding_revision():
        model_kwargs["revision"] = embedding_revision()
    return HuggingFaceEmbeddings(
        model_name=embedding_name(),
        cache_folder=str(_embedding_cache_directory()),
        model_kwargs=model_kwargs,
        encode_kwargs={"batch_size": batch_size, "normalize_embeddings": True},
    )


# ── Chunking & Multimodal Content Separation (from 8_multi_modal_rag.ipynb) ──

def create_chunks_by_title(elements):
    return chunk_by_title(elements=elements, combine_text_under_n_chars=200, max_characters=2000, new_after_n_chars=1500)


def separate_content_types(chunk):
    originals = getattr(getattr(chunk, "metadata", None), "orig_elements", None)
    elements = originals or [chunk]
    text_parts, tables, images, types = [], [], [], set()
    for el in elements:
        data = el.to_dict()
        el_type = data["type"]
        meta = data.get("metadata", {})
        types.add(el_type)
        if data.get("text"):
            text_parts.append(data["text"])
        if el_type == "Table" and meta.get("text_as_html"):
            tables.append(meta["text_as_html"])
        if el_type == "Image" and meta.get("image_base64"):
            images.append(meta["image_base64"])
    text = chunk.text if originals and hasattr(chunk, "text") else "\n\n".join(text_parts)
    return {"text": text, "tables": tables, "images": images, "types": types}


def chunks_to_documents(chunks, *, progress_callback=None):
    documents, total = [], len(chunks)
    for index, chunk in enumerate(chunks):
        data = separate_content_types(chunk)
        originals = getattr(getattr(chunk, "metadata", None), "orig_elements", None) or [chunk]
        pages = {el.metadata.page_number for el in originals if getattr(el, "metadata", None) and el.metadata.page_number is not None}
        sources = list(dict.fromkeys(el.metadata.filename for el in originals if getattr(el, "metadata", None) and el.metadata.filename))
        text = data["text"] or "\n".join(data["tables"]) or (
            f"Visual document evidence on pages {', '.join(str(p) for p in sorted(pages))}" if data["images"] else ""
        )
        if not text.strip():
            continue
        documents.append(Document(page_content=text, metadata={
            "source": ", ".join(sources) or "document",
            "pages": json.dumps(sorted(pages)),
            "content_types": ",".join(sorted(data["types"])),
            "original_content": json.dumps({"raw_text": data["text"], "tables_html": data["tables"], "images_base64": data["images"]}),
        }))
        if progress_callback:
            progress_callback(index + 1, total, "Preparing searchable text and source evidence...")
    return documents


# ── ChromaDB Vector Store (from 1_ingestion_pipeline.py & 8_multi_modal_rag.ipynb) ──

def _save_asset(asset_dir: Path, data: bytes) -> str:
    asset_dir.mkdir(parents=True, exist_ok=True)
    name = hashlib.sha256(data).hexdigest() + ".jpg"
    path = asset_dir / name
    if not path.exists():
        tmp = path.with_name(f"{name}.{uuid.uuid4().hex}.tmp")
        tmp.write_bytes(data)
        tmp.replace(path)
    return name


def _prepare_documents(documents, directory):
    asset_dir = Path(directory).resolve() / "assets"
    prepared = []
    for index, doc in enumerate(documents):
        meta = dict(doc.metadata)
        meta.pop("chunk_id", None)
        payload = document_payload(doc)
        image_paths = [_save_asset(asset_dir, base64.b64decode(b64, validate=True)) for b64 in payload["images_base64"]]
        for raw_path in payload["image_paths"]:
            p = Path(raw_path) if Path(raw_path).is_absolute() else Path(meta.get("asset_directory", asset_dir)) / raw_path
            image_paths.append(_save_asset(asset_dir, p.read_bytes()))
        if "original_content" in meta or image_paths:
            meta["original_content"] = json.dumps({
                "raw_text": payload["raw_text"], "tables_html": payload["tables_html"],
                "image_paths": list(dict.fromkeys(image_paths)),
            }, sort_keys=True)
        if image_paths:
            meta["asset_directory"] = str(asset_dir)
        fp_meta = {k: v for k, v in meta.items() if k != "asset_directory"}
        meta["chunk_id"] = hashlib.sha256(f"{index}:{doc.page_content}:{json.dumps(fp_meta, sort_keys=True, default=str)}".encode()).hexdigest()
        prepared.append(Document(page_content=doc.page_content, metadata=meta))
    return prepared


def open_vector_store(directory, embedding):
    from langchain_chroma import Chroma
    return Chroma(persist_directory=str(directory), embedding_function=embedding, collection_metadata={"hnsw:space": "cosine"})


def build_vector_store(documents, directory, embedding, *, progress_callback=None,
                       replace=False, source_fingerprint=None, document_hash=None):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    with index_lock(directory):
        documents = _prepare_documents(documents, directory)
        if not documents:
            raise ValueError("No chunks were produced; an empty index cannot answer questions.")
        db = open_vector_store(directory, embedding)
        desired_ids = [doc.metadata["chunk_id"] for doc in documents]
        existing_ids = set(db.get(include=[])["ids"])
        prev = read_index_manifest(directory)
        embedding_changed = prev and (prev.get("embedding_model") != embedding_name() or prev.get("embedding_revision", "") != embedding_revision())
        if embedding_changed and existing_ids:
            db.delete(ids=list(existing_ids))
            existing_ids = set()

        pending = [doc for doc in documents if embedding_changed or doc.metadata["chunk_id"] not in existing_ids]
        batch_size = int(os.getenv("INGESTION_BATCH_SIZE", "32"))
        for offset in range(0, len(pending), batch_size):
            batch = pending[offset:offset + batch_size]
            db.add_documents(batch, ids=[doc.metadata["chunk_id"] for doc in batch])
            if progress_callback:
                progress_callback(min(offset + len(batch), len(pending)), max(len(pending), 1), "Embedding and indexing chunks...")

        if replace and (stale := existing_ids - set(desired_ids)):
            db.delete(ids=list(stale))

        manifest = {
            "schema_version": 2, "pipeline_version": PIPELINE_VERSION, "chunk_config_version": "native-v3",
            "embedding_model": embedding_name(), "embedding_revision": embedding_revision(),
            "corpus_version": hashlib.sha256("\n".join(desired_ids).encode()).hexdigest(),
            "count": len(desired_ids), "ids": desired_ids,
        }
        if source_fingerprint is not None:
            manifest["source_fingerprint"] = source_fingerprint
        if document_hash is not None:
            manifest["document_hash"] = document_hash
        write_index_manifest(directory, manifest)
        if progress_callback:
            progress_callback(1, 1, f"Index ready: {len(desired_ids)} chunks.")
        return db


def load_vector_store(db_path):
    return open_vector_store(db_path, get_embedding_model())


def get_or_create_vector_store(elements, persist_directory, *, progress_callback=None, document_hash=None, source_documents=None):
    with index_lock(persist_directory):
        fingerprint = elements_fingerprint(elements)
        if source_documents:
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
            chunks, assigned = [], 0
            for src in source_documents:
                selected = [el for el in elements if el.metadata.page_number is not None and src["page_start"] <= el.metadata.page_number < src["page_start"] + src["page_count"]]
                assigned += len(selected)
                chunks.extend(create_chunks_by_title(selected))
            if assigned != len(elements):
                raise ValueError("Combined PDF evidence has missing page provenance.")
        else:
            chunks = create_chunks_by_title(elements)

        documents = chunks_to_documents(chunks, progress_callback=progress_callback)
        if source_documents:
            from backend.document_extraction import annotate_chunks
            annotate_chunks(documents, source_documents)
        db = build_vector_store(documents, persist_directory, get_embedding_model(), progress_callback=progress_callback, replace=True, source_fingerprint=fingerprint, document_hash=document_hash)
        if source_documents:
            write_index_manifest(Path(persist_directory), {**read_index_manifest(persist_directory), "source_documents": source_documents, "corpus_type": "combined_pdfs"})
        return db


def warm_runtime():
    started = time.perf_counter()
    print("Loading local search dependencies...", flush=True)
    get_embedding_model().embed_query("Local search initialization")
    print(f"Local search ready in {time.perf_counter() - started:.2f}s.", flush=True)
