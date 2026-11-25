"""Durable, batched indexes with image assets outside vector metadata."""

from __future__ import annotations

import base64
import hashlib
import json
import os
import threading
import uuid
from pathlib import Path

from langchain_core.documents import Document

from backend.generation_runtime import document_payload
from backend.files import read_json, write_json

PIPELINE_VERSION = "fast-v3"
MANIFEST_NAME = "index_manifest.json"
_locks_guard = threading.Lock()
_index_locks = {}


def embedding_name():
    return os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")


def embedding_revision():
    configured = os.getenv("EMBEDDING_REVISION")
    if configured:
        return configured
    if embedding_name() == "sentence-transformers/all-MiniLM-L6-v2":
        return "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
    return ""


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
    for element in elements:
        digest.update(json.dumps(element.to_dict(), sort_keys=True, default=str).encode())
        digest.update(b"\n")
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


def _write_asset(asset_directory, data):
    filename = hashlib.sha256(data).hexdigest() + ".jpg"
    path = asset_directory / filename
    if not path.exists():
        temporary = path.with_name(f"{path.name}.{uuid.uuid4().hex}.tmp")
        temporary.write_bytes(data)
        temporary.replace(path)
    return filename


def _prepare_documents(documents, directory):
    asset_directory = Path(directory).resolve() / "assets"
    prepared = []
    for index, document in enumerate(documents):
        metadata = dict(document.metadata)
        metadata.pop("chunk_id", None)
        payload = document_payload(document)
        image_paths = []
        for inline in payload["images_base64"]:
            asset_directory.mkdir(parents=True, exist_ok=True)
            data = base64.b64decode(inline, validate=True)
            image_paths.append(_write_asset(asset_directory, data))
        for image_path in payload["image_paths"]:
            path = Path(image_path)
            if not path.is_absolute():
                path = Path(metadata.get("asset_directory", asset_directory)) / path
            asset_directory.mkdir(parents=True, exist_ok=True)
            image_paths.append(_write_asset(asset_directory, path.read_bytes()))
        if "original_content" in metadata or image_paths:
            metadata["original_content"] = json.dumps({
                "raw_text": payload["raw_text"], "tables_html": payload["tables_html"],
                "image_paths": list(dict.fromkeys(image_paths)),
            }, sort_keys=True)
        if image_paths:
            metadata["asset_directory"] = str(asset_directory)
        fingerprint_metadata = {key: value for key, value in metadata.items() if key != "asset_directory"}
        chunk_id = hashlib.sha256(
            f"{index}:{document.page_content}:{json.dumps(fingerprint_metadata, sort_keys=True, default=str)}".encode()
        ).hexdigest()
        metadata["chunk_id"] = chunk_id
        prepared.append(Document(page_content=document.page_content, metadata=metadata))
    return prepared


def open_vector_store(directory, embedding):
    from langchain_chroma import Chroma

    return Chroma(
        persist_directory=str(directory), embedding_function=embedding,
        collection_metadata={"hnsw:space": "cosine"},
    )


def build_vector_store(documents, directory, embedding, *, progress_callback=None,
                       replace=False, source_fingerprint=None, document_hash=None):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    with index_lock(directory):
        documents = _prepare_documents(documents, directory)
        if not documents:
            raise ValueError("No chunks were produced; an empty index cannot answer questions.")
        db = open_vector_store(directory, embedding)
        desired_ids = [document.metadata["chunk_id"] for document in documents]
        existing_ids = set(db.get(include=[])["ids"])
        previous = read_index_manifest(directory)
        embedding_changed = previous and (
            previous.get("embedding_model") != embedding_name()
            or previous.get("embedding_revision", "") != embedding_revision()
        )
        if embedding_changed and existing_ids:
            db.delete(ids=list(existing_ids))
            existing_ids = set()
        pending = [
            document for document in documents
            if embedding_changed or document.metadata["chunk_id"] not in existing_ids
        ]
        batch_size = int(os.getenv("INGESTION_BATCH_SIZE", "32"))
        if not 1 <= batch_size <= 512:
            raise ValueError("INGESTION_BATCH_SIZE must be between 1 and 512.")
        total = len(pending)
        if progress_callback:
            progress_callback(0, max(total, 1), "Embedding document chunks locally...")
        for offset in range(0, total, batch_size):
            batch = pending[offset:offset + batch_size]
            db.add_documents(batch, ids=[document.metadata["chunk_id"] for document in batch])
            if progress_callback:
                progress_callback(min(offset + len(batch), total), total, "Embedding and indexing chunks...")
        # Replace only a document-scoped index after all desired records are written.
        if replace:
            stale = existing_ids - set(desired_ids)
            if stale:
                db.delete(ids=list(stale))
        corpus_version = hashlib.sha256("\n".join(desired_ids).encode()).hexdigest()
        manifest = {
            "schema_version": 2, "pipeline_version": PIPELINE_VERSION,
            "chunk_config_version": "native-v3", "embedding_model": embedding_name(),
            "embedding_revision": embedding_revision(),
            "corpus_version": corpus_version, "count": len(desired_ids), "ids": desired_ids,
        }
        if source_fingerprint is not None:
            manifest["source_fingerprint"] = source_fingerprint
        if document_hash is not None:
            manifest["document_hash"] = document_hash
        write_index_manifest(directory, manifest)
        if progress_callback:
            progress_callback(1, 1, f"Index ready: {len(desired_ids)} chunks.")
        return db
