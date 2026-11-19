"""Packaging must keep existing data locations and real ingestion usable."""
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from backend import generation_runtime, provider_limits, rag_pipeline
from backend.api.main import create_app
from backend.files import PROJECT_ROOT, read_json, write_json
from test_document_extraction import write_native_pdf
from test_vector_ingestion import LocalTestEmbeddings


def test_storage_defaults_remain_at_repository_root_from_another_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("RAG_EVALUATION_CACHE_DIR")
    monkeypatch.delenv("HF_HUB_CACHE", raising=False)
    monkeypatch.delenv("HF_HOME", raising=False)
    root = Path(__file__).resolve().parents[1]
    assert PROJECT_ROOT == root
    assert provider_limits.cache_root() == root / "evaluation_runs" / "cache"
    assert rag_pipeline._embedding_cache_directory() == root / ".cache" / "huggingface" / "hub"


def test_failed_json_update_retains_completed_artifact(tmp_path):
    path = tmp_path / "report.json"
    completed = {"reference": "Source café", "score": 0.8}
    write_json(path, completed)
    with pytest.raises(ValueError):
        write_json(path, {"score": float("nan")})
    assert read_json(path) == completed
    assert sorted(p.name for p in tmp_path.iterdir()) == ["report.json"]


def test_api_preparation_uses_selected_storage_root_from_another_directory(tmp_path, monkeypatch):
    root, working = tmp_path / "workspace", tmp_path / "unrelated"
    working.mkdir()
    monkeypatch.chdir(working)
    monkeypatch.setenv("PDF_EXTRACTION_MODE", "auto")
    monkeypatch.setattr(rag_pipeline, "get_embedding_model", LocalTestEmbeddings)
    monkeypatch.setattr(generation_runtime, "get_generation_model", lambda *a, **k: pytest.fail("Ingestion must be local"))
    pdf = write_native_pdf(tmp_path / "invoice.pdf", [
        "BT /F1 12 Tf 20 260 Td (Invoice total 123.45 USD.) Tj ET",
    ])
    app = create_app(root, warm=False)
    with TestClient(app) as client:
        response = client.post("/api/documents", headers={"X-RAG-Client": "1"},
                               files={"file": (pdf.name, pdf.read_bytes(), "application/pdf")})
        assert response.status_code == 202
        result = response.json()
        job_id, document_id = result["job"]["id"], result["document"]["id"]
        entry = app.state.jobs.get(client.cookies["rag_workspace"], job_id)
        entry["job"].future.result(timeout=20)
        snapshot = client.get(f"/api/jobs/{job_id}").json()
        assert snapshot["status"] == "succeeded", snapshot
        chunks = client.get(f"/api/documents/{document_id}/chunks").json()
        assert chunks["total"] == 1
        assert "123.45 USD" in chunks["items"][0]["text"]
        again = client.post("/api/documents", headers={"X-RAG-Client": "1"},
                            files={"file": (pdf.name, pdf.read_bytes(), "application/pdf")}).json()
        assert again["job"] is None
    assert app.state.catalog.path == root / "db" / "web_catalog.sqlite3"
    assert list((root / "docs").glob("*.pdf"))
    assert list((root / "db").glob("chroma_*/index_manifest.json"))
    assert list((root / ".cache" / "extractions").glob("*.json.gz"))
    assert list(working.iterdir()) == []
