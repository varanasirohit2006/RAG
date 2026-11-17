import base64
import io
import json
import time
from types import SimpleNamespace

import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.messages import AIMessage
from PIL import Image
from unstructured.documents.elements import ElementMetadata, NarrativeText, Table

import generation_runtime
import index_runtime
import rag_pipeline
import retrieval_runtime


class CountingEmbeddings(Embeddings):
    def __init__(self):
        self.document_calls = 0

    def embed_documents(self, texts):
        self.document_calls += 1
        return [[float(len(text)), 1.0, 0.5] for text in texts]

    def embed_query(self, text):
        return [float(len(text)), 1.0, 0.5]


def sample_documents():
    return [
        Document(page_content="The invoice total is 123.45 USD.", metadata={"page": 1}),
        Document(page_content="The balance is 200.00 USD.", metadata={"page": 2}),
    ]


def test_reingestion_skips_completed_embedding_batches(tmp_path, monkeypatch):
    embedding = CountingEmbeddings()
    monkeypatch.setattr(rag_pipeline, "get_embedding_model", lambda: embedding)
    db = rag_pipeline.create_vector_store(sample_documents(), str(tmp_path / "db"))
    assert len(db.get()["ids"]) == 2
    assert embedding.document_calls == 1
    reopened = rag_pipeline.create_vector_store(sample_documents(), str(tmp_path / "db"))
    assert len(reopened.get()["ids"]) == 2
    assert embedding.document_calls == 1


def test_verified_saved_index_reopens_without_rechunking(tmp_path, monkeypatch):
    embedding = CountingEmbeddings()
    monkeypatch.setattr(rag_pipeline, "get_embedding_model", lambda: embedding)
    elements = [NarrativeText(
        "The verified invoice total is 123.45 USD.",
        metadata=ElementMetadata(filename="invoice.pdf", page_number=1),
    )]
    directory = str(tmp_path / "db")
    rag_pipeline.get_or_create_vector_store(elements, directory, document_hash="source-pdf-sha")
    manifest = index_runtime.read_index_manifest(directory)
    assert manifest["document_hash"] == "source-pdf-sha"
    assert manifest["count"] == 1
    assert manifest["pipeline_version"] == rag_pipeline.PIPELINE_VERSION
    def rechunking_is_a_bug(*args, **kwargs):
        pytest.fail("Saved index must be reused")
    monkeypatch.setattr(rag_pipeline, "create_chunks_by_title", rechunking_is_a_bug)
    reopened = rag_pipeline.get_or_create_vector_store(elements, directory, document_hash="source-pdf-sha")
    assert len(reopened.get()["ids"]) == 1
    assert embedding.document_calls == 1


def test_raw_table_ingestion_preserves_evidence_without_model_calls(monkeypatch):
    def expensive_model(*args, **kwargs):
        pytest.fail("Default ingestion must not summarize with an LLM")
    monkeypatch.setattr(rag_pipeline, "create_ai_enhanced_summary", expensive_model)
    table = Table("Invoice total\t123.45 USD", metadata=ElementMetadata(
        filename="invoice.pdf", page_number=3,
        text_as_html="<table><tr><td>Total</td><td>123.45 USD</td></tr></table>",
    ))
    documents = rag_pipeline.summarise_chunks([table])
    payload = json.loads(documents[0].metadata["original_content"])
    assert "123.45 USD" in documents[0].page_content
    assert "123.45 USD" in payload["raw_text"]
    assert payload["tables_html"]
    assert json.loads(documents[0].metadata["pages"]) == [3]


def test_images_move_to_assets_and_remain_in_generation_context(tmp_path, monkeypatch):
    embedding = CountingEmbeddings()
    monkeypatch.setattr(rag_pipeline, "get_embedding_model", lambda: embedding)
    monkeypatch.setenv("GENERATION_MAX_IMAGES", "1")
    with Image.new("RGB", (1600, 600), "white") as image:
        with io.BytesIO() as output:
            image.save(output, "JPEG")
            image_base64 = base64.b64encode(output.getvalue()).decode("ascii")
    document = Document(page_content="Sales chart annual revenue.", metadata={
        "pages": "[5]", "source": "report.pdf",
        "original_content": json.dumps({
            "raw_text": "Sales chart annual revenue.", "tables_html": [],
            "images_base64": [image_base64],
        }),
    })
    db = rag_pipeline.create_vector_store([document], str(tmp_path / "db"))
    metadata = db.get()["metadatas"][0]
    payload = json.loads(metadata["original_content"])
    assert "images_base64" not in payload
    assert len(payload["image_paths"]) == 1
    assert (tmp_path / "db" / "assets" / payload["image_paths"][0]).is_file()
    retrieved = Document(page_content=document.page_content, metadata=metadata)
    context = generation_runtime.prepare_generation_context([retrieved], "What is annual revenue?", provider="google")
    assert len(context["images_base64"]) == 1
    with Image.open(io.BytesIO(base64.b64decode(context["images_base64"][0]))) as image:
        assert max(image.size) <= 1280
    assert "[Source 1; report.pdf; pages 5]" in context["text_contexts"][0]


def test_retriever_cache_reuses_keyword_index(tmp_path, monkeypatch):
    monkeypatch.setattr(rag_pipeline, "get_embedding_model", lambda: CountingEmbeddings())
    db = rag_pipeline.create_vector_store(sample_documents(), str(tmp_path / "db"))
    first = retrieval_runtime.get_retrievers(db, k=2)
    second = retrieval_runtime.get_retrievers(db, k=2)
    assert first[1] is second[1]
    retrieved, timings = retrieval_runtime.retrieve_documents(db, "invoice total", k=2, top_n=1, use_reranker=False)
    assert len(retrieved) == 1
    assert timings["reranker"] == "none"
    assert retrieved[0].metadata["chunk_id"] in db.get()["ids"]


def test_model_factory_bounds_requests_without_network(monkeypatch):
    import langchain_google_genai

    captured = {}
    class FakeGoogle:
        model_fields = {"thinking_budget": None}
        def __init__(self, **kwargs):
            captured.update(kwargs)
    monkeypatch.setattr(langchain_google_genai, "ChatGoogleGenerativeAI", FakeGoogle)
    generation_runtime._build_model.cache_clear()
    monkeypatch.setenv("MODEL_TIMEOUT_SECONDS", "60")
    monkeypatch.setenv("GENERATION_MAX_OUTPUT_TOKENS", "1024")
    monkeypatch.setenv("GENERATION_THINKING_BUDGET", "0")
    generation_runtime.get_generation_model("gemini-2.5-flash", provider="google")
    assert captured["timeout"] == 60
    assert captured["max_retries"] == 1
    assert captured["max_output_tokens"] == 1024
    assert captured["thinking_budget"] == 0
    generation_runtime._build_model.cache_clear()


def test_generator_streams_exact_prepared_context(monkeypatch):
    received, tokens = [], []
    class FakeModel:
        def stream(self, messages):
            received.extend(messages)
            return iter([AIMessage(content="The total is "), AIMessage(content="123.45 USD [Source 1].")])
    monkeypatch.setattr(generation_runtime, "get_generation_model", lambda *args, **kwargs: FakeModel())
    documents = sample_documents()
    prepared = generation_runtime.prepare_generation_context(documents, "What is the invoice total?", provider="google")
    answer = generation_runtime.generate_final_answer(
        documents, "What is the invoice total?", provider="google", token_callback=tokens.append,
    )
    assert answer == "".join(tokens)
    assert len(tokens) == 2
    assert received[0].content == prepared["messages"][0].content
    assert "123.45 USD" in answer


def test_stream_deadline_stops_run_and_closes_iterator(monkeypatch):
    original_positive = generation_runtime._positive_int
    monkeypatch.setattr(generation_runtime, "_positive_int", lambda name, default, maximum:
                        0.01 if name == "MODEL_TIMEOUT_SECONDS" else original_positive(name, default, maximum))
    closed = []
    class SlowStream:
        def __iter__(self):
            return self
        def __next__(self):
            time.sleep(0.02)
            return AIMessage(content="Late answer")
        def close(self):
            closed.append(True)
    monkeypatch.setattr(generation_runtime, "get_generation_model", lambda *args, **kwargs:
                        SimpleNamespace(stream=lambda messages: SlowStream()))
    with pytest.raises(generation_runtime.GenerationError, match="deadline"):
        generation_runtime.generate_final_answer(sample_documents(), "What is total?", provider="google",
                                                  token_callback=lambda text: None)
    assert closed == [True]


def test_provider_error_is_not_treated_as_an_answer(monkeypatch):
    def timeout(messages):
        raise TimeoutError("Provider blocked")
    monkeypatch.setattr(generation_runtime, "get_generation_model", lambda *args, **kwargs:
                        SimpleNamespace(invoke=timeout))
    with pytest.raises(generation_runtime.GenerationError, match="generation failed"):
        generation_runtime.generate_final_answer(sample_documents(), "What is total?", provider="google")


def test_context_budget_matches_exported_evidence(monkeypatch):
    monkeypatch.setenv("GENERATION_MAX_CONTEXT_CHARS", "100")
    prepared = generation_runtime.prepare_generation_context(
        [Document(page_content="a" * 500, metadata={"chunk_id": "one", "page": 1}),
         Document(page_content="b" * 500, metadata={"chunk_id": "two", "page": 2})],
        "What is present?", provider="google",
    )
    assert sum(map(len, prepared["text_contexts"])) == 100
    assert prepared["chunk_ids"] == ["one"]
    assert prepared["text_contexts"][0] in prepared["prompt"]


def test_reranker_uses_request_timeout_and_no_automatic_retries(monkeypatch):
    import cohere

    captured = {}
    class FakeClient:
        def __init__(self, **kwargs):
            captured["client"] = kwargs
        def rerank(self, **kwargs):
            captured["request"] = kwargs
            return SimpleNamespace(results=[SimpleNamespace(index=0, relevance_score=0.95)])
    monkeypatch.setattr(cohere, "ClientV2", FakeClient)
    monkeypatch.setenv("COHERE_API_KEY", "unit-test-never-sent")
    reranker = retrieval_runtime.CohereReranker("test-model", 1, 15)
    result = reranker.compress_documents(sample_documents(), "Invoice total?")
    assert captured["client"]["timeout"] == 15
    assert captured["request"]["request_options"] == {"max_retries": 0, "timeout_in_seconds": 15}
    assert result[0].metadata["relevance_score"] == 0.95


def test_retrieval_merges_duplicate_evidence_and_keeps_all_source_ids(tmp_path, monkeypatch):
    monkeypatch.setattr(rag_pipeline, "get_embedding_model", lambda: CountingEmbeddings())
    documents = [
        Document(page_content="The invoice total is 123.45 USD.", metadata={"page": 1}),
        Document(page_content="The invoice total is 123.45 USD.", metadata={"page": 2}),
        Document(page_content="The account balance is 200 USD.", metadata={"page": 3}),
    ]
    db = rag_pipeline.create_vector_store(documents, str(tmp_path / "db"))
    _, keyword, _, _ = retrieval_runtime.get_retrievers(db, k=3)
    results = keyword.invoke("invoice total")
    matches = [document for document in results if "invoice total" in document.page_content]
    assert len(matches) == 1
    assert json.loads(matches[0].metadata["pages"]) == [1, 2]
    assert len(json.loads(matches[0].metadata["duplicate_chunk_ids"])) == 2
    assert len(db.get()["ids"]) == 3


def test_same_caption_with_different_images_retains_distinct_visual_evidence(tmp_path, monkeypatch):
    monkeypatch.setattr(rag_pipeline, "get_embedding_model", lambda: CountingEmbeddings())
    documents = []
    for page, color in enumerate(("white", "black"), 1):
        with Image.new("RGB", (20, 20), color) as image:
            with io.BytesIO() as output:
                image.save(output, "JPEG")
                image_base64 = base64.b64encode(output.getvalue()).decode("ascii")
        documents.append(Document(page_content="Revenue chart.", metadata={
            "page": page, "original_content": json.dumps({
                "raw_text": "Revenue chart.", "tables_html": [], "images_base64": [image_base64],
            }),
        }))
    db = rag_pipeline.create_vector_store(documents, str(tmp_path / "db"))
    _, keyword, _, canonical = retrieval_runtime.get_retrievers(db, k=2)
    assert len(canonical) == 2
    assert len(keyword.invoke("revenue chart")) == 2


def test_changed_embedding_revision_reembeds_without_duplicating_records(tmp_path, monkeypatch):
    embedding = CountingEmbeddings()
    monkeypatch.setattr(rag_pipeline, "get_embedding_model", lambda: embedding)
    monkeypatch.setenv("EMBEDDING_REVISION", "first-revision")
    rag_pipeline.create_vector_store(sample_documents(), str(tmp_path / "db"))
    monkeypatch.setenv("EMBEDDING_REVISION", "second-revision")
    db = rag_pipeline.create_vector_store(sample_documents(), str(tmp_path / "db"))
    assert embedding.document_calls == 2
    assert len(db.get()["ids"]) == 2
    assert index_runtime.read_index_manifest(tmp_path / "db")["embedding_revision"] == "second-revision"


def test_configured_local_embedding_path_avoids_remote_model_resolution(tmp_path, monkeypatch):
    import langchain_huggingface

    path = tmp_path / "local-model"
    path.mkdir()
    (path / "modules.json").write_text("[]")
    captured = {}
    class FakeEmbeddings:
        def __init__(self, **kwargs):
            captured.update(kwargs)
    monkeypatch.setattr(langchain_huggingface, "HuggingFaceEmbeddings", FakeEmbeddings)
    monkeypatch.setenv("EMBEDDING_MODEL_PATH", str(path))
    rag_pipeline.get_embedding_model.cache_clear()
    rag_pipeline.get_embedding_model()
    assert captured["model_name"] == str(path)
    assert captured["model_kwargs"]["local_files_only"] is True
    rag_pipeline.get_embedding_model.cache_clear()


def test_long_native_element_keeps_distinct_chunk_slices_and_complete_coverage():
    original = " ".join(f"item-{number:04d}={number:04d}" for number in range(200))
    element = NarrativeText(original, metadata=ElementMetadata(filename="long.pdf", page_number=12))
    chunks = rag_pipeline.create_chunks_by_title([element])
    assert len(chunks) == 2
    documents = rag_pipeline.summarise_chunks(chunks)
    contents = [document.page_content for document in documents]
    assert contents == [chunk.text for chunk in chunks]
    assert len(set(contents)) == 2
    assert all(len(content) <= 2000 for content in contents)
    assert "".join("".join(content.split()) for content in contents) == "".join(original.split())
    for document in documents:
        assert json.loads(document.metadata["original_content"])["raw_text"] == document.page_content


def test_output_token_limit_is_reported_as_incomplete(monkeypatch):
    response = AIMessage(content="Partial answer", response_metadata={"finish_reason": "MAX_TOKENS"})
    monkeypatch.setattr(generation_runtime, "get_generation_model", lambda *args, **kwargs:
                        SimpleNamespace(invoke=lambda messages: response))
    with pytest.raises(generation_runtime.GenerationError, match="incomplete") as failure:
        generation_runtime.generate_final_answer(sample_documents(), "What is total?", provider="google")
    assert failure.value.partial_answer == "Partial answer"


def test_invalid_context_size_fails_before_search():
    with pytest.raises(ValueError, match="top_n"):
        retrieval_runtime.retrieve_documents(None, "question", k=3, top_n=4)


def test_failed_reranking_does_not_cache_failure_response(tmp_path, monkeypatch):
    monkeypatch.setattr(rag_pipeline, "get_embedding_model", lambda: CountingEmbeddings())
    db = rag_pipeline.create_vector_store(sample_documents(), str(tmp_path / "db"))
    calls = []
    class RecoveringReranker:
        def compress_documents(self, documents, query):
            calls.append(query)
            if len(calls) == 1:
                raise TimeoutError("Transient outage")
            return documents[:1]
    reranker = RecoveringReranker()
    monkeypatch.setattr(retrieval_runtime, "get_reranker", lambda **kwargs: reranker)
    _, failed = retrieval_runtime.retrieve_documents(db, "invoice total", k=2, top_n=1)
    _, recovered = retrieval_runtime.retrieve_documents(db, "invoice total", k=2, top_n=1)
    assert failed["reranking_error"] == "TimeoutError"
    assert "reranking_error" not in recovered
    assert recovered["reranker"] == "cohere"
    assert len(calls) == 2


def test_reembedding_same_corpus_rebuilds_retrievers_with_current_embedding_client(tmp_path, monkeypatch):
    directory = str(tmp_path / "db")
    first_embedding = CountingEmbeddings()
    monkeypatch.setattr(rag_pipeline, "get_embedding_model", lambda: first_embedding)
    monkeypatch.setenv("EMBEDDING_REVISION", "first-revision")
    first_db = rag_pipeline.create_vector_store(sample_documents(), directory)
    first_manifest = index_runtime.read_index_manifest(directory)
    first_retrievers = retrieval_runtime.get_retrievers(first_db, k=2)
    assert first_retrievers[0].base.vectorstore.embeddings is first_embedding

    second_embedding = CountingEmbeddings()
    monkeypatch.setattr(rag_pipeline, "get_embedding_model", lambda: second_embedding)
    monkeypatch.setenv("EMBEDDING_REVISION", "second-revision")
    second_db = rag_pipeline.create_vector_store(sample_documents(), directory)
    second_manifest = index_runtime.read_index_manifest(directory)
    second_retrievers = retrieval_runtime.get_retrievers(second_db, k=2)

    assert first_manifest["corpus_version"] == second_manifest["corpus_version"]
    assert first_manifest["count"] == second_manifest["count"]
    assert first_manifest["ids"] == second_manifest["ids"]
    assert second_retrievers is not first_retrievers
    assert second_retrievers[0].base.vectorstore.embeddings is second_embedding


def test_flash_lite_factory_uses_model_specific_minimal_thinking_without_custom_sampling(monkeypatch):
    import langchain_google_genai

    captured = {}
    class FakeGoogle:
        model_fields = {"thinking_budget": None, "reasoning_effort": None, "include_thoughts": None}
        def __init__(self, **kwargs):
            captured.update(kwargs)
    monkeypatch.setattr(langchain_google_genai, "ChatGoogleGenerativeAI", FakeGoogle)
    monkeypatch.setenv("MODEL_TIMEOUT_SECONDS", "60")
    monkeypatch.setenv("GENERATION_MAX_OUTPUT_TOKENS", "1024")
    generation_runtime._build_model.cache_clear()
    generation_runtime.get_generation_model("models/gemini-3.5-flash-lite", provider="google")
    assert captured["temperature"] is None
    assert captured["reasoning_effort"] == "minimal"
    assert captured["include_thoughts"] is False
    assert "thinking_budget" not in captured
    assert captured["timeout"] == 60
    assert captured["max_retries"] == 1
    assert captured["max_output_tokens"] == 1024
    generation_runtime._build_model.cache_clear()
