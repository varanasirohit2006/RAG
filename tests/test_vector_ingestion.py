from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from backend import rag_pipeline, index_runtime


class LocalTestEmbeddings(Embeddings):
    """A real local vector-store test without downloading models or using credentials."""

    def embed_documents(self, texts):
        return [[float(len(text)), 1.0, 0.5] for text in texts]

    def embed_query(self, text):
        return self.embed_documents([text])[0]


def test_reingestion_upserts_existing_chunks_instead_of_duplicating(tmp_path, monkeypatch):
    monkeypatch.setattr(rag_pipeline, "get_embedding_model", lambda: LocalTestEmbeddings())
    documents = [
        Document(page_content="First page invoice 123.45 USD", metadata={"page": 1}),
        Document(page_content="Second page balance 200 USD", metadata={"page": 2}),
    ]
    directory = str(tmp_path / "vectors")
    first = index_runtime.build_vector_store(documents, directory=directory, embedding=rag_pipeline.get_embedding_model())
    first_ids = set(first.get()["ids"])
    assert len(first_ids) == 2
    reopened = index_runtime.build_vector_store(documents, directory=directory, embedding=rag_pipeline.get_embedding_model())
    assert set(reopened.get()["ids"]) == first_ids
    assert len(reopened.similarity_search("invoice", k=5)) == 2
