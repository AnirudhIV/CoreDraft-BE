"""
Regression tests for Phase 0.5: multi-tenant retrieval isolation.

Before the fix, query_similar_docs / retrieve_relevant_chunks queried the shared
Chroma collection with no per-user filter, so any authenticated user could
retrieve any other user's uploaded document chunks via /compliance/ask.
"""
import uuid

import chromadb
import pytest

from app.chroma import vectorstore


@pytest.fixture
def isolated_collection(monkeypatch):
    """Swap the module-level Chroma collection for a throwaway in-memory one,
    and stub out the embedder so no real embedding API calls happen."""
    client = chromadb.EphemeralClient()
    # Unique name per test: EphemeralClient otherwise hands back the *same*
    # collection for a given name, leaking chunks between tests.
    collection = client.get_or_create_collection(f"test_compliance_docs_{uuid.uuid4().hex}")
    monkeypatch.setattr(vectorstore, "collection", collection)
    # Fixed embedding for every chunk/query: with the where-filter doing the
    # real work, similarity ranking doesn't need to be meaningful here.
    # Signature matches embed_one(text, task_type=...) — retrieval passes a
    # task_type, indexing relies on the default.
    monkeypatch.setattr(
        vectorstore, "embed_one", lambda text, task_type="retrieval_document": [1.0, 0.0, 0.0]
    )
    return collection


def test_query_similar_docs_excludes_other_users_chunks(isolated_collection):
    isolated_collection.add(
        documents=["User A's confidential contract terms."],
        embeddings=[[1.0, 0.0, 0.0]],
        metadatas=[{"doc_id": "docA", "user_id": "user-a", "is_default": False}],
        ids=["chunk-a"],
    )
    isolated_collection.add(
        documents=["User B's confidential contract terms."],
        embeddings=[[1.0, 0.0, 0.0]],
        metadatas=[{"doc_id": "docB", "user_id": "user-b", "is_default": False}],
        ids=["chunk-b"],
    )

    grouped = vectorstore.query_similar_docs("confidential contract terms", user_id="user-b", top_k=5)

    all_texts = [chunk["text"] for chunks in grouped.values() for chunk in chunks]
    assert "User B's confidential contract terms." in all_texts
    assert "User A's confidential contract terms." not in all_texts


def test_query_similar_docs_includes_shared_default_docs(isolated_collection):
    isolated_collection.add(
        documents=["Shared DPDPA baseline text."],
        embeddings=[[1.0, 0.0, 0.0]],
        metadatas=[{"doc_id": "baseline", "is_default": True}],
        ids=["chunk-baseline"],
    )
    isolated_collection.add(
        documents=["User A's confidential contract terms."],
        embeddings=[[1.0, 0.0, 0.0]],
        metadatas=[{"doc_id": "docA", "user_id": "user-a", "is_default": False}],
        ids=["chunk-a"],
    )

    grouped = vectorstore.query_similar_docs("baseline", user_id="user-b", top_k=5)

    all_texts = [chunk["text"] for chunks in grouped.values() for chunk in chunks]
    assert "Shared DPDPA baseline text." in all_texts
    assert "User A's confidential contract terms." not in all_texts


def test_retrieve_relevant_chunks_isolates_users_via_lexical_path(isolated_collection):
    """The lexical arm scans collection.get() directly; confirm it is
    tenant-filtered too, not just the vector arm."""
    isolated_collection.add(
        documents=["User A's confidential contract terms."],
        embeddings=[[1.0, 0.0, 0.0]],
        metadatas=[{"doc_id": "docA", "user_id": "user-a", "is_default": False}],
        ids=["chunk-a"],
    )
    isolated_collection.add(
        documents=["User B's confidential contract terms."],
        embeddings=[[1.0, 0.0, 0.0]],
        metadatas=[{"doc_id": "docB", "user_id": "user-b", "is_default": False}],
        ids=["chunk-b"],
    )

    docs = vectorstore.retrieve_relevant_chunks("confidential contract terms", user_id="user-b", top_k=5)

    texts = [doc.page_content for doc in docs]
    assert "User B's confidential contract terms." in texts
    assert "User A's confidential contract terms." not in texts
