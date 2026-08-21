"""
Regression tests for the embedding shape contract.

Background: every call site in vectorstore.py passed a bare `str` to
`embed_text`, which is typed `List[str]` and loops over its argument. Python
iterates a string character-by-character, so each chunk cost one Gemini call
per character and produced a nested list of vectors that ChromaDB rejected
with a ValueError — nothing was indexed and nothing was retrievable.

The existing isolation tests missed this because they monkeypatch `embed_text`
with a correctly-shaped stub. These tests assert the contract itself.
"""
import importlib
import sys
import types

import chromadb
import pytest


@pytest.fixture
def embedder(monkeypatch):
    """Import the embedder with the Gemini SDK stubbed out, and record every
    payload sent so we can assert on call count and content."""
    calls = []

    fake_genai = types.ModuleType("google.generativeai")
    fake_genai.configure = lambda **kw: None

    def fake_embed_content(model, content, task_type):
        calls.append({"content": content, "task_type": task_type})
        return {"embedding": [0.1] * 768}

    fake_genai.embed_content = fake_embed_content

    google_pkg = types.ModuleType("google")
    google_pkg.generativeai = fake_genai
    monkeypatch.setitem(sys.modules, "google", google_pkg)
    monkeypatch.setitem(sys.modules, "google.generativeai", fake_genai)

    # Reload so the module binds *this* test's stub rather than one cached
    # from an earlier test in the session.
    monkeypatch.delitem(sys.modules, "app.chroma.embedder", raising=False)
    embedder_module = importlib.import_module("app.chroma.embedder")
    embedder_module = importlib.reload(embedder_module)

    embedder_module._test_calls = calls
    return embedder_module


def test_embed_one_makes_exactly_one_api_call(embedder):
    """A single chunk must cost one Gemini call, not one per character."""
    text = "A Data Fiduciary shall intimate the Board of any breach."

    vector = embedder.embed_one(text)

    assert len(embedder._test_calls) == 1
    assert embedder._test_calls[0]["content"] == text


def test_embed_one_returns_a_flat_vector(embedder):
    """The return value must be a flat list of floats — the shape Chroma wants."""
    vector = embedder.embed_one("Consent must be free, specific and informed.")

    assert len(vector) == 768
    assert all(isinstance(component, float) for component in vector)


def test_embed_text_rejects_a_bare_string(embedder):
    """The original bug: a bare str is iterable and used to silently succeed."""
    with pytest.raises(TypeError, match="expects a list of strings"):
        embedder.embed_text("this is a single string, not a list")


def test_embed_one_passes_through_task_type(embedder):
    """Queries must be embedded as queries, documents as documents."""
    embedder.embed_one("do I need to report a breach?", task_type="retrieval_query")

    assert embedder._test_calls[0]["task_type"] == "retrieval_query"


def test_embed_one_output_is_accepted_by_chroma(embedder):
    """End-to-end shape check against real ChromaDB, which rejected the old shape."""
    collection = chromadb.EphemeralClient().get_or_create_collection("shape_probe")

    vector = embedder.embed_one("A Data Fiduciary shall notify the Board.")

    collection.add(
        documents=["A Data Fiduciary shall notify the Board."],
        embeddings=[vector],
        metadatas=[{"doc_id": "1"}],
        ids=["chunk-1"],
    )
    assert collection.count() == 1

    query_vector = embedder.embed_one("breach reporting", task_type="retrieval_query")
    results = collection.query(query_embeddings=[query_vector], n_results=1)
    assert results["documents"][0]
