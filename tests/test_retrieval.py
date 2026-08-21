"""
Tests for the hybrid retrieval path that actually serves /compliance/ask.

The pre-existing isolation tests exercised retrieve_relevant_chunks, but the
endpoint ran a second, near-identical copy of that logic inside
ai_generator.process_question_and_docs — so the tested code was not the live
code. These tests cover the merged path, including the entry point the
endpoint calls.
"""
import uuid

import chromadb
import pytest

from app.chroma import vectorstore


@pytest.fixture
def store(monkeypatch):
    """Throwaway in-memory collection + a stubbed embedder.

    The stub returns a fixed vector, so ranking is not meaningful here; these
    tests assert filtering, dedup, and merge behaviour, not similarity quality.
    """
    # Unique name per test: EphemeralClient otherwise hands back the *same*
    # collection for a given name, leaking chunks between tests.
    collection = chromadb.EphemeralClient().get_or_create_collection(
        f"test_retrieval_{uuid.uuid4().hex}"
    )
    monkeypatch.setattr(vectorstore, "collection", collection)
    monkeypatch.setattr(
        vectorstore, "embed_one",
        lambda text, task_type="retrieval_document": [1.0, 0.0, 0.0],
    )
    return collection


def _add(collection, chunk_id, text, doc_id, user_id=None, is_default=False):
    meta = {"doc_id": doc_id, "is_default": is_default}
    if user_id is not None:
        meta["user_id"] = user_id
    collection.add(
        documents=[text], embeddings=[[1.0, 0.0, 0.0]], metadatas=[meta], ids=[chunk_id]
    )


# --- lexical arm -----------------------------------------------------------

def test_lexical_terms_drops_stopwords_and_short_tokens():
    terms = vectorstore._lexical_terms("Do I need to report a breach?")

    assert "breach" in terms
    assert "report" in terms
    for noise in ("do", "i", "to", "a", "need"):
        assert noise not in terms


def test_lexical_search_finds_exact_statutory_reference(store):
    """The case dense vectors blur: an exact section number."""
    _add(store, "c1", "Section 8 requires notifying the Board of a breach.", "BASE", is_default=True)
    _add(store, "c2", "Unrelated text about office stationery.", "BASE", is_default=True)

    where = vectorstore._tenant_where("1")
    hits = vectorstore._lexical_search("Section 8 breach", where=where, limit=10)

    assert [h["id"] for h in hits] == ["c1"]


def test_lexical_search_respects_tenant_filter(store):
    """The lexical arm must be tenant-scoped, not just the vector arm."""
    _add(store, "a1", "retention schedule for payroll records", "A", user_id="1")
    _add(store, "b1", "retention schedule for payroll records", "B", user_id="2")

    where = vectorstore._tenant_where("2")
    hits = vectorstore._lexical_search("retention schedule", where=where, limit=10)

    assert [h["id"] for h in hits] == ["b1"]


def test_lexical_search_returns_nothing_for_all_stopword_query(store):
    _add(store, "c1", "Some compliance text.", "BASE", is_default=True)

    where = vectorstore._tenant_where("1")
    assert vectorstore._lexical_search("what is the", where=where, limit=10) == []


# --- merge + dedup ---------------------------------------------------------

def test_chunk_matching_both_arms_appears_once(store):
    """Previously the two arms were concatenated, duplicating shared hits."""
    _add(store, "c1", "Section 8 breach notification duty.", "BASE", is_default=True)

    docs = vectorstore.retrieve_relevant_chunks("Section 8 breach", user_id="1", top_k=5)

    matching = [d for d in docs if "Section 8" in d.page_content]
    assert len(matching) == 1


def test_retrieval_merges_both_arms(store):
    """A lexical-only match must still reach the context."""
    _add(store, "v1", "General data protection obligations.", "A", user_id="1")
    _add(store, "k1", "Reporting within 72 hours is mandatory.", "B", user_id="1")

    docs = vectorstore.retrieve_relevant_chunks("72 hours", user_id="1", top_k=5)

    assert any("72 hours" in d.page_content for d in docs)


# --- tenant isolation on the merged path -----------------------------------

def test_retrieval_excludes_other_users_chunks(store):
    _add(store, "a1", "User A confidential retention policy.", "A", user_id="1")
    _add(store, "b1", "User B confidential retention policy.", "B", user_id="2")

    docs = vectorstore.retrieve_relevant_chunks("confidential retention policy", user_id="2", top_k=5)

    texts = [d.page_content for d in docs]
    assert "User B confidential retention policy." in texts
    assert "User A confidential retention policy." not in texts


def test_retrieval_includes_shared_baseline(store):
    _add(store, "base1", "DPDPA baseline obligations.", "BASE", is_default=True)
    _add(store, "a1", "User A confidential notes.", "A", user_id="1")

    docs = vectorstore.retrieve_relevant_chunks("baseline obligations", user_id="2", top_k=5)

    texts = [d.page_content for d in docs]
    assert "DPDPA baseline obligations." in texts
    assert "User A confidential notes." not in texts


# --- distance threshold ----------------------------------------------------

def test_max_distance_drops_far_chunks(store):
    _add(store, "c1", "Some compliance text.", "A", user_id="1")

    kept = vectorstore.query_similar_docs("anything", user_id="1", top_k=5, max_distance=10.0)
    dropped = vectorstore.query_similar_docs("anything", user_id="1", top_k=5, max_distance=-1.0)

    assert kept != {}
    assert dropped == {}


# --- the live endpoint path ------------------------------------------------

def test_process_question_and_docs_uses_shared_retrieval(store, monkeypatch):
    """/compliance/ask must go through the same retrieval code these tests cover."""
    from app.utils import ai_generator

    _add(store, "base1", "DPDPA requires breach notification.", "BASE", is_default=True)
    _add(store, "a1", "Our policy omits breach notification.", "A", user_id="1")

    captured = {}

    def fake_generate(question, baseline_chunks, user_chunks, answer_style="concise"):
        captured["baseline"] = [c.page_content for c in baseline_chunks]
        captured["user"] = [c.page_content for c in user_chunks]
        return "answer", []

    monkeypatch.setattr(ai_generator, "generate_answer_from_context", fake_generate)

    answer, _ = ai_generator.process_question_and_docs(
        "breach notification", user_id="1", top_k=5, max_distance=None
    )

    assert answer == "answer"
    # The baseline/user split is what makes this gap analysis rather than chat.
    assert captured["baseline"] == ["DPDPA requires breach notification."]
    assert captured["user"] == ["Our policy omits breach notification."]


def test_process_question_and_docs_is_tenant_scoped(store, monkeypatch):
    from app.utils import ai_generator

    _add(store, "a1", "User A secret contract terms.", "A", user_id="1")
    _add(store, "b1", "User B secret contract terms.", "B", user_id="2")

    captured = {}

    def fake_generate(question, baseline_chunks, user_chunks, answer_style="concise"):
        captured["all"] = [c.page_content for c in baseline_chunks + user_chunks]
        return "answer", []

    monkeypatch.setattr(ai_generator, "generate_answer_from_context", fake_generate)

    ai_generator.process_question_and_docs(
        "secret contract terms", user_id="2", top_k=5, max_distance=None
    )

    assert "User B secret contract terms." in captured["all"]
    assert "User A secret contract terms." not in captured["all"]


def test_no_results_returns_graceful_message(store, monkeypatch):
    from app.utils import ai_generator

    monkeypatch.setattr(ai_generator, "retrieve_relevant_chunks", lambda *a, **k: [])

    answer, sources = ai_generator.process_question_and_docs("anything", user_id="1")

    assert "no relevant documents" in answer.lower()
    assert sources == []
