"""Thin RAG helper.

Kept as a convenience wrapper. The endpoint path (/compliance/ask) goes
through app.utils.ai_generator.process_question_and_docs, which additionally
splits retrieved chunks into compliance-baseline vs user documents before
prompting. Use that for compliance Q&A; this is a plain "answer from my
documents" helper with no baseline/user split.
"""
from app.chroma.vectorstore import retrieve_relevant_chunks
from app.utils.ai_generator import generate_answer_from_context


def answer_question_with_rag(question: str, user_id: str, top_k: int = 5) -> tuple:
    """Answer a question from the caller's own documents plus shared baselines.

    user_id is required: retrieval is tenant-scoped, and omitting it previously
    raised a TypeError on every call.
    """
    chunks = retrieve_relevant_chunks(question, user_id=user_id, top_k=top_k)

    if not chunks:
        return "Sorry, I couldn't find relevant information in the documents.", []

    # No baseline/user split here — treat everything retrieved as context.
    return generate_answer_from_context(question, baseline_chunks=[], user_chunks=chunks)
