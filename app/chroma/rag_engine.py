from app.chroma.embedder import embed_text
from app.chroma.vectorstore import retrieve_relevant_chunks
from app.utils.ai_generator import generate_answer_from_context  # Gemini function
from app.utils.lazy_imports import get_langchain_document  # lazy import

def answer_question_with_rag(question: str, chunks=None) -> dict:
    # Lazy-load the Document class
    Document = get_langchain_document()
    
    # Retrieve relevant chunks if not provided
    context_chunks = chunks or retrieve_relevant_chunks(question)

    if not context_chunks:
        return "Sorry, I couldn't find relevant information in the documents."

    # Join the page_content from each Document
    context = "\n\n".join(chunk.page_content for chunk in context_chunks)
    return generate_answer_from_context(question, context)
