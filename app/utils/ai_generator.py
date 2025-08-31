import os
import json
import re
import traceback
from fastapi import HTTPException
import google.generativeai as genai
from langchain.schema import Document
from app.chroma.vectorstore import query_similar_docs
from chromadb import PersistentClient
from typing import List, Tuple


chroma_client = PersistentClient(path="./chroma_db")
COLLECTION_NAME = "compliance_docs"
collection = chroma_client.get_or_create_collection(name=COLLECTION_NAME)

# --- API KEY Setup ---
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not set")

genai.configure(api_key=GEMINI_API_KEY)  # ✅ Use environment variable, not hardcoded key

# --- Model Initialization ---
model = genai.GenerativeModel("gemini-2.0-flash")

# --- Document Generation ---
def generate_document_from_prompt(prompt: str) -> dict:
    try:
        full_prompt = (
            "You are a compliance assistant. Generate a compliance document in JSON format "
            "with the following structure:\n\n"
            "{\n"
            "  \"title\": \"...\",\n"
            "  \"content\": \"...\",\n"
            "  \"tags\": [\"...\", \"...\", \"...\"]\n"
            "}\n\n"
            f"Prompt: {prompt}"
        )

        response = model.generate_content(full_prompt)
        raw_text = response.text.strip()
        print("RAW AI RESPONSE:\n", raw_text)

        # Clean markdown if needed
        if raw_text.startswith("```json"):
            raw_text = re.sub(r"```json|```", "", raw_text).strip()

        generated = json.loads(raw_text)

        if not all(k in generated for k in ("title", "content", "tags")):
            print("Missing required fields in response")
            raise ValueError("Missing fields")

        return generated

    except json.JSONDecodeError:
        print("❌ JSON Decode Error")
        traceback.print_exc()
        raise HTTPException(status_code=422, detail="AI response is not valid JSON.")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")

# --- Tag Suggestion ---
def suggest_tags_ai(content: str) -> list[str]:
    prompt = f"Suggest 3 to 5 relevant tags for this compliance document as a pure JSON list of strings with no explanations:\n\n{content}"
    response = model.generate_content(prompt)
    
    # ✅ Ensure we always send a string to parse_json_tags
    if hasattr(response, "text"):
        ai_text = response.text
    else:
        ai_text = str(response)  # fallback
    
    return parse_json_tags(ai_text)


# --- Summary Generation ---
def summarize_doc_ai(content: str) -> str:
    prompt = f"Summarize the following compliance document in 3–5 lines, with each line as its own paragraph:\n\n{content}"
    response = model.generate_content(prompt)

    # Clean and format output
    summary = response.text.strip()

    # Replace single line breaks or sentence endings with double breaks for paragraph separation
    summary = summary.replace(". ", ".\n\n")

    return summary


# --- Tag Parser ---
def parse_json_tags(text: str) -> list[str]:
    try:
        json_like = re.search(r"\[.*\]", text, re.DOTALL).group()
        return json.loads(json_like)
    except Exception:
        return [tag.strip() for tag in text.split(",") if tag.strip()]




from typing import List, Tuple


def retrieve_and_classify_docs(all_docs: List[Document]) -> Tuple[List[Document], List[Document]]:
    """
    Separate docs into baseline (default compliance) and user docs.
    'is_default' metadata field = True for baseline docs.
    """
    baseline_docs = [doc for doc in all_docs if doc.metadata.get('is_default', False)]
    user_docs = [doc for doc in all_docs if not doc.metadata.get('is_default', False)]
    return baseline_docs, user_docs

def generate_answer_from_context(
    question: str,
    baseline_chunks: List[Document],
    user_chunks: List[Document],
    answer_style: str = "concise"  # "concise" or "detailed"
):
    """
    Uses Gemini to answer compliance questions based on the baseline (e.g., DPDPA)
    and user's company document chunks. Returns the answer + metadata of source chunks used.
    """

    baseline_context = "\n".join([doc.page_content for doc in baseline_chunks])
    user_context = "\n".join([doc.page_content for doc in user_chunks])

    if answer_style == "concise":
        style_instructions = """
- Keep your answer within approximately 200 words.
- Write in clear, well-structured paragraphs separated by blank lines.
- Use bullet points ("- " or "* ") to highlight key obligations, gaps, or next steps.
- Avoid rigid sections like 'Executive Summary'.
- Mention DPDPA-specific obligations where relevant (e.g., DPO appointment, DPIAs, breach timelines).
- If the question asks for a definition, provide a brief definition (1-2 sentences) with 2-3 related bullet points.
- Write in a conversational, chatbot-friendly style.
"""
    else:
        style_instructions = """
- Provide a more detailed answer with well-organized paragraphs.
- Include bullet point lists (up to 8-10 items) for all key obligations and recommendations.
- Use professional, clear formatting to improve readability.
- Add 3-4 practical next steps as bullet points.
- Leave one line after each paragraph and bullet point for clarity.
"""

    prompt = f"""
You are a helpful and knowledgeable compliance assistant AI.

You have two sets of documents:
1. Compliance Baseline (e.g., DPDPA legislation and rules):
{baseline_context}

2. User's Company Documents:
{user_context}

Your task:
- Compare the user's company documents against the compliance baseline.
- Identify gaps, missing obligations, or inconsistencies.
- Summarize new or removed responsibilities.
- Answer the question: {question}

Formatting Guidelines:
{style_instructions}

Answer:
"""

    try:
        response = model.generate_content(prompt)
        answer = response.text.strip()
        sources = [doc.metadata for doc in baseline_chunks + user_chunks]
        return answer, sources
    except Exception:
        return "There was an error generating the answer.", []

def process_question_and_docs(question: str, top_k: int = 5, answer_style: str = "concise"):
    """
    Combines hybrid retrieval (vector + keyword), classification,
    and answer generation for maximum accuracy in compliance Q&A.
    """

    if not isinstance(top_k, int):
        top_k = int(top_k)

    # Step 1: Retrieve relevant chunks using hybrid method
    vector_results = query_similar_docs(question, top_k=top_k)

    # Keyword search fallback
    try:
        keyword_results = collection.query(
            query_texts=[question],
            n_results=top_k * 5
        )
        keyword_docs = [
            Document(page_content=doc, metadata=meta)
            for doc, meta in zip(keyword_results["documents"][0], keyword_results["metadatas"][0])
        ]
    except Exception:
        keyword_docs = []
        stored_docs = collection.get(include=["documents", "metadatas"])
        if stored_docs and stored_docs.get("documents"):
            for doc, meta in zip(stored_docs["documents"], stored_docs["metadatas"]):
                if question.lower() in doc.lower():
                    keyword_docs.append(Document(page_content=doc, metadata=meta))

    # Convert vector results to Documents
    retrieved_docs = []
    for doc_id, chunks in vector_results.items():
        for chunk in chunks:
            retrieved_docs.append(Document(page_content=chunk["text"], metadata=chunk["metadata"]))

    # Add keyword docs if new
    retrieved_docs.extend(keyword_docs)

    if not retrieved_docs:
        return "Sorry, no relevant documents found.", []

    # Step 2: Classify docs into baseline vs user docs
    baseline_chunks, user_chunks = retrieve_and_classify_docs(retrieved_docs)

    # Step 3: Generate AI answer from classified docs
    answer, sources = generate_answer_from_context(
        question, baseline_chunks, user_chunks, answer_style=answer_style
    )

    return answer, sources
