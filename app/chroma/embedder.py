import os
import google.generativeai as genai
from dotenv import load_dotenv
import time
from typing import List

# Load environment variables
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

def embed_text(texts: List[str], task_type: str = "retrieval_document") -> List[List[float]]:
    """
    Generates embeddings for a list of texts using Gemini's embedding model.

    Args:
        texts (List[str]): List of texts to embed.
        task_type (str): Either 'retrieval_document' or 'retrieval_query'.

    Returns:
        List[List[float]]: List of embedding vectors corresponding to input texts.
    """
    # Guard: a bare str is iterable, so passing one here used to silently embed
    # the text character-by-character (one Gemini call per character) and return
    # a nested list that ChromaDB then rejected. Fail loudly instead.
    if isinstance(texts, str):
        raise TypeError(
            "embed_text() expects a list of strings; got a bare str. "
            "Use embed_one() to embed a single text."
        )

    embeddings = []
    for text in texts:
        if not text.strip():
            embeddings.append([])
            continue

        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = genai.embed_content(
                    model="models/embedding-001",
                    content=text,
                    task_type=task_type
                )
                embeddings.append(response.get("embedding", []))
                break
            except Exception as e:
                print(f"Attempt {attempt+1} ❌ Error embedding text: {e}")
                time.sleep(2 ** attempt)  # exponential backoff
        else:
            # If all retries fail, append empty embedding
            embeddings.append([])

    return embeddings


def embed_one(text: str, task_type: str = "retrieval_document") -> List[float]:
    """
    Embeds a single text and returns one flat vector.

    Use this wherever a single chunk or query is embedded — it returns the
    shape ChromaDB expects for one item, rather than a list of vectors.

    Args:
        text (str): The text to embed.
        task_type (str): 'retrieval_document' when indexing, 'retrieval_query'
            when embedding a search query. Gemini produces different vectors
            per task type, so matching the call site improves retrieval.

    Returns:
        List[float]: A single embedding vector, or [] if embedding failed.
    """
    return embed_text([text], task_type=task_type)[0]
