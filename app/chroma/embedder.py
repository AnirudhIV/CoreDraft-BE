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
