# app/chroma/embedder.py
from functools import lru_cache
from sentence_transformers import SentenceTransformer

MODEL_NAME = "all-mpnet-base-v2"

@lru_cache(maxsize=1)
def get_model() -> SentenceTransformer:
    """
    Lazy-load and cache the SentenceTransformer model.
    Will only be loaded the first time it's called.
    """
    return SentenceTransformer(MODEL_NAME)

def embed_text(text: str) -> list[float]:
    """
    Generates an embedding vector for the input text using a lazily-loaded
    SentenceTransformer model.

    Args:
        text (str): The text to embed.

    Returns:
        list[float]: The embedding vector.
    """
    if not text.strip():
        raise ValueError("Text is empty for embedding.")

    model = get_model()  # Load model only on demand
    embedding = model.encode(text, show_progress_bar=False)

    return embedding.tolist() if hasattr(embedding, "tolist") else embedding
