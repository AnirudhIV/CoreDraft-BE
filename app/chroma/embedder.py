# app/chroma/embedder.py
from sentence_transformers import SentenceTransformer

# Use a model that outputs 768-dimensional embeddings
MODEL_NAME = "all-mpnet-base-v2"
model = SentenceTransformer(MODEL_NAME)

def embed_text(text: str) -> list[float]:
    """
    Generates an embedding vector for the input text using a local SentenceTransformer model.

    Args:
        text (str): The text to embed.

    Returns:
        list[float]: The embedding vector.
    """
    if not text.strip():
        raise ValueError("Text is empty for embedding.")

    embedding = model.encode(text, show_progress_bar=False)
    return embedding.tolist() if hasattr(embedding, "tolist") else embedding
