import google.generativeai as genai

genai.configure(api_key="AIzaSyAApeCzigkoqk20PH1D5k10MDCzM6dVYdo")  # Replace with your actual API key

def embed_text(text: str, task_type: str = "retrieval_document") -> list[float]:
    """
    Generates an embedding vector for the input text using Gemini's embedding model.

    Args:
        text (str): The text to embed.
        task_type (str): Either 'retrieval_document' or 'retrieval_query'.

    Returns:
        list[float]: The embedding vector.
    """
    try:
        if not text.strip():
            raise ValueError("Text is empty for embedding.")

        response = genai.embed_content(
            model="models/embedding-001",
            content=text,
            task_type=task_type
        )
        return response["embedding"]

    except Exception as e:
        print(f"❌ Error embedding text: {e}")
        return []
