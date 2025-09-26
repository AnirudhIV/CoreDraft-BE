# app/utils/lazy_imports.py
from functools import lru_cache

# -------------------------------
# Sentence Transformers (embeddings)
# -------------------------------
@lru_cache(maxsize=1)
def get_sentence_transformer_model(model_name: str = "all-mpnet-base-v2"):
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(model_name)


# -------------------------------
# SpaCy NLP
# -------------------------------
@lru_cache(maxsize=1)
def get_spacy_nlp(model: str = "en_core_web_sm"):
    import spacy
    return spacy.load(model)


# -------------------------------
# ChromaDB client
# -------------------------------
@lru_cache(maxsize=1)
def get_chromadb_client(path: str = "chroma_db"):
    from chromadb import PersistentClient
    return PersistentClient(path=path)


# -------------------------------
# PyMuPDF (fitz)
# -------------------------------
@lru_cache(maxsize=1)
def get_fitz():
    import fitz
    return fitz


# -------------------------------
# Google Generative AI
# -------------------------------
@lru_cache(maxsize=1)
def get_google_genai():
    import google.generativeai as genai
    return genai


# -------------------------------
# LangChain Document schema
# -------------------------------
@lru_cache(maxsize=1)
def get_langchain_document():
    from langchain.schema import Document
    return Document


# -------------------------------
# Torch
# -------------------------------
@lru_cache(maxsize=1)
def get_torch():
    import torch
    return torch
