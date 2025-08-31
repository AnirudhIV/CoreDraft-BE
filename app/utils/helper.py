# app/utils/helpers.py

import hashlib

def chunk_text(text: str, chunk_size: int = 300, overlap: int = 50) -> list[str]:
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = words[i:i + chunk_size]
        chunks.append(" ".join(chunk))
    return chunks

def hash_file_content(file_bytes: bytes) -> str:
    return hashlib.sha256(file_bytes).hexdigest()
