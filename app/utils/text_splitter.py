import re
import spacy
from typing import List, Dict, Union

nlp = spacy.load("en_core_web_sm")


def spacy_sent_tokenize(text: str) -> List[str]:
    doc = nlp(text)
    return [sent.text.strip() for sent in doc.sents]


# === 1. SECTION SPLITTER ===
def split_by_section(text: str) -> List[Dict]:
    """
    Splits text using patterns like 'Section 1', 'Chapter 2', etc.
    Returns list of dicts with 'text' and 'metadata'.
    """
    section_pattern = r"(Section\s+\d+|Chapter\s+\d+)(.*?)(?=(Section\s+\d+|Chapter\s+\d+|\Z))"
    matches = re.findall(section_pattern, text, re.DOTALL | re.IGNORECASE)

    chunks = []
    for match in matches:
        header = match[0].strip()
        body = match[1].strip()
        if body:
            chunks.append({
                "text": body,
                "metadata": {
                    "section_title": header
                }
            })
    return chunks


# === 1. SEMANTIC SPLITTER (sentence-based) ===
def semantic_split(
    text: str,
    max_chunk_tokens: int = 1000,     # bigger for legal docs
    overlap_tokens: int = 150
) -> List[str]:
    """
    Splits text semantically by sentences.
    Preserves sentence integrity and includes token overlap.
    """
    sentences = spacy_sent_tokenize(text)  # you must have spacy sent_tokenizer loaded
    chunks = []
    current_chunk = []
    current_len = 0

    for sentence in sentences:
        token_count = len(sentence.split())

        # If adding the sentence would exceed chunk limit → save current chunk
        if current_len + token_count > max_chunk_tokens and current_chunk:
            chunks.append(" ".join(current_chunk))

            # Apply overlap (from end of last chunk)
            if overlap_tokens:
                overlap_words = " ".join(current_chunk).split()[-int(overlap_tokens):]
                current_chunk = [" ".join(overlap_words)]
                current_len = len(overlap_words)
            else:
                current_chunk = []
                current_len = 0

        current_chunk.append(sentence)
        current_len += token_count

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks


# === 2. FIXED SPLITTER (word-based) ===
def split_text(
    content: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 150
) -> List[str]:
    """
    Splits text into fixed word-count chunks with overlap.
    """
    # Ensure chunk_size and chunk_overlap are integers
    chunk_size = int(chunk_size)
    chunk_overlap = int(chunk_overlap)

    words = content.split()
    chunks = []
    start = 0

    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += chunk_size - chunk_overlap

    return chunks


# === 3. HYBRID SPLITTER (section → semantic → fixed) ===
def hybrid_split_text(text: str) -> List[Dict[str, Union[str, Dict]]]:
    """
    Combines all strategies:
    1. Try splitting by section (if possible)
    2. Inside long sections, use semantic split
    3. If no sections found, try semantic split for the whole doc
    4. Fallback to fixed word splitting
    Returns list of dicts: { text: ..., metadata: {...} }
    """
    chunks = []

    # Step 1: Split by section (you need to define split_by_section elsewhere)
    section_chunks = split_by_section(text)

    if section_chunks:
        for section in section_chunks:
            section_text = section['text']
            metadata = section.get('metadata', {})

            # Step 2: For long sections → semantic split
            if len(section_text.split()) > 600:
                semantic_chunks = semantic_split(
                    section_text,
                    max_chunk_tokens=1500,
                    overlap_tokens=200
                )
                for ch in semantic_chunks:
                    chunks.append({"text": ch, "metadata": metadata})
            else:
                chunks.append({"text": section_text, "metadata": metadata})
        return chunks

    # Step 3: If no section structure → try semantic split
    semantic_chunks = semantic_split(
        text,
        max_chunk_tokens=1500,
        overlap_tokens=200
    )
    if semantic_chunks:
        for ch in semantic_chunks:
            chunks.append({"text": ch, "metadata": {}})
        return chunks

    # Step 4: Fallback to fixed chunking
    fixed_chunks = split_text(
        text,
        chunk_size=1500,
        chunk_overlap=200
    )
    for ch in fixed_chunks:
        chunks.append({"text": ch, "metadata": {}})
    return chunks
