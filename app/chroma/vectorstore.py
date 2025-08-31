import uuid
from chromadb import PersistentClient
from langchain.schema import Document
from app.chroma.embedder import embed_text
from app.utils.text_splitter import split_text
import chromadb
from typing import List
from langchain.schema import Document



# 🔧 Keep your global Chroma client and collection as is
chroma_client = PersistentClient(path="./chroma_db")
COLLECTION_NAME = "compliance_docs"
collection = chroma_client.get_or_create_collection(name=COLLECTION_NAME)


def add_doc_to_vectorstore(
    doc_id: str,
    content: str,
    metadata: dict = {},
    chunk_size: int = 500,
    use_section_split: bool = False
):
    """ Adds a document to ChromaDB with optional chunking """
    from app.utils.text_splitter import split_text, split_by_section

    # Split text into chunks
    if use_section_split:
        chunks = split_by_section(content)
    else:
        chunks = split_text(content, chunk_size=chunk_size)

    chunks = [{"text": chunk, "metadata": {}} for chunk in chunks]

    documents, ids, embeddings, metadatas = [], [], [], []

    for i, chunk_data in enumerate(chunks):
        text = chunk_data["text"]
        chunk_metadata = chunk_data.get("metadata", {})
        chunk_id = f"{doc_id}_chunk_{i}"
        embedding = embed_text(text)

        documents.append(text)
        ids.append(chunk_id)
        embeddings.append(embedding)
        metadatas.append({
            **metadata,
            **chunk_metadata,
            "doc_id": doc_id,
            "chunk_index": i
        })

    collection.add(
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas,
        ids=ids
    )




def add_text_to_chroma(text: str, doc_id: str, metadata: dict = None):
    """ Adds or updates a single text chunk in ChromaDB """
    embedding = embed_text(text)
    chunk_id = f"{doc_id}_{uuid.uuid4().hex}"   # unique ID per chunk
    
    # ensure metadata carries doc_id
    metadata = metadata or {}
    metadata["doc_id"] = doc_id  

    collection.upsert(
        documents=[text],
        embeddings=[embedding],
        metadatas=[metadata],
        ids=[chunk_id]
    )
    print(f"✅ Upserted text chunk with chunk_id={chunk_id} (doc_id={doc_id})")



def delete(doc_id: str):
    """ Deletes all chunks belonging to a specific document ID """
    collection.delete(where={"doc_id": doc_id})
    print(f"✅ Deleted all chunks for doc_id: {doc_id}")


def query_similar_docs(query: str, top_k: int = 3) -> dict:
    """Retrieve top_k chunks per document using vector similarity search."""
    embedding = embed_text(query)

    results = collection.query(
        query_embeddings=[embedding],
        n_results=50  # fetch more initially to allow grouping by doc
    )

    # Group results by doc_id
    grouped = {}
    for i, meta in enumerate(results["metadatas"][0]):
        doc_id = meta.get("doc_id", "unknown")
        if doc_id not in grouped:
            grouped[doc_id] = []
        grouped[doc_id].append({
            "text": results["documents"][0][i],
            "score": results["distances"][0][i],
            "metadata": meta,
        })

    # Keep only top_k per doc
    for doc_id in grouped:
        grouped[doc_id] = sorted(grouped[doc_id], key=lambda x: x["score"])[:top_k]

    return grouped


def retrieve_relevant_chunks(query: str, top_k: int = 5) -> List[Document]:
    """Retrieve top_k chunks per document using hybrid retrieval (vector + keyword)."""

    # --- Vector search (grouped per doc) ---
    vector_results = query_similar_docs(query, top_k=top_k)

    # --- Keyword search fallback ---
    try:
        keyword_results = collection.query(
            query_texts=[query],
            n_results=top_k * 5
        )
        keyword_docs = [
            Document(page_content=doc, metadata=meta)
            for doc, meta in zip(keyword_results["documents"][0], keyword_results["metadatas"][0])
        ]
    except Exception:
        # fallback: brute force substring match
        keyword_docs = []
        stored_docs = collection.get(include=["documents", "metadatas"])
        if stored_docs and stored_docs.get("documents"):
            for doc, meta in zip(stored_docs["documents"], stored_docs["metadatas"]):
                if query.lower() in doc.lower():
                    keyword_docs.append(Document(page_content=doc, metadata=meta))

    # --- Convert vector results into Document objects ---
    final_docs = []
    for doc_id, chunks in vector_results.items():
        for chunk in chunks:
            final_docs.append(Document(page_content=chunk["text"], metadata=chunk["metadata"]))

    # Add keyword results (optional, only if new docs are found)
    final_docs.extend(keyword_docs)

    return final_docs



def update_document_embedding(doc_id: str, content: str):
    """ Updates the embedding for a document in Chroma """
    embedding = embed_text(content)
    collection.update(
        ids=[doc_id],
        embeddings=[embedding],
        documents=[content]
    )
    print(f"✅ Updated embedding in Chroma for doc_id {doc_id}")

def get_chroma_collection():
    return collection