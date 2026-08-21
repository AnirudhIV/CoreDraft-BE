import re
import uuid
from chromadb import PersistentClient
from langchain.schema import Document
from app.chroma.embedder import embed_one
from app.utils.text_splitter import split_text
import chromadb
from typing import List
from langchain.schema import Document



# 🔧 Keep your global Chroma client and collection as is
chroma_client = PersistentClient(path="./chroma_db")
COLLECTION_NAME = "compliance_docs"
collection = chroma_client.get_or_create_collection(name=COLLECTION_NAME)


def _tenant_where(user_id: str) -> dict:
    """Scope retrieval to the requesting user's own chunks, plus shared default/baseline chunks."""
    return {
        "$or": [
            {"user_id": {"$eq": str(user_id)}},
            {"is_default": {"$eq": True}},
        ]
    }


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
        embedding = embed_one(text)

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
    embedding = embed_one(text)
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


def _lexical_terms(query: str) -> List[str]:
    """Extract distinctive terms from a query for lexical matching.

    Drops very short tokens and common stopwords so that a question like
    "do I need to report a breach?" reduces to ["need", "report", "breach"]
    rather than matching on "do"/"i"/"a".
    """
    stopwords = {
        "a", "an", "and", "are", "as", "at", "be", "by", "can", "do", "does",
        "for", "from", "has", "have", "how", "i", "if", "in", "is", "it", "my",
        "need", "of", "on", "or", "our", "shall", "should", "that", "the",
        "there", "this", "to", "we", "what", "when", "which", "who", "why",
        "will", "with", "you", "your",
    }
    tokens = re.findall(r"[a-z0-9]+", query.lower())
    return [t for t in tokens if len(t) > 2 and t not in stopwords]


def _lexical_search(query: str, where: dict, limit: int) -> List[dict]:
    """Score user-visible chunks by how many distinct query terms they contain.

    This is the lexical half of hybrid retrieval. It exists to catch exact
    strings that dense embeddings blur together — statutory references like
    "Section 8", or fixed figures like "72 hours" — which vector similarity
    alone reliably misses.

    Note: this loads every chunk the caller is allowed to see. That is fine at
    the current corpus size (a handful of documents per user) but is the first
    thing to replace with a real inverted index as the corpus grows.
    """
    terms = _lexical_terms(query)
    if not terms:
        return []

    try:
        stored = collection.get(include=["documents", "metadatas"], where=where)
    except Exception as exc:
        print(f"⚠️ Lexical search unavailable: {exc}")
        return []

    documents = (stored or {}).get("documents") or []
    metadatas = (stored or {}).get("metadatas") or []
    ids = (stored or {}).get("ids") or [None] * len(documents)

    scored = []
    for chunk_id, text, meta in zip(ids, documents, metadatas):
        if not text:
            continue
        haystack = text.lower()
        hits = sum(1 for term in terms if term in haystack)
        if hits:
            scored.append({
                "id": chunk_id,
                "text": text,
                "metadata": meta or {},
                # Fraction of query terms present, so it is comparable across
                # queries of different lengths.
                "lexical_score": hits / len(terms),
            })

    scored.sort(key=lambda c: c["lexical_score"], reverse=True)
    return scored[:limit]


def query_similar_docs(
    query: str,
    user_id: str,
    top_k: int = 3,
    max_distance: float | None = None,
) -> dict:
    """Retrieve top_k chunks per document using vector similarity search, scoped to user_id (plus shared default docs).

    Over-fetches (n_results=50) and then keeps only the closest top_k chunks
    *per document*. Without the per-document cap, one long document could fill
    every slot and the baseline-vs-user comparison would have nothing to
    compare against.

    Chroma returns distances, so lower is closer. `max_distance`, when set,
    drops chunks farther away than that cutoff.
    """
    # retrieval_query: Gemini embeds questions differently from documents, and
    # matching the task type to the call site improves recall.
    embedding = embed_one(query, task_type="retrieval_query")

    results = collection.query(
        query_embeddings=[embedding],
        n_results=50,  # fetch more initially to allow grouping by doc
        where=_tenant_where(user_id)
    )

    metadatas = (results.get("metadatas") or [[]])[0]
    documents = (results.get("documents") or [[]])[0]
    distances = (results.get("distances") or [[]])[0]
    ids = (results.get("ids") or [[]])[0] or [None] * len(documents)

    # Group results by doc_id
    grouped = {}
    for i, meta in enumerate(metadatas):
        meta = meta or {}
        distance = distances[i]
        if max_distance is not None and distance > max_distance:
            continue
        doc_id = meta.get("doc_id", "unknown")
        grouped.setdefault(doc_id, []).append({
            "id": ids[i],
            "text": documents[i],
            "score": distance,
            "metadata": meta,
        })

    # Keep only top_k per doc (ascending: Chroma distances are lower-is-closer)
    for doc_id in grouped:
        grouped[doc_id] = sorted(grouped[doc_id], key=lambda x: x["score"])[:top_k]

    return grouped


def retrieve_relevant_chunks(
    query: str,
    user_id: str,
    top_k: int = 5,
    max_distance: float | None = None,
) -> List[Document]:
    """Hybrid retrieval (vector + lexical), scoped to user_id plus shared default docs.

    This is the single retrieval entry point — /compliance/ask reaches it via
    process_question_and_docs. Results from both arms are merged and
    deduplicated, so a chunk matching both ways appears in the context once.
    """
    where = _tenant_where(user_id)

    # --- Vector search (grouped per doc) ---
    vector_results = query_similar_docs(
        query, user_id=user_id, top_k=top_k, max_distance=max_distance
    )

    final_docs: List[Document] = []
    seen: set = set()

    def _remember(chunk_id, text, metadata) -> bool:
        """Track chunks already added. Falls back to the text itself when a
        chunk carries no id, so dedup still works for hand-built collections."""
        key = chunk_id if chunk_id is not None else text
        if key in seen:
            return False
        seen.add(key)
        final_docs.append(Document(page_content=text, metadata=metadata or {}))
        return True

    for chunks in vector_results.values():
        for chunk in chunks:
            _remember(chunk["id"], chunk["text"], chunk["metadata"])

    # --- Lexical search (catches exact terms vector search blurs) ---
    for chunk in _lexical_search(query, where=where, limit=top_k * 5):
        _remember(chunk["id"], chunk["text"], chunk["metadata"])

    return final_docs


def update_document_embedding(doc_id: str, content: str):
    """ Updates the embedding for a document in Chroma """
    embedding = embed_one(content)
    collection.update(
        ids=[doc_id],
        embeddings=[embedding],
        documents=[content]
    )
    print(f"✅ Updated embedding in Chroma for doc_id {doc_id}")

def get_chroma_collection():
    return collection