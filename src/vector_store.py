"""Stores the chunks and their embeddings in ChromaDB, a local database folder.

Why: the database is a plain folder (`chroma_db/`) that students can open and
inspect. Each row holds the chunk text, its metadata, and its embedding, so the
whole pipeline is visible instead of hidden behind an API.
"""

import chromadb

from src.config import CHROMA_DIR, COLLECTION_NAME, ROOT
from src.embedder import embed_texts


def _client() -> chromadb.PersistentClient:
    return chromadb.PersistentClient(path=str(ROOT / CHROMA_DIR))


def build_collection(chunks: list[dict]) -> None:
    """Rebuild the collection from scratch with all chunks. Safe to run again: it deletes the old one first."""
    client = _client()
    try:
        client.delete_collection(COLLECTION_NAME)
    except (ValueError, chromadb.errors.NotFoundError):
        # Only "it does not exist yet" is expected on the first run. Anything else should crash.
        pass
    collection = client.create_collection(COLLECTION_NAME)

    ids = [chunk["id"] for chunk in chunks]
    texts = [chunk["text"] for chunk in chunks]
    metadatas = [chunk["metadata"] for chunk in chunks]
    embeddings = embed_texts(texts)

    collection.add(ids=ids, documents=texts, metadatas=metadatas, embeddings=embeddings)


def query_collection(query: str, k: int) -> list[dict]:
    """Return the k closest chunks. Each result has its text, metadata, and Chroma's distance (smaller is closer)."""
    collection = _client().get_collection(COLLECTION_NAME)
    result = collection.query(
        query_embeddings=embed_texts([query]),
        n_results=k,
    )
    matches = []
    for text, metadata, distance in zip(
        result["documents"][0], result["metadatas"][0], result["distances"][0]
    ):
        matches.append({"text": text, "metadata": metadata, "distance": distance})
    return matches


def inspect() -> None:
    """Print how many rows are stored and show three sample rows, as proof the data landed."""
    collection = _client().get_collection(COLLECTION_NAME)
    count = collection.count()
    print(f"Collection '{COLLECTION_NAME}' has {count} rows.")
    sample = collection.peek(limit=3)
    for row_id, text, metadata in zip(sample["ids"], sample["documents"], sample["metadatas"]):
        print(f"\n  id: {row_id}")
        print(f"  text: {text[:120]}{'...' if len(text) > 120 else ''}")
        print(f"  metadata: {metadata}")
