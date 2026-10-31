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
    """Return the k chunks whose embeddings are closest to the question.

    Each result has the chunk's text, its metadata, and its embedding, so the
    caller can compute cosine similarity itself.
    """
    collection = _client().get_collection(COLLECTION_NAME)
    result = collection.query(
        query_embeddings=embed_texts([query]),
        n_results=k,
        include=["documents", "metadatas", "embeddings"],
    )
    matches = []
    for text, metadata, embedding in zip(
        result["documents"][0], result["metadatas"][0], result["embeddings"][0]
    ):
        matches.append({"text": text, "metadata": metadata, "embedding": [float(x) for x in embedding]})
    return matches


def inspect(numbers_per_line: int = 8) -> None:
    """Print the row count, then one full sample row: its text, labels, and every number in its embedding."""
    collection = _client().get_collection(COLLECTION_NAME)
    print(f"Collection '{COLLECTION_NAME}' has {collection.count()} rows.")

    sample = collection.get(limit=1, include=["documents", "metadatas", "embeddings"])
    row_id = sample["ids"][0]
    embedding = [float(number) for number in sample["embeddings"][0]]

    print("\nSample row")
    print("-" * 72)
    print(f"  id:        {row_id}")
    print(f"  text:      {sample['documents'][0]}")
    print(f"  metadata:  {sample['metadatas'][0]}")
    print(f"\n  embedding: {len(embedding)} numbers, one per dimension (index: value)")
    for start in range(0, len(embedding), numbers_per_line):
        chunk = embedding[start : start + numbers_per_line]
        cells = "  ".join(f"{start + offset:>3}: {value:+.4f}" for offset, value in enumerate(chunk))
        print(f"    {cells}")
    print("-" * 72)
    print("  Each number is one coordinate of the chunk's meaning. Similar chunks have similar numbers.")
