"""Finds the stored chunks that best match a question, and reports how confident each match is.

Why: retrieval is the "R" in RAG. Students should see which chunks were chosen
and why, so this module prints a ranked list with a 0-1 score on request.

Score formula: ChromaDB returns a squared L2 distance. Our embeddings are
normalized to length 1, so for any two of them
    distance = |a - b|^2 = 2 - 2 * cos(a, b)
which means cos = 1 - distance / 2. We report that cosine value as the
similarity score, so 1.0 means "same meaning" and values near 0 mean "unrelated".
"""

from src.config import TOP_K
from src.vector_store import query_collection


def distance_to_similarity(distance: float) -> float:
    """Convert Chroma's squared L2 distance to a cosine similarity (valid because embeddings are normalized)."""
    return 1 - distance / 2


def retrieve(query: str, k: int = TOP_K, print_debug: bool = False) -> list[dict]:
    """Return the top-k chunks for `query`, each with a `score` between roughly 0 and 1."""
    matches = query_collection(query, k)
    results = [
        {
            "text": match["text"],
            "metadata": match["metadata"],
            "distance": match["distance"],
            "score": round(distance_to_similarity(match["distance"]), 3),
        }
        for match in matches
    ]

    if print_debug:
        print(f"\nQuery: {query!r}  (top {k})")
        for rank, item in enumerate(results, start=1):
            label = item["metadata"].get("name") or item["metadata"].get("section")
            print(f"  {rank}. score={item['score']:.3f}  [{label}]  {item['text'][:90]}")
    return results


if __name__ == "__main__":
    import sys

    sys.stdout.reconfigure(encoding="utf-8")
    for question in [
        "spicy chicken dish",
        "food came cold want refund",
        "who won the IPL",
    ]:
        retrieve(question, print_debug=True)
