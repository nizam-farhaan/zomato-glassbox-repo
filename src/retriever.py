"""Finds the stored chunks that best match a question, and scores each one with cosine similarity.

Why: retrieval is the "R" in RAG. Students should see which chunks were chosen
and why, so this module prints a ranked list with each chunk's cosine similarity.

Cosine similarity measures the angle between two embedding vectors:
  1.0  -> same direction (same meaning)
  0.0  -> right angle (unrelated)
 -1.0  -> opposite direction (opposite meaning)
"""

from src.config import TOP_K
from src.embedder import cosine_similarity, embed_texts
from src.vector_store import query_collection


def retrieve(query: str, k: int = TOP_K, print_debug: bool = False) -> list[dict]:
    """Return the top-k chunks for `query`, best first, each with a cosine `score`."""
    question_vector = embed_texts([query])[0]
    results = []
    for match in query_collection(query, k):
        results.append(
            {
                "text": match["text"],
                "metadata": match["metadata"],
                "score": round(cosine_similarity(question_vector, match["embedding"]), 3),
            }
        )
    results.sort(key=lambda item: item["score"], reverse=True)

    if print_debug:
        print(f"\nQuery: {query!r}  (top {k}, cosine similarity)")
        for rank, item in enumerate(results, start=1):
            label = item["metadata"].get("name") or item["metadata"].get("section")
            print(f"  {rank}. cosine={item['score']:.3f}  [{label}]  {item['text'][:90]}")
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
