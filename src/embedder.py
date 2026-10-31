"""Turns text into lists of numbers (embeddings) so that similar meanings get similar numbers.

Why: the computer cannot compare meanings directly. An embedding model maps each
sentence to a point in 384-dimensional space, and sentences that mean similar
things land close together. This runs fully on your laptop, with no API key.

Run `python -m src.embedder` to see the similarity scores for yourself.
"""

import math
from functools import lru_cache

from sentence_transformers import SentenceTransformer

from src.config import EMBEDDING_MODEL


@lru_cache(maxsize=1)
def get_embedder() -> SentenceTransformer:
    """Load the model once. The first call downloads about 90 MB; later calls reuse it from memory."""
    return SentenceTransformer(EMBEDDING_MODEL)


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Return one 384-number vector per input text."""
    vectors = get_embedder().encode(texts, normalize_embeddings=True)
    return [vector.tolist() for vector in vectors]


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Return how closely two vectors point the same way: 1.0 is identical direction, 0.0 is unrelated."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    return dot / (norm_a * norm_b)


if __name__ == "__main__":
    import sys

    sys.stdout.reconfigure(encoding="utf-8")
    samples = ["Paneer Tikka", "Chicken Tikka", "refund policy"]
    vectors = embed_texts(samples)
    print(f"Embedding size: {len(vectors[0])} numbers per text\n")
    for i in range(len(samples)):
        for j in range(i + 1, len(samples)):
            score = cosine_similarity(vectors[i], vectors[j])
            print(f"{samples[i]!r:20} vs {samples[j]!r:20} similarity = {score:.3f}")
