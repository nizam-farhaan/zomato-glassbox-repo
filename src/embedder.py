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
    samples = {
        "A": ("food", "Chunks of cottage cheese marinated in spiced yogurt and grilled in a tandoor, a vegetarian starter."),
        "B": ("food", "Boneless chicken pieces marinated in red chilli, ginger and garlic, then charred in a tandoor."),
        "C": ("policy", "Refunds are eligible when an order arrives cold, damaged, or more than 45 minutes late, and must be requested within 24 hours."),
    }
    vectors = dict(zip(samples, embed_texts([text for _, text in samples.values()])))

    print("=" * 72)
    print("EMBEDDING CHECK: do similar meanings get similar scores?")
    print("=" * 72)
    print(f"Each sentence becomes {len(next(iter(vectors.values())))} numbers.\n")

    print("Samples")
    for label, (topic, text) in samples.items():
        print(f"  [{label}] ({topic}) {text}")

    print("\nPairs (score runs from -1 to 1; higher means closer in meaning)")
    print(f"  {'pair':<8}{'topics':<16}{'score':>8}   {'bar':<22}")
    print("  " + "-" * 60)
    labels = list(samples)
    for i, first in enumerate(labels):
        for second in labels[i + 1 :]:
            score = cosine_similarity(vectors[first], vectors[second])
            topics = f"{samples[first][0]} vs {samples[second][0]}"
            bar = "#" * max(0, round(score * 20))
            print(f"  {first} vs {second:<4}{topics:<16}{score:>8.3f}   {bar:<22}")

    print("\nWhat to notice: food vs food scores high, food vs policy scores near zero.")
