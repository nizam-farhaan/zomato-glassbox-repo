"""Checks the cosine similarity math and the ordering of retrieved chunks.

No network, no API key, no database: the question embedding and the search are replaced with fakes.
"""

import pytest

from src import retriever
from src.embedder import cosine_similarity


@pytest.mark.parametrize(
    "first, second, expected",
    [
        ([1.0, 0.0], [1.0, 0.0], 1.0),  # same direction
        ([1.0, 0.0], [0.0, 1.0], 0.0),  # right angle
        ([1.0, 0.0], [-1.0, 0.0], -1.0),  # opposite direction
    ],
)
def test_cosine_similarity(first, second, expected):
    assert cosine_similarity(first, second) == pytest.approx(expected)


def test_retrieve_sorts_best_match_first(monkeypatch):
    monkeypatch.setattr(retriever, "embed_texts", lambda texts: [[1.0, 0.0]])
    fake_matches = [
        {"text": "far", "metadata": {}, "embedding": [0.0, 1.0]},
        {"text": "close", "metadata": {}, "embedding": [1.0, 0.0]},
    ]
    monkeypatch.setattr(retriever, "query_collection", lambda query, k: fake_matches)

    results = retriever.retrieve("anything", k=2)

    assert [item["text"] for item in results] == ["close", "far"]
    assert results[0]["score"] == pytest.approx(1.0)
    assert results[1]["score"] == pytest.approx(0.0)
