"""Checks the distance-to-score conversion in the retriever.

No network, no API key, no database: this is pure arithmetic.
"""

import pytest

from src.retriever import distance_to_similarity


@pytest.mark.parametrize(
    "distance, expected",
    [(0.0, 1.0), (1.0, 0.5), (2.0, 0.0)],
)
def test_distance_to_similarity(distance, expected):
    assert distance_to_similarity(distance) == pytest.approx(expected)
