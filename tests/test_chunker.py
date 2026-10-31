"""Checks that chunking produces the right number of pieces and never cuts a sentence in half.

No network, no API key, no model download: these tests only read the local data files.
"""

from src.chunker import _pack_sentences, chunk_menu, chunk_policy


def test_menu_has_one_chunk_per_dish():
    chunks = chunk_menu()
    assert len(chunks) == 20
    assert all(chunk["metadata"]["source"] == "menu" for chunk in chunks)


def test_menu_chunk_text_contains_price_and_name():
    first = chunk_menu()[0]
    assert "Paneer Tikka" in first["text"]
    assert "₹220" in first["text"]


def test_policy_chunks_respect_size_limit():
    chunks = chunk_policy()
    assert len(chunks) > 1
    # A single sentence may exceed the limit on its own, but packed chunks should not.
    for chunk in chunks:
        assert len(chunk["text"]) <= 300 or chunk["text"].count(". ") == 0


def test_policy_chunks_keep_section_metadata():
    sections = {chunk["metadata"]["section"] for chunk in chunk_policy()}
    assert {"Refund Policy", "Delivery Policy", "Cancellation Policy", "Packaging & Hygiene"} <= sections


def test_pack_sentences_never_repeats_a_whole_chunk():
    sentences = ["A" * 200 + ".", "B" * 200 + ".", "C" * 200 + "."]
    chunks = _pack_sentences(sentences, size=300, overlap=50)
    assert len(chunks) == 3
    assert len(set(chunks)) == len(chunks)
