"""Turns raw files into small, labelled pieces of text ("chunks") that can be searched.

Why: a language model and a vector search both work better on short, focused
pieces. The menu is naturally one chunk per dish. The policy document is long,
so it is cut into overlapping pieces, always on sentence boundaries so no
sentence is torn in half.

Run `python -m src.chunker` to print every chunk, so you can see exactly what
will be embedded.
"""

import json
import re
import sys
from pathlib import Path

from src.config import CHUNK_OVERLAP, CHUNK_SIZE, MENU_PATH, POLICY_PATH

SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


def chunk_menu(menu_json_path: Path = MENU_PATH) -> list[dict]:
    """Return one chunk per dish. The text is a plain English sentence built from the fields."""
    with open(menu_json_path, encoding="utf-8") as f:
        dishes = json.load(f)

    chunks = []
    for dish in dishes:
        diet = "veg" if dish["veg"] else "non-veg"
        text = (
            f"{dish['name']} is a {dish['category']}, {diet}, priced at ₹{dish['price']}, "
            f"spice level {dish['spice_level']}. {dish['description']}"
        )
        metadata = {**dish, "veg": dish["veg"], "source": "menu"}
        chunks.append({"id": f"menu-{dish['id']}", "text": text, "metadata": metadata})
    return chunks


def chunk_policy(policy_md_path: Path = POLICY_PATH) -> list[dict]:
    """Split the policy file by its `## ` sections, then pack sentences into overlapping chunks."""
    with open(policy_md_path, encoding="utf-8") as f:
        raw = f.read()

    chunks = []
    for section, body in _split_sections(raw):
        sentences = [s.strip() for s in SENTENCE_END.split(body.strip()) if s.strip()]
        for piece in _pack_sentences(sentences, CHUNK_SIZE, CHUNK_OVERLAP):
            index = len(chunks) + 1
            chunks.append(
                {
                    "id": f"policy-{index:03d}",
                    "text": piece,
                    "metadata": {"source": "policy", "section": section},
                }
            )
    return chunks


def _split_sections(markdown: str) -> list[tuple[str, str]]:
    """Return (header, body) pairs for each `## ` section. Text before the first header is skipped."""
    sections = []
    header = None
    lines = []
    for line in markdown.splitlines():
        if line.startswith("## "):
            if header is not None:
                sections.append((header, "\n".join(lines)))
            header = line[3:].strip()
            lines = []
        elif header is not None:
            lines.append(line)
    if header is not None:
        sections.append((header, "\n".join(lines)))
    return sections


def _pack_sentences(sentences: list[str], size: int, overlap: int) -> list[str]:
    """Greedily fill chunks up to `size` characters. Carry the last sentences (up to `overlap`) forward."""
    chunks = []
    current: list[str] = []
    for sentence in sentences:
        if current and len(" ".join(current + [sentence])) > size:
            chunks.append(" ".join(current))
            carry = []
            for previous in reversed(current):
                if len(" ".join([previous] + carry)) > overlap:
                    break
                carry.insert(0, previous)
            # Never carry the whole chunk forward, or the next chunk would repeat it exactly.
            if len(carry) == len(current):
                carry = []
            current = carry
        current.append(sentence)
    if current:
        chunks.append(" ".join(current))
    return chunks


def _print_chunks(title: str, chunks: list[dict]) -> None:
    print(f"\n=== {title}: {len(chunks)} chunks ===")
    for chunk in chunks:
        print(f"\n[{chunk['id']}] {len(chunk['text'])} chars")
        print(f"  text:     {chunk['text']}")
        print(f"  metadata: {chunk['metadata']}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    _print_chunks("Menu chunks", chunk_menu())
    _print_chunks("Policy chunks", chunk_policy())
