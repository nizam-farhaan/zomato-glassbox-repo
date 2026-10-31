"""Live demo script: reads the data files, chunks them, and loads them into the vector database.

Why: this is the "build the library" step of RAG. Run it once, or again after
editing data/menu.json or data/policies.md. It narrates each step so the class
can follow along.

Usage (from the project root):  python ingest.py
"""

import sys

from src import vector_store
from src.chunker import chunk_menu, chunk_policy


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")

    print("Step 1/3: chunking the menu (one chunk per dish)...")
    menu_chunks = chunk_menu()
    print(f"  -> {len(menu_chunks)} menu chunks")

    print("Step 2/3: chunking the policies (sentence-aware, overlapping)...")
    policy_chunks = chunk_policy()
    print(f"  -> {len(policy_chunks)} policy chunks")

    print("Step 3/3: embedding every chunk and storing it in chroma_db/...")
    vector_store.build_collection(menu_chunks + policy_chunks)

    print("\nDone. Here is what landed in the database:")
    vector_store.inspect()


if __name__ == "__main__":
    main()
