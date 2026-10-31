"""Central place for settings, so every other file reads the same numbers and key.

Why: when a student changes chunk size or the model name, they should only have
to look in one file. The API key is read from a `.env` file, never hardcoded.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

# Project root = the folder that contains `src/`. Data paths are built from this,
# so scripts work no matter which folder you run them from.
ROOT = Path(__file__).resolve().parent.parent

load_dotenv(ROOT / ".env")

CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "zomato_menu"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
CHUNK_SIZE = 300  # characters per policy chunk
CHUNK_OVERLAP = 150  # characters of trailing sentences repeated in the next chunk (about one sentence)
TOP_K = 3  # how many chunks to retrieve per question

MENU_PATH = ROOT / "data" / "menu.json"
POLICY_PATH = ROOT / "data" / "policies.md"
SYSTEM_PROMPT_PATH = ROOT / "prompts" / "system_prompt.txt"

GEMINI_MODEL = "gemini-3.5-flash"


def require_gemini_key() -> str:
    """Return the Gemini API key, or stop with a clear message if it is missing.

    Why a function and not a check at import time: ingestion and retrieval run
    fully offline, so they must work without a key. Only the answer step needs it.
    """
    key = os.getenv("GEMINI_API_KEY", "").strip()
    if not key:
        raise RuntimeError(
            "GEMINI_API_KEY is missing. Copy .env.example to .env and paste your key "
            "from https://aistudio.google.com/apikey"
        )
    return key
