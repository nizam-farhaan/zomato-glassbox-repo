"""Runs one full question-and-answer cycle: check the question, retrieve, build the prompt, ask the LLM.

Why: this is where the RAG steps are stitched together in plain sight. The
result also carries the retrieved chunks and the exact prompt, so the UI can
show students everything that happened.
"""

import logging
import re

from src import llm
from src.config import SYSTEM_PROMPT_PATH, TOP_K
from src.retriever import retrieve

logger = logging.getLogger(__name__)

# Phrases that try to override the rules. Matching is case-insensitive.
INJECTION_PATTERNS = [
    r"\bignore\b",
    r"disregard\s+(your|the|all)\s+rules",
    r"100\s*%\s*discount",
    r"system\s+prompt",
]
INJECTION_REFUSAL = (
    "I can't follow instructions that try to change how I work. "
    "I can only answer questions about our menu, prices, and policies."
)


def load_system_prompt() -> str:
    """Read the rules from prompts/system_prompt.txt. The file has a {context} placeholder."""
    with open(SYSTEM_PROMPT_PATH, encoding="utf-8") as f:
        return f.read()


def looks_like_injection(message: str) -> bool:
    """Return True when the message contains a known instruction-override phrase."""
    return any(re.search(pattern, message, re.IGNORECASE) for pattern in INJECTION_PATTERNS)


def answer(question: str, k: int = TOP_K) -> dict:
    """Answer a question from retrieved context only.

    Returns {"answer", "chunks", "prompt", "blocked"}. `prompt` is the exact text
    the model was given, and `blocked` is True when the injection check refused.
    """
    if looks_like_injection(question):
        logger.warning("Blocked a possible prompt injection: %r", question)
        return {"answer": INJECTION_REFUSAL, "chunks": [], "prompt": "", "blocked": True}

    chunks = retrieve(question, k=k)
    context = "\n\n".join(f"[{chunk['metadata'].get('source')}] {chunk['text']}" for chunk in chunks)

    system_prompt = load_system_prompt()
    filled_rules = system_prompt.replace("{context}", context)
    prompt = f"SYSTEM RULES:\n{filled_rules}\n\nUSER QUESTION:\n{question}"

    text = llm.generate_answer(system_prompt, question, context)
    return {"answer": text, "chunks": chunks, "prompt": prompt, "blocked": False}
