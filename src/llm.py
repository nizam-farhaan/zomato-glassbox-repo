"""The only place the app calls Gemini to write an answer.

Why: keeping the paid-or-network call in one small file makes it easy to see
exactly what leaves the laptop: the system rules with the retrieved context,
and the customer's question. Nothing else is sent.
"""

from google import genai
from google.genai import types

from src.config import GEMINI_MODEL, require_gemini_key


def generate_answer(system_prompt: str, user_question: str, context: str) -> str:
    """Ask Gemini to answer `user_question`, using `context` filled into `system_prompt`.

    Raises an exception on API failure. It never returns a made-up answer.
    """
    filled_rules = system_prompt.replace("{context}", context)
    client = genai.Client(api_key=require_gemini_key())
    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=user_question,
        config=types.GenerateContentConfig(
            system_instruction=filled_rules,
            temperature=0,
        ),
    )
    if not response.text:
        raise RuntimeError("Gemini returned an empty response.")
    return response.text.strip()
