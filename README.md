# Zomato Glass-Box RAG Chatbot

A teaching project for a beginner GenAI class. It is a small restaurant assistant
that answers questions about a menu and its policies, and it shows every step of
the RAG pipeline so you can see how the answer was produced.

## The problem

A chatbot that answers from memory can invent prices, dishes, and refund rules.
Retrieval-Augmented Generation (RAG) fixes this by looking up the relevant facts
first and asking the language model to answer only from them. This project makes
each step visible: chunking, embedding, storage, retrieval, prompt, and answer.

## Architecture

```
data/menu.json ─┐
                ├─► chunker ─► embedder ─► ChromaDB (chroma_db/)     ← ingest.py (run once)
data/policies.md┘

question ─► injection check ─► retriever (top 3 chunks + scores)
                                   │
                                   ▼
                    system rules + chunks + question
                                   │
                                   ▼
                      Gemini (answer only, temperature 0)
                                   │
                                   ▼
                 answer + chunks + exact prompt ─► Streamlit "Under the hood"
```

| File | Job |
|---|---|
| `src/config.py` | Settings and the API key check |
| `src/chunker.py` | Menu: one chunk per dish. Policies: sentence-aware overlapping chunks |
| `src/embedder.py` | Local embeddings with `all-MiniLM-L6-v2` (no API key needed) |
| `src/vector_store.py` | Stores and queries chunks in a local ChromaDB folder |
| `src/retriever.py` | Returns the top chunks, scored by cosine similarity |
| `src/llm.py` | The single Gemini call |
| `src/rag.py` | Injection check, retrieval, prompt, and answer in one place |
| `src/schema.py` | Pydantic checks for food orders |
| `src/tools.py` | Mock order-status lookup |
| `app.py` | Streamlit chat UI, "Under the hood" panel, and order tools |
| `ingest.py` | Live demo script that builds the database |

## Setup

You need Python 3.10 or newer and a free Gemini API key from
https://aistudio.google.com/apikey.

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
```

Open `.env` and paste your key after `GEMINI_API_KEY=`.

## How to run

```bash
python ingest.py                 # build chroma_db/ (run once, or after editing data/)
streamlit run app.py             # open the chat at http://localhost:8501
python -m pytest                 # run the tests (no network or API key needed)
```

Helpful demo commands:

```bash
python -m src.chunker            # print every chunk and its metadata
python -m src.embedder           # print similarity between sample sentences
python -m src.retriever          # print ranked chunks for three sample questions
```

## Before and after retrieval

The same question, shown with and without retrieval. Without retrieval the model
has no menu or policy text, so it can only guess. With retrieval it gets the
right chunks, with their scores.

| Question | Without retrieval | With retrieval (top 3, score) |
|---|---|---|
| "spicy chicken dish" | Could suggest any chicken dish, with invented prices | Chilli Chicken 0.616, Chicken Tikka 0.590, Butter Chicken 0.517 |
| "food came cold want refund" | Could invent a refund rule or timeline | Refund Policy 0.625, Refund Policy 0.619, Refund Policy 0.568 |
| "who won the IPL" | Would answer from general knowledge | Best match is only 0.099, so the assistant replies "I don't know, let me connect you with support." |

## Tests

`python -m pytest` covers the chunker, the score conversion, the order schema,
the mock order tool, and the RAG flow (with the LLM replaced by a fake). None of
these tests call the network or need an API key.

## Limitations

- The injection check is a simple phrase list. It catches obvious attempts, not every clever one.
- The policy overlap is measured in characters, and it carries whole sentences only. The overlap is set to 150 characters because a typical sentence is about 90 characters, so a smaller limit would never copy anything. It is a good discussion point: try changing `CHUNK_OVERLAP` in `src/config.py` and watch the chunks change.
- Order status comes from an in-memory mock, not a real order system.
