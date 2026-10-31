"""Chat screen for the restaurant assistant. Run with: streamlit run app.py

Why: a chat box is what customers see, and the "Under the hood" expander under
each answer shows what the assistant looked at and what it was told, so students
can check every answer themselves.
"""

import streamlit as st

from src import rag

st.set_page_config(page_title="Zomato Glass-Box RAG", page_icon="🍛")
st.title("🍛 Zomato Glass-Box Chatbot")
st.caption("Answers come only from the menu and policy files. Expand 'Under the hood' to see how.")

if "history" not in st.session_state:
    st.session_state.history = []


def show_under_the_hood(result: dict) -> None:
    """Show the retrieved chunks with scores and the exact prompt sent to Gemini."""
    with st.expander("Under the hood"):
        if result["blocked"]:
            st.warning("The injection check blocked this message. Gemini was never called.")
            return

        st.subheader("Retrieved chunks")
        st.table(
            [
                {
                    "rank": rank,
                    "score": chunk["score"],
                    "source": chunk["metadata"].get("source"),
                    "name or section": chunk["metadata"].get("name") or chunk["metadata"].get("section"),
                    "text": chunk["text"],
                }
                for rank, chunk in enumerate(result["chunks"], start=1)
            ]
        )

        st.subheader("Exact prompt sent to Gemini")
        st.code(result["prompt"], language=None)


for turn in st.session_state.history:
    with st.chat_message(turn["role"]):
        st.markdown(turn["content"])
        if turn["role"] == "assistant" and "result" in turn:
            show_under_the_hood(turn["result"])

question = st.chat_input("Ask about a dish, a price, or a policy")
if question:
    st.session_state.history.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        try:
            with st.spinner("Searching the menu and policies..."):
                result = rag.answer(question)
        except Exception as error:  # Show any failure in the chat instead of crashing the page.
            st.error(f"Sorry, I couldn't get an answer right now. Details: {error}")
            st.session_state.history.append({"role": "assistant", "content": f"Error: {error}"})
        else:
            st.markdown(result["answer"])
            show_under_the_hood(result)
            st.session_state.history.append(
                {"role": "assistant", "content": result["answer"], "result": result}
            )
