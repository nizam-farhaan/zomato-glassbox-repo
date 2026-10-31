"""Chat screen for the restaurant assistant. Run with: streamlit run app.py

Why: a chat box is what customers see, and the "Under the hood" expander under
each answer shows what the assistant looked at and what it was told, so students
can check every answer themselves.
"""

import json

import streamlit as st
from pydantic import ValidationError

from src import rag
from src.schema import Order
from src.tools import get_order_status

st.set_page_config(page_title="Zomato Glass-Box RAG", page_icon="🍛")
st.title("🍛 Zomato Glass-Box Chatbot")
st.caption("Answers come only from the menu and policy files. Expand 'Under the hood' to see how.")

if "history" not in st.session_state:
    st.session_state.history = []

with st.sidebar:
    st.header("Orders")

    st.subheader("Check order status")
    order_id = st.text_input("Order ID", placeholder="e.g. ORD1002")
    if st.button("Check order status") and order_id:
        st.json(get_order_status(order_id))

    st.subheader("Validate an order")
    st.caption("Paste an order as JSON. It is checked against the menu before it is accepted.")
    order_json = st.text_area(
        "Order JSON",
        value='{"items": [{"dish_id": "D01", "quantity": 2}], "delivery_address": "12 MG Road"}',
        height=140,
    )
    if st.button("Validate order"):
        try:
            order = Order.model_validate(json.loads(order_json))
        except json.JSONDecodeError as error:
            st.error(f"Not valid JSON: {error}")
        except ValidationError as error:
            st.error("Order rejected:")
            for problem in error.errors():
                location = ".".join(str(part) for part in problem["loc"])
                st.write(f"- `{location}`: {problem['msg']}")
        else:
            st.success(f"Order accepted: {len(order.items)} item(s) for {order.delivery_address}")


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
