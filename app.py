from __future__ import annotations

import os
import streamlit as st
from dotenv import load_dotenv

from rag import Assistant

load_dotenv()

st.set_page_config(page_title="Recipe Companion", page_icon="🍽️", layout="centered")
st.title("Recipe Companion")
st.caption("Tell me what ingredients you have and I'll find the perfect recipe for you!")


@st.cache_resource(show_spinner="Loading recipe database, please wait...")
def get_assistant() -> Assistant:
    config = {
        "api_key": os.getenv("OPENAI_API_KEY"),
        "base_url": os.getenv("OPENAI_BASE_URL"),
        "model": os.getenv("MODEL"),
        "embedding_model": os.getenv("EMBEDDING_MODEL"),
        "top_k": os.getenv("TOP_K"),
        "top_n": os.getenv("TOP_N"),
    }
    return Assistant.from_config(config)


if "messages" not in st.session_state:
    st.session_state.messages = []

assistant = get_assistant()

with st.sidebar:
    st.header("Options")

    diet_filter = st.selectbox(
        "Dietary filter",
        ["None", "Vegan", "Vegetarian", "Gluten-free", "Dairy-free", "Keto"],
    )

    st.divider()
    if st.button("Clear conversation", use_container_width=True):
        assistant.clear_history()
        st.session_state.messages = []
        st.rerun()
    st.divider()
    st.markdown("**How to use:**")
    st.markdown("- Tell me ingredients you have available")
    st.markdown("- Ask for recipes by cuisine, diet, or time")
    st.markdown("- I'll suggest recipes only from my database")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("e.g. I have chicken, garlic and lemon..."):
    filtered_prompt = prompt
    if diet_filter != "None":
        filtered_prompt = f"{prompt} (only {diet_filter} recipes)"

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Finding a recipe..."):
            try:
                reply = assistant.ask(filtered_prompt)
            except Exception as e:
                reply = f"Error: {e}"
        st.markdown(reply)

    st.session_state.messages.append({"role": "assistant", "content": reply})