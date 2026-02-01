"""
Chat Flow

Handles message history, user input, chain invocation, and response rendering.
"""

import streamlit as st

from src.chain.rag_chain import PawAidChain
from src.chain.response_handler import ChainResponse
from .components import render_assistant_message


def handle_chat_input(chain: PawAidChain) -> None:
    """
    Main chat loop.

    1. Re-render all existing messages from session state.
    2. Capture new user input via st.chat_input().
    3. Invoke the chain, render the response, and persist to history.
    """
    # --- replay history ---
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            with st.chat_message("user"):
                st.markdown(msg["content"])
        else:
            with st.chat_message("assistant", avatar="\U0001f43e"):
                response: ChainResponse = msg["response"]
                render_assistant_message(response)

    # --- new user input ---
    prompt = st.chat_input("Describe your pet's situation...")
    if prompt is None:
        return

    # Show & store user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Invoke chain & render assistant response
    with st.chat_message("assistant", avatar="\U0001f43e"):
        with st.spinner("Thinking..."):
            response = chain.invoke(
                query=prompt,
                animal_type=st.session_state.get("animal_type"),
            )
        render_assistant_message(response)

    # Persist assistant response for replay
    st.session_state.messages.append({"role": "assistant", "response": response})
