"""
Session State & Chain Lifecycle

Manages Streamlit session state and provides a cached PawAidChain instance.
"""

import logging
from typing import Optional

import streamlit as st

from src.chain.rag_chain import PawAidChain
from .config import get_chain_config

logger = logging.getLogger(__name__)


def init_session_state() -> None:
    """Set session-state defaults (runs every rerun, only writes missing keys)."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "animal_type" not in st.session_state:
        st.session_state.animal_type = "dog"
    if "chain_ready" not in st.session_state:
        st.session_state.chain_ready = False
    if "chain_error" not in st.session_state:
        st.session_state.chain_error = None


@st.cache_resource(show_spinner="Loading PawAid AI (vector store + LLM)...")
def load_chain() -> PawAidChain:
    """Create and cache the PawAidChain (expensive: ChromaDB + LLM client)."""
    config = get_chain_config()
    return PawAidChain(chain_config=config)


def get_chain() -> Optional[PawAidChain]:
    """Return the cached chain, or set an error in session state."""
    try:
        chain = load_chain()
        st.session_state.chain_ready = True
        st.session_state.chain_error = None
        return chain
    except Exception as exc:
        logger.exception("Failed to initialise PawAidChain")
        st.session_state.chain_ready = False
        st.session_state.chain_error = str(exc)
        return None


def clear_chat(chain: Optional[PawAidChain]) -> None:
    """Reset conversation history in both UI and chain memory."""
    st.session_state.messages = []
    if chain is not None:
        chain.clear_memory()
