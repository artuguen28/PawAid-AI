"""
PawAid AI — Streamlit Entry Point

Configures page, injects CSS, initialises the chain, and launches the chat UI.
"""

import sys
from pathlib import Path

# ChromaDB requires sqlite3 >= 3.35.0; use pysqlite3 as a drop-in replacement
# when the system sqlite3 is too old (must run before any chromadb import).
__import__("pysqlite3")
sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")

# Ensure project root is on sys.path so `src.*` imports work
_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from dotenv import load_dotenv  # noqa: E402

load_dotenv()

import streamlit as st  # noqa: E402

from app.config import APP_ICON, APP_TITLE, PAGE_LAYOUT  # noqa: E402
from app.styles import GLOBAL_CSS  # noqa: E402
from app.state import init_session_state, get_chain, clear_chat  # noqa: E402
from app.components import render_sidebar  # noqa: E402
from app.chat import handle_chat_input  # noqa: E402

# --- Page config (must be the first Streamlit command) ---
st.set_page_config(
    page_title=APP_TITLE,
    page_icon=APP_ICON,
    layout=PAGE_LAYOUT,
)

# --- Global CSS ---
st.markdown(GLOBAL_CSS, unsafe_allow_html=True)

# --- Session state ---
init_session_state()

# --- Sidebar ---
render_sidebar()

# --- Handle "New Chat" request set by sidebar button ---
if st.session_state.pop("_clear_requested", False):
    clear_chat(get_chain())
    st.rerun()

# --- Header ---
st.title(f"{APP_ICON} {APP_TITLE}")
st.caption("AI-powered first-aid guidance for dogs & cats")

# --- Chain initialisation ---
chain = get_chain()

if chain is None:
    st.error(
        f"**Failed to start PawAid AI.**\n\n"
        f"`{st.session_state.chain_error}`\n\n"
        "Make sure your `.env` file contains a valid `OPENAI_API_KEY` "
        "and the vector store has been built (`make ingest`)."
    )
    st.stop()

# --- Chat ---
handle_chat_input(chain)
