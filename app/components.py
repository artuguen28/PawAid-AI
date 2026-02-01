"""
UI Components

Reusable Streamlit rendering functions for the PawAid chat interface.
"""

from typing import Optional

import streamlit as st

from src.chain.response_handler import ChainResponse
from src.prompts.config import UrgencyLevel
from src.prompts.urgency import UrgencyAssessment
from src.retrival.citation import CitationList

from .config import (
    APP_ICON,
    APP_TITLE,
    ANIMAL_OPTIONS,
    DISCLAIMER_TEXT,
    URGENCY_COLORS,
    URGENCY_EMOJIS,
)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

def render_sidebar() -> None:
    """Render the full sidebar: branding, pet selector, new-chat button, about."""
    with st.sidebar:
        st.markdown(
            f'<div class="sidebar-title">{APP_ICON} {APP_TITLE}</div>'
            '<div class="sidebar-subtitle">'
            "AI-powered first-aid guidance for dogs &amp; cats"
            "</div>",
            unsafe_allow_html=True,
        )
        st.divider()
        render_pet_selector()
        st.divider()

        if st.button("New Chat", use_container_width=True):
            # Defer actual clearing to main — just set a flag
            st.session_state["_clear_requested"] = True

        st.divider()
        with st.expander("About"):
            st.markdown(
                "**PawAid AI** provides first-aid guidance for common dog and "
                "cat health situations.\n\n"
                "It is **not** a substitute for professional veterinary care. "
                "Always consult a licensed veterinarian for diagnosis and treatment."
            )


def render_pet_selector() -> None:
    """Dog / Cat radio selector — updates session state on change."""
    labels = list(ANIMAL_OPTIONS.keys())
    current_value = st.session_state.get("animal_type", "dog")
    current_index = list(ANIMAL_OPTIONS.values()).index(current_value) if current_value in ANIMAL_OPTIONS.values() else 0

    choice = st.radio(
        "Pet type",
        labels,
        index=current_index,
        horizontal=True,
    )
    st.session_state.animal_type = ANIMAL_OPTIONS[choice]


# ---------------------------------------------------------------------------
# Message components
# ---------------------------------------------------------------------------

def render_urgency_badge(urgency: Optional[UrgencyAssessment]) -> None:
    """Render a color-coded urgency pill."""
    if urgency is None:
        return
    color = URGENCY_COLORS.get(urgency.level, "#888")
    emoji = URGENCY_EMOJIS.get(urgency.level, "")
    st.markdown(
        f'<span class="urgency-badge" style="background:{color}">'
        f"{emoji} {urgency.label}</span>",
        unsafe_allow_html=True,
    )


def render_emergency_banner(urgency: Optional[UrgencyAssessment]) -> None:
    """Show a prominent warning banner for CRITICAL / HIGH urgency."""
    if urgency is None:
        return
    if urgency.level == UrgencyLevel.CRITICAL:
        st.error(f"\u26a0\ufe0f {urgency.action}")
    elif urgency.level == UrgencyLevel.HIGH:
        st.warning(f"\u26a0\ufe0f {urgency.action}")


def render_source_panel(citations: Optional[CitationList]) -> None:
    """Collapsible panel listing source documents."""
    if citations is None or citations.is_empty:
        return
    label = f"Sources ({citations.num_sources} document{'s' if citations.num_sources != 1 else ''})"
    with st.expander(label):
        for i, citation in enumerate(citations.citations, 1):
            st.markdown(
                f'<div class="source-item">{i}. {citation.format_short()}</div>',
                unsafe_allow_html=True,
            )


def render_disclaimer() -> None:
    """Muted veterinary disclaimer below each assistant response."""
    st.markdown(
        f'<div class="disclaimer-text">{DISCLAIMER_TEXT}</div>',
        unsafe_allow_html=True,
    )


def render_assistant_message(response: ChainResponse) -> None:
    """
    Orchestrate all sub-components for a single assistant message.

    Order:
    1. Emergency banner (CRITICAL / HIGH only)
    2. Urgency badge
    3. Answer text
    4. Source panel
    5. Disclaimer
    """
    if response.was_refused:
        # Safety refusal — show plain text, no structured UI
        st.markdown(response.answer)
        return

    render_emergency_banner(response.urgency)
    render_urgency_badge(response.urgency)
    st.markdown(response.answer)
    render_source_panel(response.citations)
    render_disclaimer()
