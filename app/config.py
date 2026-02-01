"""
Streamlit App Configuration

Constants, color mappings, and chain configuration factory
for the PawAid AI frontend.
"""

from src.chain.config import (
    ChainConfig,
    LLMConfig,
    MemoryConfig,
    ResponseHandlerConfig,
)
from src.prompts.config import UrgencyLevel

# Page
APP_TITLE = "PawAid AI"
APP_ICON = "\U0001f43e"  # paw prints
PAGE_LAYOUT = "centered"

# Pet selector
ANIMAL_OPTIONS = {"Dog": "dog", "Cat": "cat"}

# Urgency level -> hex color
URGENCY_COLORS = {
    UrgencyLevel.CRITICAL: "#dc3545",
    UrgencyLevel.HIGH: "#fd7e14",
    UrgencyLevel.MODERATE: "#ffc107",
    UrgencyLevel.LOW: "#28a745",
    UrgencyLevel.INFORMATIONAL: "#17a2b8",
}

# Urgency level -> emoji (matches UrgencyAssessment.format_header)
URGENCY_EMOJIS = {
    UrgencyLevel.CRITICAL: "\U0001f534",
    UrgencyLevel.HIGH: "\U0001f7e0",
    UrgencyLevel.MODERATE: "\U0001f7e1",
    UrgencyLevel.LOW: "\U0001f7e2",
    UrgencyLevel.INFORMATIONAL: "\u2139\ufe0f",
}

DISCLAIMER_TEXT = (
    "This is first-aid guidance only, not a veterinary diagnosis. "
    "Please consult a licensed veterinarian for proper medical care."
)


def get_chain_config() -> ChainConfig:
    """Build a ChainConfig with text formatting disabled (UI handles display)."""
    return ChainConfig(
        llm=LLMConfig(),
        response_handler=ResponseHandlerConfig(
            include_urgency_header=False,
            include_citations_footer=False,
            include_disclaimer=False,
            validate_safety=True,
        ),
        memory=MemoryConfig(enabled=True, max_history_turns=5),
    )
