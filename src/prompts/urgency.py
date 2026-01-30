"""
Urgency Classification Module

Provides prompt templates for classifying the urgency of pet health queries
into severity levels with appropriate action guidance.
"""

import logging
from dataclasses import dataclass
from typing import Optional, List

from .config import UrgencyLevel, UrgencyConfig


logger = logging.getLogger(__name__)


# Urgency level descriptions and guidance
URGENCY_DEFINITIONS = {
    UrgencyLevel.CRITICAL: {
        "label": "CRITICAL — Life-threatening emergency",
        "description": (
            "The animal may die or suffer permanent harm without immediate "
            "professional veterinary intervention."
        ),
        "action": (
            "Seek emergency veterinary care IMMEDIATELY. Do not wait. "
            "Call your vet or emergency animal hospital now."
        ),
        "examples": [
            "Not breathing or struggling to breathe",
            "Unconscious or unresponsive",
            "Active seizures",
            "Severe uncontrolled bleeding",
            "Known ingestion of highly toxic substance",
            "Hit by car or major trauma",
            "Choking",
        ],
    },
    UrgencyLevel.HIGH: {
        "label": "HIGH — Urgent veterinary attention needed",
        "description": (
            "The animal needs veterinary care within hours. The condition "
            "could worsen significantly without treatment."
        ),
        "action": (
            "Contact your veterinarian as soon as possible, ideally within "
            "1-2 hours. If after hours, consider an emergency clinic."
        ),
        "examples": [
            "Suspected broken bone or fracture",
            "Burns or scalds",
            "Eye injuries",
            "Persistent vomiting with blood",
            "Ingestion of potentially toxic substance",
            "Snake or insect bite with swelling",
            "Signs of heatstroke",
            "Difficulty urinating (especially male cats)",
        ],
    },
    UrgencyLevel.MODERATE: {
        "label": "MODERATE — Veterinary visit recommended",
        "description": (
            "The animal should be seen by a veterinarian soon but is not "
            "in immediate danger."
        ),
        "action": (
            "Schedule a veterinary appointment within 24-48 hours. "
            "Monitor for worsening symptoms."
        ),
        "examples": [
            "Vomiting or diarrhea (without blood, not prolonged)",
            "Minor limping or lameness",
            "Mild skin irritation or rash",
            "Loss of appetite lasting more than a day",
            "Minor wounds that have stopped bleeding",
            "Excessive scratching or ear shaking",
        ],
    },
    UrgencyLevel.LOW: {
        "label": "LOW — Monitor and consult if needed",
        "description": (
            "The situation can likely be managed at home with monitoring, "
            "but a vet visit is advisable if symptoms persist."
        ),
        "action": (
            "Monitor your pet closely. If symptoms persist beyond 24-48 hours "
            "or worsen, consult your veterinarian."
        ),
        "examples": [
            "Single episode of vomiting (pet otherwise normal)",
            "Minor behavioral changes",
            "Mild lethargy (eating and drinking normally)",
            "Small superficial scrape",
            "Occasional soft stool",
        ],
    },
    UrgencyLevel.INFORMATIONAL: {
        "label": "INFORMATIONAL — General wellness query",
        "description": (
            "This is a general information or preventive care question, "
            "not an active health concern."
        ),
        "action": (
            "No immediate action needed. Consider discussing with your vet "
            "at your next routine visit."
        ),
        "examples": [
            "General diet and nutrition questions",
            "Preventive care and vaccination questions",
            "Breed-specific information",
            "Training and behavior questions",
            "General pet safety and toxin awareness",
        ],
    },
}


_CLASSIFICATION_PROMPT = """Based on the user's description of their pet's situation, \
classify the urgency level as one of:

1. CRITICAL — Life-threatening emergency requiring immediate veterinary care
2. HIGH — Urgent, needs veterinary attention within hours
3. MODERATE — Should see a vet within 24-48 hours
4. LOW — Monitor at home, consult vet if symptoms persist
5. INFORMATIONAL — General wellness or information query

Consider these factors:
- Is the animal's life in immediate danger?
- Could the condition worsen rapidly without treatment?
- Are there signs of severe pain, trauma, or toxin exposure?
- How long have symptoms been present?
- Is the animal still eating, drinking, and responsive?

Respond with ONLY the urgency level (CRITICAL, HIGH, MODERATE, LOW, or \
INFORMATIONAL) followed by a brief justification."""


@dataclass
class UrgencyAssessment:
    """Result of an urgency assessment."""

    level: UrgencyLevel
    label: str
    description: str
    action: str
    pre_classified: bool = False

    def format_header(self) -> str:
        """Format as a display header for the response."""
        if self.level == UrgencyLevel.CRITICAL:
            return f"🔴 {self.label}"
        elif self.level == UrgencyLevel.HIGH:
            return f"🟠 {self.label}"
        elif self.level == UrgencyLevel.MODERATE:
            return f"🟡 {self.label}"
        elif self.level == UrgencyLevel.LOW:
            return f"🟢 {self.label}"
        else:
            return f"ℹ️ {self.label}"

    def format_action_banner(self) -> str:
        """Format the action guidance as a banner."""
        if self.level in (UrgencyLevel.CRITICAL, UrgencyLevel.HIGH):
            return f"⚠️ {self.action}"
        return self.action


class UrgencyClassifier:
    """
    Classifies the urgency of pet health queries.

    Provides:
    - Keyword-based pre-classification for immediate triage
    - LLM prompt templates for detailed classification
    - Urgency-appropriate response formatting
    """

    def __init__(self, config: Optional[UrgencyConfig] = None):
        """
        Initialize the urgency classifier.

        Args:
            config: Urgency classification configuration. Uses defaults if not provided.
        """
        self.config = config or UrgencyConfig()
        logger.debug("UrgencyClassifier initialized")

    def pre_classify(self, query: str) -> UrgencyAssessment:
        """
        Pre-classify urgency based on keyword matching.

        This provides a fast, initial urgency assessment before LLM
        classification. It is intentionally conservative (biased toward
        higher urgency).

        Args:
            query: The user's query string.

        Returns:
            UrgencyAssessment with the pre-classified level.
        """
        query_lower = query.lower()

        # Check critical keywords first
        for keyword in self.config.critical_keywords:
            if keyword in query_lower:
                logger.info(f"Pre-classified as CRITICAL (keyword: '{keyword}')")
                return self._make_assessment(UrgencyLevel.CRITICAL, pre_classified=True)

        # Check high urgency keywords
        for keyword in self.config.high_keywords:
            if keyword in query_lower:
                logger.info(f"Pre-classified as HIGH (keyword: '{keyword}')")
                return self._make_assessment(UrgencyLevel.HIGH, pre_classified=True)

        # Default to configured default level
        logger.debug(
            f"No urgency keywords found, using default: {self.config.default_level.value}"
        )
        return self._make_assessment(self.config.default_level, pre_classified=True)

    def get_classification_prompt(self, query: str) -> str:
        """
        Get the LLM prompt for urgency classification.

        Args:
            query: The user's query to classify.

        Returns:
            Formatted prompt string for the LLM.
        """
        return f"{_CLASSIFICATION_PROMPT}\n\nUser query: \"{query}\""

    def get_level_info(self, level: UrgencyLevel) -> dict:
        """
        Get detailed information for an urgency level.

        Args:
            level: The urgency level to look up.

        Returns:
            Dictionary with label, description, action, and examples.
        """
        return URGENCY_DEFINITIONS[level]

    def _make_assessment(
        self,
        level: UrgencyLevel,
        pre_classified: bool = False,
    ) -> UrgencyAssessment:
        """Create an UrgencyAssessment from a level."""
        info = URGENCY_DEFINITIONS[level]
        return UrgencyAssessment(
            level=level,
            label=info["label"],
            description=info["description"],
            action=info["action"],
            pre_classified=pre_classified,
        )

    def parse_llm_response(self, response: str) -> UrgencyAssessment:
        """
        Parse an LLM's urgency classification response.

        Args:
            response: The LLM's response text.

        Returns:
            UrgencyAssessment parsed from the response.
        """
        response_upper = response.strip().upper()

        for level in UrgencyLevel:
            if level.value.upper() in response_upper:
                logger.info(f"Parsed LLM urgency classification: {level.value}")
                return self._make_assessment(level)

        # Fallback to default if parsing fails
        logger.warning(
            f"Could not parse urgency from LLM response, "
            f"using default: {self.config.default_level.value}"
        )
        return self._make_assessment(self.config.default_level)


def classify_urgency(query: str) -> UrgencyAssessment:
    """
    Quick urgency classification using keyword pre-classification.

    Args:
        query: The user's query string.

    Returns:
        UrgencyAssessment with the classified level.
    """
    classifier = UrgencyClassifier()
    return classifier.pre_classify(query)
