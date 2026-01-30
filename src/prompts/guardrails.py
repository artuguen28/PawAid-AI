"""
Safety Guardrails Module

Builds safety instructions that prevent the LLM from providing
dangerous advice, making diagnoses, or recommending prescriptions.
"""

import logging
import re
from dataclasses import dataclass
from typing import Optional, List

from .config import GuardrailConfig


logger = logging.getLogger(__name__)


# Patterns that indicate the LLM attempted to diagnose
DIAGNOSIS_PATTERNS = [
    r"(?:your|the) (?:dog|cat|pet|puppy|kitten) (?:has|is suffering from|is diagnosed with) ",
    r"(?:this|it) (?:is|looks like|appears to be|sounds like) (?:a case of )?(?:\w+ ){0,2}(?:disease|syndrome|infection|disorder|condition)",
    r"I (?:can )?diagnose",
    r"(?:the )?diagnosis (?:is|would be)",
]

# Patterns that indicate the LLM attempted to prescribe
PRESCRIPTION_PATTERNS = [
    r"(?:give|administer|prescribe|take|use) (?:\d+ ?(?:mg|ml|cc|tablet|capsule|pill))",
    r"(?:dosage|dose) (?:of |is |should be )?\d+",
    r"(?:prescribe|prescribed|prescription) (?:of |for )",
    r"(?:take|give|administer) (?:amoxicillin|metronidazole|prednisone|tramadol|gabapentin|carprofen|meloxicam|benadryl|ibuprofen|acetaminophen|tylenol|advil)",
]

# Patterns that indicate surgery advice
SURGERY_PATTERNS = [
    r"(?:perform|do|attempt|make) (?:a |the |an )?(?:surgery|operation|procedure|incision|cut)",
    r"(?:surgically|operatively) (?:remove|repair|fix)",
    r"(?:you (?:can|should|need to) )?(?:cut|incise|suture|stitch)",
    r"(?:drain|lance|open) (?:the |a )?(?:wound|abscess|cyst)",
]

# Refusal messages
REFUSAL_MESSAGES = {
    "diagnosis": (
        "I cannot diagnose medical conditions. Only a licensed veterinarian "
        "can provide a proper diagnosis after examining your pet. "
        "Please consult your vet for an accurate diagnosis."
    ),
    "prescription": (
        "I cannot recommend specific medications or dosages. Medications must "
        "be prescribed by a licensed veterinarian who has examined your pet. "
        "Incorrect medications or dosages can be dangerous."
    ),
    "surgery": (
        "I cannot provide surgical or advanced medical procedure advice. "
        "These must only be performed by a licensed veterinarian. "
        "Please consult your vet for any procedures your pet may need."
    ),
    "dosage": (
        "I cannot recommend medication dosages. The correct dosage depends on "
        "your pet's weight, age, health conditions, and other medications. "
        "Only your veterinarian can determine the safe dosage."
    ),
    "out_of_scope": (
        "This question is outside my area of expertise. I can only help with "
        "first-aid guidance for dogs and cats. Please consult a qualified "
        "professional for this question."
    ),
}

_GUARDRAIL_PROMPT = """SAFETY GUARDRAILS — You MUST follow these rules at all times:

{refusal_rules}
{redirect_rules}
{scope_rules}

If you are unsure whether a response violates these rules, err on the side \
of caution and recommend consulting a veterinarian."""


@dataclass
class GuardrailCheckResult:
    """Result of a guardrail check on generated content."""

    passed: bool
    violations: List[str]
    refusal_message: str = ""

    @property
    def has_violations(self) -> bool:
        """Check if any violations were detected."""
        return len(self.violations) > 0


class SafetyGuardrails:
    """
    Manages safety guardrails for the veterinary AI assistant.

    Provides:
    - Prompt instructions that establish safety boundaries
    - Post-generation checks for policy violations
    - Refusal messages for out-of-scope requests
    """

    def __init__(self, config: Optional[GuardrailConfig] = None):
        """
        Initialize safety guardrails.

        Args:
            config: Guardrail configuration. Uses defaults if not provided.
        """
        self.config = config or GuardrailConfig()
        self._compiled_patterns = self._compile_patterns()
        logger.debug("SafetyGuardrails initialized")

    def build_guardrail_prompt(self) -> str:
        """
        Build the safety guardrail section of the system prompt.

        Returns:
            Formatted guardrail instructions for the LLM.
        """
        refusal_rules = self._build_refusal_rules()
        redirect_rules = self._build_redirect_rules()
        scope_rules = self._build_scope_rules()

        prompt = _GUARDRAIL_PROMPT.format(
            refusal_rules=refusal_rules,
            redirect_rules=redirect_rules,
            scope_rules=scope_rules,
        )

        logger.debug(f"Built guardrail prompt: {len(prompt)} chars")
        return prompt

    def check_response(self, response: str) -> GuardrailCheckResult:
        """
        Check a generated response for safety violations.

        This is a post-generation check that detects if the LLM
        output contains content that violates safety policies.

        Args:
            response: The LLM-generated response text.

        Returns:
            GuardrailCheckResult indicating pass/fail and violations.
        """
        violations = []

        if self.config.refuse_diagnosis:
            if self._check_patterns(response, "diagnosis"):
                violations.append("Attempted diagnosis detected")

        if self.config.refuse_prescriptions or self.config.refuse_dosage:
            if self._check_patterns(response, "prescription"):
                violations.append("Medication recommendation detected")

        if self.config.refuse_surgery_advice:
            if self._check_patterns(response, "surgery"):
                violations.append("Surgical advice detected")

        if violations:
            logger.warning(f"Guardrail violations detected: {violations}")
            refusal = self._build_combined_refusal(violations)
            return GuardrailCheckResult(
                passed=False,
                violations=violations,
                refusal_message=refusal,
            )

        logger.debug("Response passed guardrail check")
        return GuardrailCheckResult(passed=True, violations=[])

    def get_refusal_message(self, category: str) -> str:
        """
        Get a refusal message for a specific category.

        Args:
            category: The refusal category ("diagnosis", "prescription",
                "surgery", "dosage", or "out_of_scope").

        Returns:
            Formatted refusal message string.
        """
        return REFUSAL_MESSAGES.get(category, REFUSAL_MESSAGES["out_of_scope"])

    def _build_refusal_rules(self) -> str:
        """Build the refusal rules section."""
        rules = []

        if self.config.refuse_diagnosis:
            rules.append(
                "- NEVER diagnose conditions, diseases, or disorders. "
                "Do not say \"your pet has...\" or \"this is a case of...\""
            )

        if self.config.refuse_prescriptions:
            rules.append(
                "- NEVER recommend specific medications by name. "
                "Do not suggest any prescription or over-the-counter drugs."
            )

        if self.config.refuse_dosage:
            rules.append(
                "- NEVER provide medication dosages (mg, ml, tablets, etc.). "
                "Dosages must come from a veterinarian."
            )

        if self.config.refuse_surgery_advice:
            rules.append(
                "- NEVER advise on surgical procedures, incisions, or "
                "advanced medical procedures."
            )

        if not rules:
            return ""

        return "NEVER do the following:\n" + "\n".join(rules)

    def _build_redirect_rules(self) -> str:
        """Build the redirect rules section."""
        rules = []

        if self.config.always_suggest_vet:
            rules.append(
                "- ALWAYS include a recommendation to consult a veterinarian "
                "for proper diagnosis and treatment."
            )

        if self.config.emergency_redirect:
            rules.append(
                "- For ANY life-threatening situation, your FIRST instruction must be "
                "to contact a veterinarian or emergency animal hospital IMMEDIATELY."
            )

        if not rules:
            return ""

        return "ALWAYS do the following:\n" + "\n".join(rules)

    def _build_scope_rules(self) -> str:
        """Build the scope restriction rules."""
        rules = []

        if self.config.restrict_to_cats_and_dogs:
            rules.append(
                "- Only provide advice for DOGS and CATS. For other animals, "
                "politely decline and suggest consulting an appropriate specialist."
            )

        if self.config.restrict_to_first_aid:
            rules.append(
                "- Only provide FIRST-AID and immediate care guidance. "
                "For long-term treatment plans, defer to a veterinarian."
            )

        if not rules:
            return ""

        return "SCOPE RESTRICTIONS:\n" + "\n".join(rules)

    def _compile_patterns(self) -> dict:
        """Pre-compile regex patterns for performance."""
        compiled = {}

        if self.config.refuse_diagnosis:
            compiled["diagnosis"] = [
                re.compile(p, re.IGNORECASE) for p in DIAGNOSIS_PATTERNS
            ]

        if self.config.refuse_prescriptions or self.config.refuse_dosage:
            compiled["prescription"] = [
                re.compile(p, re.IGNORECASE) for p in PRESCRIPTION_PATTERNS
            ]

        if self.config.refuse_surgery_advice:
            compiled["surgery"] = [
                re.compile(p, re.IGNORECASE) for p in SURGERY_PATTERNS
            ]

        return compiled

    def _check_patterns(self, text: str, category: str) -> bool:
        """Check if text matches any patterns in a category."""
        patterns = self._compiled_patterns.get(category, [])
        for pattern in patterns:
            if pattern.search(text):
                logger.debug(
                    f"Pattern match in '{category}': {pattern.pattern}"
                )
                return True
        return False

    def _build_combined_refusal(self, violations: List[str]) -> str:
        """Build a combined refusal message for multiple violations."""
        parts = [
            "I need to correct my response for your safety:\n"
        ]

        for violation in violations:
            if "diagnosis" in violation.lower():
                parts.append(f"- {REFUSAL_MESSAGES['diagnosis']}")
            elif "medication" in violation.lower():
                parts.append(f"- {REFUSAL_MESSAGES['prescription']}")
            elif "surgical" in violation.lower():
                parts.append(f"- {REFUSAL_MESSAGES['surgery']}")

        parts.append(
            "\nPlease consult your veterinarian for proper diagnosis and treatment."
        )

        return "\n".join(parts)


def build_guardrail_prompt() -> str:
    """
    Build guardrail instructions with default configuration.

    Returns:
        Formatted guardrail instructions string.
    """
    guardrails = SafetyGuardrails()
    return guardrails.build_guardrail_prompt()
