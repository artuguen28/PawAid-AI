"""
System Prompt Module

Builds the safety-critical veterinary persona system prompt for the LLM.
"""

import logging
from typing import Optional, List

from .config import SystemPromptConfig, GuardrailConfig


logger = logging.getLogger(__name__)


# Core system prompt components
_PERSONA_TEMPLATE = """You are {persona_name}, a {persona_role} for {animals}.

Your purpose is to provide immediate, practical first-aid guidance when a pet \
owner's {animals_lower} is sick, injured, or has been exposed to something \
potentially dangerous.

You speak in a {tone} tone. {simple_language_note}"""

_SAFETY_PREAMBLE = """CRITICAL SAFETY RULES (you must ALWAYS follow these):
- You are NOT a veterinarian. You provide first-aid guidance only.
- NEVER diagnose conditions or diseases.
- NEVER recommend specific medications, drugs, or dosages.
- NEVER provide surgical advice or procedures.
- ALWAYS recommend consulting a veterinarian for proper diagnosis and treatment.
- If the situation sounds life-threatening, clearly state this is an EMERGENCY \
and urge the owner to contact a veterinarian or emergency animal hospital \
IMMEDIATELY.
- If you are unsure or the question is outside your scope, say so honestly \
and recommend professional veterinary care."""

_SCOPE_INSTRUCTIONS = """YOUR SCOPE:
- First-aid and immediate care guidance for {animals}
- Recognizing signs that require emergency veterinary care
- General wellness and preventive care information
- Identifying common household toxins and dangers for pets
- Basic wound care, stabilization, and comfort measures

OUT OF SCOPE (you must refuse these):
- Medical diagnosis of conditions or diseases
- Prescription medication recommendations or dosages
- Surgical procedures or advanced medical treatments
- Care for animals other than {animals_lower}
- Replacing professional veterinary consultation"""

_RESPONSE_GUIDELINES = """RESPONSE GUIDELINES:
- Base your answers ONLY on the provided veterinary reference material.
- If the reference material does not contain relevant information, say so \
clearly rather than guessing.
- Structure your responses with clear, numbered steps when giving first-aid \
instructions.
- Always include "when to see a vet" guidance.
- Include warnings about what NOT to do when relevant.
- Cite your sources using the reference numbers provided in the context."""

_EMERGENCY_DISCLAIMER = """
EMERGENCY DISCLAIMER:
If at any point you believe the animal may be in a life-threatening situation, \
your FIRST response must be:
"⚠️ This sounds like it could be an emergency. Please contact your \
veterinarian or nearest emergency animal hospital immediately."
Then provide any safe, immediate first-aid steps the owner can take while \
seeking professional help."""


class SystemPromptBuilder:
    """
    Builds the system prompt for the PawAid veterinary assistant.

    The system prompt establishes:
    - The AI's veterinary first-aid persona
    - Safety-critical boundaries and guardrails
    - Response formatting guidelines
    - Emergency handling protocols
    """

    def __init__(
        self,
        config: Optional[SystemPromptConfig] = None,
        guardrail_config: Optional[GuardrailConfig] = None,
    ):
        """
        Initialize the system prompt builder.

        Args:
            config: System prompt configuration. Uses defaults if not provided.
            guardrail_config: Guardrail configuration for safety rules.
        """
        self.config = config or SystemPromptConfig()
        self.guardrail_config = guardrail_config or GuardrailConfig()
        logger.debug("SystemPromptBuilder initialized")

    def build(self, animal_type: Optional[str] = None) -> str:
        """
        Build the complete system prompt.

        Args:
            animal_type: Optional specific animal type ("dog" or "cat")
                to specialize the prompt. If None, covers all supported animals.

        Returns:
            Complete system prompt string.
        """
        parts = []

        # 1. Persona
        parts.append(self._build_persona(animal_type))

        # 2. Safety rules
        parts.append(_SAFETY_PREAMBLE)

        # 3. Scope
        parts.append(self._build_scope(animal_type))

        # 4. Response guidelines
        parts.append(_RESPONSE_GUIDELINES)

        # 5. Emergency disclaimer
        if self.config.emergency_disclaimer:
            parts.append(_EMERGENCY_DISCLAIMER)

        prompt = "\n\n".join(parts)

        logger.info(
            f"Built system prompt: {len(prompt)} chars, "
            f"animal_type={animal_type or 'all'}"
        )

        return prompt

    def _build_persona(self, animal_type: Optional[str] = None) -> str:
        """Build the persona section of the system prompt."""
        animals = self._format_animals(animal_type)
        animals_lower = animals.lower()

        simple_language_note = ""
        if self.config.use_simple_language:
            simple_language_note = (
                "Use simple, non-technical language that any pet owner can "
                "understand. When you must use medical terms, briefly explain them."
            )

        return _PERSONA_TEMPLATE.format(
            persona_name=self.config.persona_name,
            persona_role=self.config.persona_role,
            animals=animals,
            animals_lower=animals_lower,
            tone=self.config.tone,
            simple_language_note=simple_language_note,
        )

    def _build_scope(self, animal_type: Optional[str] = None) -> str:
        """Build the scope section of the system prompt."""
        animals = self._format_animals(animal_type)
        animals_lower = animals.lower()

        return _SCOPE_INSTRUCTIONS.format(
            animals=animals,
            animals_lower=animals_lower,
        )

    def _format_animals(self, animal_type: Optional[str] = None) -> str:
        """Format the animal type string for prompt insertion."""
        if animal_type:
            return animal_type.capitalize() + "s"

        # Format all supported animals
        animals = self.config.supported_animals
        if len(animals) == 1:
            return animals[0].capitalize() + "s"
        elif len(animals) == 2:
            return f"{animals[0].capitalize()}s and {animals[1].capitalize()}s"
        else:
            formatted = ", ".join(a.capitalize() + "s" for a in animals[:-1])
            return f"{formatted}, and {animals[-1].capitalize()}s"

    def build_with_context(
        self,
        context_text: str,
        animal_type: Optional[str] = None,
        is_emergency: bool = False,
    ) -> str:
        """
        Build system prompt with retrieved context injected.

        Args:
            context_text: Formatted context from the retrieval system.
            animal_type: Optional specific animal type.
            is_emergency: Whether this is an emergency query.

        Returns:
            Complete system prompt with context.
        """
        base_prompt = self.build(animal_type)

        context_section = "\n\n--- REFERENCE MATERIAL ---\n"
        if is_emergency:
            context_section += (
                "⚠️ EMERGENCY QUERY — Prioritize immediate safety actions.\n\n"
            )
        context_section += context_text
        context_section += "\n--- END REFERENCE MATERIAL ---"

        return base_prompt + context_section


def build_system_prompt(
    animal_type: Optional[str] = None,
    is_emergency: bool = False,
) -> str:
    """
    Build a system prompt with default configuration.

    Args:
        animal_type: Optional specific animal type ("dog" or "cat").
        is_emergency: Whether this is an emergency query.

    Returns:
        Complete system prompt string.
    """
    builder = SystemPromptBuilder()
    return builder.build(animal_type)
