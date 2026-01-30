"""
Response Templates Module

Provides structured response templates for different types of veterinary
first-aid responses, ensuring consistent and safe formatting.
"""

import logging
from dataclasses import dataclass, field
from typing import Optional, List

from .config import ResponseStyle, TemplateConfig, UrgencyLevel
from .urgency import UrgencyAssessment


logger = logging.getLogger(__name__)


_EMERGENCY_TEMPLATE = """⚠️ EMERGENCY — SEEK VETERINARY CARE IMMEDIATELY ⚠️

{urgency_banner}

While you seek professional help, here is what you can do right now:

{steps}
{do_not_section}
{vet_section}
{sources_section}"""

_FIRST_AID_TEMPLATE = """{urgency_header}

{summary}

**Recommended Steps:**

{steps}
{do_not_section}
{vet_section}
{sources_section}"""

_GENERAL_TEMPLATE = """{urgency_header}

{summary}
{detail_section}
{vet_section}
{sources_section}"""

_NO_INFO_TEMPLATE = """I don't have specific information about this topic in my \
veterinary reference materials.

{vet_redirect}

{safety_note}"""


@dataclass
class ResponseTemplate:
    """A structured response template with sections."""

    style: ResponseStyle
    urgency_header: str = ""
    urgency_banner: str = ""
    summary: str = ""
    steps_instruction: str = ""
    do_not_instruction: str = ""
    vet_instruction: str = ""
    sources_instruction: str = ""
    detail_instruction: str = ""

    def render(self) -> str:
        """
        Render the template into a complete LLM instruction.

        Returns:
            Formatted instruction string for the LLM.
        """
        if self.style == ResponseStyle.EMERGENCY:
            return self._render_emergency()
        elif self.style == ResponseStyle.FIRST_AID:
            return self._render_first_aid()
        elif self.style == ResponseStyle.GENERAL:
            return self._render_general()
        else:
            return self._render_no_info()

    def _render_emergency(self) -> str:
        """Render emergency template."""
        return _EMERGENCY_TEMPLATE.format(
            urgency_banner=self.urgency_banner,
            steps=self.steps_instruction,
            do_not_section=self.do_not_instruction,
            vet_section=self.vet_instruction,
            sources_section=self.sources_instruction,
        ).strip()

    def _render_first_aid(self) -> str:
        """Render first-aid template."""
        return _FIRST_AID_TEMPLATE.format(
            urgency_header=self.urgency_header,
            summary=self.summary,
            steps=self.steps_instruction,
            do_not_section=self.do_not_instruction,
            vet_section=self.vet_instruction,
            sources_section=self.sources_instruction,
        ).strip()

    def _render_general(self) -> str:
        """Render general information template."""
        return _GENERAL_TEMPLATE.format(
            urgency_header=self.urgency_header,
            summary=self.summary,
            detail_section=self.detail_instruction,
            vet_section=self.vet_instruction,
            sources_section=self.sources_instruction,
        ).strip()

    def _render_no_info(self) -> str:
        """Render no-information template."""
        return _NO_INFO_TEMPLATE.format(
            vet_redirect=self.vet_instruction,
            safety_note=self.sources_instruction,
        ).strip()


class TemplateManager:
    """
    Manages response templates for different query types.

    Produces structured LLM instructions that guide the model to
    format responses consistently with appropriate safety sections.
    """

    def __init__(self, config: Optional[TemplateConfig] = None):
        """
        Initialize the template manager.

        Args:
            config: Template configuration. Uses defaults if not provided.
        """
        self.config = config or TemplateConfig()
        logger.debug("TemplateManager initialized")

    def get_response_instruction(
        self,
        style: ResponseStyle,
        assessment: Optional[UrgencyAssessment] = None,
        animal_type: Optional[str] = None,
        has_context: bool = True,
    ) -> str:
        """
        Get LLM instructions for formatting the response.

        This returns a prompt instruction block that tells the LLM
        how to structure its response.

        Args:
            style: The response style to use.
            assessment: Optional urgency assessment.
            animal_type: Optional detected animal type.
            has_context: Whether retrieved context is available.

        Returns:
            Instruction string for the LLM.
        """
        if not has_context:
            style = ResponseStyle.NO_INFORMATION

        template = self._build_template(style, assessment, animal_type)

        instruction = self._build_formatting_instruction(style, assessment)

        logger.info(
            f"Generated response instruction: style={style.value}, "
            f"urgency={assessment.level.value if assessment else 'none'}"
        )

        return instruction

    def select_style(
        self,
        assessment: Optional[UrgencyAssessment] = None,
        has_context: bool = True,
    ) -> ResponseStyle:
        """
        Select the appropriate response style based on urgency and context.

        Args:
            assessment: Optional urgency assessment.
            has_context: Whether retrieved context is available.

        Returns:
            The appropriate ResponseStyle.
        """
        if not has_context:
            return ResponseStyle.NO_INFORMATION

        if assessment is None:
            return ResponseStyle.GENERAL

        if assessment.level in (UrgencyLevel.CRITICAL, UrgencyLevel.HIGH):
            return ResponseStyle.EMERGENCY
        elif assessment.level == UrgencyLevel.MODERATE:
            return ResponseStyle.FIRST_AID
        else:
            return ResponseStyle.GENERAL

    def _build_template(
        self,
        style: ResponseStyle,
        assessment: Optional[UrgencyAssessment],
        animal_type: Optional[str],
    ) -> ResponseTemplate:
        """Build a ResponseTemplate for the given parameters."""
        template = ResponseTemplate(style=style)

        if assessment:
            template.urgency_header = assessment.format_header()
            template.urgency_banner = assessment.format_action_banner()

        animal_str = animal_type or "pet"

        template.steps_instruction = self._steps_instruction(animal_str)
        template.do_not_instruction = self._do_not_instruction()
        template.vet_instruction = self._vet_instruction(style, assessment)
        template.sources_instruction = self._sources_instruction()
        template.detail_instruction = self._detail_instruction()

        return template

    def _build_formatting_instruction(
        self,
        style: ResponseStyle,
        assessment: Optional[UrgencyAssessment],
    ) -> str:
        """Build the LLM formatting instruction."""
        parts = ["FORMAT YOUR RESPONSE AS FOLLOWS:"]

        if style == ResponseStyle.EMERGENCY:
            parts.append(self._emergency_format_instruction(assessment))
        elif style == ResponseStyle.FIRST_AID:
            parts.append(self._first_aid_format_instruction(assessment))
        elif style == ResponseStyle.GENERAL:
            parts.append(self._general_format_instruction(assessment))
        else:
            parts.append(self._no_info_format_instruction())

        return "\n\n".join(parts)

    def _emergency_format_instruction(
        self,
        assessment: Optional[UrgencyAssessment],
    ) -> str:
        """Build formatting instructions for emergency responses."""
        header = ""
        if assessment:
            header = f"\n1. Start with: \"{assessment.format_header()}\""
        else:
            header = "\n1. Start with an emergency warning header"

        return f"""This is an EMERGENCY response. Structure it as:
{header}
2. State the urgency: advise the owner to seek emergency veterinary care IMMEDIATELY
3. Provide IMMEDIATE first-aid steps (numbered, max {self.config.max_steps} steps) \
the owner can take right now while seeking help
4. Include a "⛔ Do NOT" section listing dangerous actions to avoid
5. End with "📞 When to see a vet" — in this case, IMMEDIATELY
6. Cite sources using [1], [2], etc. matching the reference numbers"""

    def _first_aid_format_instruction(
        self,
        assessment: Optional[UrgencyAssessment],
    ) -> str:
        """Build formatting instructions for first-aid responses."""
        header = ""
        if assessment:
            header = f"\n1. Start with: \"{assessment.format_header()}\""
        else:
            header = "\n1. Start with the urgency level"

        return f"""Structure your response as a first-aid guide:
{header}
2. Briefly summarize the situation (1-2 sentences)
3. Provide step-by-step first-aid instructions (numbered, max {self.config.max_steps} steps)
4. Include a "⛔ Do NOT" section listing things to avoid
5. Include "📞 When to see a vet" with specific signs to watch for
6. Cite sources using [1], [2], etc. matching the reference numbers"""

    def _general_format_instruction(
        self,
        assessment: Optional[UrgencyAssessment],
    ) -> str:
        """Build formatting instructions for general responses."""
        header = ""
        if assessment:
            header = f"\n1. Start with: \"{assessment.format_header()}\""
        else:
            header = "\n1. Start with a brief header"

        return f"""Structure your response as an informational guide:
{header}
2. Provide a clear, concise answer to the question
3. Include relevant details from the reference material
4. If applicable, mention preventive care tips
5. Include "📞 When to see a vet" if relevant health concerns are mentioned
6. Cite sources using [1], [2], etc. matching the reference numbers"""

    def _no_info_format_instruction(self) -> str:
        """Build formatting instructions for no-information responses."""
        return """The reference material does not contain relevant information for \
this query. Structure your response as:

1. Clearly state that you don't have specific information on this topic
2. Do NOT guess or make up information
3. Recommend the owner consult their veterinarian
4. If the query sounds urgent, advise seeking immediate veterinary care
5. If you can offer any general safety advice that is well-established, \
do so with a clear caveat"""

    def _steps_instruction(self, animal_str: str) -> str:
        """Build the steps instruction."""
        return (
            f"Provide clear, numbered first-aid steps (max {self.config.max_steps}) "
            f"for the {animal_str}."
        )

    def _do_not_instruction(self) -> str:
        """Build the 'do not' instruction."""
        if not self.config.include_what_not_to_do:
            return ""
        return (
            "Include a section of things the owner should NOT do, "
            "as incorrect first aid can cause further harm."
        )

    def _vet_instruction(
        self,
        style: ResponseStyle,
        assessment: Optional[UrgencyAssessment],
    ) -> str:
        """Build the vet recommendation instruction."""
        if not self.config.include_when_to_see_vet:
            return ""

        if style == ResponseStyle.EMERGENCY:
            return "Emphasize that the owner should seek veterinary care IMMEDIATELY."
        elif assessment and assessment.level == UrgencyLevel.HIGH:
            return "Advise the owner to see a vet within 1-2 hours."

        return (
            "Include guidance on when the owner should consult a veterinarian, "
            "with specific warning signs to watch for."
        )

    def _sources_instruction(self) -> str:
        """Build the sources instruction."""
        return "Cite your sources using reference numbers [1], [2], etc."

    def _detail_instruction(self) -> str:
        """Build the detail section instruction."""
        return "Provide relevant details organized in clear paragraphs."


def get_response_instruction(
    is_emergency: bool = False,
    has_context: bool = True,
    assessment: Optional[UrgencyAssessment] = None,
    animal_type: Optional[str] = None,
) -> str:
    """
    Get response formatting instructions with default configuration.

    Args:
        is_emergency: Whether the query is an emergency.
        has_context: Whether retrieved context is available.
        assessment: Optional urgency assessment.
        animal_type: Optional detected animal type.

    Returns:
        LLM instruction string.
    """
    manager = TemplateManager()

    if not has_context:
        style = ResponseStyle.NO_INFORMATION
    elif is_emergency:
        style = ResponseStyle.EMERGENCY
    else:
        style = manager.select_style(assessment, has_context)

    return manager.get_response_instruction(
        style=style,
        assessment=assessment,
        animal_type=animal_type,
        has_context=has_context,
    )
