"""
Response Handler Module

Parses LLM output, validates safety using guardrails, and formats
the final response with urgency headers, citations, and disclaimers.
"""

import logging
from dataclasses import dataclass, field
from typing import Optional, List

from src.prompts import (
    SafetyGuardrails,
    GuardrailCheckResult,
    GuardrailConfig,
    UrgencyAssessment,
)
from src.retrival import CitationList
from .config import ResponseHandlerConfig


logger = logging.getLogger(__name__)


@dataclass
class ChainResponse:
    """Final response from the RAG chain."""

    answer: str
    urgency: Optional[UrgencyAssessment] = None
    citations: Optional[CitationList] = None
    safety_check: Optional[GuardrailCheckResult] = None
    was_refused: bool = False
    source_query: str = ""

    @property
    def is_safe(self) -> bool:
        """Check if the response passed safety validation."""
        if self.safety_check is None:
            return True
        return self.safety_check.passed

    @property
    def has_citations(self) -> bool:
        """Check if citations are available."""
        return self.citations is not None and not self.citations.is_empty


class ResponseHandler:
    """
    Handles post-processing of LLM responses.

    Responsibilities:
    - Validate responses against safety guardrails
    - Replace unsafe responses with refusal messages
    - Format final response with urgency header, citations, disclaimer
    """

    def __init__(
        self,
        config: Optional[ResponseHandlerConfig] = None,
        guardrail_config: Optional[GuardrailConfig] = None,
    ):
        """
        Initialize the response handler.

        Args:
            config: Response handler configuration.
            guardrail_config: Guardrail configuration for safety checks.
        """
        self.config = config or ResponseHandlerConfig()
        self.guardrails = SafetyGuardrails(guardrail_config)
        logger.debug("ResponseHandler initialized")

    def process(
        self,
        llm_output: str,
        query: str,
        urgency: Optional[UrgencyAssessment] = None,
        citations: Optional[CitationList] = None,
    ) -> ChainResponse:
        """
        Process an LLM response through safety validation and formatting.

        Args:
            llm_output: Raw text output from the LLM.
            query: The original user query.
            urgency: Urgency assessment for the query.
            citations: Citation list from retrieval.

        Returns:
            ChainResponse with validated and formatted answer.
        """
        # Step 1: Safety validation
        safety_check = None
        was_refused = False
        answer = llm_output.strip()

        if self.config.validate_safety:
            safety_check = self.guardrails.check_response(answer)

            if not safety_check.passed:
                logger.warning(
                    f"Response failed safety check: {safety_check.violations}"
                )
                answer = safety_check.refusal_message
                was_refused = True

        # Step 2: Format the response
        if not was_refused:
            answer = self._format_response(answer, urgency, citations)

        return ChainResponse(
            answer=answer,
            urgency=urgency,
            citations=citations,
            safety_check=safety_check,
            was_refused=was_refused,
            source_query=query,
        )

    def _format_response(
        self,
        answer: str,
        urgency: Optional[UrgencyAssessment],
        citations: Optional[CitationList],
    ) -> str:
        """Format the response with optional headers and footers."""
        parts = []

        # Urgency header
        if self.config.include_urgency_header and urgency:
            parts.append(urgency.format_header())
            parts.append(urgency.format_action_banner())
            parts.append("")

        # Main answer
        parts.append(answer)

        # Citations footer
        if (
            self.config.include_citations_footer
            and citations
            and not citations.is_empty
        ):
            parts.append("")
            parts.append(citations.format_list())

        # Disclaimer
        if self.config.include_disclaimer:
            parts.append("")
            parts.append(f"---\n{self.config.disclaimer_text}")

        return "\n".join(parts)
