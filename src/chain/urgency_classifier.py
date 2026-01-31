"""
LLM-based Urgency Classifier

Wraps the prompt-engineering UrgencyClassifier with actual LLM calls
for detailed urgency classification beyond keyword matching.
"""

import logging
from typing import Optional

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

from src.prompts import UrgencyClassifier, UrgencyAssessment, UrgencyConfig
from .config import LLMConfig


logger = logging.getLogger(__name__)


_CLASSIFICATION_SYSTEM = (
    "You are a veterinary triage assistant. Your only job is to classify "
    "the urgency of pet health queries. Respond with ONLY the urgency level "
    "(CRITICAL, HIGH, MODERATE, LOW, or INFORMATIONAL) followed by a one-sentence "
    "justification. Do not provide any medical advice."
)


class LLMUrgencyClassifier:
    """
    Urgency classifier that combines keyword pre-classification with
    LLM-based detailed classification for accurate triage.

    The two-stage approach:
    1. Fast keyword pre-classification (no LLM call) for obvious cases
    2. LLM-based classification for ambiguous queries
    """

    def __init__(
        self,
        llm_config: Optional[LLMConfig] = None,
        urgency_config: Optional[UrgencyConfig] = None,
        llm: Optional[ChatOpenAI] = None,
    ):
        """
        Initialize the LLM urgency classifier.

        Args:
            llm_config: LLM configuration. Uses defaults if not provided.
            urgency_config: Urgency classification configuration.
            llm: Existing ChatOpenAI instance to reuse.
        """
        self.llm_config = llm_config or LLMConfig()
        self.keyword_classifier = UrgencyClassifier(urgency_config)

        if llm:
            self.llm = llm
        else:
            self.llm = ChatOpenAI(
                model=self.llm_config.model_name,
                temperature=0.0,
                max_tokens=100,
                request_timeout=self.llm_config.request_timeout,
            )

        logger.info("LLMUrgencyClassifier initialized")

    def classify(
        self,
        query: str,
        use_llm: bool = True,
    ) -> UrgencyAssessment:
        """
        Classify the urgency of a pet health query.

        First performs keyword-based pre-classification. If the result
        is CRITICAL or HIGH (clear from keywords), returns immediately.
        Otherwise, uses the LLM for more nuanced classification.

        Args:
            query: The user's query string.
            use_llm: Whether to use LLM for ambiguous cases.

        Returns:
            UrgencyAssessment with the classified level.
        """
        # Stage 1: Keyword pre-classification
        pre_assessment = self.keyword_classifier.pre_classify(query)

        # For clear-cut emergencies, skip LLM
        if pre_assessment.level.value in ("critical", "high"):
            logger.info(
                f"Urgency pre-classified as {pre_assessment.level.value}, "
                "skipping LLM classification"
            )
            return pre_assessment

        # Stage 2: LLM classification for ambiguous cases
        if use_llm:
            return self._classify_with_llm(query, pre_assessment)

        return pre_assessment

    def _classify_with_llm(
        self,
        query: str,
        fallback: UrgencyAssessment,
    ) -> UrgencyAssessment:
        """
        Classify urgency using the LLM.

        Args:
            query: The user's query.
            fallback: Fallback assessment if LLM call fails.

        Returns:
            UrgencyAssessment from LLM classification.
        """
        classification_prompt = self.keyword_classifier.get_classification_prompt(query)

        try:
            messages = [
                SystemMessage(content=_CLASSIFICATION_SYSTEM),
                HumanMessage(content=classification_prompt),
            ]

            response = self.llm.invoke(messages)
            response_text = response.content

            logger.debug(f"LLM urgency response: {response_text[:100]}")

            assessment = self.keyword_classifier.parse_llm_response(response_text)
            return assessment

        except Exception as e:
            logger.warning(
                f"LLM urgency classification failed: {e}, "
                f"using pre-classification: {fallback.level.value}"
            )
            return fallback
