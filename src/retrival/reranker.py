"""
Reranker Module

Optional relevance reranking for improved retrieval quality.
"""

import logging
from typing import Optional, List, Dict, Any, Callable

from .config import RerankerConfig, RerankerStrategy
from .retriever import RetrievalResult, RetrievalResponse
from .query_processor import ProcessedQuery


logger = logging.getLogger(__name__)


# Section type relevance mapping
SECTION_RELEVANCE = {
    "emergency": 1.0,
    "treatment": 0.9,
    "symptoms": 0.8,
    "diagnosis": 0.7,
    "prevention": 0.5,
    "general": 0.4,
}

# Query intent to section type mapping
QUERY_INTENT_SECTIONS = {
    "what to do": ["treatment", "emergency"],
    "how to treat": ["treatment"],
    "symptoms of": ["symptoms", "diagnosis"],
    "signs of": ["symptoms"],
    "is it dangerous": ["emergency", "symptoms"],
    "can dogs eat": ["treatment", "prevention"],
    "can cats eat": ["treatment", "prevention"],
    "poisoning": ["emergency", "treatment"],
    "toxic": ["emergency", "treatment"],
    "emergency": ["emergency"],
    "urgent": ["emergency"],
}


class Reranker:
    """
    Reranks retrieval results based on domain-specific relevance signals.

    Supports multiple reranking strategies:
    - NONE: No reranking (passthrough)
    - URGENCY_BOOST: Boost high-urgency content
    - ANIMAL_MATCH: Boost content matching detected animal type
    - COMBINED: Apply all boosting strategies
    """

    def __init__(self, config: Optional[RerankerConfig] = None):
        """
        Initialize the reranker.

        Args:
            config: Reranker configuration. Uses defaults if not provided.
        """
        self.config = config or RerankerConfig()
        logger.debug(f"Reranker initialized (strategy={self.config.strategy.value})")

    def rerank(
        self,
        response: RetrievalResponse,
        top_k: Optional[int] = None,
    ) -> RetrievalResponse:
        """
        Rerank retrieval results.

        Args:
            response: The original retrieval response.
            top_k: Optional limit on number of results to return.

        Returns:
            New RetrievalResponse with reranked results.
        """
        if not response.has_results:
            return response

        if self.config.strategy == RerankerStrategy.NONE:
            logger.debug("Reranking skipped (strategy=none)")
            results = response.results
        else:
            # Apply reranking based on strategy
            results = self._apply_reranking(response.results, response.query)

        # Sort by adjusted score
        results = sorted(results, key=lambda r: r.score, reverse=True)

        # Apply top_k limit if specified
        if top_k is not None and top_k < len(results):
            results = results[:top_k]

        logger.info(f"Reranked {len(response.results)} results")

        return RetrievalResponse(
            query=response.query,
            results=results,
            total_found=response.total_found,
            filtered_count=response.filtered_count + (len(response.results) - len(results))
        )

    def _apply_reranking(
        self,
        results: List[RetrievalResult],
        query: ProcessedQuery
    ) -> List[RetrievalResult]:
        """Apply reranking strategy to results."""
        reranked = []

        for result in results:
            adjusted_score = result.score

            if self.config.strategy in (
                RerankerStrategy.URGENCY_BOOST,
                RerankerStrategy.COMBINED
            ):
                adjusted_score = self._apply_urgency_boost(
                    adjusted_score, result, query
                )

            if self.config.strategy in (
                RerankerStrategy.ANIMAL_MATCH,
                RerankerStrategy.COMBINED
            ):
                adjusted_score = self._apply_animal_boost(
                    adjusted_score, result, query
                )

            if self.config.strategy == RerankerStrategy.COMBINED:
                adjusted_score = self._apply_section_boost(
                    adjusted_score, result, query
                )

            # Create new result with adjusted score
            reranked.append(RetrievalResult(
                content=result.content,
                score=adjusted_score,
                metadata=result.metadata
            ))

        return reranked

    def _apply_urgency_boost(
        self,
        score: float,
        result: RetrievalResult,
        query: ProcessedQuery
    ) -> float:
        """Boost score for urgent content when query indicates emergency."""
        if not query.is_emergency:
            return score

        urgency = result.urgency_level
        if urgency == "high":
            return score * self.config.urgency_boost
        elif urgency == "medium":
            return score * (1.0 + (self.config.urgency_boost - 1.0) * 0.5)

        return score

    def _apply_animal_boost(
        self,
        score: float,
        result: RetrievalResult,
        query: ProcessedQuery
    ) -> float:
        """Boost score when document animal type matches query."""
        if not query.detected_animal:
            return score

        animal_types = result.animal_types
        if not animal_types:
            return score

        # Exact match with detected animal
        if query.detected_animal in animal_types:
            return score * self.config.animal_match_boost

        # Penalize if document is for the other animal only
        other_animal = "cat" if query.detected_animal == "dog" else "dog"
        if other_animal in animal_types and query.detected_animal not in animal_types:
            return score * self.config.low_relevance_penalty

        return score

    def _apply_section_boost(
        self,
        score: float,
        result: RetrievalResult,
        query: ProcessedQuery
    ) -> float:
        """Boost score based on section type relevance to query."""
        section_type = result.section_type
        if not section_type:
            return score

        # Check if section matches query intent
        query_lower = query.cleaned_query.lower()
        matched_sections = []

        for intent, sections in QUERY_INTENT_SECTIONS.items():
            if intent in query_lower:
                matched_sections.extend(sections)

        if matched_sections and section_type in matched_sections:
            return score * self.config.section_type_boost

        # Apply general section relevance
        base_relevance = SECTION_RELEVANCE.get(section_type, 0.5)
        if base_relevance >= 0.8:
            return score * (1.0 + (self.config.section_type_boost - 1.0) * 0.5)

        return score


def create_custom_reranker(
    scoring_fn: Callable[[RetrievalResult, ProcessedQuery], float]
) -> Reranker:
    """
    Create a reranker with a custom scoring function.

    Args:
        scoring_fn: Function that takes (result, query) and returns adjusted score.

    Returns:
        Configured Reranker instance.

    Example:
        def my_scorer(result, query):
            if "chocolate" in query.cleaned_query and "poison" in result.content:
                return result.score * 2.0
            return result.score

        reranker = create_custom_reranker(my_scorer)
    """
    reranker = Reranker(RerankerConfig(strategy=RerankerStrategy.NONE))

    # Override the rerank method to use custom scoring
    original_rerank = reranker.rerank

    def custom_rerank(
        response: RetrievalResponse,
        top_k: Optional[int] = None,
    ) -> RetrievalResponse:
        if not response.has_results:
            return response

        reranked_results = []
        for result in response.results:
            new_score = scoring_fn(result, response.query)
            reranked_results.append(RetrievalResult(
                content=result.content,
                score=new_score,
                metadata=result.metadata
            ))

        # Sort by new score
        reranked_results.sort(key=lambda r: r.score, reverse=True)

        if top_k is not None and top_k < len(reranked_results):
            reranked_results = reranked_results[:top_k]

        return RetrievalResponse(
            query=response.query,
            results=reranked_results,
            total_found=response.total_found,
            filtered_count=response.filtered_count
        )

    reranker.rerank = custom_rerank
    return reranker
