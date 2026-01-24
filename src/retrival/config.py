"""
Retrieval Configuration Module

Configuration dataclasses for the retrieval system.
"""

from dataclasses import dataclass, field
from typing import Optional, List
from enum import Enum


class RerankerStrategy(Enum):
    """Available reranking strategies."""

    NONE = "none"
    URGENCY_BOOST = "urgency_boost"
    ANIMAL_MATCH = "animal_match"
    COMBINED = "combined"


@dataclass
class QueryConfig:
    """Configuration for query processing."""

    # Query cleaning
    normalize_whitespace: bool = True
    lowercase: bool = True
    remove_punctuation: bool = False

    # Query expansion
    expand_synonyms: bool = True
    expand_abbreviations: bool = True

    # Animal detection
    detect_animal_type: bool = True
    default_animal_type: Optional[str] = None  # "dog", "cat", or None for both

    # Urgency detection
    detect_urgency: bool = True


@dataclass
class RetrieverConfig:
    """Configuration for the retriever."""

    # Search parameters
    default_k: int = 5
    max_k: int = 20
    min_similarity_score: float = 0.0

    # Metadata filtering
    filter_by_animal: bool = True
    filter_by_urgency: bool = False

    def __post_init__(self):
        if self.default_k < 1:
            raise ValueError("default_k must be at least 1")
        if self.max_k < self.default_k:
            raise ValueError("max_k must be greater than or equal to default_k")
        if not 0.0 <= self.min_similarity_score <= 1.0:
            raise ValueError("min_similarity_score must be between 0.0 and 1.0")


@dataclass
class RerankerConfig:
    """Configuration for the reranker."""

    strategy: RerankerStrategy = RerankerStrategy.COMBINED

    # Boost factors
    urgency_boost: float = 1.5
    animal_match_boost: float = 1.3
    section_type_boost: float = 1.2

    # Penalty factors
    low_relevance_penalty: float = 0.8

    def __post_init__(self):
        if self.urgency_boost < 1.0:
            raise ValueError("urgency_boost must be at least 1.0")
        if self.animal_match_boost < 1.0:
            raise ValueError("animal_match_boost must be at least 1.0")
        if not 0.0 < self.low_relevance_penalty <= 1.0:
            raise ValueError("low_relevance_penalty must be between 0.0 and 1.0")


@dataclass
class ContextConfig:
    """Configuration for context building."""

    # Context limits
    max_context_length: int = 4000  # characters
    max_chunks: int = 5

    # Formatting
    include_source: bool = True
    include_page_number: bool = True
    include_section_type: bool = True
    chunk_separator: str = "\n\n---\n\n"

    # Truncation
    truncate_long_chunks: bool = True
    chunk_max_length: int = 1000

    def __post_init__(self):
        if self.max_context_length < 100:
            raise ValueError("max_context_length must be at least 100")
        if self.max_chunks < 1:
            raise ValueError("max_chunks must be at least 1")


@dataclass
class CitationConfig:
    """Configuration for citation handling."""

    # Citation format
    include_page_numbers: bool = True
    include_section_type: bool = True
    include_chunk_index: bool = False

    # Grouping
    group_by_source: bool = True
    deduplicate: bool = True


@dataclass
class RetrievalConfig:
    """Combined configuration for the full retrieval pipeline."""

    query: QueryConfig = field(default_factory=QueryConfig)
    retriever: RetrieverConfig = field(default_factory=RetrieverConfig)
    reranker: RerankerConfig = field(default_factory=RerankerConfig)
    context: ContextConfig = field(default_factory=ContextConfig)
    citation: CitationConfig = field(default_factory=CitationConfig)
    verbose: bool = False

    @classmethod
    def default(cls) -> "RetrievalConfig":
        """Create a default configuration."""
        return cls()

    @classmethod
    def for_testing(cls) -> "RetrievalConfig":
        """Create a configuration suitable for testing."""
        return cls(
            retriever=RetrieverConfig(default_k=3, max_k=10),
            context=ContextConfig(max_chunks=3, max_context_length=2000),
            verbose=True
        )

    @classmethod
    def for_emergency(cls) -> "RetrievalConfig":
        """Create a configuration optimized for emergency queries."""
        return cls(
            retriever=RetrieverConfig(default_k=7, filter_by_urgency=True),
            reranker=RerankerConfig(
                strategy=RerankerStrategy.COMBINED,
                urgency_boost=2.0
            ),
            context=ContextConfig(max_chunks=7),
            verbose=False
        )
