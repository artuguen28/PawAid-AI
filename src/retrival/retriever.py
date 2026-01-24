"""
Retriever Module

Core retrieval functionality for similarity search with metadata filtering.
"""

import logging
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any

from langchain_core.documents import Document as LCDocument

from src.embeddings import ChromaManager, ChromaConfig
from .config import RetrieverConfig, QueryConfig
from .query_processor import QueryProcessor, ProcessedQuery, create_metadata_filter


logger = logging.getLogger(__name__)


@dataclass
class RetrievalResult:
    """A single retrieval result with score and metadata."""

    content: str
    score: float
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def source(self) -> str:
        """Get the source document path."""
        return self.metadata.get("source", "unknown")

    @property
    def chunk_index(self) -> int:
        """Get the chunk index within the source document."""
        return self.metadata.get("chunk_index", 0)

    @property
    def animal_types(self) -> List[str]:
        """Get the animal types this chunk applies to."""
        animals = self.metadata.get("animal_types", "")
        if isinstance(animals, str):
            return [a.strip() for a in animals.split(",") if a.strip()]
        return animals if animals else []

    @property
    def urgency_level(self) -> Optional[str]:
        """Get the urgency level if available."""
        return self.metadata.get("urgency_level")

    @property
    def section_type(self) -> Optional[str]:
        """Get the section type if available."""
        return self.metadata.get("section_type")


@dataclass
class RetrievalResponse:
    """Response from the retriever containing results and metadata."""

    query: ProcessedQuery
    results: List[RetrievalResult]
    total_found: int
    filtered_count: int = 0

    @property
    def has_results(self) -> bool:
        """Check if any results were found."""
        return len(self.results) > 0

    @property
    def top_result(self) -> Optional[RetrievalResult]:
        """Get the top-scoring result."""
        return self.results[0] if self.results else None


class Retriever:
    """
    Retrieves relevant document chunks from the vector store.

    Combines query processing with similarity search and optional
    metadata filtering for accurate, context-aware retrieval.
    """

    def __init__(
        self,
        config: Optional[RetrieverConfig] = None,
        query_config: Optional[QueryConfig] = None,
        chroma_config: Optional[ChromaConfig] = None,
        chroma_manager: Optional[ChromaManager] = None,
    ):
        """
        Initialize the retriever.

        Args:
            config: Retriever configuration. Uses defaults if not provided.
            query_config: Query processing configuration.
            chroma_config: ChromaDB configuration for creating a new manager.
            chroma_manager: Existing ChromaManager instance to use.
        """
        self.config = config or RetrieverConfig()
        self.query_processor = QueryProcessor(query_config)

        # Use provided manager or create one
        if chroma_manager:
            self.chroma_manager = chroma_manager
        else:
            self.chroma_manager = ChromaManager(config=chroma_config)

        logger.info(
            f"Retriever initialized (k={self.config.default_k}, "
            f"filter_by_animal={self.config.filter_by_animal})"
        )

    def retrieve(
        self,
        query: str,
        k: Optional[int] = None,
        filter: Optional[Dict[str, Any]] = None,
        use_query_filter: bool = True,
    ) -> RetrievalResponse:
        """
        Retrieve relevant documents for a query.

        Args:
            query: User query string.
            k: Number of results to return. Uses default_k if not provided.
            filter: Optional explicit metadata filter (overrides auto-filter).
            use_query_filter: Whether to auto-generate filter from query analysis.

        Returns:
            RetrievalResponse with results and metadata.
        """
        if not query or not query.strip():
            logger.warning("Empty query provided")
            return RetrievalResponse(
                query=ProcessedQuery(
                    original_query=query,
                    cleaned_query="",
                    expanded_query=""
                ),
                results=[],
                total_found=0
            )

        # Determine k value
        k = k or self.config.default_k
        k = min(k, self.config.max_k)

        # Process the query
        processed_query = self.query_processor.process(query)

        # Build metadata filter
        metadata_filter = filter
        if metadata_filter is None and use_query_filter:
            metadata_filter = create_metadata_filter(
                processed_query,
                filter_by_animal=self.config.filter_by_animal
            )

        logger.debug(
            f"Searching with k={k}, filter={metadata_filter is not None}"
        )

        # Perform similarity search with scores
        try:
            raw_results = self.chroma_manager.similarity_search_with_score(
                query=processed_query.expanded_query,
                k=k,
                filter=metadata_filter
            )
        except Exception as e:
            logger.error(f"Search failed: {e}")
            raise RetrieverError(f"Failed to retrieve documents: {e}") from e

        # Convert to RetrievalResult objects
        results = []
        for doc, score in raw_results:
            # ChromaDB returns cosine distance (0 = identical, 2 = opposite)
            # Convert to normalized similarity score (1 = identical, 0 = opposite)
            similarity = max(0.0, (2.0 - score) / 2.0)

            if similarity >= self.config.min_similarity_score:
                results.append(RetrievalResult(
                    content=doc.page_content,
                    score=similarity,
                    metadata=dict(doc.metadata)
                ))

        filtered_count = len(raw_results) - len(results)

        logger.info(
            f"Retrieved {len(results)} results "
            f"(filtered {filtered_count} below threshold)"
        )

        return RetrievalResponse(
            query=processed_query,
            results=results,
            total_found=len(raw_results),
            filtered_count=filtered_count
        )

    def retrieve_for_animal(
        self,
        query: str,
        animal_type: str,
        k: Optional[int] = None,
    ) -> RetrievalResponse:
        """
        Retrieve documents filtered for a specific animal type.

        Args:
            query: User query string.
            animal_type: Either "dog" or "cat".
            k: Number of results to return.

        Returns:
            RetrievalResponse with filtered results.
        """
        if animal_type not in ("dog", "cat"):
            raise ValueError(f"animal_type must be 'dog' or 'cat', got: {animal_type}")

        # Create explicit animal filter
        animal_filter = {
            "$or": [
                {"animal_types": {"$eq": animal_type}},
                {"animal_types": {"$eq": "dog, cat"}},
                {"animal_types": {"$eq": "cat, dog"}},
            ]
        }

        return self.retrieve(
            query=query,
            k=k,
            filter=animal_filter,
            use_query_filter=False
        )

    def retrieve_emergency(
        self,
        query: str,
        k: Optional[int] = None,
    ) -> RetrievalResponse:
        """
        Retrieve documents with emergency/urgent content prioritized.

        Args:
            query: User query string.
            k: Number of results to return.

        Returns:
            RetrievalResponse with emergency-relevant results.
        """
        k = k or self.config.default_k
        # Fetch more results to allow for filtering
        expanded_k = min(k * 2, self.config.max_k)

        response = self.retrieve(query=query, k=expanded_k)

        # Sort results by urgency (high urgency first)
        def urgency_key(result: RetrievalResult) -> tuple:
            urgency = result.urgency_level
            urgency_order = {"high": 0, "medium": 1, "low": 2, None: 3}
            return (urgency_order.get(urgency, 3), -result.score)

        sorted_results = sorted(response.results, key=urgency_key)

        return RetrievalResponse(
            query=response.query,
            results=sorted_results[:k],
            total_found=response.total_found,
            filtered_count=len(sorted_results) - k if len(sorted_results) > k else 0
        )

    def get_stats(self) -> Dict[str, Any]:
        """
        Get retriever and collection statistics.

        Returns:
            Dictionary with statistics.
        """
        collection_stats = self.chroma_manager.get_collection_stats()
        return {
            "retriever_config": {
                "default_k": self.config.default_k,
                "max_k": self.config.max_k,
                "min_similarity_score": self.config.min_similarity_score,
                "filter_by_animal": self.config.filter_by_animal,
            },
            "collection": collection_stats
        }


class RetrieverError(Exception):
    """Base exception for retriever errors."""

    pass


class NoResultsError(RetrieverError):
    """Raised when no results are found for a query."""

    pass
