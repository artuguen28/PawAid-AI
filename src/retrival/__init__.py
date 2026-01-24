"""
PawAid Retrieval Module

This module provides the RAG (Retrieval-Augmented Generation) retrieval
system for finding relevant veterinary information from the vector store.

Components:
    - QueryProcessor: Clean and expand user queries
    - Retriever: Similarity search with metadata filtering
    - Reranker: Optional relevance reranking
    - ContextBuilder: Format chunks for LLM input
    - CitationManager: Track and return source references

Configuration:
    - QueryConfig: Query processing settings
    - RetrieverConfig: Retrieval parameters
    - RerankerConfig: Reranking strategy settings
    - ContextConfig: Context formatting options
    - CitationConfig: Citation handling settings
    - RetrievalConfig: Combined configuration

Example:
    from src.retrival import Retriever, ContextBuilder, CitationManager

    # Initialize retriever
    retriever = Retriever()

    # Retrieve relevant documents
    response = retriever.retrieve("my dog ate chocolate")
    print(f"Found {len(response.results)} results")

    # Build context for LLM
    builder = ContextBuilder()
    context = builder.build(response)
    print(context.context_text)

    # Get citations
    citation_mgr = CitationManager()
    citations = citation_mgr.extract_citations(response)
    print(citations.format_list())
"""

from .config import (
    QueryConfig,
    RetrieverConfig,
    RerankerConfig,
    RerankerStrategy,
    ContextConfig,
    CitationConfig,
    RetrievalConfig,
)

from .query_processor import (
    QueryProcessor,
    ProcessedQuery,
    create_metadata_filter,
)

from .retriever import (
    Retriever,
    RetrievalResult,
    RetrievalResponse,
    RetrieverError,
    NoResultsError,
)

from .reranker import (
    Reranker,
    create_custom_reranker,
)

from .context_builder import (
    ContextBuilder,
    FormattedContext,
    build_context_with_instructions,
)

from .citation import (
    Citation,
    CitationList,
    CitationManager,
    create_citation_block,
)


__all__ = [
    # Configuration
    "QueryConfig",
    "RetrieverConfig",
    "RerankerConfig",
    "RerankerStrategy",
    "ContextConfig",
    "CitationConfig",
    "RetrievalConfig",
    # Query Processing
    "QueryProcessor",
    "ProcessedQuery",
    "create_metadata_filter",
    # Retriever
    "Retriever",
    "RetrievalResult",
    "RetrievalResponse",
    "RetrieverError",
    "NoResultsError",
    # Reranker
    "Reranker",
    "create_custom_reranker",
    # Context Builder
    "ContextBuilder",
    "FormattedContext",
    "build_context_with_instructions",
    # Citations
    "Citation",
    "CitationList",
    "CitationManager",
    "create_citation_block",
]
