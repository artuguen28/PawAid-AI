"""
PawAid Embeddings Module

This module provides tools for embedding veterinary document chunks
and storing them in a ChromaDB vector store for semantic retrieval.

Components:
    - EmbeddingService: OpenAI embeddings with retry logic
    - ChromaManager: ChromaDB collection management
    - BatchProcessor: Rate-limited batch embedding
    - IndexBuilder: High-level orchestration

Configuration:
    - EmbeddingConfig: Embedding model and retry settings
    - ChromaConfig: Vector store settings
    - IndexConfig: Combined configuration

Example:
    from src.embeddings import IndexBuilder, IndexConfig

    # Build index from chunks JSON
    builder = IndexBuilder()
    result = builder.build_from_json("data/chunks.json")
    print(f"Indexed {result.successful_documents} documents")

    # Test retrieval
    results = builder.test_retrieval("chocolate poisoning in dogs")
    for r in results:
        print(f"Score: {r['score']}, Source: {r['source']}")
"""

from .config import (
    EmbeddingConfig,
    ChromaConfig,
    IndexConfig,
)

from .converters import (
    chunk_to_langchain_document,
    chunks_to_langchain_documents,
    langchain_document_to_chunk,
    langchain_documents_to_chunks,
    generate_document_id,
    generate_document_ids,
    sanitize_metadata_for_chroma,
)

from .embedding_service import (
    EmbeddingService,
    EmbeddingError,
    EmbeddingRateLimitError,
    EmbeddingAPIError,
)

from .chroma_manager import ChromaManager

from .batch_processor import (
    BatchProcessor,
    BatchResult,
    ProcessingResult,
    create_progress_printer,
)

from .index_builder import (
    IndexBuilder,
    build_index_from_json,
)

__all__ = [
    # Configuration
    "EmbeddingConfig",
    "ChromaConfig",
    "IndexConfig",
    # Converters
    "chunk_to_langchain_document",
    "chunks_to_langchain_documents",
    "langchain_document_to_chunk",
    "langchain_documents_to_chunks",
    "generate_document_id",
    "generate_document_ids",
    "sanitize_metadata_for_chroma",
    # Embedding Service
    "EmbeddingService",
    "EmbeddingError",
    "EmbeddingRateLimitError",
    "EmbeddingAPIError",
    # ChromaDB Manager
    "ChromaManager",
    # Batch Processing
    "BatchProcessor",
    "BatchResult",
    "ProcessingResult",
    "create_progress_printer",
    # Index Builder
    "IndexBuilder",
    "build_index_from_json",
]
