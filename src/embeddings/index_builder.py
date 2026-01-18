"""
Index Builder Module

High-level orchestrator for building and managing the vector index.
"""

import json
import logging
from pathlib import Path
from typing import List, Optional, Dict, Any

from langchain_core.documents import Document as LCDocument

from src.ingestion.splitter import Chunk

from .config import IndexConfig, EmbeddingConfig, ChromaConfig
from .converters import chunks_to_langchain_documents, generate_document_ids
from .embedding_service import EmbeddingService
from .chroma_manager import ChromaManager
from .batch_processor import BatchProcessor, ProcessingResult, create_progress_printer


logger = logging.getLogger(__name__)


class IndexBuilder:
    """
    High-level orchestrator for building the vector index.

    Provides methods to build from chunks, JSON files, or rebuild entirely.
    """

    def __init__(self, config: Optional[IndexConfig] = None):
        """
        Initialize the index builder.

        Args:
            config: Full index configuration. Uses defaults if not provided.
        """
        self.config = config or IndexConfig.default()

        # Initialize components
        self.embedding_service = EmbeddingService(self.config.embedding)
        self.chroma_manager = ChromaManager(
            config=self.config.chroma,
            embedding_service=self.embedding_service
        )
        self.batch_processor = BatchProcessor(
            chroma_manager=self.chroma_manager,
            config=self.config.embedding
        )

    def build_from_chunks(
        self,
        chunks: List[Chunk],
        progress_callback=None
    ) -> ProcessingResult:
        """
        Build the index from Chunk objects.

        Args:
            chunks: List of PawAid Chunk objects.
            progress_callback: Optional progress callback.

        Returns:
            ProcessingResult with build statistics.
        """
        if not chunks:
            logger.warning("No chunks provided to build index")
            return ProcessingResult(
                total_documents=0,
                successful_documents=0,
                failed_documents=0
            )

        logger.info(f"Building index from {len(chunks)} chunks")

        # Convert to LangChain documents
        documents = chunks_to_langchain_documents(chunks)

        # Generate deterministic IDs
        ids = generate_document_ids(chunks)

        # Process in batches
        result = self.batch_processor.process_documents(
            documents=documents,
            ids=ids,
            progress_callback=progress_callback
        )

        return result

    def build_from_json(
        self,
        json_path: str | Path,
        progress_callback=None
    ) -> ProcessingResult:
        """
        Build the index from a JSON file containing chunks.

        Args:
            json_path: Path to the JSON file.
            progress_callback: Optional progress callback.

        Returns:
            ProcessingResult with build statistics.

        Raises:
            FileNotFoundError: If the JSON file doesn't exist.
            ValueError: If the JSON format is invalid.
        """
        json_path = Path(json_path)

        if not json_path.exists():
            raise FileNotFoundError(f"Chunks file not found: {json_path}")

        logger.info(f"Loading chunks from {json_path}")

        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Convert JSON to Chunk objects
        chunks = []
        for item in data:
            if not isinstance(item, dict):
                raise ValueError("Invalid chunk format: expected dict")

            content = item.get("content", "")
            metadata = item.get("metadata", {})

            chunks.append(Chunk(content=content, metadata=metadata))

        logger.info(f"Loaded {len(chunks)} chunks from JSON")

        return self.build_from_chunks(chunks, progress_callback)

    def rebuild_index(
        self,
        chunks: Optional[List[Chunk]] = None,
        json_path: Optional[str | Path] = None,
        progress_callback=None
    ) -> ProcessingResult:
        """
        Clear the existing index and rebuild from scratch.

        Either chunks or json_path must be provided.

        Args:
            chunks: List of Chunk objects to index.
            json_path: Path to JSON file with chunks.
            progress_callback: Optional progress callback.

        Returns:
            ProcessingResult with build statistics.

        Raises:
            ValueError: If neither chunks nor json_path is provided.
        """
        if chunks is None and json_path is None:
            raise ValueError("Either chunks or json_path must be provided")

        logger.info("Rebuilding index (clearing existing data)")

        # Reset the collection
        self.chroma_manager.reset()

        # Build new index
        if chunks is not None:
            return self.build_from_chunks(chunks, progress_callback)
        else:
            return self.build_from_json(json_path, progress_callback)

    def test_retrieval(
        self,
        query: str,
        k: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Test retrieval with a sample query.

        Args:
            query: Search query.
            k: Number of results to return.

        Returns:
            List of result dictionaries with content and metadata.
        """
        logger.info(f"Testing retrieval with query: '{query}'")

        results = self.chroma_manager.similarity_search_with_score(query, k=k)

        formatted_results = []
        for doc, score in results:
            formatted_results.append({
                "content": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
                "score": round(score, 4),
                "source": doc.metadata.get("source", "unknown"),
                "filename": doc.metadata.get("filename", "unknown"),
                "animal_types": doc.metadata.get("animal_types", []),
                "urgency_level": doc.metadata.get("urgency_level"),
            })

        return formatted_results

    def get_stats(self) -> Dict[str, Any]:
        """
        Get combined statistics from all components.

        Returns:
            Dictionary with index statistics.
        """
        collection_stats = self.chroma_manager.get_collection_stats()
        embedding_stats = self.embedding_service.get_stats()

        return {
            **collection_stats,
            "embedding_service": embedding_stats,
        }


def build_index_from_json(
    json_path: str | Path,
    config: Optional[IndexConfig] = None,
    verbose: bool = False,
    rebuild: bool = False
) -> ProcessingResult:
    """
    Convenience function to build index from a JSON file.

    Args:
        json_path: Path to the chunks JSON file.
        config: Optional index configuration.
        verbose: Print progress updates.
        rebuild: Clear existing index before building.

    Returns:
        ProcessingResult with build statistics.
    """
    builder = IndexBuilder(config)

    progress_callback = create_progress_printer(verbose)

    if rebuild:
        return builder.rebuild_index(json_path=json_path, progress_callback=progress_callback)
    else:
        return builder.build_from_json(json_path, progress_callback)
