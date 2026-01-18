"""
ChromaDB Manager Module

Manages ChromaDB collections for vector storage and retrieval.
"""

import logging
from pathlib import Path
from typing import List, Optional, Dict, Any

import chromadb
from chromadb.config import Settings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document as LCDocument

from .config import ChromaConfig
from .embedding_service import EmbeddingService


logger = logging.getLogger(__name__)


class ChromaManager:
    """
    Manager for ChromaDB vector store operations.

    Handles collection initialization, document storage, and similarity search.
    """

    def __init__(
        self,
        config: Optional[ChromaConfig] = None,
        embedding_service: Optional[EmbeddingService] = None
    ):
        """
        Initialize the ChromaDB manager.

        Args:
            config: ChromaDB configuration. Uses defaults if not provided.
            embedding_service: Embedding service instance. Creates one if not provided.
        """
        self.config = config or ChromaConfig()
        self.embedding_service = embedding_service or EmbeddingService()

        # Ensure persist directory exists
        self.config.ensure_directory()

        # Initialize ChromaDB client with persistence
        self._client = chromadb.PersistentClient(
            path=str(self.config.persist_path),
            settings=Settings(anonymized_telemetry=False)
        )

        # Initialize the vector store
        self._vectorstore: Optional[Chroma] = None

    @property
    def vectorstore(self) -> Chroma:
        """Get or create the vector store instance."""
        if self._vectorstore is None:
            self._vectorstore = Chroma(
                client=self._client,
                collection_name=self.config.collection_name,
                embedding_function=self.embedding_service._embeddings,
            )
        return self._vectorstore

    def add_documents(
        self,
        documents: List[LCDocument],
        ids: Optional[List[str]] = None
    ) -> List[str]:
        """
        Add documents to the vector store.

        Args:
            documents: List of LangChain documents to add.
            ids: Optional list of document IDs. Auto-generated if not provided.

        Returns:
            List of document IDs that were added.
        """
        if not documents:
            logger.warning("No documents to add")
            return []

        logger.info(f"Adding {len(documents)} documents to collection '{self.config.collection_name}'")

        added_ids = self.vectorstore.add_documents(documents, ids=ids)

        logger.info(f"Successfully added {len(added_ids)} documents")
        return added_ids

    def similarity_search(
        self,
        query: str,
        k: int = 5,
        filter: Optional[Dict[str, Any]] = None
    ) -> List[LCDocument]:
        """
        Perform similarity search on the vector store.

        Args:
            query: Search query text.
            k: Number of results to return.
            filter: Optional metadata filter.

        Returns:
            List of matching documents with their content and metadata.
        """
        logger.debug(f"Searching for: '{query[:50]}...' (k={k})")

        results = self.vectorstore.similarity_search(
            query=query,
            k=k,
            filter=filter
        )

        logger.debug(f"Found {len(results)} results")
        return results

    def similarity_search_with_score(
        self,
        query: str,
        k: int = 5,
        filter: Optional[Dict[str, Any]] = None
    ) -> List[tuple[LCDocument, float]]:
        """
        Perform similarity search and return scores.

        Args:
            query: Search query text.
            k: Number of results to return.
            filter: Optional metadata filter.

        Returns:
            List of (document, score) tuples.
        """
        logger.debug(f"Searching with scores for: '{query[:50]}...' (k={k})")

        results = self.vectorstore.similarity_search_with_score(
            query=query,
            k=k,
            filter=filter
        )

        logger.debug(f"Found {len(results)} results with scores")
        return results

    def get_collection_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the collection.

        Returns:
            Dictionary with collection information.
        """
        try:
            collection = self._client.get_collection(self.config.collection_name)
            document_count = collection.count()
        except Exception:
            document_count = 0

        return {
            "collection_name": self.config.collection_name,
            "persist_directory": str(self.config.persist_path),
            "document_count": document_count,
            "distance_metric": self.config.distance_metric,
            "embedding_model": self.embedding_service.model,
            "embedding_dimensions": self.embedding_service.dimensions,
        }

    def delete_collection(self) -> None:
        """
        Delete the entire collection.

        Warning: This permanently removes all documents in the collection.
        """
        logger.warning(f"Deleting collection '{self.config.collection_name}'")

        try:
            self._client.delete_collection(self.config.collection_name)
            self._vectorstore = None
            logger.info(f"Collection '{self.config.collection_name}' deleted")
        except (ValueError, Exception) as e:
            # Handle both ValueError and chromadb.errors.NotFoundError
            if "does not exist" in str(e).lower() or "not found" in str(e).lower():
                logger.info(f"Collection '{self.config.collection_name}' does not exist")
            else:
                raise

    def collection_exists(self) -> bool:
        """
        Check if the collection exists.

        Returns:
            True if collection exists, False otherwise.
        """
        try:
            collections = self._client.list_collections()
            return any(c.name == self.config.collection_name for c in collections)
        except Exception:
            return False

    def reset(self) -> None:
        """
        Reset the collection by deleting and recreating it.

        Useful for rebuilding the index from scratch.
        """
        logger.info(f"Resetting collection '{self.config.collection_name}'")

        self.delete_collection()

        # Force recreation on next access
        self._vectorstore = None

        logger.info("Collection reset complete")
