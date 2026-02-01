"""
Embeddings Configuration Module

Configuration dataclasses for embedding and vector store settings.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class EmbeddingConfig:
    """Configuration for the embedding service."""

    model: str = "text-embedding-3-small"
    batch_size: int = 100
    max_retries: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    requests_per_minute: int = 3000
    tokens_per_minute: int = 1000000

    def __post_init__(self):
        if self.batch_size < 1:
            raise ValueError("batch_size must be at least 1")
        if self.max_retries < 0:
            raise ValueError("max_retries cannot be negative")
        if self.base_delay <= 0:
            raise ValueError("base_delay must be positive")


@dataclass
class ChromaConfig:
    """Configuration for ChromaDB vector store."""

    persist_directory: str = "vector_store"
    collection_name: str = "pawaid_veterinary"
    distance_metric: str = "cosine"

    def __post_init__(self):
        if self.distance_metric not in ("cosine", "l2", "ip"):
            raise ValueError(
                f"distance_metric must be one of: cosine, l2, ip. "
                f"Got: {self.distance_metric}"
            )

    @property
    def persist_path(self) -> Path:
        """Get the persist directory as a Path object."""
        return Path(self.persist_directory)

    def ensure_directory(self) -> None:
        """Create the persist directory if it doesn't exist."""
        self.persist_path.mkdir(parents=True, exist_ok=True)


@dataclass
class IndexConfig:
    """Combined configuration for the full indexing pipeline."""

    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    chroma: ChromaConfig = field(default_factory=ChromaConfig)
    verbose: bool = False

    @classmethod
    def default(cls) -> "IndexConfig":
        """Create a default configuration."""
        return cls()

    @classmethod
    def for_testing(cls) -> "IndexConfig":
        """Create a configuration suitable for testing."""
        return cls(
            embedding=EmbeddingConfig(batch_size=10, max_retries=1),
            chroma=ChromaConfig(
                persist_directory="vector_store_test",
                collection_name="pawaid_test"
            ),
            verbose=True
        )
