"""
Embedding Service Module

Wrapper around OpenAI embeddings with retry logic and error handling.
"""

import time
import random
import logging
from typing import List, Optional

from langchain_openai import OpenAIEmbeddings
from openai import RateLimitError, APIConnectionError, APIError

from .config import EmbeddingConfig


logger = logging.getLogger(__name__)


class EmbeddingError(Exception):
    """Base exception for embedding-related errors."""
    pass


class EmbeddingRateLimitError(EmbeddingError):
    """Raised when rate limit is exceeded after all retries."""
    pass


class EmbeddingAPIError(EmbeddingError):
    """Raised when API error occurs after all retries."""
    pass


class EmbeddingService:
    """
    Wrapper around OpenAI embeddings with retry and rate limiting.

    Provides exponential backoff for transient failures and
    handles rate limit headers for optimal retry timing.
    """

    def __init__(self, config: Optional[EmbeddingConfig] = None):
        """
        Initialize the embedding service.

        Args:
            config: Embedding configuration. Uses defaults if not provided.
        """
        self.config = config or EmbeddingConfig()
        self._embeddings = OpenAIEmbeddings(model=self.config.model)
        self._total_requests = 0
        self._total_retries = 0

    @property
    def model(self) -> str:
        """Get the embedding model name."""
        return self.config.model

    @property
    def dimensions(self) -> int:
        """Get the embedding dimensions for the model."""
        model_dimensions = {
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072,
            "text-embedding-ada-002": 1536,
        }
        return model_dimensions.get(self.config.model, 1536)

    def embed_query(self, text: str) -> List[float]:
        """
        Embed a single query text with retry logic.

        Args:
            text: Text to embed.

        Returns:
            List of floats representing the embedding.

        Raises:
            EmbeddingRateLimitError: If rate limit exceeded after retries.
            EmbeddingAPIError: If API error after retries.
        """
        return self._embed_with_retry(
            lambda: self._embeddings.embed_query(text)
        )

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Embed multiple documents with retry logic.

        Args:
            texts: List of texts to embed.

        Returns:
            List of embeddings (each a list of floats).

        Raises:
            EmbeddingRateLimitError: If rate limit exceeded after retries.
            EmbeddingAPIError: If API error after retries.
        """
        if not texts:
            return []

        return self._embed_with_retry(
            lambda: self._embeddings.embed_documents(texts)
        )

    def _embed_with_retry(self, operation):
        """
        Execute an embedding operation with exponential backoff retry.

        Args:
            operation: Callable that performs the embedding.

        Returns:
            Result of the operation.

        Raises:
            EmbeddingRateLimitError: If rate limit exceeded after retries.
            EmbeddingAPIError: If API error after retries.
        """
        last_exception = None

        for attempt in range(self.config.max_retries + 1):
            try:
                self._total_requests += 1
                result = operation()
                return result

            except RateLimitError as e:
                last_exception = e
                self._total_retries += 1

                if attempt >= self.config.max_retries:
                    logger.error(
                        f"Rate limit exceeded after {self.config.max_retries} retries"
                    )
                    raise EmbeddingRateLimitError(
                        f"Rate limit exceeded after {self.config.max_retries} retries: {e}"
                    ) from e

                wait_time = self._calculate_wait_time(e, attempt)
                logger.warning(
                    f"Rate limit hit, waiting {wait_time:.1f}s (attempt {attempt + 1}/{self.config.max_retries + 1})"
                )
                time.sleep(wait_time)

            except APIConnectionError as e:
                last_exception = e
                self._total_retries += 1

                if attempt >= self.config.max_retries:
                    logger.error(
                        f"API connection error after {self.config.max_retries} retries"
                    )
                    raise EmbeddingAPIError(
                        f"API connection error after {self.config.max_retries} retries: {e}"
                    ) from e

                wait_time = self._calculate_backoff(attempt)
                logger.warning(
                    f"Connection error, retrying in {wait_time:.1f}s (attempt {attempt + 1}/{self.config.max_retries + 1})"
                )
                time.sleep(wait_time)

            except APIError as e:
                last_exception = e
                self._total_retries += 1

                if attempt >= self.config.max_retries:
                    logger.error(
                        f"API error after {self.config.max_retries} retries"
                    )
                    raise EmbeddingAPIError(
                        f"API error after {self.config.max_retries} retries: {e}"
                    ) from e

                wait_time = self._calculate_backoff(attempt)
                logger.warning(
                    f"API error, retrying in {wait_time:.1f}s (attempt {attempt + 1}/{self.config.max_retries + 1})"
                )
                time.sleep(wait_time)

        # Should not reach here, but just in case
        raise EmbeddingAPIError(f"Unexpected error: {last_exception}")

    def _calculate_wait_time(self, error: RateLimitError, attempt: int) -> float:
        """
        Calculate wait time, preferring retry-after header if available.

        Args:
            error: The rate limit error.
            attempt: Current attempt number (0-indexed).

        Returns:
            Number of seconds to wait.
        """
        # Try to extract retry-after from error response
        retry_after = None
        if hasattr(error, 'response') and error.response is not None:
            retry_after = error.response.headers.get('retry-after')
            if retry_after:
                try:
                    return min(float(retry_after), self.config.max_delay)
                except (ValueError, TypeError):
                    pass

        # Fall back to exponential backoff
        return self._calculate_backoff(attempt)

    def _calculate_backoff(self, attempt: int) -> float:
        """
        Calculate exponential backoff with jitter.

        Args:
            attempt: Current attempt number (0-indexed).

        Returns:
            Number of seconds to wait.
        """
        # Exponential backoff: base_delay * 2^attempt
        delay = self.config.base_delay * (2 ** attempt)

        # Add jitter (up to 25% of delay)
        jitter = delay * 0.25 * random.random()
        delay += jitter

        # Cap at max_delay
        return min(delay, self.config.max_delay)

    def get_stats(self) -> dict:
        """
        Get service statistics.

        Returns:
            Dictionary with request and retry counts.
        """
        return {
            "model": self.config.model,
            "dimensions": self.dimensions,
            "total_requests": self._total_requests,
            "total_retries": self._total_retries,
        }
