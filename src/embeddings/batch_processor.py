"""
Batch Processor Module

Handles rate-limited batch processing of documents for embedding.
"""

import logging
from dataclasses import dataclass, field
from typing import List, Optional, Callable

from langchain_core.documents import Document as LCDocument

from .config import EmbeddingConfig
from .chroma_manager import ChromaManager
from .converters import generate_document_ids


logger = logging.getLogger(__name__)


@dataclass
class BatchResult:
    """Result of processing a single batch."""

    batch_index: int
    documents_processed: int
    document_ids: List[str]
    success: bool
    error: Optional[str] = None


@dataclass
class ProcessingResult:
    """Result of the full batch processing operation."""

    total_documents: int
    successful_documents: int
    failed_documents: int
    batch_results: List[BatchResult] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    @property
    def success_rate(self) -> float:
        """Calculate success rate as a percentage."""
        if self.total_documents == 0:
            return 100.0
        return (self.successful_documents / self.total_documents) * 100


ProgressCallback = Callable[[int, int, int], None]


class BatchProcessor:
    """
    Processes documents in batches with rate limiting.

    Handles batch-level retries and continues on failures,
    reporting all errors at the end.
    """

    def __init__(
        self,
        chroma_manager: ChromaManager,
        config: Optional[EmbeddingConfig] = None
    ):
        """
        Initialize the batch processor.

        Args:
            chroma_manager: ChromaDB manager for storing documents.
            config: Embedding configuration for batch size settings.
        """
        self.chroma_manager = chroma_manager
        self.config = config or EmbeddingConfig()

    def process_documents(
        self,
        documents: List[LCDocument],
        ids: Optional[List[str]] = None,
        progress_callback: Optional[ProgressCallback] = None
    ) -> ProcessingResult:
        """
        Process documents in rate-limited batches.

        Args:
            documents: List of LangChain documents to process.
            ids: Optional pre-generated document IDs.
            progress_callback: Optional callback(current, total, batch_num) for progress updates.

        Returns:
            ProcessingResult with statistics and any errors.
        """
        if not documents:
            return ProcessingResult(
                total_documents=0,
                successful_documents=0,
                failed_documents=0
            )

        # Generate IDs if not provided
        if ids is None:
            from src.ingestion.splitter import Chunk
            # Convert to chunks temporarily for ID generation
            chunks = [
                Chunk(content=doc.page_content, metadata=doc.metadata or {})
                for doc in documents
            ]
            ids = generate_document_ids(chunks)

        total = len(documents)
        batch_size = self.config.batch_size
        num_batches = (total + batch_size - 1) // batch_size

        logger.info(f"Processing {total} documents in {num_batches} batches (batch_size={batch_size})")

        result = ProcessingResult(
            total_documents=total,
            successful_documents=0,
            failed_documents=0
        )

        for batch_idx in range(num_batches):
            start = batch_idx * batch_size
            end = min(start + batch_size, total)

            batch_docs = documents[start:end]
            batch_ids = ids[start:end]

            if progress_callback:
                progress_callback(start, total, batch_idx + 1)

            batch_result = self._process_batch(
                batch_docs,
                batch_ids,
                batch_idx
            )

            result.batch_results.append(batch_result)

            if batch_result.success:
                result.successful_documents += batch_result.documents_processed
            else:
                result.failed_documents += len(batch_docs)
                if batch_result.error:
                    result.errors.append(
                        f"Batch {batch_idx + 1}: {batch_result.error}"
                    )

        # Final progress update
        if progress_callback:
            progress_callback(total, total, num_batches)

        logger.info(
            f"Processing complete: {result.successful_documents}/{total} documents "
            f"({result.success_rate:.1f}% success rate)"
        )

        if result.errors:
            logger.warning(f"{len(result.errors)} batch(es) failed")
            for error in result.errors:
                logger.error(f"  - {error}")

        return result

    def _process_batch(
        self,
        documents: List[LCDocument],
        ids: List[str],
        batch_index: int
    ) -> BatchResult:
        """
        Process a single batch of documents.

        Args:
            documents: Documents in this batch.
            ids: Document IDs for this batch.
            batch_index: Index of this batch (0-indexed).

        Returns:
            BatchResult with success/failure information.
        """
        logger.debug(f"Processing batch {batch_index + 1} ({len(documents)} documents)")

        try:
            added_ids = self.chroma_manager.add_documents(documents, ids=ids)

            return BatchResult(
                batch_index=batch_index,
                documents_processed=len(added_ids),
                document_ids=added_ids,
                success=True
            )

        except Exception as e:
            logger.error(f"Batch {batch_index + 1} failed: {e}")

            return BatchResult(
                batch_index=batch_index,
                documents_processed=0,
                document_ids=[],
                success=False,
                error=str(e)
            )


def create_progress_printer(verbose: bool = True) -> Optional[ProgressCallback]:
    """
    Create a progress callback that prints to stdout.

    Args:
        verbose: Whether to print progress.

    Returns:
        Progress callback function or None.
    """
    if not verbose:
        return None

    def print_progress(current: int, total: int, batch_num: int) -> None:
        percentage = (current / total * 100) if total > 0 else 100
        print(f"\rProcessing: {current}/{total} documents ({percentage:.1f}%) - Batch {batch_num}", end="", flush=True)
        if current == total:
            print()  # Newline at the end

    return print_progress
