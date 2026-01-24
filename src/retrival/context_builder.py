"""
Context Builder Module

Formats retrieved chunks into context suitable for LLM input.
"""

import logging
from dataclasses import dataclass, field
from typing import Optional, List

from .config import ContextConfig
from .retriever import RetrievalResult, RetrievalResponse


logger = logging.getLogger(__name__)


@dataclass
class FormattedContext:
    """Formatted context ready for LLM input."""

    context_text: str
    num_chunks: int
    total_characters: int
    truncated: bool = False
    sources_used: List[str] = field(default_factory=list)

    @property
    def is_empty(self) -> bool:
        """Check if context is empty."""
        return not self.context_text or self.num_chunks == 0


class ContextBuilder:
    """
    Builds formatted context from retrieval results for LLM input.

    Handles:
    - Chunk formatting with metadata
    - Context length limiting
    - Source tracking
    - Chunk deduplication
    """

    def __init__(self, config: Optional[ContextConfig] = None):
        """
        Initialize the context builder.

        Args:
            config: Context configuration. Uses defaults if not provided.
        """
        self.config = config or ContextConfig()
        logger.debug(
            f"ContextBuilder initialized (max_length={self.config.max_context_length})"
        )

    def build(self, response: RetrievalResponse) -> FormattedContext:
        """
        Build formatted context from retrieval response.

        Args:
            response: Retrieval response containing results.

        Returns:
            FormattedContext ready for LLM input.
        """
        if not response.has_results:
            logger.warning("No results to build context from")
            return FormattedContext(
                context_text="",
                num_chunks=0,
                total_characters=0
            )

        return self.build_from_results(response.results)

    def build_from_results(
        self,
        results: List[RetrievalResult],
        max_chunks: Optional[int] = None
    ) -> FormattedContext:
        """
        Build formatted context from a list of results.

        Args:
            results: List of retrieval results.
            max_chunks: Override for maximum chunks to include.

        Returns:
            FormattedContext ready for LLM input.
        """
        max_chunks = max_chunks or self.config.max_chunks
        chunks_to_use = results[:max_chunks]

        formatted_chunks = []
        sources_used = set()
        total_length = 0
        truncated = False

        for i, result in enumerate(chunks_to_use):
            # Format the chunk
            formatted = self._format_chunk(result, i + 1)

            # Check if adding this chunk would exceed limit
            chunk_length = len(formatted)
            if total_length + chunk_length > self.config.max_context_length:
                if not formatted_chunks:
                    # At least include first chunk (truncated if needed)
                    formatted = formatted[:self.config.max_context_length]
                    truncated = True
                else:
                    truncated = True
                    break

            formatted_chunks.append(formatted)
            total_length += chunk_length
            sources_used.add(result.source)

        # Join chunks with separator
        context_text = self.config.chunk_separator.join(formatted_chunks)

        logger.info(
            f"Built context: {len(formatted_chunks)} chunks, "
            f"{len(context_text)} chars, truncated={truncated}"
        )

        return FormattedContext(
            context_text=context_text,
            num_chunks=len(formatted_chunks),
            total_characters=len(context_text),
            truncated=truncated,
            sources_used=sorted(sources_used)
        )

    def _format_chunk(self, result: RetrievalResult, index: int) -> str:
        """Format a single chunk with metadata."""
        parts = []

        # Header with metadata
        header_parts = [f"[{index}]"]

        if self.config.include_source:
            source = self._format_source(result.source)
            header_parts.append(f"Source: {source}")

        if self.config.include_page_number:
            page = result.metadata.get("page_number")
            if page is not None:
                header_parts.append(f"Page {page}")

        if self.config.include_section_type and result.section_type:
            header_parts.append(f"({result.section_type})")

        header = " | ".join(header_parts)
        parts.append(header)

        # Content
        content = result.content
        if self.config.truncate_long_chunks and len(content) > self.config.chunk_max_length:
            content = content[:self.config.chunk_max_length] + "..."

        parts.append(content)

        return "\n".join(parts)

    def _format_source(self, source: str) -> str:
        """Format source path for display."""
        # Remove common prefixes and clean up
        source = source.replace("\\", "/")

        # Extract filename if it's a path
        if "/" in source:
            parts = source.split("/")
            # Keep last two parts (folder/filename) if available
            if len(parts) >= 2:
                return "/".join(parts[-2:])
            return parts[-1]

        return source

    def build_minimal(self, results: List[RetrievalResult]) -> str:
        """
        Build minimal context with just content (no metadata).

        Args:
            results: List of retrieval results.

        Returns:
            Plain text context.
        """
        chunks = []
        total_length = 0

        for result in results[:self.config.max_chunks]:
            content = result.content

            if self.config.truncate_long_chunks:
                content = content[:self.config.chunk_max_length]

            if total_length + len(content) > self.config.max_context_length:
                break

            chunks.append(content)
            total_length += len(content)

        return "\n\n".join(chunks)


def build_context_with_instructions(
    context: FormattedContext,
    query: str,
    animal_type: Optional[str] = None,
    is_emergency: bool = False,
) -> str:
    """
    Build full context with retrieval instructions for the LLM.

    Args:
        context: The formatted context.
        query: Original user query.
        animal_type: Detected animal type.
        is_emergency: Whether the query is an emergency.

    Returns:
        Complete context string with instructions.
    """
    parts = []

    # Context header
    parts.append("=== RETRIEVED VETERINARY INFORMATION ===")
    parts.append("")

    if context.is_empty:
        parts.append("No relevant information found in the knowledge base.")
        parts.append("Please advise the user to consult a veterinarian.")
    else:
        # Add context metadata
        meta_parts = [f"{context.num_chunks} relevant excerpts"]
        if animal_type:
            meta_parts.append(f"for {animal_type}s")
        if is_emergency:
            meta_parts.append("(EMERGENCY QUERY)")

        parts.append(f"Found {', '.join(meta_parts)}:")
        parts.append("")

        # Add the actual context
        parts.append(context.context_text)

    parts.append("")
    parts.append("=== END OF RETRIEVED INFORMATION ===")

    return "\n".join(parts)
