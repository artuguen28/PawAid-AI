"""
Citation Module

Tracks and formats source citations for retrieval results.
"""

import logging
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from collections import defaultdict

from .config import CitationConfig
from .retriever import RetrievalResult, RetrievalResponse


logger = logging.getLogger(__name__)


@dataclass
class Citation:
    """A single source citation."""

    source: str
    page_numbers: List[int] = field(default_factory=list)
    chunk_indices: List[int] = field(default_factory=list)
    section_types: List[str] = field(default_factory=list)
    relevance_scores: List[float] = field(default_factory=list)

    @property
    def filename(self) -> str:
        """Get just the filename from the source path."""
        source = self.source.replace("\\", "/")
        if "/" in source:
            return source.split("/")[-1]
        return source

    @property
    def average_relevance(self) -> float:
        """Get the average relevance score."""
        if not self.relevance_scores:
            return 0.0
        return sum(self.relevance_scores) / len(self.relevance_scores)

    @property
    def top_relevance(self) -> float:
        """Get the highest relevance score."""
        return max(self.relevance_scores) if self.relevance_scores else 0.0

    @property
    def num_references(self) -> int:
        """Get the number of times this source was cited."""
        return len(self.chunk_indices)

    def format_short(self) -> str:
        """Format as a short citation string."""
        parts = [self.filename]

        if self.page_numbers:
            unique_pages = sorted(set(self.page_numbers))
            if len(unique_pages) == 1:
                parts.append(f"p. {unique_pages[0]}")
            else:
                parts.append(f"pp. {unique_pages[0]}-{unique_pages[-1]}")

        return ", ".join(parts)

    def format_full(self) -> str:
        """Format as a full citation with all details."""
        lines = [f"Source: {self.source}"]

        if self.page_numbers:
            unique_pages = sorted(set(self.page_numbers))
            lines.append(f"  Pages: {', '.join(map(str, unique_pages))}")

        if self.section_types:
            unique_sections = sorted(set(s for s in self.section_types if s))
            if unique_sections:
                lines.append(f"  Sections: {', '.join(unique_sections)}")

        lines.append(f"  References: {self.num_references}")
        lines.append(f"  Relevance: {self.top_relevance:.2f}")

        return "\n".join(lines)


@dataclass
class CitationList:
    """Collection of citations with formatting methods."""

    citations: List[Citation] = field(default_factory=list)

    @property
    def is_empty(self) -> bool:
        """Check if there are no citations."""
        return len(self.citations) == 0

    @property
    def num_sources(self) -> int:
        """Get the number of unique sources."""
        return len(self.citations)

    @property
    def total_references(self) -> int:
        """Get total number of references across all citations."""
        return sum(c.num_references for c in self.citations)

    def format_inline(self, max_citations: int = 3) -> str:
        """
        Format citations for inline display.

        Example: "[cat-poisons.pdf, p. 2; dog-toxins.pdf, p. 5]"
        """
        if self.is_empty:
            return ""

        formatted = [c.format_short() for c in self.citations[:max_citations]]
        result = "[" + "; ".join(formatted)

        if len(self.citations) > max_citations:
            result += f" +{len(self.citations) - max_citations} more"

        return result + "]"

    def format_list(self) -> str:
        """Format citations as a numbered list."""
        if self.is_empty:
            return "No sources cited."

        lines = ["Sources:"]
        for i, citation in enumerate(self.citations, 1):
            lines.append(f"  {i}. {citation.format_short()}")

        return "\n".join(lines)

    def format_detailed(self) -> str:
        """Format citations with full details."""
        if self.is_empty:
            return "No sources cited."

        lines = [f"=== CITATIONS ({self.num_sources} sources) ===", ""]

        for i, citation in enumerate(self.citations, 1):
            lines.append(f"[{i}] {citation.format_full()}")
            lines.append("")

        return "\n".join(lines)


class CitationManager:
    """
    Manages citation extraction and formatting from retrieval results.

    Handles:
    - Citation extraction from results
    - Deduplication and grouping
    - Multiple output formats
    """

    def __init__(self, config: Optional[CitationConfig] = None):
        """
        Initialize the citation manager.

        Args:
            config: Citation configuration. Uses defaults if not provided.
        """
        self.config = config or CitationConfig()
        logger.debug("CitationManager initialized")

    def extract_citations(
        self,
        response: RetrievalResponse
    ) -> CitationList:
        """
        Extract citations from a retrieval response.

        Args:
            response: Retrieval response containing results.

        Returns:
            CitationList with extracted citations.
        """
        return self.extract_from_results(response.results)

    def extract_from_results(
        self,
        results: List[RetrievalResult]
    ) -> CitationList:
        """
        Extract citations from a list of results.

        Args:
            results: List of retrieval results.

        Returns:
            CitationList with extracted citations.
        """
        if not results:
            return CitationList()

        if self.config.group_by_source:
            citations = self._group_by_source(results)
        else:
            citations = self._individual_citations(results)

        # Sort by relevance (top relevance descending)
        citations.sort(key=lambda c: c.top_relevance, reverse=True)

        logger.info(f"Extracted {len(citations)} citations from {len(results)} results")

        return CitationList(citations=citations)

    def _group_by_source(
        self,
        results: List[RetrievalResult]
    ) -> List[Citation]:
        """Group results by source document."""
        grouped: Dict[str, Citation] = {}

        for result in results:
            source = result.source

            if source not in grouped:
                grouped[source] = Citation(source=source)

            citation = grouped[source]

            # Add page number
            if self.config.include_page_numbers:
                page = result.metadata.get("page_number")
                if page is not None:
                    citation.page_numbers.append(page)

            # Add chunk index
            if self.config.include_chunk_index:
                citation.chunk_indices.append(result.chunk_index)
            else:
                # Still track for counting references
                citation.chunk_indices.append(result.chunk_index)

            # Add section type
            if self.config.include_section_type and result.section_type:
                citation.section_types.append(result.section_type)

            # Add relevance score
            citation.relevance_scores.append(result.score)

        return list(grouped.values())

    def _individual_citations(
        self,
        results: List[RetrievalResult]
    ) -> List[Citation]:
        """Create individual citations for each result."""
        citations = []
        seen = set()

        for result in results:
            # Create unique key for deduplication
            if self.config.deduplicate:
                key = (result.source, result.chunk_index)
                if key in seen:
                    continue
                seen.add(key)

            page_numbers = []
            if self.config.include_page_numbers:
                page = result.metadata.get("page_number")
                if page is not None:
                    page_numbers = [page]

            section_types = []
            if self.config.include_section_type and result.section_type:
                section_types = [result.section_type]

            chunk_indices = []
            if self.config.include_chunk_index:
                chunk_indices = [result.chunk_index]

            citations.append(Citation(
                source=result.source,
                page_numbers=page_numbers,
                chunk_indices=chunk_indices or [result.chunk_index],
                section_types=section_types,
                relevance_scores=[result.score]
            ))

        return citations

    def format_for_response(
        self,
        citation_list: CitationList,
        style: str = "list"
    ) -> str:
        """
        Format citations for inclusion in response.

        Args:
            citation_list: The citations to format.
            style: Format style - "inline", "list", or "detailed".

        Returns:
            Formatted citation string.
        """
        if style == "inline":
            return citation_list.format_inline()
        elif style == "detailed":
            return citation_list.format_detailed()
        else:
            return citation_list.format_list()


def create_citation_block(
    results: List[RetrievalResult],
    header: str = "References"
) -> str:
    """
    Create a formatted citation block from results.

    Args:
        results: List of retrieval results.
        header: Header text for the citation block.

    Returns:
        Formatted citation block string.
    """
    manager = CitationManager()
    citations = manager.extract_from_results(results)

    if citations.is_empty:
        return ""

    lines = [f"### {header}", ""]
    for i, citation in enumerate(citations.citations, 1):
        lines.append(f"{i}. {citation.format_short()}")

    return "\n".join(lines)
