"""
Citation Injection Module

Injects source citations from the retrieval system into prompts,
instructing the LLM on how to reference and attribute sources.
"""

import logging
from typing import Optional, List

from .config import CitationInjectionConfig


logger = logging.getLogger(__name__)


_CITATION_INSTRUCTION_NUMBERED = """SOURCE CITATION RULES:
- Each piece of reference material is labeled with a number like [1], [2], etc.
- When you use information from a source, cite it using its number: [1], [2], etc.
- Place citations at the end of the relevant sentence or paragraph.
- If multiple sources support the same point, list them together: [1][3].
- Only cite sources that you actually used in your response.
- Do NOT fabricate citations or reference numbers not provided."""

_CITATION_INSTRUCTION_INLINE = """SOURCE CITATION RULES:
- Reference materials include source filenames and page numbers.
- When you use information from a source, cite it inline with the filename.
- Format: (source: filename, p. X)
- Only cite sources that you actually used in your response.
- Do NOT fabricate source names or page numbers not provided."""

_CITATION_INSTRUCTION_FOOTNOTE = """SOURCE CITATION RULES:
- Each piece of reference material is labeled with a number like [1], [2], etc.
- When you use information from a source, add a footnote marker in superscript.
- At the end of your response, include a "References" section listing all cited sources.
- Only cite sources that you actually used in your response.
- Do NOT fabricate citations or reference numbers not provided."""

_CITATION_INSTRUCTIONS = {
    "numbered": _CITATION_INSTRUCTION_NUMBERED,
    "inline": _CITATION_INSTRUCTION_INLINE,
    "footnote": _CITATION_INSTRUCTION_FOOTNOTE,
}

_NO_SOURCES_NOTE = """NOTE: No reference material was found for this query. \
Do NOT cite any sources. Clearly state that you could not find relevant \
information in your knowledge base and recommend consulting a veterinarian."""

_SOURCE_ATTRIBUTION_REMINDER = """
IMPORTANT: Your response must be grounded in the provided reference material. \
Do not provide information that is not supported by the sources. If the \
reference material does not cover part of the question, explicitly say so \
rather than guessing."""


class CitationInjector:
    """
    Injects citation instructions and source references into prompts.

    Integrates with the retrieval module's output to:
    - Add citation format instructions to the system prompt
    - Format source references for context injection
    - Enforce source attribution in responses
    """

    def __init__(self, config: Optional[CitationInjectionConfig] = None):
        """
        Initialize the citation injector.

        Args:
            config: Citation injection configuration. Uses defaults if not provided.
        """
        self.config = config or CitationInjectionConfig()
        logger.debug(
            f"CitationInjector initialized (style={self.config.citation_style})"
        )

    def build_citation_instruction(self, has_sources: bool = True) -> str:
        """
        Build citation instructions for the system prompt.

        Args:
            has_sources: Whether source material is available.

        Returns:
            Citation instruction string for the LLM.
        """
        if not self.config.inject_citations:
            return ""

        if not has_sources:
            return _NO_SOURCES_NOTE

        parts = [_CITATION_INSTRUCTIONS[self.config.citation_style]]

        if self.config.require_source_attribution:
            parts.append(_SOURCE_ATTRIBUTION_REMINDER)

        instruction = "\n".join(parts)

        logger.debug(f"Built citation instruction: {len(instruction)} chars")
        return instruction

    def format_sources_for_prompt(
        self,
        sources: List[str],
        page_numbers: Optional[List[Optional[int]]] = None,
    ) -> str:
        """
        Format a list of source references for injection into the prompt.

        Args:
            sources: List of source file paths or names.
            page_numbers: Optional list of page numbers corresponding to sources.

        Returns:
            Formatted source reference string.
        """
        if not sources:
            return ""

        max_sources = min(len(sources), self.config.max_citations)
        lines = ["Available sources:"]

        for i in range(max_sources):
            source = sources[i]
            # Clean up the source path for display
            display_name = self._format_source_name(source)

            if page_numbers and i < len(page_numbers) and page_numbers[i] is not None:
                lines.append(f"  [{i + 1}] {display_name}, p. {page_numbers[i]}")
            else:
                lines.append(f"  [{i + 1}] {display_name}")

        if len(sources) > max_sources:
            lines.append(f"  ... and {len(sources) - max_sources} more sources")

        return "\n".join(lines)

    def build_citation_block_from_results(
        self,
        results: list,
    ) -> str:
        """
        Build a citation block from retrieval results.

        This extracts sources and page numbers from RetrievalResult objects
        (from src.retrival.retriever) and formats them for prompt injection.

        Args:
            results: List of RetrievalResult objects.

        Returns:
            Formatted citation block string.
        """
        if not results:
            return ""

        # Deduplicate sources while preserving order
        seen_sources = {}
        for result in results:
            source = result.source
            if source not in seen_sources:
                page = result.metadata.get("page_number")
                seen_sources[source] = page

        sources = list(seen_sources.keys())
        page_numbers = list(seen_sources.values())

        return self.format_sources_for_prompt(sources, page_numbers)

    def _format_source_name(self, source: str) -> str:
        """Format a source path into a clean display name."""
        source = source.replace("\\", "/")

        if "/" in source:
            parts = source.split("/")
            if len(parts) >= 2:
                return "/".join(parts[-2:])
            return parts[-1]

        return source


def build_citation_prompt(
    sources: List[str],
    style: str = "numbered",
) -> str:
    """
    Build citation instructions with source list.

    Args:
        sources: List of source references.
        style: Citation style ("numbered", "inline", or "footnote").

    Returns:
        Complete citation prompt section.
    """
    config = CitationInjectionConfig(citation_style=style)
    injector = CitationInjector(config)

    parts = []
    parts.append(injector.build_citation_instruction(has_sources=bool(sources)))

    if sources:
        parts.append("")
        parts.append(injector.format_sources_for_prompt(sources))

    return "\n".join(parts)
