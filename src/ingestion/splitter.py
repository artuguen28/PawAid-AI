"""
Text Splitter Module

Handles chunking of documents into smaller pieces for embedding.
Uses LangChain's text splitters with configurable overlap.
"""

from typing import List, Optional
from dataclasses import dataclass, field

from langchain_text_splitters import RecursiveCharacterTextSplitter

from .loader import Document


@dataclass
class Chunk:
    """Represents a chunk of text from a document."""

    content: str
    metadata: dict = field(default_factory=dict)

    @property
    def source(self) -> str:
        return self.metadata.get("source", "unknown")

    @property
    def chunk_index(self) -> int:
        return self.metadata.get("chunk_index", 0)

    def __len__(self) -> int:
        return len(self.content)


class TextSplitter:
    """
    Splits documents into chunks for embedding.

    Uses recursive character splitting which tries to split on natural
    boundaries (paragraphs, sentences) before falling back to character limits.
    """

    # Default separators optimized for veterinary/medical text
    DEFAULT_SEPARATORS = [
        "\n\n\n",    # Major section breaks
        "\n\n",      # Paragraph breaks
        "\n",        # Line breaks
        ". ",        # Sentence endings
        "! ",        # Exclamation endings
        "? ",        # Question endings
        "; ",        # Semicolon breaks
        ", ",        # Comma breaks
        " ",         # Word breaks
        ""           # Character level (last resort)
    ]

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        separators: Optional[List[str]] = None
    ):
        """
        Initialize the text splitter.

        Args:
            chunk_size: Maximum size of each chunk in characters.
            chunk_overlap: Number of overlapping characters between chunks.
            separators: Custom list of separators to use for splitting.
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or self.DEFAULT_SEPARATORS

        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=self.separators,
            length_function=len,
            is_separator_regex=False
        )

    def split_document(self, document: Document) -> List[Chunk]:
        """
        Split a single document into chunks.

        Args:
            document: Document object to split.

        Returns:
            List of Chunk objects.
        """
        text_chunks = self._splitter.split_text(document.content)

        chunks = []
        for i, text in enumerate(text_chunks):
            chunk_metadata = {
                **document.metadata,
                "chunk_index": i,
                "chunk_count": len(text_chunks),
                "chunk_size": len(text),
            }
            chunks.append(Chunk(content=text, metadata=chunk_metadata))

        return chunks

    def split_documents(self, documents: List[Document]) -> List[Chunk]:
        """
        Split multiple documents into chunks.

        Args:
            documents: List of Document objects to split.

        Returns:
            List of all Chunk objects from all documents.
        """
        all_chunks = []

        for doc in documents:
            chunks = self.split_document(doc)
            all_chunks.extend(chunks)

        return all_chunks

    def estimate_chunk_count(self, text: str) -> int:
        """
        Estimate how many chunks a text will produce.

        Args:
            text: Text to estimate.

        Returns:
            Estimated number of chunks.
        """
        if not text:
            return 0

        text_length = len(text)
        effective_chunk_size = self.chunk_size - self.chunk_overlap

        if effective_chunk_size <= 0:
            return 1

        return max(1, (text_length + effective_chunk_size - 1) // effective_chunk_size)


class SemanticSplitter(TextSplitter):
    """
    Enhanced splitter that tries to preserve semantic boundaries.

    Specifically tuned for veterinary/medical content with section awareness.
    """

    # Separators that preserve medical document structure
    MEDICAL_SEPARATORS = [
        "\n## ",      # Markdown H2 headers
        "\n# ",       # Markdown H1 headers
        "\n### ",     # Markdown H3 headers
        "\n\n\n",     # Major section breaks
        "\n\n",       # Paragraph breaks
        "\n• ",       # Bullet points
        "\n- ",       # Dash lists
        "\n* ",       # Asterisk lists
        "\n",         # Line breaks
        ". ",         # Sentence endings
        "! ",
        "? ",
        "; ",
        ", ",
        " ",
        ""
    ]

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200
    ):
        """
        Initialize the semantic splitter with medical-optimized separators.

        Args:
            chunk_size: Maximum size of each chunk in characters.
            chunk_overlap: Number of overlapping characters between chunks.
        """
        super().__init__(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=self.MEDICAL_SEPARATORS
        )
