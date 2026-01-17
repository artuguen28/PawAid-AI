"""
Preprocessor Module

Handles text cleaning, normalization, and metadata extraction.
Prepares documents for embedding by removing noise and standardizing format.
"""

import re
from typing import List, Optional, Dict, Any
from dataclasses import dataclass

from .loader import Document
from .splitter import Chunk


@dataclass
class PreprocessingConfig:
    """Configuration for text preprocessing."""

    # Text cleaning options
    remove_extra_whitespace: bool = True
    normalize_unicode: bool = True
    remove_page_markers: bool = False  # Keep page markers for citation
    lowercase: bool = False  # Keep original case for medical terms

    # Content filtering
    min_chunk_length: int = 50  # Minimum characters to keep a chunk
    remove_empty_lines: bool = True

    # Medical-specific options
    preserve_medical_terms: bool = True
    preserve_dosage_info: bool = True


class TextPreprocessor:
    """
    Preprocesses text content for optimal embedding quality.

    Handles cleaning, normalization, and filtering of document content
    while preserving important medical/veterinary terminology.
    """

    # Patterns to preserve (medical terms, dosages, etc.)
    PRESERVE_PATTERNS = [
        r'\d+\s*(mg|kg|ml|cc|g|mcg|IU|mEq)',  # Dosages
        r'\d+\.?\d*\s*%',                       # Percentages
        r'\d+\s*-\s*\d+',                       # Ranges
        r'[A-Z]{2,}',                           # Acronyms
    ]

    # Patterns to clean
    NOISE_PATTERNS = [
        (r'\s+', ' '),                          # Multiple spaces to single
        (r'\n{3,}', '\n\n'),                    # Multiple newlines to double
        (r'[\x00-\x08\x0b\x0c\x0e-\x1f]', ''),  # Control characters
        (r'[^\S\n]+', ' '),                     # Normalize whitespace (keep newlines)
    ]

    def __init__(self, config: Optional[PreprocessingConfig] = None):
        """
        Initialize the preprocessor.

        Args:
            config: Preprocessing configuration. Uses defaults if not provided.
        """
        self.config = config or PreprocessingConfig()

    def clean_text(self, text: str) -> str:
        """
        Clean and normalize text content.

        Args:
            text: Raw text to clean.

        Returns:
            Cleaned text.
        """
        if not text:
            return ""

        # Apply noise cleaning patterns
        cleaned = text
        for pattern, replacement in self.NOISE_PATTERNS:
            cleaned = re.sub(pattern, replacement, cleaned)

        # Normalize unicode if configured
        if self.config.normalize_unicode:
            import unicodedata
            cleaned = unicodedata.normalize('NFKC', cleaned)

        # Remove extra whitespace
        if self.config.remove_extra_whitespace:
            cleaned = re.sub(r'[ \t]+', ' ', cleaned)
            cleaned = re.sub(r'\n[ \t]+', '\n', cleaned)
            cleaned = re.sub(r'[ \t]+\n', '\n', cleaned)

        # Remove empty lines if configured
        if self.config.remove_empty_lines:
            lines = cleaned.split('\n')
            lines = [line for line in lines if line.strip()]
            cleaned = '\n'.join(lines)

        # Lowercase if configured (usually not for medical text)
        if self.config.lowercase:
            cleaned = cleaned.lower()

        return cleaned.strip()

    def preprocess_document(self, document: Document) -> Document:
        """
        Preprocess a single document.

        Args:
            document: Document to preprocess.

        Returns:
            Preprocessed Document with cleaned content.
        """
        cleaned_content = self.clean_text(document.content)

        # Update metadata with preprocessing info
        updated_metadata = {
            **document.metadata,
            "preprocessed": True,
            "original_length": len(document.content),
            "cleaned_length": len(cleaned_content),
        }

        return Document(content=cleaned_content, metadata=updated_metadata)

    def preprocess_chunk(self, chunk: Chunk) -> Optional[Chunk]:
        """
        Preprocess a single chunk.

        Args:
            chunk: Chunk to preprocess.

        Returns:
            Preprocessed Chunk, or None if chunk should be filtered out.
        """
        cleaned_content = self.clean_text(chunk.content)

        # Filter out chunks that are too short
        if len(cleaned_content) < self.config.min_chunk_length:
            return None

        # Update metadata
        updated_metadata = {
            **chunk.metadata,
            "preprocessed": True,
            "original_chunk_size": len(chunk.content),
            "cleaned_chunk_size": len(cleaned_content),
        }

        return Chunk(content=cleaned_content, metadata=updated_metadata)

    def preprocess_chunks(self, chunks: List[Chunk]) -> List[Chunk]:
        """
        Preprocess multiple chunks, filtering out invalid ones.

        Args:
            chunks: List of chunks to preprocess.

        Returns:
            List of preprocessed chunks (filtered).
        """
        processed = []

        for chunk in chunks:
            result = self.preprocess_chunk(chunk)
            if result is not None:
                processed.append(result)

        return processed


class MetadataExtractor:
    """
    Extracts and enriches metadata from documents and chunks.

    Identifies sections, topics, and other structural elements
    that can improve retrieval quality.
    """

    # Common section headers in veterinary documents
    SECTION_PATTERNS = [
        (r'(?i)symptoms?\s*[:\-]?', 'symptoms'),
        (r'(?i)treatment\s*[:\-]?', 'treatment'),
        (r'(?i)diagnosis\s*[:\-]?', 'diagnosis'),
        (r'(?i)prevention\s*[:\-]?', 'prevention'),
        (r'(?i)first\s*aid\s*[:\-]?', 'first_aid'),
        (r'(?i)emergency\s*[:\-]?', 'emergency'),
        (r'(?i)warning\s*[:\-]?', 'warning'),
        (r'(?i)caution\s*[:\-]?', 'caution'),
        (r'(?i)dosage\s*[:\-]?', 'dosage'),
        (r'(?i)contraindications?\s*[:\-]?', 'contraindications'),
    ]

    # Animal type patterns
    ANIMAL_PATTERNS = [
        (r'(?i)\b(dog|canine|puppy|puppies)\b', 'dog'),
        (r'(?i)\b(cat|feline|kitten|kittens)\b', 'cat'),
    ]

    # Urgency indicators
    URGENCY_PATTERNS = [
        (r'(?i)emergency|immediate|urgent|critical|life.?threatening', 'high'),
        (r'(?i)should see a vet|veterinary attention|contact.{0,20}vet', 'medium'),
        (r'(?i)monitor|watch for|keep an eye', 'low'),
    ]

    def extract_section_type(self, text: str) -> Optional[str]:
        """
        Identify the section type from text content.

        Args:
            text: Text to analyze.

        Returns:
            Section type identifier or None.
        """
        for pattern, section_type in self.SECTION_PATTERNS:
            if re.search(pattern, text[:200]):  # Check beginning of text
                return section_type
        return None

    def extract_animal_types(self, text: str) -> List[str]:
        """
        Identify which animals the text applies to.

        Args:
            text: Text to analyze.

        Returns:
            List of animal types mentioned.
        """
        animals = set()
        for pattern, animal_type in self.ANIMAL_PATTERNS:
            if re.search(pattern, text):
                animals.add(animal_type)
        return list(animals)

    def extract_urgency_level(self, text: str) -> Optional[str]:
        """
        Determine urgency level from text content.

        Args:
            text: Text to analyze.

        Returns:
            Urgency level (high, medium, low) or None.
        """
        for pattern, urgency in self.URGENCY_PATTERNS:
            if re.search(pattern, text):
                return urgency
        return None

    def enrich_chunk_metadata(self, chunk: Chunk) -> Chunk:
        """
        Enrich a chunk's metadata with extracted information.

        Args:
            chunk: Chunk to enrich.

        Returns:
            Chunk with enriched metadata.
        """
        text = chunk.content

        enriched_metadata = {
            **chunk.metadata,
            "section_type": self.extract_section_type(text),
            "animal_types": self.extract_animal_types(text),
            "urgency_level": self.extract_urgency_level(text),
            "word_count": len(text.split()),
            "has_dosage_info": bool(re.search(r'\d+\s*(mg|kg|ml)', text, re.I)),
        }

        return Chunk(content=chunk.content, metadata=enriched_metadata)

    def enrich_chunks(self, chunks: List[Chunk]) -> List[Chunk]:
        """
        Enrich multiple chunks with metadata.

        Args:
            chunks: List of chunks to enrich.

        Returns:
            List of enriched chunks.
        """
        return [self.enrich_chunk_metadata(chunk) for chunk in chunks]
