"""
PawAid Document Ingestion Module

This module provides tools for loading, splitting, and preprocessing
veterinary documents for the RAG pipeline.

Components:
    - PDFLoader: Load PDF documents with metadata extraction
    - TextSplitter: Split documents into chunks for embedding
    - SemanticSplitter: Semantic-aware splitting for medical content
    - TextPreprocessor: Clean and normalize text
    - MetadataExtractor: Extract veterinary-specific metadata

Example:
    from src.ingestion import PDFLoader, SemanticSplitter, TextPreprocessor

    # Load documents
    loader = PDFLoader()
    documents = loader.load_directory()

    # Split into chunks
    splitter = SemanticSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(documents)

    # Preprocess
    preprocessor = TextPreprocessor()
    chunks = preprocessor.preprocess_chunks(chunks)
"""

from .loader import PDFLoader, Document
from .splitter import TextSplitter, SemanticSplitter, Chunk
from .preprocessor import (
    TextPreprocessor,
    MetadataExtractor,
    PreprocessingConfig
)

__all__ = [
    # Loader
    "PDFLoader",
    "Document",
    # Splitter
    "TextSplitter",
    "SemanticSplitter",
    "Chunk",
    # Preprocessor
    "TextPreprocessor",
    "MetadataExtractor",
    "PreprocessingConfig",
]
