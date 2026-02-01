"""
Document Loader Module

Handles loading PDF documents from the data directory using PyPDF2.
Supports single file and batch loading with metadata extraction.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional
from dataclasses import dataclass, field
from datetime import datetime

from PyPDF2 import PdfReader


@dataclass
class Document:
    """Represents a loaded document with content and metadata."""

    content: str
    metadata: dict = field(default_factory=dict)

    @property
    def page_count(self) -> int:
        return self.metadata.get("page_count", 0)

    @property
    def source(self) -> str:
        return self.metadata.get("source", "unknown")


class PDFLoader:
    """Loads PDF documents and extracts text content with metadata."""

    def __init__(self, data_dir: Optional[str] = None):
        """
        Initialize the PDF loader.

        Args:
            data_dir: Directory containing PDF files. Defaults to project's data/ directory.
        """
        if data_dir is None:
            project_root = Path(__file__).parent.parent.parent
            data_dir = project_root / "data"

        self.data_dir = Path(data_dir)

    def load_file(self, file_path: str | Path) -> Document:
        """
        Load a single PDF file.

        Args:
            file_path: Path to the PDF file (absolute or relative to data root).

        Returns:
            Document object with extracted text and metadata.

        Raises:
            FileNotFoundError: If the file doesn't exist.
            ValueError: If the file is not a PDF.
        """
        file_path = Path(file_path).resolve()

        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        if file_path.suffix.lower() != ".pdf":
            raise ValueError(f"Expected PDF file, got: {file_path.suffix}")

        reader = PdfReader(file_path)

        # Extract text from all pages
        pages_text = []
        for page_num, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            pages_text.append({
                "page_num": page_num,
                "text": text
            })

        # Combine all pages into single content
        full_text = "\n\n".join(
            f"[Page {p['page_num']}]\n{p['text']}"
            for p in pages_text
        )

        # Compute path relative to data root (for portability)
        try:
            relative_source = str(file_path.relative_to(self.data_dir))
        except ValueError:
            # Fallback if file is outside data root
            relative_source = file_path.name

        pdf_metadata = reader.metadata or {}

        metadata = {
            "source": relative_source,
            "filename": file_path.name,
            "page_count": len(reader.pages),
            "title": pdf_metadata.get("/Title", file_path.stem),
            "author": pdf_metadata.get("/Author", "Unknown"),
            "creation_date": pdf_metadata.get("/CreationDate", ""),
            "loaded_at": datetime.now().isoformat(),
            "file_size_bytes": file_path.stat().st_size,
        }

        return Document(content=full_text, metadata=metadata)

    def load_directory(
        self,
        directory: Optional[str | Path] = None,
        recursive: bool = False
    ) -> List[Document]:
        """
        Load all PDF files from a directory.

        Args:
            directory: Directory to load from. Defaults to data_dir.
            recursive: If True, search subdirectories as well.

        Returns:
            List of Document objects.
        """
        directory = Path(directory) if directory else self.data_dir

        if not directory.exists():
            raise FileNotFoundError(f"Directory not found: {directory}")

        pattern = "**/*.pdf" if recursive else "*.pdf"
        pdf_files = list(directory.glob(pattern))

        documents = []
        for pdf_file in sorted(pdf_files):
            try:
                doc = self.load_file(pdf_file)
                documents.append(doc)
            except Exception as e:
                print(f"Warning: Failed to load {pdf_file}: {e}")
                continue

        return documents

    def list_available_files(self, recursive: bool = False) -> List[Path]:
        """
        List all available PDF files in the data directory.

        Args:
            recursive: If True, search subdirectories as well.

        Returns:
            List of PDF file paths.
        """
        if not self.data_dir.exists():
            return []

        pattern = "**/*.pdf" if recursive else "*.pdf"
        return sorted(self.data_dir.glob(pattern))
