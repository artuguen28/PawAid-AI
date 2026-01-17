#!/usr/bin/env python3
"""
Document Ingestion Script

CLI tool for ingesting veterinary documents into the PawAid knowledge base.
Processes PDFs, splits into chunks, and prepares for embedding.

Usage:
    python scripts/ingest_documents.py [OPTIONS]

Examples:
    # Ingest all PDFs from data/ directory
    python scripts/ingest_documents.py

    # Ingest specific file
    python scripts/ingest_documents.py --file data/vet_guide.pdf

    # Ingest with custom chunk size
    python scripts/ingest_documents.py --chunk-size 500 --chunk-overlap 100

    # Dry run to see what would be processed
    python scripts/ingest_documents.py --dry-run
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.ingestion.loader import PDFLoader, Document
from src.ingestion.splitter import TextSplitter, SemanticSplitter, Chunk
from src.ingestion.preprocessor import (
    TextPreprocessor,
    MetadataExtractor,
    PreprocessingConfig
)


class IngestionPipeline:
    """
    Complete document ingestion pipeline.

    Orchestrates loading, splitting, preprocessing, and metadata enrichment.
    """

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        use_semantic_splitter: bool = True,
        min_chunk_length: int = 50,
        data_dir: Optional[str] = None
    ):
        """
        Initialize the ingestion pipeline.

        Args:
            chunk_size: Maximum chunk size in characters.
            chunk_overlap: Overlap between chunks.
            use_semantic_splitter: Use semantic-aware splitting.
            min_chunk_length: Minimum chunk length to keep.
            data_dir: Directory containing source documents.
        """
        self.loader = PDFLoader(data_dir=data_dir)

        if use_semantic_splitter:
            self.splitter = SemanticSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )
        else:
            self.splitter = TextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )

        self.preprocessor = TextPreprocessor(
            config=PreprocessingConfig(min_chunk_length=min_chunk_length)
        )
        self.metadata_extractor = MetadataExtractor()

    def process_file(self, file_path: str | Path) -> List[Chunk]:
        """
        Process a single file through the pipeline.

        Args:
            file_path: Path to the PDF file.

        Returns:
            List of processed chunks.
        """
        # Load document
        document = self.loader.load_file(file_path)

        # Preprocess document
        document = self.preprocessor.preprocess_document(document)

        # Split into chunks
        chunks = self.splitter.split_document(document)

        # Preprocess chunks (filtering)
        chunks = self.preprocessor.preprocess_chunks(chunks)

        # Enrich metadata
        chunks = self.metadata_extractor.enrich_chunks(chunks)

        return chunks

    def process_directory(
        self,
        directory: Optional[str | Path] = None,
        recursive: bool = False
    ) -> List[Chunk]:
        """
        Process all PDFs in a directory.

        Args:
            directory: Directory to process. Defaults to data_dir.
            recursive: Search subdirectories.

        Returns:
            List of all processed chunks.
        """
        # Load all documents
        documents = self.loader.load_directory(
            directory=directory,
            recursive=recursive
        )

        all_chunks = []
        for doc in documents:
            # Preprocess document
            doc = self.preprocessor.preprocess_document(doc)

            # Split into chunks
            chunks = self.splitter.split_document(doc)

            # Preprocess chunks
            chunks = self.preprocessor.preprocess_chunks(chunks)

            # Enrich metadata
            chunks = self.metadata_extractor.enrich_chunks(chunks)

            all_chunks.extend(chunks)

        return all_chunks

    def get_statistics(self, chunks: List[Chunk]) -> dict:
        """
        Get statistics about processed chunks.

        Args:
            chunks: List of processed chunks.

        Returns:
            Dictionary with statistics.
        """
        if not chunks:
            return {"total_chunks": 0}

        sources = set(c.metadata.get("source", "") for c in chunks)
        total_chars = sum(len(c.content) for c in chunks)
        avg_chunk_size = total_chars / len(chunks) if chunks else 0

        urgency_counts = {"high": 0, "medium": 0, "low": 0, "none": 0}
        animal_counts = {"dog": 0, "cat": 0, "both": 0, "unspecified": 0}

        for chunk in chunks:
            urgency = chunk.metadata.get("urgency_level")
            urgency_counts[urgency or "none"] += 1

            animals = chunk.metadata.get("animal_types", [])
            if "dog" in animals and "cat" in animals:
                animal_counts["both"] += 1
            elif "dog" in animals:
                animal_counts["dog"] += 1
            elif "cat" in animals:
                animal_counts["cat"] += 1
            else:
                animal_counts["unspecified"] += 1

        return {
            "total_chunks": len(chunks),
            "total_documents": len(sources),
            "total_characters": total_chars,
            "average_chunk_size": round(avg_chunk_size, 1),
            "urgency_distribution": urgency_counts,
            "animal_distribution": animal_counts,
        }


def save_chunks_to_json(chunks: List[Chunk], output_path: Path) -> None:
    """Save processed chunks to a JSON file."""
    data = [
        {
            "content": chunk.content,
            "metadata": chunk.metadata
        }
        for chunk in chunks
    ]

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def main():
    parser = argparse.ArgumentParser(
        description="Ingest veterinary documents into PawAid knowledge base",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument(
        "--file", "-f",
        type=str,
        help="Process a single PDF file"
    )

    parser.add_argument(
        "--data-dir", "-d",
        type=str,
        default=None,
        help="Directory containing PDF files (default: data/)"
    )

    parser.add_argument(
        "--recursive", "-r",
        action="store_true",
        help="Search subdirectories for PDFs"
    )

    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1000,
        help="Maximum chunk size in characters (default: 1000)"
    )

    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=200,
        help="Overlap between chunks (default: 200)"
    )

    parser.add_argument(
        "--min-chunk-length",
        type=int,
        default=50,
        help="Minimum chunk length to keep (default: 50)"
    )

    parser.add_argument(
        "--no-semantic",
        action="store_true",
        help="Disable semantic-aware splitting"
    )

    parser.add_argument(
        "--output", "-o",
        type=str,
        help="Output JSON file for processed chunks"
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be processed without doing it"
    )

    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output"
    )

    args = parser.parse_args()

    # Initialize pipeline
    pipeline = IngestionPipeline(
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        use_semantic_splitter=not args.no_semantic,
        min_chunk_length=args.min_chunk_length,
        data_dir=args.data_dir
    )

    # Dry run - just list files
    if args.dry_run:
        print("Dry run - files that would be processed:")
        if args.file:
            print(f"  - {args.file}")
        else:
            files = pipeline.loader.list_available_files(recursive=args.recursive)
            if not files:
                print("  No PDF files found in data directory")
            for f in files:
                print(f"  - {f}")
        return

    # Process files
    try:
        if args.file:
            print(f"Processing file: {args.file}")
            chunks = pipeline.process_file(args.file)
        else:
            print(f"Processing directory: {pipeline.loader.data_dir}")
            files = pipeline.loader.list_available_files(recursive=args.recursive)

            if not files:
                print("No PDF files found in data directory.")
                print("Add veterinary PDF documents to the data/ directory and try again.")
                return

            print(f"Found {len(files)} PDF file(s)")
            chunks = pipeline.process_directory(recursive=args.recursive)

    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error processing documents: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)

    # Print statistics
    stats = pipeline.get_statistics(chunks)
    print("\n" + "=" * 50)
    print("Ingestion Complete!")
    print("=" * 50)
    print(f"Documents processed: {stats['total_documents']}")
    print(f"Chunks created: {stats['total_chunks']}")
    print(f"Total characters: {stats['total_characters']:,}")
    print(f"Average chunk size: {stats['average_chunk_size']} chars")
    print(f"\nUrgency distribution:")
    for level, count in stats['urgency_distribution'].items():
        print(f"  {level}: {count}")
    print(f"\nAnimal distribution:")
    for animal, count in stats['animal_distribution'].items():
        print(f"  {animal}: {count}")

    # Save output if requested
    if args.output:
        output_path = Path(args.output)
        save_chunks_to_json(chunks, output_path)
        print(f"\nChunks saved to: {output_path}")

    # Verbose: show sample chunks
    if args.verbose and chunks:
        print("\n" + "=" * 50)
        print("Sample chunks (first 3):")
        print("=" * 50)
        for i, chunk in enumerate(chunks[:3]):
            print(f"\n--- Chunk {i + 1} ---")
            print(f"Source: {chunk.metadata.get('filename', 'unknown')}")
            print(f"Section: {chunk.metadata.get('section_type', 'unknown')}")
            print(f"Animals: {chunk.metadata.get('animal_types', [])}")
            print(f"Urgency: {chunk.metadata.get('urgency_level', 'unknown')}")
            print(f"Content preview: {chunk.content[:200]}...")


if __name__ == "__main__":
    main()
