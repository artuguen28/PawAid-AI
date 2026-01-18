#!/usr/bin/env python3
"""
Vector Index Builder Script

CLI tool for building and managing the PawAid vector index.
Embeds document chunks and stores them in ChromaDB for semantic retrieval.

Usage:
    python scripts/build_index.py [OPTIONS]

Examples:
    # Build index from default chunks file
    python scripts/build_index.py

    # Build from specific file
    python scripts/build_index.py --input data/chunks.json

    # Rebuild existing index (clear and recreate)
    python scripts/build_index.py --rebuild

    # Test retrieval after building
    python scripts/build_index.py --test-query "dog ate chocolate"

    # View collection stats only
    python scripts/build_index.py --stats-only

    # Verbose output with progress
    python scripts/build_index.py --verbose
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv

from src.embeddings import (
    IndexBuilder,
    IndexConfig,
    EmbeddingConfig,
    ChromaConfig,
    create_progress_printer,
)


def print_stats(builder: IndexBuilder) -> None:
    """Print collection statistics."""
    try:
        stats = builder.get_stats()
        print("\n" + "=" * 50)
        print("Collection Statistics")
        print("=" * 50)
        print(f"Collection: {stats['collection_name']}")
        print(f"Documents: {stats['document_count']}")
        print(f"Persist directory: {stats['persist_directory']}")
        print(f"Distance metric: {stats['distance_metric']}")
        print(f"Embedding model: {stats['embedding_model']}")
        print(f"Embedding dimensions: {stats['embedding_dimensions']}")
    except Exception as e:
        print(f"Error getting stats: {e}")


def print_test_results(results: list, query: str) -> None:
    """Print test retrieval results."""
    print("\n" + "=" * 50)
    print(f"Test Query: \"{query}\"")
    print("=" * 50)

    if not results:
        print("No results found")
        return

    for i, result in enumerate(results, 1):
        print(f"\n--- Result {i} (score: {result['score']}) ---")
        print(f"Source: {result['filename']}")
        print(f"Animals: {result['animal_types']}")
        print(f"Urgency: {result['urgency_level']}")
        print(f"Content: {result['content']}")


def main():
    parser = argparse.ArgumentParser(
        description="Build and manage the PawAid vector index",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument(
        "--input", "-i",
        type=str,
        default="data/chunks.json",
        help="Path to chunks JSON file (default: data/chunks.json)"
    )

    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Clear existing index and rebuild from scratch"
    )

    parser.add_argument(
        "--test-query", "-t",
        type=str,
        help="Test retrieval with a query after building"
    )

    parser.add_argument(
        "--stats-only",
        action="store_true",
        help="Only show collection statistics, don't build"
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Batch size for embedding (default: 100)"
    )

    parser.add_argument(
        "--collection-name",
        type=str,
        default="pawaid_veterinary",
        help="ChromaDB collection name (default: pawaid_veterinary)"
    )

    parser.add_argument(
        "--persist-dir",
        type=str,
        default="data/chroma",
        help="ChromaDB persist directory (default: data/chroma)"
    )

    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output with progress"
    )

    args = parser.parse_args()

    # Load environment variables
    load_dotenv()

    # Create configuration
    config = IndexConfig(
        embedding=EmbeddingConfig(batch_size=args.batch_size),
        chroma=ChromaConfig(
            persist_directory=args.persist_dir,
            collection_name=args.collection_name
        ),
        verbose=args.verbose
    )

    # Initialize builder
    builder = IndexBuilder(config)

    # Stats only mode
    if args.stats_only:
        print_stats(builder)
        return

    # Check input file
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        print("Run the ingestion pipeline first:")
        print("  python scripts/ingest_documents.py --output data/chunks.json")
        sys.exit(1)

    # Build index
    print("=" * 50)
    print("PawAid Vector Index Builder")
    print("=" * 50)
    print(f"Input: {input_path}")
    print(f"Collection: {args.collection_name}")
    print(f"Persist directory: {args.persist_dir}")
    print(f"Batch size: {args.batch_size}")

    if args.rebuild:
        print("Mode: REBUILD (clearing existing data)")
    else:
        print("Mode: ADD (appending to existing)")

    print()

    try:
        progress_callback = create_progress_printer(args.verbose)

        if args.rebuild:
            result = builder.rebuild_index(
                json_path=input_path,
                progress_callback=progress_callback
            )
        else:
            result = builder.build_from_json(
                json_path=input_path,
                progress_callback=progress_callback
            )

        # Print results
        print("\n" + "=" * 50)
        print("Build Complete!")
        print("=" * 50)
        print(f"Total documents: {result.total_documents}")
        print(f"Successful: {result.successful_documents}")
        print(f"Failed: {result.failed_documents}")
        print(f"Success rate: {result.success_rate:.1f}%")

        if result.errors:
            print("\nErrors:")
            for error in result.errors:
                print(f"  - {error}")

    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error building index: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)

    # Show stats
    print_stats(builder)

    # Test query if provided
    if args.test_query:
        try:
            results = builder.test_retrieval(args.test_query, k=3)
            print_test_results(results, args.test_query)
        except Exception as e:
            print(f"Error running test query: {e}")

    print("\n" + "=" * 50)
    print("Done!")
    print("=" * 50)


if __name__ == "__main__":
    main()
