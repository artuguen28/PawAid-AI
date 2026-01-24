#!/usr/bin/env python3
"""
Retrieval Query Script

CLI tool for testing the PawAid retrieval system.
Searches the vector store and returns relevant veterinary information.

Usage:
    python scripts/retrieve.py "your query here" [OPTIONS]

Examples:
    # Basic query
    python scripts/retrieve.py "my dog ate chocolate"

    # Query with more results
    python scripts/retrieve.py "cat vomiting" --k 10

    # Filter by animal type
    python scripts/retrieve.py "poisoning symptoms" --animal dog

    # Emergency query (prioritizes urgent content)
    python scripts/retrieve.py "dog not breathing" --emergency

    # Show full context (for LLM input)
    python scripts/retrieve.py "chocolate toxicity" --context

    # Show citations
    python scripts/retrieve.py "flea treatment" --citations

    # Interactive mode
    python scripts/retrieve.py --interactive

    # Verbose output
    python scripts/retrieve.py "toxic plants" --verbose
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv

from src.embeddings import ChromaConfig
from src.retrival import (
    Retriever,
    RetrieverConfig,
    Reranker,
    RerankerConfig,
    RerankerStrategy,
    ContextBuilder,
    ContextConfig,
    CitationManager,
    RetrievalConfig,
    build_context_with_instructions,
)


def print_header(text: str, char: str = "=") -> None:
    """Print a formatted header."""
    print(char * 50)
    print(text)
    print(char * 50)


def print_results(response, verbose: bool = False) -> None:
    """Print retrieval results."""
    query = response.query

    print(f"\nQuery: \"{query.original_query}\"")
    if query.cleaned_query != query.original_query:
        print(f"Cleaned: \"{query.cleaned_query}\"")

    print(f"Detected animal: {query.detected_animal or 'none'}")
    print(f"Emergency: {'Yes' if query.is_emergency else 'No'}")

    if verbose and query.keywords:
        print(f"Keywords: {', '.join(query.keywords)}")

    print(f"\nFound {len(response.results)} results")
    print("-" * 50)

    if not response.has_results:
        print("No results found. Try:")
        print("  - Using different keywords")
        print("  - Removing animal filter (--animal none)")
        print("  - Checking if index is built")
        return

    for i, result in enumerate(response.results, 1):
        print(f"\n[{i}] Score: {result.score:.3f}")
        print(f"    Source: {result.source}")
        print(f"    Animals: {', '.join(result.animal_types) or 'unspecified'}")

        if result.urgency_level:
            print(f"    Urgency: {result.urgency_level}")
        if result.section_type:
            print(f"    Section: {result.section_type}")

        # Print content preview
        content = result.content
        if len(content) > 300:
            content = content[:300] + "..."
        print(f"    Content: {content}")


def print_context(context, query_info) -> None:
    """Print formatted context for LLM input."""
    print_header("Context for LLM", "-")

    full_context = build_context_with_instructions(
        context,
        query=query_info.original_query,
        animal_type=query_info.detected_animal,
        is_emergency=query_info.is_emergency
    )

    print(full_context)

    print("-" * 50)
    print(f"Chunks: {context.num_chunks}")
    print(f"Characters: {context.total_characters}")
    print(f"Truncated: {'Yes' if context.truncated else 'No'}")
    print(f"Sources: {', '.join(context.sources_used)}")


def print_citations(citations) -> None:
    """Print citation information."""
    print_header("Citations", "-")
    print(citations.format_detailed())


def run_interactive(retriever, reranker, context_builder, citation_manager, verbose: bool) -> None:
    """Run interactive query mode."""
    print_header("Interactive Mode")
    print("Enter queries to search. Type 'quit' or 'exit' to stop.")
    print("Commands: 'stats' - show collection stats")
    print()

    while True:
        try:
            query = input("Query> ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting...")
            break

        if not query:
            continue

        if query.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break

        if query.lower() == "stats":
            stats = retriever.get_stats()
            print(f"\nCollection: {stats['collection']['collection_name']}")
            print(f"Documents: {stats['collection']['document_count']}")
            print()
            continue

        # Run query
        try:
            response = retriever.retrieve(query)
            response = reranker.rerank(response)
            print_results(response, verbose)
            print()
        except Exception as e:
            print(f"Error: {e}")
            print()


def main():
    parser = argparse.ArgumentParser(
        description="Search the PawAid veterinary knowledge base",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument(
        "query",
        nargs="?",
        type=str,
        help="Search query (e.g., 'my dog ate chocolate')"
    )

    parser.add_argument(
        "--k", "-k",
        type=int,
        default=5,
        help="Number of results to return (default: 5)"
    )

    parser.add_argument(
        "--animal", "-a",
        type=str,
        choices=["dog", "cat", "none"],
        help="Filter by animal type (dog, cat, or none to disable)"
    )

    parser.add_argument(
        "--emergency", "-e",
        action="store_true",
        help="Prioritize emergency/urgent content"
    )

    parser.add_argument(
        "--no-rerank",
        action="store_true",
        help="Disable reranking"
    )

    parser.add_argument(
        "--context", "-c",
        action="store_true",
        help="Show formatted context for LLM input"
    )

    parser.add_argument(
        "--citations",
        action="store_true",
        help="Show detailed citations"
    )

    parser.add_argument(
        "--interactive", "-i",
        action="store_true",
        help="Run in interactive mode"
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
        help="Verbose output"
    )

    args = parser.parse_args()

    # Load environment variables
    load_dotenv()

    # Check for query or interactive mode
    if not args.query and not args.interactive:
        parser.print_help()
        print("\nError: Please provide a query or use --interactive mode")
        sys.exit(1)

    # Create configuration
    chroma_config = ChromaConfig(
        persist_directory=args.persist_dir,
        collection_name=args.collection_name
    )

    retriever_config = RetrieverConfig(
        default_k=args.k,
        filter_by_animal=(args.animal != "none")
    )

    reranker_config = RerankerConfig(
        strategy=RerankerStrategy.NONE if args.no_rerank else RerankerStrategy.COMBINED,
        urgency_boost=2.0 if args.emergency else 1.5
    )

    context_config = ContextConfig(
        max_chunks=args.k,
        max_context_length=4000
    )

    # Initialize components
    try:
        retriever = Retriever(
            config=retriever_config,
            chroma_config=chroma_config
        )
        reranker = Reranker(config=reranker_config)
        context_builder = ContextBuilder(config=context_config)
        citation_manager = CitationManager()
    except Exception as e:
        print(f"Error initializing retriever: {e}")
        print("\nMake sure you have built the index:")
        print("  python scripts/build_index.py --input data/chunks.json")
        sys.exit(1)

    # Interactive mode
    if args.interactive:
        run_interactive(
            retriever, reranker, context_builder, citation_manager, args.verbose
        )
        return

    # Single query mode
    print_header("PawAid Retrieval")

    try:
        # Handle animal filter
        if args.animal == "dog":
            response = retriever.retrieve_for_animal(args.query, "dog", k=args.k)
        elif args.animal == "cat":
            response = retriever.retrieve_for_animal(args.query, "cat", k=args.k)
        elif args.emergency:
            response = retriever.retrieve_emergency(args.query, k=args.k)
        else:
            response = retriever.retrieve(args.query, k=args.k)

        # Rerank
        response = reranker.rerank(response, top_k=args.k)

        # Print results
        print_results(response, args.verbose)

        # Print context if requested
        if args.context:
            context = context_builder.build(response)
            print()
            print_context(context, response.query)

        # Print citations if requested
        if args.citations:
            citations = citation_manager.extract_citations(response)
            print()
            print_citations(citations)

    except Exception as e:
        print(f"Error during retrieval: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)

    print()
    print_header("Done!", "-")


if __name__ == "__main__":
    main()
