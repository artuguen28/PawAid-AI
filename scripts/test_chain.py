#!/usr/bin/env python3
"""
LLM Integration Test Script

CLI tool for testing and demonstrating the PawAid RAG chain module.
Tests the full pipeline: retrieval, prompt assembly, LLM call, and
safety validation.

Usage:
    python scripts/test_chain.py [COMMAND] [OPTIONS]

Examples:
    # Ask a single question
    python scripts/test_chain.py ask "my dog ate chocolate"

    # Ask with explicit animal type
    python scripts/test_chain.py ask "ate a lily plant" --animal cat

    # Classify urgency using LLM
    python scripts/test_chain.py urgency "my cat is not breathing"

    # Interactive multi-turn conversation
    python scripts/test_chain.py chat

    # Show chain configuration and stats
    python scripts/test_chain.py info

    # Run all demos with sample queries
    python scripts/test_chain.py demo

    # Verbose output
    python scripts/test_chain.py demo --verbose
"""

import argparse
import logging
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv

load_dotenv()

from src.chain import (
    PawAidChain,
    ChainConfig,
    LLMConfig,
    LLMUrgencyClassifier,
    ChainResponse,
)


def print_header(text: str, char: str = "=") -> None:
    """Print a formatted header."""
    print(char * 60)
    print(text)
    print(char * 60)


def print_section(text: str) -> None:
    """Print a section header."""
    print(f"\n{'─' * 40}")
    print(f"  {text}")
    print(f"{'─' * 40}\n")


def print_response(response: ChainResponse) -> None:
    """Print a formatted chain response."""
    print(response.answer)

    if response.was_refused:
        print("\n[Response was refused by safety guardrails]")

    if response.urgency:
        print(f"\n[Urgency: {response.urgency.level.value}]")

    if response.has_citations:
        print(f"[Sources: {response.citations.num_sources}]")

    print(f"[Safe: {'Yes' if response.is_safe else 'No'}]")


def create_chain(args) -> PawAidChain:
    """Create a PawAidChain with configuration from args."""
    llm_config = LLMConfig(
        model_name=getattr(args, "model", "gpt-4o"),
        temperature=getattr(args, "temperature", 0.2),
    )

    chain_config = ChainConfig(
        llm=llm_config,
        verbose=getattr(args, "verbose", False),
    )

    return PawAidChain(chain_config=chain_config)


def cmd_ask(args) -> None:
    """Ask a single question."""
    print_header("PawAid RAG Chain — Single Query")

    chain = create_chain(args)

    print(f"Query: \"{args.query}\"")
    if args.animal:
        print(f"Animal: {args.animal}")
    print()

    response = chain.invoke(
        query=args.query,
        animal_type=args.animal,
    )

    print_response(response)


def cmd_urgency(args) -> None:
    """Classify urgency using LLM."""
    print_header("LLM Urgency Classification")

    llm_config = LLMConfig(
        model_name=getattr(args, "model", "gpt-4o"),
    )

    classifier = LLMUrgencyClassifier(llm_config=llm_config)

    print(f"Query: \"{args.query}\"")
    print()

    # Show keyword pre-classification
    pre_assessment = classifier.keyword_classifier.pre_classify(args.query)
    print(f"Keyword pre-classification: {pre_assessment.format_header()}")

    # Show LLM classification
    print("\nCalling LLM for detailed classification...")
    assessment = classifier.classify(args.query, use_llm=True)

    print(f"\nFinal classification: {assessment.format_header()}")
    print(f"Description: {assessment.description}")
    print(f"Action: {assessment.format_action_banner()}")
    print(f"Pre-classified only: {'Yes' if assessment.pre_classified else 'No'}")


def cmd_chat(args) -> None:
    """Interactive multi-turn conversation."""
    print_header("PawAid Interactive Chat")
    print("\nType your pet health questions. Type 'quit' or 'exit' to stop.")
    print("Type 'clear' to reset conversation memory.\n")

    llm_config = LLMConfig(
        model_name=getattr(args, "model", "gpt-4o"),
    )

    chain_config = ChainConfig.with_memory()
    chain_config.llm = llm_config
    chain_config.verbose = getattr(args, "verbose", False)

    chain = PawAidChain(chain_config=chain_config)

    turn = 0
    while True:
        try:
            query = input(f"\n[{turn + 1}] You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not query:
            continue

        if query.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break

        if query.lower() == "clear":
            chain.clear_memory()
            turn = 0
            print("Conversation memory cleared.")
            continue

        response = chain.invoke(query)
        print(f"\nPawAid: {response.answer}")

        if response.urgency:
            print(f"\n  [{response.urgency.level.value.upper()}]", end="")
        if response.was_refused:
            print("  [REFUSED]", end="")
        print()

        turn += 1


def cmd_info(args) -> None:
    """Show chain configuration and stats."""
    print_header("Chain Configuration & Stats")

    chain = create_chain(args)
    stats = chain.get_stats()

    print_section("Chain Config")
    for key, value in stats["chain_config"].items():
        print(f"  {key}: {value}")

    print_section("Retriever Stats")
    retriever_stats = stats["retriever"]
    for key, value in retriever_stats["retriever_config"].items():
        print(f"  {key}: {value}")

    print_section("Collection Stats")
    for key, value in retriever_stats["collection"].items():
        print(f"  {key}: {value}")

    print(f"\nMemory turns: {stats['memory_turns']}")


def cmd_demo(args) -> None:
    """Run demo with sample queries."""
    print_header("PawAid RAG Chain — Full Demo")

    chain = create_chain(args)

    sample_queries = [
        ("my dog ate chocolate and is vomiting", "dog"),
        ("cat ate a lily plant yesterday", "cat"),
        ("my puppy has a small scrape on his paw", "dog"),
        ("what foods are toxic for cats", None),
    ]

    for i, (query, animal) in enumerate(sample_queries, 1):
        print_section(f"Query {i}/{len(sample_queries)}")
        print(f"Query: \"{query}\"")
        if animal:
            print(f"Animal: {animal}")
        print()

        response = chain.invoke(query=query, animal_type=animal)
        print_response(response)

    print()
    print_header("Demo Complete", "-")


def main():
    parser = argparse.ArgumentParser(
        description="Test the PawAid LLM integration chain",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Global options
    parser.add_argument(
        "--model", "-m", type=str, default="gpt-4o",
        help="OpenAI model to use (default: gpt-4o)"
    )
    parser.add_argument(
        "--temperature", "-t", type=float, default=0.2,
        help="LLM temperature (default: 0.2)"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Enable verbose logging"
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # ask command
    ask_parser = subparsers.add_parser(
        "ask", help="Ask a single question"
    )
    ask_parser.add_argument("query", type=str, help="Pet health question")
    ask_parser.add_argument(
        "--animal", "-a", type=str, choices=["dog", "cat"],
        help="Animal type"
    )

    # urgency command
    urg_parser = subparsers.add_parser(
        "urgency", help="Classify urgency using LLM"
    )
    urg_parser.add_argument("query", type=str, help="Query to classify")

    # chat command
    subparsers.add_parser(
        "chat", help="Interactive multi-turn conversation"
    )

    # info command
    subparsers.add_parser(
        "info", help="Show chain configuration and stats"
    )

    # demo command
    subparsers.add_parser(
        "demo", help="Run demo with sample queries"
    )

    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.WARNING)

    if not args.command:
        parser.print_help()
        print("\nError: Please specify a command")
        sys.exit(1)

    commands = {
        "ask": cmd_ask,
        "urgency": cmd_urgency,
        "chat": cmd_chat,
        "info": cmd_info,
        "demo": cmd_demo,
    }

    commands[args.command](args)


if __name__ == "__main__":
    main()
