#!/usr/bin/env python3
"""
Prompt Engineering Test Script

CLI tool for testing and demonstrating the PawAid prompt engineering module.
Generates and displays system prompts, urgency classifications, response
templates, guardrail checks, and citation injection.

Usage:
    python scripts/test_prompts.py [COMMAND] [OPTIONS]

Examples:
    # Show complete system prompt
    python scripts/test_prompts.py system-prompt

    # System prompt for a specific animal
    python scripts/test_prompts.py system-prompt --animal dog

    # Classify urgency of a query
    python scripts/test_prompts.py classify "my dog ate chocolate"

    # Show response template for a query
    python scripts/test_prompts.py template "cat is vomiting blood"

    # Check guardrails against sample text
    python scripts/test_prompts.py guardrails --check "Your dog has parvo"

    # Show citation instructions
    python scripts/test_prompts.py citations --style numbered

    # Run all demos
    python scripts/test_prompts.py demo

    # Verbose output
    python scripts/test_prompts.py demo --verbose
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.prompts import (
    SystemPromptBuilder,
    UrgencyClassifier,
    UrgencyAssessment,
    UrgencyLevel,
    TemplateManager,
    ResponseStyle,
    SafetyGuardrails,
    GuardrailCheckResult,
    CitationInjector,
    PromptConfig,
    URGENCY_DEFINITIONS,
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


def cmd_system_prompt(args) -> None:
    """Display the system prompt."""
    print_header("System Prompt")

    builder = SystemPromptBuilder()
    prompt = builder.build(animal_type=args.animal)

    print(prompt)
    print(f"\n[{len(prompt)} characters]")


def cmd_classify(args) -> None:
    """Classify query urgency."""
    print_header("Urgency Classification")

    classifier = UrgencyClassifier()
    assessment = classifier.pre_classify(args.query)

    print(f"Query: \"{args.query}\"")
    print(f"\nClassification: {assessment.format_header()}")
    print(f"Description: {assessment.description}")
    print(f"Action: {assessment.format_action_banner()}")
    print(f"Pre-classified: {'Yes' if assessment.pre_classified else 'No'}")

    if args.verbose:
        print_section("LLM Classification Prompt")
        print(classifier.get_classification_prompt(args.query))


def cmd_template(args) -> None:
    """Show response template for a query."""
    print_header("Response Template")

    classifier = UrgencyClassifier()
    assessment = classifier.pre_classify(args.query)
    manager = TemplateManager()

    style = manager.select_style(assessment, has_context=True)

    print(f"Query: \"{args.query}\"")
    print(f"Urgency: {assessment.format_header()}")
    print(f"Selected style: {style.value}")

    print_section("LLM Formatting Instructions")

    instruction = manager.get_response_instruction(
        style=style,
        assessment=assessment,
        animal_type=args.animal,
        has_context=True,
    )
    print(instruction)

    # Also show no-context variant
    if args.verbose:
        print_section("No-Context Variant")
        no_ctx_instruction = manager.get_response_instruction(
            style=ResponseStyle.NO_INFORMATION,
            assessment=assessment,
            has_context=False,
        )
        print(no_ctx_instruction)


def cmd_guardrails(args) -> None:
    """Show guardrails and optionally check text."""
    print_header("Safety Guardrails")

    guardrails = SafetyGuardrails()

    print_section("Guardrail Prompt Instructions")
    print(guardrails.build_guardrail_prompt())

    if args.check:
        print_section("Guardrail Check")
        print(f"Checking: \"{args.check}\"")

        result = guardrails.check_response(args.check)

        if result.passed:
            print("\n✅ PASSED — No violations detected")
        else:
            print(f"\n❌ FAILED — {len(result.violations)} violation(s):")
            for v in result.violations:
                print(f"  - {v}")
            print(f"\nRefusal message:\n{result.refusal_message}")


def cmd_citations(args) -> None:
    """Show citation injection instructions."""
    print_header("Citation Injection")

    injector = CitationInjector()

    print_section(f"Citation Instructions (style: {args.style})")
    from src.prompts.config import CitationInjectionConfig
    injector = CitationInjector(CitationInjectionConfig(citation_style=args.style))
    print(injector.build_citation_instruction(has_sources=True))

    # Demo with sample sources
    sample_sources = [
        "first_aid_guides/dog-emergency-first-aid.pdf",
        "toxic_substances_database/chocolate-toxicity.pdf",
        "breed-specific_health_information/labrador-health.pdf",
    ]
    sample_pages = [5, 12, None]

    print_section("Sample Source Block")
    print(injector.format_sources_for_prompt(sample_sources, sample_pages))

    if args.verbose:
        print_section("No-Sources Variant")
        print(injector.build_citation_instruction(has_sources=False))


def cmd_demo(args) -> None:
    """Run complete demo with sample queries."""
    print_header("PawAid Prompt Engineering — Full Demo")

    sample_queries = [
        ("my dog is choking and can't breathe", "dog"),
        ("cat ate a lily plant", "cat"),
        ("dog has a small scrape on his paw", "dog"),
        ("what food is toxic for cats", None),
    ]

    builder = SystemPromptBuilder()
    classifier = UrgencyClassifier()
    templates = TemplateManager()
    guardrails = SafetyGuardrails()
    injector = CitationInjector()

    # 1. System prompt
    print_section("1. System Prompt (excerpt)")
    prompt = builder.build()
    # Show first 500 chars
    print(prompt[:500] + "...\n")
    print(f"[Full prompt: {len(prompt)} characters]")

    # 2. Urgency classification for each query
    print_section("2. Urgency Classification")
    for query, animal in sample_queries:
        assessment = classifier.pre_classify(query)
        print(f"  \"{query}\"")
        print(f"    → {assessment.format_header()}")
        print()

    # 3. Response templates
    print_section("3. Response Templates")
    for query, animal in sample_queries:
        assessment = classifier.pre_classify(query)
        style = templates.select_style(assessment, has_context=True)
        print(f"  \"{query}\"")
        print(f"    Urgency: {assessment.level.value} → Style: {style.value}")
        print()

    # 4. Guardrail checks
    print_section("4. Guardrail Checks")
    test_responses = [
        ("Your dog has parvo virus disease.", False),
        ("Give 25mg of Benadryl twice daily.", False),
        ("Apply gentle pressure with a clean cloth.", True),
        ("You should perform an incision to drain the wound.", False),
    ]

    for text, expected_pass in test_responses:
        result = guardrails.check_response(text)
        status = "✅ PASS" if result.passed else "❌ FAIL"
        expected = "expected" if result.passed == expected_pass else "UNEXPECTED"
        print(f"  {status} ({expected}): \"{text}\"")
        if result.violations:
            for v in result.violations:
                print(f"         Violation: {v}")
        print()

    # 5. Citation injection
    print_section("5. Citation Injection")
    print(injector.build_citation_instruction(has_sources=True))

    # 6. Complete assembled prompt
    if args.verbose:
        print_section("6. Full Assembled Prompt (Emergency Example)")
        query = "my dog ate chocolate and is vomiting"
        assessment = classifier.pre_classify(query)
        style = templates.select_style(assessment, has_context=True)

        print("System prompt + guardrails + template + citations:\n")
        print(builder.build(animal_type="dog"))
        print()
        print(guardrails.build_guardrail_prompt())
        print()
        print(templates.get_response_instruction(
            style=style, assessment=assessment, animal_type="dog"
        ))
        print()
        print(injector.build_citation_instruction())

    print()
    print_header("Demo Complete", "-")


def main():
    parser = argparse.ArgumentParser(
        description="Test the PawAid prompt engineering module",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # system-prompt command
    sp_parser = subparsers.add_parser(
        "system-prompt", help="Display the system prompt"
    )
    sp_parser.add_argument(
        "--animal", "-a", type=str, choices=["dog", "cat"],
        help="Specialize prompt for animal type"
    )
    sp_parser.add_argument("--verbose", "-v", action="store_true")

    # classify command
    cl_parser = subparsers.add_parser(
        "classify", help="Classify urgency of a query"
    )
    cl_parser.add_argument("query", type=str, help="Query to classify")
    cl_parser.add_argument("--verbose", "-v", action="store_true")

    # template command
    tp_parser = subparsers.add_parser(
        "template", help="Show response template for a query"
    )
    tp_parser.add_argument("query", type=str, help="Query to template")
    tp_parser.add_argument(
        "--animal", "-a", type=str, choices=["dog", "cat"],
        help="Animal type"
    )
    tp_parser.add_argument("--verbose", "-v", action="store_true")

    # guardrails command
    gr_parser = subparsers.add_parser(
        "guardrails", help="Show guardrails and check text"
    )
    gr_parser.add_argument(
        "--check", type=str,
        help="Text to check against guardrails"
    )
    gr_parser.add_argument("--verbose", "-v", action="store_true")

    # citations command
    ct_parser = subparsers.add_parser(
        "citations", help="Show citation injection instructions"
    )
    ct_parser.add_argument(
        "--style", "-s", type=str, default="numbered",
        choices=["numbered", "inline", "footnote"],
        help="Citation style (default: numbered)"
    )
    ct_parser.add_argument("--verbose", "-v", action="store_true")

    # demo command
    dm_parser = subparsers.add_parser(
        "demo", help="Run complete demo with sample queries"
    )
    dm_parser.add_argument("--verbose", "-v", action="store_true")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        print("\nError: Please specify a command")
        sys.exit(1)

    commands = {
        "system-prompt": cmd_system_prompt,
        "classify": cmd_classify,
        "template": cmd_template,
        "guardrails": cmd_guardrails,
        "citations": cmd_citations,
        "demo": cmd_demo,
    }

    commands[args.command](args)


if __name__ == "__main__":
    main()
