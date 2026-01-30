"""
PawAid Prompt Engineering Module

This module provides the prompt engineering system for the PawAid
veterinary first-aid assistant, including system prompts, urgency
classification, response templates, safety guardrails, and citation injection.

Components:
    - SystemPromptBuilder: Build safety-critical veterinary persona prompts
    - UrgencyClassifier: Classify query urgency (critical to informational)
    - TemplateManager: Structured first-aid response templates
    - SafetyGuardrails: Refuse diagnosis/prescriptions, enforce boundaries
    - CitationInjector: Include source references in prompts

Configuration:
    - SystemPromptConfig: Persona and tone settings
    - UrgencyConfig: Urgency classification settings
    - TemplateConfig: Response formatting options
    - GuardrailConfig: Safety policy settings
    - CitationInjectionConfig: Citation format settings
    - PromptConfig: Combined configuration

Example:
    from src.prompts import (
        SystemPromptBuilder,
        UrgencyClassifier,
        TemplateManager,
        SafetyGuardrails,
        CitationInjector,
    )

    # Build system prompt
    builder = SystemPromptBuilder()
    system_prompt = builder.build(animal_type="dog")

    # Classify urgency
    classifier = UrgencyClassifier()
    assessment = classifier.pre_classify("my dog ate chocolate")

    # Get response format instructions
    templates = TemplateManager()
    style = templates.select_style(assessment)
    instructions = templates.get_response_instruction(style, assessment)

    # Build guardrails
    guardrails = SafetyGuardrails()
    guardrail_prompt = guardrails.build_guardrail_prompt()

    # Add citation instructions
    injector = CitationInjector()
    citation_prompt = injector.build_citation_instruction()
"""

from .config import (
    UrgencyLevel,
    ResponseStyle,
    SystemPromptConfig,
    UrgencyConfig,
    TemplateConfig,
    GuardrailConfig,
    CitationInjectionConfig,
    PromptConfig,
)

from .system_prompt import (
    SystemPromptBuilder,
    build_system_prompt,
)

from .urgency import (
    UrgencyClassifier,
    UrgencyAssessment,
    URGENCY_DEFINITIONS,
    classify_urgency,
)

from .templates import (
    ResponseTemplate,
    TemplateManager,
    get_response_instruction,
)

from .guardrails import (
    SafetyGuardrails,
    GuardrailCheckResult,
    build_guardrail_prompt,
)

from .citation_injection import (
    CitationInjector,
    build_citation_prompt,
)


__all__ = [
    # Configuration
    "UrgencyLevel",
    "ResponseStyle",
    "SystemPromptConfig",
    "UrgencyConfig",
    "TemplateConfig",
    "GuardrailConfig",
    "CitationInjectionConfig",
    "PromptConfig",
    # System Prompt
    "SystemPromptBuilder",
    "build_system_prompt",
    # Urgency Classification
    "UrgencyClassifier",
    "UrgencyAssessment",
    "URGENCY_DEFINITIONS",
    "classify_urgency",
    # Response Templates
    "ResponseTemplate",
    "TemplateManager",
    "get_response_instruction",
    # Safety Guardrails
    "SafetyGuardrails",
    "GuardrailCheckResult",
    "build_guardrail_prompt",
    # Citation Injection
    "CitationInjector",
    "build_citation_prompt",
]
