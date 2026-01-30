"""
Prompt Engineering Configuration Module

Configuration dataclasses for the prompt engineering system.
"""

from dataclasses import dataclass, field
from typing import Optional, List
from enum import Enum


class UrgencyLevel(Enum):
    """Urgency levels for pet emergency classification."""

    CRITICAL = "critical"
    HIGH = "high"
    MODERATE = "moderate"
    LOW = "low"
    INFORMATIONAL = "informational"


class ResponseStyle(Enum):
    """Available response formatting styles."""

    EMERGENCY = "emergency"
    FIRST_AID = "first_aid"
    GENERAL = "general"
    NO_INFORMATION = "no_information"


@dataclass
class SystemPromptConfig:
    """Configuration for the system prompt builder."""

    # Persona
    persona_name: str = "PawAid"
    persona_role: str = "veterinary first-aid assistant"
    supported_animals: List[str] = field(default_factory=lambda: ["dog", "cat"])

    # Tone
    tone: str = "calm, clear, and compassionate"
    use_simple_language: bool = True

    # Safety emphasis
    always_recommend_vet: bool = True
    emergency_disclaimer: bool = True


@dataclass
class UrgencyConfig:
    """Configuration for urgency classification."""

    # Classification
    enable_classification: bool = True
    default_level: UrgencyLevel = UrgencyLevel.MODERATE

    # Thresholds for keyword-based pre-classification
    critical_keywords: List[str] = field(default_factory=lambda: [
        "not breathing", "unconscious", "seizure", "convulsion",
        "choking", "severe bleeding", "poisoned", "hit by car",
        "collapsed", "unresponsive", "blue gums", "pale gums",
    ])
    high_keywords: List[str] = field(default_factory=lambda: [
        "bleeding", "broken", "fracture", "burn", "swallowed",
        "toxic", "poison", "difficulty breathing", "vomiting blood",
        "severe pain", "eye injury", "snake bite", "heatstroke",
        "chocolate", "lily", "xylitol", "antifreeze",
        "rat poison", "ate grapes", "ate raisins", "ate onion",
    ])

    def __post_init__(self):
        if not isinstance(self.default_level, UrgencyLevel):
            raise ValueError("default_level must be an UrgencyLevel enum value")


@dataclass
class TemplateConfig:
    """Configuration for response templates."""

    # Formatting
    include_warning_header: bool = True
    include_steps: bool = True
    include_what_not_to_do: bool = True
    include_when_to_see_vet: bool = True
    max_steps: int = 8

    # Sections
    section_separator: str = "\n\n"

    def __post_init__(self):
        if self.max_steps < 1:
            raise ValueError("max_steps must be at least 1")


@dataclass
class GuardrailConfig:
    """Configuration for safety guardrails."""

    # Refusal policies
    refuse_diagnosis: bool = True
    refuse_prescriptions: bool = True
    refuse_surgery_advice: bool = True
    refuse_dosage: bool = True

    # Redirects
    always_suggest_vet: bool = True
    emergency_redirect: bool = True

    # Scope boundaries
    restrict_to_cats_and_dogs: bool = True
    restrict_to_first_aid: bool = True


@dataclass
class CitationInjectionConfig:
    """Configuration for citation injection into prompts."""

    # Citation format
    inject_citations: bool = True
    citation_style: str = "numbered"  # "numbered", "inline", "footnote"
    max_citations: int = 5

    # Instructions
    instruct_llm_to_cite: bool = True
    require_source_attribution: bool = True

    def __post_init__(self):
        if self.citation_style not in ("numbered", "inline", "footnote"):
            raise ValueError(
                f"citation_style must be 'numbered', 'inline', or 'footnote', "
                f"got: {self.citation_style}"
            )
        if self.max_citations < 1:
            raise ValueError("max_citations must be at least 1")


@dataclass
class PromptConfig:
    """Combined configuration for the full prompt engineering pipeline."""

    system_prompt: SystemPromptConfig = field(default_factory=SystemPromptConfig)
    urgency: UrgencyConfig = field(default_factory=UrgencyConfig)
    template: TemplateConfig = field(default_factory=TemplateConfig)
    guardrail: GuardrailConfig = field(default_factory=GuardrailConfig)
    citation: CitationInjectionConfig = field(default_factory=CitationInjectionConfig)
    verbose: bool = False

    @classmethod
    def default(cls) -> "PromptConfig":
        """Create a default configuration."""
        return cls()

    @classmethod
    def for_testing(cls) -> "PromptConfig":
        """Create a configuration suitable for testing."""
        return cls(
            template=TemplateConfig(max_steps=5),
            citation=CitationInjectionConfig(max_citations=3),
            verbose=True,
        )

    @classmethod
    def for_emergency(cls) -> "PromptConfig":
        """Create a configuration optimized for emergency queries."""
        return cls(
            urgency=UrgencyConfig(
                default_level=UrgencyLevel.HIGH,
            ),
            template=TemplateConfig(
                include_warning_header=True,
                include_what_not_to_do=True,
            ),
            guardrail=GuardrailConfig(
                emergency_redirect=True,
                always_suggest_vet=True,
            ),
        )
