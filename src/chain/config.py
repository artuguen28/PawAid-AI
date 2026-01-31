"""
LLM Integration Configuration Module

Configuration dataclasses for the RAG chain, response handling,
urgency classification, and conversation memory.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class LLMConfig:
    """Configuration for the LLM (ChatOpenAI)."""

    model_name: str = "gpt-4o"
    temperature: float = 0.2
    max_tokens: int = 1024
    request_timeout: int = 60

    def __post_init__(self):
        if not 0.0 <= self.temperature <= 2.0:
            raise ValueError("temperature must be between 0.0 and 2.0")
        if self.max_tokens < 1:
            raise ValueError("max_tokens must be at least 1")


@dataclass
class ResponseHandlerConfig:
    """Configuration for the response handler."""

    # Safety validation
    validate_safety: bool = True
    max_retries_on_violation: int = 1

    # Response formatting
    include_urgency_header: bool = True
    include_citations_footer: bool = True
    include_disclaimer: bool = True

    disclaimer_text: str = (
        "This is first-aid guidance only, not a veterinary diagnosis. "
        "Please consult a licensed veterinarian for proper medical care."
    )


@dataclass
class MemoryConfig:
    """Configuration for conversation memory."""

    enabled: bool = False
    max_history_turns: int = 5
    memory_key: str = "chat_history"
    return_messages: bool = True

    def __post_init__(self):
        if self.max_history_turns < 1:
            raise ValueError("max_history_turns must be at least 1")


@dataclass
class ChainConfig:
    """Combined configuration for the full LLM integration pipeline."""

    llm: LLMConfig = field(default_factory=LLMConfig)
    response_handler: ResponseHandlerConfig = field(
        default_factory=ResponseHandlerConfig
    )
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    verbose: bool = False

    @classmethod
    def default(cls) -> "ChainConfig":
        """Create a default configuration."""
        return cls()

    @classmethod
    def for_testing(cls) -> "ChainConfig":
        """Create a configuration suitable for testing."""
        return cls(
            llm=LLMConfig(temperature=0.0, max_tokens=512),
            memory=MemoryConfig(enabled=False),
            verbose=True,
        )

    @classmethod
    def with_memory(cls) -> "ChainConfig":
        """Create a configuration with conversation memory enabled."""
        return cls(
            memory=MemoryConfig(enabled=True, max_history_turns=5),
        )
