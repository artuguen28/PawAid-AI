"""
PawAid LLM Integration Module

This module provides the complete RAG chain that ties together retrieval,
prompt engineering, LLM calls, and safety validation into a single pipeline.

Components:
    - PawAidChain: End-to-end RAG chain for pet health queries
    - ResponseHandler: Post-processing with safety validation
    - LLMUrgencyClassifier: LLM-based urgency classification
    - ConversationMemory: Optional multi-turn conversation memory

Configuration:
    - ChainConfig: Combined configuration
    - LLMConfig: LLM model settings
    - ResponseHandlerConfig: Response processing settings
    - MemoryConfig: Conversation memory settings

Example:
    from src.chain import PawAidChain, ChainConfig

    # Initialize with defaults
    chain = PawAidChain()

    # Ask a question
    response = chain.invoke("my dog ate chocolate")
    print(response.answer)
    print(f"Urgency: {response.urgency.level.value}")
    print(f"Safe: {response.is_safe}")

    # With memory enabled
    chain = PawAidChain(chain_config=ChainConfig.with_memory())
    response1 = chain.invoke("my cat is limping")
    response2 = chain.invoke("should I wrap the leg?")
"""

from .config import (
    LLMConfig,
    ResponseHandlerConfig,
    MemoryConfig,
    ChainConfig,
)

from .rag_chain import PawAidChain

from .response_handler import (
    ResponseHandler,
    ChainResponse,
)

from .urgency_classifier import LLMUrgencyClassifier

from .memory import ConversationMemory


__all__ = [
    # Configuration
    "LLMConfig",
    "ResponseHandlerConfig",
    "MemoryConfig",
    "ChainConfig",
    # Core Chain
    "PawAidChain",
    # Response Handling
    "ResponseHandler",
    "ChainResponse",
    # Urgency Classification
    "LLMUrgencyClassifier",
    # Memory
    "ConversationMemory",
]
