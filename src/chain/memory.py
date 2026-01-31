"""
Conversation Memory Module

Provides optional multi-turn conversation memory for the RAG chain,
allowing the assistant to reference previous exchanges.
"""

import logging
from typing import Optional, List, Dict

from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    AIMessage,
    SystemMessage,
)

from .config import MemoryConfig


logger = logging.getLogger(__name__)


class ConversationMemory:
    """
    Manages conversation history for multi-turn interactions.

    Stores a sliding window of recent exchanges (human + AI messages)
    and formats them for inclusion in the LLM prompt.
    """

    def __init__(self, config: Optional[MemoryConfig] = None):
        """
        Initialize conversation memory.

        Args:
            config: Memory configuration. Uses defaults if not provided.
        """
        self.config = config or MemoryConfig()
        self._messages: List[BaseMessage] = []
        logger.debug(
            f"ConversationMemory initialized "
            f"(enabled={self.config.enabled}, "
            f"max_turns={self.config.max_history_turns})"
        )

    @property
    def is_enabled(self) -> bool:
        """Check if memory is enabled."""
        return self.config.enabled

    @property
    def messages(self) -> List[BaseMessage]:
        """Get the current message history."""
        return list(self._messages)

    @property
    def num_turns(self) -> int:
        """Get the number of conversation turns (human+AI pairs)."""
        return len(self._messages) // 2

    def add_exchange(self, human_message: str, ai_message: str) -> None:
        """
        Add a human-AI exchange to memory.

        Args:
            human_message: The user's message.
            ai_message: The assistant's response.
        """
        if not self.config.enabled:
            return

        self._messages.append(HumanMessage(content=human_message))
        self._messages.append(AIMessage(content=ai_message))

        # Trim to max turns (each turn = 2 messages)
        max_messages = self.config.max_history_turns * 2
        if len(self._messages) > max_messages:
            self._messages = self._messages[-max_messages:]

        logger.debug(
            f"Added exchange to memory (turns: {self.num_turns}/"
            f"{self.config.max_history_turns})"
        )

    def get_messages_for_prompt(self) -> List[BaseMessage]:
        """
        Get conversation history formatted for LLM prompt.

        Returns:
            List of messages to include before the current query.
        """
        if not self.config.enabled or not self._messages:
            return []

        return list(self._messages)

    def get_context_summary(self) -> str:
        """
        Get a text summary of conversation history.

        Returns:
            Formatted string of previous exchanges, or empty string.
        """
        if not self.config.enabled or not self._messages:
            return ""

        lines = ["Previous conversation:"]
        for msg in self._messages:
            if isinstance(msg, HumanMessage):
                lines.append(f"  Owner: {msg.content}")
            elif isinstance(msg, AIMessage):
                # Truncate long AI responses in summary
                content = msg.content
                if len(content) > 200:
                    content = content[:200] + "..."
                lines.append(f"  PawAid: {content}")

        return "\n".join(lines)

    def clear(self) -> None:
        """Clear all conversation history."""
        self._messages.clear()
        logger.debug("Conversation memory cleared")
