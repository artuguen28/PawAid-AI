"""
RAG Chain Module

Core chain that wires together the full pipeline: retrieval, reranking,
context building, prompt assembly, LLM call, and safety validation.
"""

import logging
from typing import Optional

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

from src.embeddings import ChromaManager, ChromaConfig
from src.retrival import (
    Retriever,
    Reranker,
    ContextBuilder,
    CitationManager,
    FormattedContext,
    RetrievalResponse,
    RetrieverConfig,
    RerankerConfig,
    ContextConfig,
    CitationConfig,
    RetrievalConfig,
)
from src.retrival.context_builder import build_context_with_instructions
from src.prompts import (
    SystemPromptBuilder,
    TemplateManager,
    CitationInjector,
    SafetyGuardrails,
    PromptConfig,
)
from .config import ChainConfig, LLMConfig
from .urgency_classifier import LLMUrgencyClassifier
from .response_handler import ResponseHandler, ChainResponse
from .memory import ConversationMemory


logger = logging.getLogger(__name__)


class PawAidChain:
    """
    End-to-end RAG chain for the PawAid veterinary first-aid assistant.

    Pipeline:
    1. Classify query urgency (keyword + optional LLM)
    2. Retrieve relevant documents from vector store
    3. Rerank results for relevance
    4. Build context from top results
    5. Assemble full prompt (system + guardrails + context + template + citations)
    6. Call LLM
    7. Validate response safety
    8. Format and return final response
    """

    def __init__(
        self,
        chain_config: Optional[ChainConfig] = None,
        retrieval_config: Optional[RetrievalConfig] = None,
        prompt_config: Optional[PromptConfig] = None,
        chroma_manager: Optional[ChromaManager] = None,
    ):
        """
        Initialize the PawAid RAG chain.

        Args:
            chain_config: Chain configuration (LLM, response handler, memory).
            retrieval_config: Retrieval pipeline configuration.
            prompt_config: Prompt engineering configuration.
            chroma_manager: Existing ChromaManager instance to reuse.
        """
        self.chain_config = chain_config or ChainConfig()
        self.retrieval_config = retrieval_config or RetrievalConfig()
        self.prompt_config = prompt_config or PromptConfig()

        # Initialize LLM
        self.llm = ChatOpenAI(
            model=self.chain_config.llm.model_name,
            temperature=self.chain_config.llm.temperature,
            max_tokens=self.chain_config.llm.max_tokens,
            request_timeout=self.chain_config.llm.request_timeout,
        )

        # Initialize retrieval pipeline
        self.retriever = Retriever(
            config=self.retrieval_config.retriever,
            query_config=self.retrieval_config.query,
            chroma_manager=chroma_manager,
        )
        self.reranker = Reranker(config=self.retrieval_config.reranker)
        self.context_builder = ContextBuilder(config=self.retrieval_config.context)
        self.citation_manager = CitationManager(config=self.retrieval_config.citation)

        # Initialize prompt components
        self.system_prompt_builder = SystemPromptBuilder(
            config=self.prompt_config.system_prompt,
            guardrail_config=self.prompt_config.guardrail,
        )
        self.template_manager = TemplateManager(config=self.prompt_config.template)
        self.citation_injector = CitationInjector(config=self.prompt_config.citation)
        self.guardrails = SafetyGuardrails(config=self.prompt_config.guardrail)

        # Initialize urgency classifier (reuses same LLM)
        self.urgency_classifier = LLMUrgencyClassifier(
            llm_config=self.chain_config.llm,
            urgency_config=self.prompt_config.urgency,
            llm=self.llm,
        )

        # Initialize response handler
        self.response_handler = ResponseHandler(
            config=self.chain_config.response_handler,
            guardrail_config=self.prompt_config.guardrail,
        )

        # Initialize memory
        self.memory = ConversationMemory(config=self.chain_config.memory)

        logger.info(
            f"PawAidChain initialized "
            f"(model={self.chain_config.llm.model_name}, "
            f"memory={self.chain_config.memory.enabled})"
        )

    def invoke(
        self,
        query: str,
        animal_type: Optional[str] = None,
        use_llm_urgency: bool = True,
    ) -> ChainResponse:
        """
        Run the full RAG pipeline for a user query.

        Args:
            query: The user's pet health question.
            animal_type: Optional explicit animal type ("dog" or "cat").
            use_llm_urgency: Whether to use LLM for urgency classification.

        Returns:
            ChainResponse with the final answer and metadata.
        """
        logger.info(f"Processing query: '{query[:80]}...'")

        # Step 1: Classify urgency
        urgency = self.urgency_classifier.classify(
            query, use_llm=use_llm_urgency
        )
        logger.info(f"Urgency: {urgency.level.value}")

        # Step 2: Retrieve relevant documents
        if urgency.level.value in ("critical", "high"):
            retrieval_response = self.retriever.retrieve_emergency(query)
        elif animal_type:
            retrieval_response = self.retriever.retrieve_for_animal(
                query, animal_type
            )
        else:
            retrieval_response = self.retriever.retrieve(query)

        # Use detected animal from query processing if not explicitly provided
        detected_animal = animal_type or retrieval_response.query.detected_animal
        is_emergency = retrieval_response.query.is_emergency or urgency.level.value in (
            "critical", "high"
        )

        # Step 3: Rerank results
        reranked_response = self.reranker.rerank(retrieval_response)

        # Step 4: Build context
        context = self.context_builder.build(reranked_response)
        has_context = not context.is_empty

        # Step 5: Extract citations
        citations = self.citation_manager.extract_citations(reranked_response)

        # Step 6: Assemble prompt
        messages = self._assemble_prompt(
            query=query,
            context=context,
            urgency=urgency,
            animal_type=detected_animal,
            is_emergency=is_emergency,
            has_context=has_context,
        )

        # Step 7: Call LLM
        try:
            llm_response = self.llm.invoke(messages)
            llm_output = llm_response.content
            logger.debug(f"LLM response: {len(llm_output)} chars")
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            llm_output = (
                "I'm sorry, I'm having trouble processing your request right now. "
                "If this is an emergency, please contact your veterinarian or "
                "nearest emergency animal hospital immediately."
            )

        # Step 8: Process response (safety check + formatting)
        chain_response = self.response_handler.process(
            llm_output=llm_output,
            query=query,
            urgency=urgency,
            citations=citations,
        )

        # Step 9: Update memory
        if self.memory.is_enabled:
            self.memory.add_exchange(query, chain_response.answer)

        return chain_response

    def _assemble_prompt(
        self,
        query: str,
        context: FormattedContext,
        urgency,
        animal_type: Optional[str],
        is_emergency: bool,
        has_context: bool,
    ) -> list:
        """
        Assemble the full prompt from all components.

        Returns:
            List of LangChain messages ready for the LLM.
        """
        # Build system prompt with context
        if has_context:
            context_with_instructions = build_context_with_instructions(
                context=context,
                query=query,
                animal_type=animal_type,
                is_emergency=is_emergency,
            )
            system_prompt = self.system_prompt_builder.build_with_context(
                context_text=context_with_instructions,
                animal_type=animal_type,
                is_emergency=is_emergency,
            )
        else:
            system_prompt = self.system_prompt_builder.build(
                animal_type=animal_type
            )

        # Add guardrails
        guardrail_prompt = self.guardrails.build_guardrail_prompt()

        # Add response template instructions
        style = self.template_manager.select_style(urgency, has_context)
        template_instruction = self.template_manager.get_response_instruction(
            style=style,
            assessment=urgency,
            animal_type=animal_type,
            has_context=has_context,
        )

        # Add citation instructions
        citation_instruction = self.citation_injector.build_citation_instruction(
            has_sources=has_context,
        )

        # Combine system prompt parts
        full_system = "\n\n".join(
            part for part in [
                system_prompt,
                guardrail_prompt,
                template_instruction,
                citation_instruction,
            ] if part
        )

        messages = [SystemMessage(content=full_system)]

        # Add conversation history if enabled
        if self.memory.is_enabled:
            history = self.memory.get_messages_for_prompt()
            messages.extend(history)

        # Add user query
        messages.append(HumanMessage(content=query))

        if self.chain_config.verbose:
            logger.debug(
                f"Assembled prompt: {len(full_system)} chars system, "
                f"{len(messages)} messages total"
            )

        return messages

    def clear_memory(self) -> None:
        """Clear conversation memory."""
        self.memory.clear()

    def get_stats(self) -> dict:
        """Get chain statistics."""
        return {
            "chain_config": {
                "model": self.chain_config.llm.model_name,
                "temperature": self.chain_config.llm.temperature,
                "memory_enabled": self.chain_config.memory.enabled,
            },
            "retriever": self.retriever.get_stats(),
            "memory_turns": self.memory.num_turns,
        }
