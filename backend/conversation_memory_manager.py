from typing import Any, List, Optional, Tuple, Union, TypeAlias
from enum import Enum
import asyncio
import structlog
from langchain_openai import ChatOpenAI
from langchain.memory import (
    ConversationBufferMemory,
    ConversationSummaryBufferMemory,
    ConversationEntityMemory,
    CombinedMemory,
)
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_community.retrievers import TimeWeightedVectorStoreRetriever
from langchain_core.documents import Document
from backend.llm_token_counter import LLMTokenCounter
from backend.vector_store_manager import VectorStoreManager, DocumentList

logger = structlog.get_logger(__name__)

# Type aliases for clarity
MessagePair: TypeAlias = Tuple[str, str]
MemoryType: TypeAlias = Union[ConversationBufferMemory, ConversationSummaryBufferMemory, CombinedMemory]

# Constants
SHORT_TERM_TOKEN_LIMIT: int = 1000
MEDIUM_TERM_TOKEN_LIMIT: int = 3000
VECTOR_RETRIEVER_DECAY_RATE: float = 0.5
VECTOR_RETRIEVER_K: int = 4
MEMORY_UPDATE_TIMEOUT_SECONDS: float = 60.0  # Timeout for memory updates

class MemoryStrategy(Enum):
    """Enum defining available memory strategies for conversation history management."""
    SHORT_TERM = "buffer_memory"
    MEDIUM_TERM = "summarized_memory"
    ADVANCED = "enhanced_context_memory"
    FALLBACK = "minimal_context_memory"

class ConversationMemoryManager:
    """Manages conversation history and memory strategies asynchronously based on token count.

    Dynamically selects memory strategies (short-term, medium-term, advanced, or fallback)
    based on the token count of the conversation history, ensuring efficient memory usage
    within the model's context window.

    Attributes:
        llm: ChatOpenAI instance for language model interactions.
        model_name: Name of the LLM model (e.g., 'gpt-4o').
        openai_api_key: OpenAI API key for embeddings.
        llm_token_encoder: LLMTokenCounter instance for counting tokens.
        max_context_tokens: Maximum tokens allowed in the context window.
        fallback_token_limit: Token limit for the fallback memory strategy.
        vector_store_manager: Manager for FAISS vector store operations.
        history: List of (user_content, assistant_content) tuples.
        llm_token_count: Total token count of the conversation history.
        memory: Current memory object (buffer, summary, or combined).
        memory_strategy: Current memory strategy in use.
    """

    def __init__(self) -> None:
        """Initialize ConversationMemoryManager without dependencies.

        Dependencies must be set via async_initialize to support async vector store initialization.
        """
        self.llm: Optional[ChatOpenAI] = None
        self.model_name: Optional[str] = None
        self.openai_api_key: Optional[str] = None
        self.llm_token_encoder: Optional[LLMTokenCounter] = None
        self.max_context_tokens: Optional[int] = None
        self.fallback_token_limit: Optional[int] = None
        self.vector_store_manager: Optional[VectorStoreManager] = None
        self.logger = logger.bind(class_name=self.__class__.__name__)
        self.history: List[MessagePair] = []
        self.llm_token_count: int = 0
        self.memory: Optional[MemoryType] = None
        self.memory_strategy: Optional[MemoryStrategy] = None

    async def async_initialize(
        self,
        llm: ChatOpenAI,
        model_name: str,
        openai_api_key: str,
        llm_token_encoder: LLMTokenCounter,
        max_context_tokens: int,
        fallback_token_limit: int,
        vector_store_manager: VectorStoreManager
    ) -> None:
        """Asynchronously initialize ConversationMemoryManager with dependencies.

        Args:
            llm: Initialized ChatOpenAI instance for language model interactions.
            model_name: Name of the LLM model (e.g., 'gpt-4o').
            openai_api_key: OpenAI API key for embeddings.
            llm_token_encoder: LLMTokenCounter instance for counting tokens.
            max_context_tokens: Maximum tokens allowed in the context window.
            fallback_token_limit: Token limit for the fallback memory strategy.
            vector_store_manager: Manager for FAISS vector store operations.

        Raises:
            ValueError: If configuration is invalid (e.g., negative token limits).
        """
        if max_context_tokens <= 0:
            raise ValueError(f"max_context_tokens must be positive, got {max_context_tokens}")
        if fallback_token_limit <= 0:
            raise ValueError(f"fallback_token_limit must be positive, got {fallback_token_limit}")

        self.llm = llm
        self.model_name = model_name
        self.openai_api_key = openai_api_key
        self.llm_token_encoder = llm_token_encoder
        self.max_context_tokens = max_context_tokens
        self.fallback_token_limit = fallback_token_limit
        self.vector_store_manager = vector_store_manager
        self.memory, self.memory_strategy = await self._initialize_short_term_memory([])

    async def add_message(self, message_pair: MessagePair) -> None:
        """Add a user-assistant message pair to history and update memory strategy if needed.

        Args:
            message_pair: Tuple of (user_content, assistant_content) strings.

        Raises:
            ValueError: If message pair is invalid, token counting fails, or memory is not initialized.
            asyncio.TimeoutError: If memory update exceeds MEMORY_UPDATE_TIMEOUT_SECONDS.
        """
        if not self.llm_token_encoder or not self.memory:
            self.logger.error("ConversationMemoryManager not initialized")
            raise ValueError("ConversationMemoryManager not initialized")

        try:
            async def update_memory() -> None:
                user_content, assistant_content = message_pair
                msg_tokens: int = self.llm_token_encoder.count_tokens([message_pair])
                self.history.append(message_pair)
                self.llm_token_count += msg_tokens
                self.logger.debug(
                    "Added message pair to history",
                    user_tokens=self.llm_token_encoder.count_tokens([(user_content, "")]),
                    assistant_tokens=self.llm_token_encoder.count_tokens([("", assistant_content)]),
                    total_tokens=self.llm_token_count
                )

                expected_strategy: MemoryStrategy = self._get_expected_strategy(self.llm_token_count)
                if expected_strategy != self.memory_strategy:
                    self.logger.info(
                        "Switching memory strategy",
                        current_strategy=self.memory_strategy.value,
                        new_strategy=expected_strategy.value,
                        llm_token_count=self.llm_token_count
                    )
                    self.memory, self.memory_strategy = await self._initialize_memory(
                        self.llm_token_count, self.history
                    )

            await asyncio.wait_for(
                update_memory(),
                timeout=MEMORY_UPDATE_TIMEOUT_SECONDS
            )
        except asyncio.TimeoutError:
            self.logger.error("Memory update timed out", timeout=MEMORY_UPDATE_TIMEOUT_SECONDS)
            raise ValueError(f"Memory update timed out after {MEMORY_UPDATE_TIMEOUT_SECONDS} seconds")
        except Exception as e:
            self.logger.error("Failed to add message pair", message_pair=message_pair, error=str(e), exc_info=True)
            raise ValueError(f"Failed to add message pair: {str(e)}") from e

    async def asimilarity_search(self, query: str, k: int = VECTOR_RETRIEVER_K) -> DocumentList:
        """Perform an asynchronous similarity search on the vector store.

        Args:
            query: Query string to search for.
            k: Number of results to return (defaults to VECTOR_RETRIEVER_K).

        Returns:
            DocumentList: List of FAISS Document objects.

        Raises:
            ValueError: If vector store manager is not initialized or search fails.
            asyncio.TimeoutError: If search exceeds timeout.
        """
        if not self.vector_store_manager:
            self.logger.error("VectorStoreManager not initialized")
            raise ValueError("VectorStoreManager not initialized")
        return await self.vector_store_manager.asimilarity_search(query, k)

    def _get_expected_strategy(self, llm_token_count: int) -> MemoryStrategy:
        """Determine the appropriate memory strategy based on token count.

        Args:
            llm_token_count: Current total token count of the conversation history.

        Returns:
            MemoryStrategy: The memory strategy to use based on token thresholds.
        """
        if llm_token_count > self.max_context_tokens:
            return MemoryStrategy.FALLBACK
        elif llm_token_count <= SHORT_TERM_TOKEN_LIMIT:
            return MemoryStrategy.SHORT_TERM
        elif llm_token_count <= MEDIUM_TERM_TOKEN_LIMIT:
            return MemoryStrategy.MEDIUM_TERM
        return MemoryStrategy.ADVANCED

    async def _initialize_fallback_memory(
        self, llm_token_count: int, history: List[MessagePair]
    ) -> Tuple[ConversationSummaryBufferMemory, MemoryStrategy]:
        """Initialize fallback memory strategy asynchronously for excessive token counts.

        Args:
            llm_token_count: Total token count of the conversation history.
            history: List of (user_content, assistant_content) tuples.

        Returns:
            Tuple[ConversationSummaryBufferMemory, MemoryStrategy]: Fallback memory and strategy.

        Raises:
            asyncio.TimeoutError: If memory initialization exceeds MEMORY_UPDATE_TIMEOUT_SECONDS.
        """
        self.logger.warning(
            "Token count exceeds context limit, using FALLBACK strategy",
            llm_token_count=llm_token_count
        )
        memory = ConversationSummaryBufferMemory(
            llm=self.llm,
            max_token_limit=self.fallback_token_limit,
            memory_key="chat_history",
            chat_memory=ChatMessageHistory()
        )
        recent_messages: List[MessagePair] = []
        current_tokens: int = 0
        for user_content, assistant_content in reversed(history):
            msg_tokens: int = self.llm_token_encoder.count_tokens([(user_content, assistant_content)])
            if current_tokens + msg_tokens <= self.fallback_token_limit:
                recent_messages.append((user_content, assistant_content))
                current_tokens += msg_tokens
            else:
                break
        recent_messages.reverse()

        async def load_messages() -> None:
            for user_content, assistant_content in recent_messages:
                memory.chat_memory.add_user_message(user_content)
                memory.chat_memory.add_ai_message(assistant_content)

        await asyncio.wait_for(
            asyncio.to_thread(load_messages),
            timeout=MEMORY_UPDATE_TIMEOUT_SECONDS
        )
        self.logger.info("Initialized FALLBACK memory", total_tokens=current_tokens)
        return memory, MemoryStrategy.FALLBACK

    async def _initialize_short_term_memory(
        self, history: List[MessagePair]
    ) -> Tuple[ConversationBufferMemory, MemoryStrategy]:
        """Initialize short-term memory asynchronously for low token counts.

        Args:
            history: List of (user_content, assistant_content) tuples.

        Returns:
            Tuple[ConversationBufferMemory, MemoryStrategy]: Short-term memory and strategy.

        Raises:
            asyncio.TimeoutError: If memory initialization exceeds MEMORY_UPDATE_TIMEOUT_SECONDS.
        """
        self.logger.info("Initialized SHORT_TERM memory", max_token_limit=SHORT_TERM_TOKEN_LIMIT)
        memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            chat_memory=ChatMessageHistory()
        )

        async def load_messages() -> None:
            for user_content, assistant_content in history:
                memory.chat_memory.add_user_message(user_content)
                memory.chat_memory.add_ai_message(assistant_content)

        await asyncio.wait_for(
            asyncio.to_thread(load_messages),
            timeout=MEMORY_UPDATE_TIMEOUT_SECONDS
        )
        return memory, MemoryStrategy.SHORT_TERM

    async def _initialize_medium_term_memory(
        self, history: List[MessagePair]
    ) -> Tuple[CombinedMemory, MemoryStrategy]:
        """Initialize medium-term memory asynchronously for moderate token counts.

        Args:
            history: List of (user_content, assistant_content) tuples.

        Returns:
            Tuple[CombinedMemory, MemoryStrategy]: Medium-term memory and strategy.

        Raises:
            asyncio.TimeoutError: If memory initialization exceeds MEMORY_UPDATE_TIMEOUT_SECONDS.
        """
        self.logger.info("Initialized MEDIUM_TERM memory", max_token_limit=MEDIUM_TERM_TOKEN_LIMIT)
        buffer_memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            chat_memory=ChatMessageHistory()
        )
        summary_memory = ConversationSummaryBufferMemory(
            llm=self.llm,
            max_token_limit=SHORT_TERM_TOKEN_LIMIT,
            memory_key="chat_history",
            chat_memory=ChatMessageHistory()
        )

        async def load_messages() -> None:
            for user_content, assistant_content in history:
                buffer_memory.chat_memory.add_user_message(user_content)
                buffer_memory.chat_memory.add_ai_message(assistant_content)
                summary_memory.chat_memory.add_user_message(user_content)
                summary_memory.chat_memory.add_ai_message(assistant_content)

        await asyncio.wait_for(
            asyncio.to_thread(load_messages),
            timeout=MEMORY_UPDATE_TIMEOUT_SECONDS
        )
        combined_memory = CombinedMemory(
            memories=[buffer_memory, summary_memory],
            memory_key="chat_history"
        )
        return combined_memory, MemoryStrategy.MEDIUM_TERM

    async def _initialize_advanced_memory(
        self, history: List[MessagePair]
    ) -> Tuple[CombinedMemory, MemoryStrategy]:
        """Initialize advanced memory asynchronously with entities and vector store for high token counts.

        Args:
            history: List of (user_content, assistant_content) tuples.

        Returns:
            Tuple[CombinedMemory, MemoryStrategy]: Advanced memory and strategy.

        Raises:
            asyncio.TimeoutError: If memory initialization exceeds MEMORY_UPDATE_TIMEOUT_SECONDS.
            ValueError: If vector store manager is not initialized.
        """
        if not self.vector_store_manager or not self.vector_store_manager.vector_store:
            self.logger.error("VectorStoreManager or vector store not initialized")
            raise ValueError("VectorStoreManager or vector store not initialized")

        self.logger.info("Initialized ADVANCED memory", max_token_limit=MEDIUM_TERM_TOKEN_LIMIT)
        buffer_memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            chat_memory=ChatMessageHistory()
        )
        summary_memory = ConversationSummaryBufferMemory(
            llm=self.llm,
            max_token_limit=MEDIUM_TERM_TOKEN_LIMIT,
            memory_key="chat_history",
            chat_memory=ChatMessageHistory()
        )
        entity_memory = ConversationEntityMemory(
            llm=self.llm,
            memory_key="chat_history"
        )
        retriever = TimeWeightedVectorStoreRetriever(
            vectorstore=self.vector_store_manager.vector_store,
            decay_rate=VECTOR_RETRIEVER_DECAY_RATE,
            k=VECTOR_RETRIEVER_K
        )
        vector_memory = retriever.as_memory(memory_key="chat_history")

        async def load_messages() -> None:
            for user_content, assistant_content in history:
                buffer_memory.chat_memory.add_user_message(user_content)
                buffer_memory.chat_memory.add_ai_message(assistant_content)
                summary_memory.chat_memory.add_user_message(user_content)
                summary_memory.chat_memory.add_ai_message(assistant_content)
                entity_memory.save_context({"input": user_content}, {"output": assistant_content})
                vector_memory.save_context({"input": user_content}, {"output": assistant_content})

        await asyncio.wait_for(
            asyncio.to_thread(load_messages),
            timeout=MEMORY_UPDATE_TIMEOUT_SECONDS
        )
        combined_memory = CombinedMemory(
            memories=[buffer_memory, summary_memory, entity_memory, vector_memory],
            memory_key="chat_history"
        )
        return combined_memory, MemoryStrategy.ADVANCED

    async def _initialize_memory(
        self, llm_token_count: int, history: List[MessagePair]
    ) -> Tuple[MemoryType, MemoryStrategy]:
        """Initialize the appropriate memory strategy asynchronously based on token count.

        Args:
            llm_token_count: Total token count of the conversation history.
            history: List of (user_content, assistant_content) tuples.

        Returns:
            Tuple[MemoryType, MemoryStrategy]: Initialized memory and strategy.

        Raises:
            asyncio.TimeoutError: If memory initialization exceeds MEMORY_UPDATE_TIMEOUT_SECONDS.
        """
        self.logger.info("Selecting memory strategy", llm_token_count=llm_token_count)
        if llm_token_count > self.max_context_tokens:
            return await self._initialize_fallback_memory(llm_token_count, history)
        elif llm_token_count <= SHORT_TERM_TOKEN_LIMIT:
            return await self._initialize_short_term_memory(history)
        elif llm_token_count <= MEDIUM_TERM_TOKEN_LIMIT:
            return await self._initialize_medium_term_memory(history)
        return await self._initialize_advanced_memory(history)

    async def aget_memory(self) -> Tuple[MemoryType, MemoryStrategy]:
        """Retrieve the current memory strategy and object asynchronously.

        Returns:
            Tuple[MemoryType, MemoryStrategy]: Current memory object and strategy.

        Raises:
            ValueError: If memory is not initialized.
        """
        if not self.memory or not self.memory_strategy:
            self.logger.error("Memory not initialized")
            raise ValueError("Memory not initialized")
        return self.memory, self.memory_strategy