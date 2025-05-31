from typing import List, Optional, Tuple, Union, TypeAlias, Dict, Any, Callable
from enum import Enum
import asyncio
import structlog
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.memory import (
    CombinedMemory,
    ConversationSummaryBufferMemory,
    ConversationEntityMemory,
)
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain.retrievers import TimeWeightedVectorStoreRetriever
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.messages import trim_messages
from langchain_core.memory import BaseMemory
from backend.llm_token_counter import LLMTokenCounter

logger = structlog.get_logger(__name__)

# Type aliases for clarity
MessagePair: TypeAlias = Tuple[str, str]
DocumentList: TypeAlias = List[Document]

# Constants for memory configuration
# Token limits for different memory strategies
SHORT_TERM_TOKEN_LIMIT: int = 300  # Maximum tokens for short-term memory
MEDIUM_TERM_TOKEN_LIMIT: int = 1500  # Maximum tokens for medium-term memory

# Vector retriever settings
VECTOR_RETRIEVER_DECAY_RATE: float = (
    0.5  # Decay rate for time-weighted vector retrieval
)
VECTOR_RETRIEVER_K: int = 4  # Number of results to retrieve from vector store

# Buffer window settings
DEFAULT_BUFFER_WINDOW_SIZE: int = (
    10  # Default number of messages to keep in buffer window
)
SLIDING_WINDOW_EXCHANGES: int = 4  # Number of exchanges to keep in sliding window
DEFAULT_TOKEN_LIMIT: int = 1000  # Default token limit for buffer window memory

# Performance settings
MEMORY_UPDATE_TIMEOUT_SECONDS: float = 60.0  # Timeout for memory updates


class CustomBufferWindowMemory(BaseMemory):
    """
    Memory implementation that keeps a window of conversation history based on token count.

    This implementation maintains recent conversation history up to a specified token limit,
    rather than a fixed number of messages. It's useful for short-term memory in conversation
    systems where context needs to be managed within token constraints.

    Attributes:
        chat_memory: Storage for the conversation messages
        max_token_limit: Maximum number of tokens to retain (default: DEFAULT_TOKEN_LIMIT)
        k: Legacy parameter for number of messages (used as fallback if token counting fails)
        memory_key: Key to use when returning the memory in load_memory_variables
        return_messages: Whether to return messages directly or as a formatted string
        input_key: Key for identifying the input field in the inputs dictionary
        output_key: Key for identifying the output field in the outputs dictionary
        token_counter: Function to count tokens in messages (defaults to None)
    """

    chat_memory: ChatMessageHistory
    max_token_limit: int = DEFAULT_TOKEN_LIMIT
    k: int = DEFAULT_BUFFER_WINDOW_SIZE  # Fallback if token counting fails
    memory_key: str = "history"
    return_messages: bool = True
    input_key: Optional[str] = None
    output_key: Optional[str] = None
    token_counter: Optional[Callable] = None

    @property
    def memory_variables(self) -> List[str]:
        """
        Return the memory variables that are accessed in this memory.

        Returns:
            List[str]: List containing the memory key used by this memory component
        """
        return [self.memory_key]

    def load_memory_variables(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Load memory variables from the chat history, keeping only the most recent messages.

        This method retrieves the most recent messages from the chat history,
        preserving any system message at the beginning. It uses the trim_messages
        function to select messages while maintaining conversation integrity.
        
        The sliding window approach ensures we only keep the last 4 exchanges (8 messages)
        to prevent context bloat and improve performance.

        Returns:
            Dict[str, Any]: Dictionary with memory_key mapped to either a list of
                           messages or a formatted string (if return_messages=False)

        Raises:
            NotImplementedError: If return_messages=False, as string formatting is
                                not fully implemented
        """
        messages = self.chat_memory.messages
                
        # Use token counter if provided, otherwise fall back to message count
        token_counter_fn = self.token_counter
        max_tokens = self.max_token_limit

        # If no token counter is provided, fall back to message count
        if token_counter_fn is None:
            token_counter_fn = len  # Count messages instead of tokens
            max_tokens = self.k  # Use k as the maximum number of messages

        # Trim messages based on token count or message count
        windowed_messages = trim_messages(
            messages,
            max_tokens=max_tokens,
            token_counter=token_counter_fn,
            strategy="last",
            start_on="human",  # Default, good for most chat models
            include_system=True,  # Preserves first system message if any
            allow_partial=False,  # Don't allow partial messages
        )

        if self.return_messages:
            return {self.memory_key: windowed_messages}
        else:
            # This part mimics how ConversationBufferMemory would format if not return_messages
            # For now, we assume return_messages=True as per original usage
            # If needed, this can be expanded later.
            # For simplicity with return_messages=True, this branch might not be hit
            # if ConversationBufferWindowMemory's string formatting is complex.
            # The primary goal is to replicate the message windowing.
            # buffer_string = get_buffer_string(windowed_messages, human_prefix=..., ai_prefix=...)
            # return {self.memory_key: buffer_string}
            # For now, let's stick to the return_messages=True path as that's what was used.
            raise NotImplementedError(
                "String buffer formatting not fully implemented for CustomBufferWindowMemory if return_messages=False"
            )

    def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, str]) -> None:
        """
        Save the context of this conversation turn to the memory.

        This method extracts the user input and AI output from the provided dictionaries
        and adds them to the chat history. It intelligently handles cases where input_key
        and output_key are specified or when they need to be inferred from the dictionaries.

        Args:
            inputs (Dict[str, Any]): Dictionary containing the user's input message
            outputs (Dict[str, str]): Dictionary containing the AI's output message

        Note:
            This logic assumes inputs has one key (e.g. "input" or "human_input")
            and outputs has one key (e.g. "output" or "ai_output").
            This matches how LLMChain and agents typically save context.
        """
        # Determine input message string
        if self.input_key is None:
            input_str = next(iter(inputs.values()))  # Get the first value
        else:
            input_str = inputs[self.input_key]

        # Determine output message string
        if self.output_key is None:
            output_str = next(iter(outputs.values()))  # Get the first value
        else:
            output_str = outputs[self.output_key]

        self.chat_memory.add_user_message(input_str)
        self.chat_memory.add_ai_message(output_str)

    def clear(self) -> None:
        """
        Clear all messages from the chat memory.

        This method removes all stored messages from the chat history,
        effectively resetting the conversation memory to its initial state.
        """
        self.chat_memory.clear()


MemoryType: TypeAlias = Union[
    CustomBufferWindowMemory, ConversationSummaryBufferMemory, CombinedMemory
]


class MemoryStrategy(Enum):
    """Enum defining available memory strategies for conversation history management."""

    SHORT_TERM = "buffer_memory"
    MEDIUM_TERM = "summarized_memory"
    ADVANCED = "enhanced_context_memory"
    FALLBACK = "fallback_context_memory"


class ConversationMemoryManager:
    """Manages conversation history and memory strategies asynchronously based on token count.

    Dynamically selects memory strategies (short-term, medium-term, advanced, or fallback)
    based on the token count of the conversation history, ensuring efficient memory usage
    within the model's context window.

    Attributes:
        llm: ChatOpenAI instance for language model interactions.
        model_name: Name of the LLM model (e.g., 'gpt-4o').
        llm_token_encoder: LLMTokenCounter instance for counting tokens.
        max_context_tokens: Maximum tokens allowed in the context window.
        fallback_token_limit: Token limit for the fallback memory strategy.
        embeddings: OpenAIEmbeddings instance for vector store initialization.
        history: List of (user_content, assistant_content) tuples.
        llm_token_count: Total token count of the conversation history.
        memory: Current memory object (buffer, summary, or combined).
        memory_strategy: Current memory strategy in use.
    """

    def __init__(self) -> None:
        """Initialize ConversationMemoryManager without dependencies.

        Dependencies must be set via async_initialize to support async initialization.
        """
        self.llm: Optional[ChatOpenAI] = None
        self.model_name: Optional[str] = None
        self.llm_token_encoder: Optional[LLMTokenCounter] = None
        self.max_context_tokens: Optional[int] = None
        self.fallback_token_limit: Optional[int] = None
        self.embeddings: Optional[OpenAIEmbeddings] = None
        self.logger = logger.bind(class_name=self.__class__.__name__)
        self.history: List[MessagePair] = []
        self.llm_token_count: int = 0
        self.memory: Optional[MemoryType] = None
        self.memory_strategy: Optional[MemoryStrategy] = None

    async def async_initialize(
        self,
        llm: ChatOpenAI,
        model_name: str,
        llm_token_encoder: LLMTokenCounter,
        max_context_tokens: int,
        fallback_token_limit: int,
        embeddings: OpenAIEmbeddings,
    ) -> None:
        """Asynchronously initialize ConversationMemoryManager with dependencies.

        Args:
            llm: Initialized ChatOpenAI instance for language model interactions.
            model_name: Name of the LLM model (e.g., 'gpt-4o').
            llm_token_encoder: LLMTokenCounter instance for counting tokens.
            max_context_tokens: Maximum tokens allowed in the context window.
            fallback_token_limit: Token limit for the fallback memory strategy.
            embeddings: Pre-initialized OpenAIEmbeddings instance for vector store.

        Raises:
            ValueError: If configuration is invalid (handled by caller).
        """
        self.llm = llm
        self.model_name = model_name
        self.llm_token_encoder = llm_token_encoder
        self.max_context_tokens = max_context_tokens
        self.fallback_token_limit = fallback_token_limit
        self.embeddings = embeddings
        self.memory, self.memory_strategy = await self._initialize_short_term_memory([])

    async def add_message(self, message_pair: MessagePair) -> None:
        """Add a user-assistant message pair to history and update memory strategy if needed.

        Args:
            message_pair: Tuple of (user_content, assistant_content) strings.

        Raises:
            ValueError: If message pair is invalid, token counting fails, or manager is not initialized.
            asyncio.TimeoutError: If memory update exceeds MEMORY_UPDATE_TIMEOUT_SECONDS.
        """
        try:
            user_content, assistant_content = message_pair
            
            # First, always add the message to the current memory object
            self.memory.chat_memory.add_user_message(user_content)
            self.memory.chat_memory.add_ai_message(assistant_content)

            async def update_memory() -> None:
                msg_tokens: int = self.llm_token_encoder.count_tokens([message_pair])
                self.history.append(message_pair)
                # Keep only the last exchanges for sliding window memory
                if len(self.history) > SLIDING_WINDOW_EXCHANGES:
                    removed = self.history[0]
                    self.history = self.history[-SLIDING_WINDOW_EXCHANGES:]
                    self.logger.info(
                        "Applied sliding window memory - removed oldest message",
                        removed_message=removed[0][:30] + "..." if len(removed[0]) > 30 else removed[0],
                        window_size=SLIDING_WINDOW_EXCHANGES,
                    )
                self.llm_token_count += msg_tokens
                self.logger.debug(
                    "Added message pair to history",
                    user_tokens=self.llm_token_encoder.count_tokens(
                        [(user_content, "")]
                    ),
                    assistant_tokens=self.llm_token_encoder.count_tokens(
                        [("", assistant_content)]
                    ),
                    total_tokens=self.llm_token_count,
                    history_size=len(self.history),
                )

                expected_strategy: MemoryStrategy = self._get_expected_strategy(
                    self.llm_token_count
                )
                if expected_strategy != self.memory_strategy:
                    self.logger.info(
                        "Switching memory strategy",
                        current_strategy=self.memory_strategy.value,
                        new_strategy=expected_strategy.value,
                        llm_token_count=self.llm_token_count,
                    )
                    self.memory, self.memory_strategy = await self._initialize_memory(
                        self.llm_token_count, self.history
                    )

            await asyncio.wait_for(
                asyncio.create_task(update_memory()),
                timeout=MEMORY_UPDATE_TIMEOUT_SECONDS,
            )
        except asyncio.TimeoutError:
            self.logger.error(
                "Memory update timed out", timeout=MEMORY_UPDATE_TIMEOUT_SECONDS
            )
            raise ValueError(
                f"Memory update timed out after {MEMORY_UPDATE_TIMEOUT_SECONDS} seconds"
            )
        except Exception as e:
            self.logger.error(
                "Failed to add message pair",
                message_pair=message_pair,
                error=str(e),
                exc_info=True,
            )
            raise ValueError(f"Failed to add message pair: {str(e)}") from e

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
            llm_token_count=llm_token_count,
        )
        memory = ConversationSummaryBufferMemory(
            llm=self.llm,
            max_token_limit=self.fallback_token_limit,
            memory_key="history",
            chat_memory=ChatMessageHistory(),
            return_messages=True,
        )
        recent_messages: List[MessagePair] = []
        current_tokens: int = 0
        for user_content, assistant_content in reversed(history):
            msg_tokens: int = self.llm_token_encoder.count_tokens(
                [(user_content, assistant_content)]
            )
            if current_tokens + msg_tokens <= self.fallback_token_limit:
                recent_messages.append((user_content, assistant_content))
                current_tokens += msg_tokens
            else:
                break
        recent_messages.reverse()

        def load_messages() -> None:
            for user_content, assistant_content in recent_messages:
                memory.chat_memory.add_user_message(user_content)
                memory.chat_memory.add_ai_message(assistant_content)

        await asyncio.wait_for(
            asyncio.to_thread(load_messages), timeout=MEMORY_UPDATE_TIMEOUT_SECONDS
        )
        self.logger.info("Initialized FALLBACK memory", total_tokens=current_tokens)
        return memory, MemoryStrategy.FALLBACK

    async def _initialize_short_term_memory(
        self, history: List[MessagePair]
    ) -> Tuple[CustomBufferWindowMemory, MemoryStrategy]:
        """
        Initialize short-term memory asynchronously for low token counts.

        Args:
            history: List of (user_content, assistant_content) tuples.

        Returns:
            Tuple[CustomBufferWindowMemory, MemoryStrategy]: Short-term memory and strategy.

        Raises:
            asyncio.TimeoutError: If memory initialization exceeds MEMORY_UPDATE_TIMEOUT_SECONDS.
        """
        self.logger.info(
            "Initializing SHORT_TERM memory", max_token_limit=SHORT_TERM_TOKEN_LIMIT
        )
        memory = CustomBufferWindowMemory(
            chat_memory=ChatMessageHistory(),
            max_token_limit=SHORT_TERM_TOKEN_LIMIT,  # Use token-based limit
            k=DEFAULT_BUFFER_WINDOW_SIZE,  # Fallback if token counting fails
            memory_key="history",  # Explicitly set
            return_messages=True,  # Explicitly set
            token_counter=(
                self.llm_token_encoder.count_message_tokens
                if self.llm_token_encoder
                else None
            ),
        )

        def load_messages() -> None:
            for user_content, assistant_content in history:
                memory.chat_memory.add_user_message(user_content)
                memory.chat_memory.add_ai_message(assistant_content)

        await asyncio.wait_for(
            asyncio.to_thread(load_messages), timeout=MEMORY_UPDATE_TIMEOUT_SECONDS
        )
        self.logger.info("Initialized SHORT_TERM memory", message_count=len(history))
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
        self.logger.info(
            "Initializing MEDIUM_TERM memory", max_token_limit=MEDIUM_TERM_TOKEN_LIMIT
        )
        buffer_memory = CustomBufferWindowMemory(
            chat_memory=ChatMessageHistory(),
            max_token_limit=SHORT_TERM_TOKEN_LIMIT,  # Use token-based limit
            k=DEFAULT_BUFFER_WINDOW_SIZE,  # Fallback if token counting fails
            memory_key="buffer_history",  # Explicitly set
            return_messages=True,  # Explicitly set
            token_counter=(
                self.llm_token_encoder.count_message_tokens
                if self.llm_token_encoder
                else None
            ),
        )
        summary_memory = ConversationSummaryBufferMemory(
            llm=self.llm,
            max_token_limit=SHORT_TERM_TOKEN_LIMIT,
            memory_key="summary_history",
            chat_memory=ChatMessageHistory(),
            return_messages=True,
        )

        def load_messages() -> None:
            for user_content, assistant_content in history:
                buffer_memory.chat_memory.add_user_message(user_content)
                buffer_memory.chat_memory.add_ai_message(assistant_content)
                summary_memory.chat_memory.add_user_message(user_content)
                summary_memory.chat_memory.add_ai_message(assistant_content)

        await asyncio.wait_for(
            asyncio.to_thread(load_messages), timeout=MEMORY_UPDATE_TIMEOUT_SECONDS
        )
        combined_memory = CombinedMemory(memories=[buffer_memory, summary_memory])
        self.logger.info("Initialized MEDIUM_TERM memory", message_count=len(history))
        return combined_memory, MemoryStrategy.MEDIUM_TERM

    async def _initialize_advanced_memory(
        self, history: List[MessagePair]
    ) -> Tuple[CombinedMemory, MemoryStrategy]:
        """Initialize advanced memory asynchronously with entities and vector store for high token counts.

        Creates a FAISS vector store for conversation history using provided embeddings,
        and combines buffer, summary, entity, and vector memory for enhanced context.

        Args:
            history: List of (user_content, assistant_content) tuples.

        Returns:
            Tuple[CombinedMemory, MemoryStrategy]: Advanced memory and strategy.

        Raises:
            asyncio.TimeoutError: If memory initialization exceeds MEMORY_UPDATE_TIMEOUT_SECONDS.
            ValueError: If vector store initialization fails or embeddings are not initialized.
        """
        self.logger.info(
            "Initializing ADVANCED memory", max_token_limit=MEDIUM_TERM_TOKEN_LIMIT
        )

        # Initialize memory components
        buffer_memory = CustomBufferWindowMemory(
            chat_memory=ChatMessageHistory(),
            max_token_limit=SHORT_TERM_TOKEN_LIMIT,  # Use token-based limit
            k=DEFAULT_BUFFER_WINDOW_SIZE,  # Fallback if token counting fails
            memory_key="buffer_history",  # Explicitly set
            return_messages=True,  # Explicitly set
            token_counter=(
                self.llm_token_encoder.count_message_tokens
                if self.llm_token_encoder
                else None
            ),
        )
        summary_memory = ConversationSummaryBufferMemory(
            llm=self.llm,
            max_token_limit=MEDIUM_TERM_TOKEN_LIMIT,
            memory_key="summary_history",
            chat_memory=ChatMessageHistory(),
            return_messages=True,
        )
        entity_memory = ConversationEntityMemory(
            llm=self.llm, memory_key="entity_history", return_messages=True
        )

        # Load messages and build documents
        documents: DocumentList = []

        def load_messages() -> None:
            for user_content, assistant_content in history:
                buffer_memory.chat_memory.add_user_message(user_content)
                buffer_memory.chat_memory.add_ai_message(assistant_content)
                summary_memory.chat_memory.add_user_message(user_content)
                summary_memory.chat_memory.add_ai_message(assistant_content)
                entity_memory.save_context(
                    {"input": user_content}, {"output": assistant_content}
                )
                documents.append(
                    Document(
                        page_content=f"User: {user_content}\nAssistant: {assistant_content}",
                        metadata={
                            "user_content": user_content,
                            "assistant_content": assistant_content,
                        },
                    )
                )

        await asyncio.wait_for(
            asyncio.to_thread(load_messages), timeout=MEMORY_UPDATE_TIMEOUT_SECONDS
        )

        # Create FAISS vector store for conversation history
        try:

            def init_vector_store() -> FAISS:
                if not self.embeddings:
                    raise ValueError("Embeddings not initialized")
                return FAISS.from_documents(documents, self.embeddings)

            vector_store = await asyncio.wait_for(
                asyncio.to_thread(init_vector_store),
                timeout=MEMORY_UPDATE_TIMEOUT_SECONDS,
            )
        except Exception as e:
            self.logger.error(
                "Failed to initialize FAISS vector store", error=str(e), exc_info=True
            )
            raise ValueError(
                f"Failed to initialize vector store for conversation history: {str(e)}"
            ) from e

        # Use TimeWeightedVectorStoreRetriever for conversation history searches
        retriever = TimeWeightedVectorStoreRetriever(
            vectorstore=vector_store,
            decay_rate=VECTOR_RETRIEVER_DECAY_RATE,
            k=VECTOR_RETRIEVER_K,
        )
        vector_memory = retriever.as_memory(
            memory_key="vector_history", return_messages=True
        )

        combined_memory = CombinedMemory(
            memories=[buffer_memory, summary_memory, entity_memory, vector_memory]
        )
        self.logger.info("Initialized ADVANCED memory", message_count=len(history))
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
            ValueError: If manager is not initialized.
        """
        if self.memory is None or self.memory_strategy is None:
            raise ValueError("Memory not initialized")
        return self.memory, self.memory_strategy
