from typing import Dict, Any, Optional, List, Tuple
import os
import structlog
import asyncio
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import Tool
from langchain.agents import create_openai_tools_agent, AgentExecutor
from langchain.callbacks.base import BaseCallbackHandler
from langchain_core.runnables import RunnableMap
import duckdb
from backend.conversation_memory_manager import ConversationMemoryManager
from backend.vector_store_manager import VectorStoreManager
from backend.telemetry_schema import TELEMETRY_SCHEMA

logger = structlog.get_logger(__name__)

# Load environment variables from .env file
load_dotenv()

# Constants
MAX_MODEL_TOKENS: int = 8192
RESERVED_RESPONSE_TOKENS: int = 1024
MAX_TOKEN_SAFETY_LIMIT: int = 16384
FALLBACK_TOKEN_FACTOR: int = 2
LLM_TEMPERATURE: float = 0.0
VECTOR_RETRIEVER_K: int = 4
QUERY_TIMEOUT_SECONDS: float = 30.0  # Timeout for DuckDB queries
CHAT_TIMEOUT_SECONDS: float = 30.0  # Timeout for agent execution
SYSTEM_PROMPT: str = """You are a telemetry data analysis assistant. Use the provided tools to query telemetry data from DuckDB tables.
Tables available: {tables}.

For each question:
1. Identify the relevant table(s) using the vector store.
2. Generate a precise SQL query.
3. Execute the query using the query_duckdb tool.
4. Format the response clearly.

If the question is ambiguous, ask for clarification and suggest possible interpretations.
If data is missing, inform the user and suggest alternatives."""

# Clarification phrases for detecting ambiguous user input (case-insensitive)
CLARIFICATION_PHRASES: List[str] = [
    "could you clarify",
    "can you clarify",
    "can you specify",
    "what do you mean",
    "please clarify",
    "could you specify",
    "more details needed",
    "can you provide more"
]

# Validate token-related constants at module level
if MAX_MODEL_TOKENS <= 0 or RESERVED_RESPONSE_TOKENS < 0:
    logger.error(
        "Invalid token limits",
        max_model_tokens=MAX_MODEL_TOKENS,
        reserved_response_tokens=RESERVED_RESPONSE_TOKENS
    )
    raise ValueError("MAX_MODEL_TOKENS must be positive and RESERVED_RESPONSE_TOKENS non-negative")
if MAX_MODEL_TOKENS <= RESERVED_RESPONSE_TOKENS:
    logger.error(
        "MAX_MODEL_TOKENS must be greater than RESERVED_RESPONSE_TOKENS",
        max_model_tokens=MAX_MODEL_TOKENS,
        reserved_response_tokens=RESERVED_RESPONSE_TOKENS
    )
    raise ValueError(
        f"MAX_MODEL_TOKENS ({MAX_MODEL_TOKENS}) must be greater than "
        f"RESERVED_RESPONSE_TOKENS ({RESERVED_RESPONSE_TOKENS})"
    )
if MAX_TOKEN_SAFETY_LIMIT <= MAX_MODEL_TOKENS:
    logger.error(
        "MAX_TOKEN_SAFETY_LIMIT must be greater than MAX_MODEL_TOKENS",
        max_token_safety_limit=MAX_TOKEN_SAFETY_LIMIT,
        max_model_tokens=MAX_MODEL_TOKENS
    )
    raise ValueError(
        f"MAX_TOKEN_SAFETY_LIMIT ({MAX_TOKEN_SAFETY_LIMIT}) must be greater than "
        f"MAX_MODEL_TOKENS ({MAX_MODEL_TOKENS})"
    )

@Tool.from_function(
    func=None,  # Will define async_run
    name="query_duckdb",
    description="Execute a SQL query on a DuckDB database and return the results."
)
async def query_duckdb(db_path: str, query: str) -> Dict[str, Any]:
    """Execute a SQL query asynchronously on a DuckDB database and return the results.

    Args:
        db_path: Path to the DuckDB database file.
        query: SQL query string to execute.

    Returns:
        Dict[str, Any]: Dictionary containing query results or error details.
            - status: "success" or "error".
            - data: List of dictionaries mapping column names to row values (if success).
            - row_count: Number of rows returned (if success).
            - message: Error message (if error).

    Raises:
        asyncio.TimeoutError: If query execution exceeds QUERY_TIMEOUT_SECONDS.
        ValueError: If db_path or query is invalid.
    """
    try:
        # Prevent path traversal attacks
        if ".." in os.path.relpath(db_path):
            logger.error("Invalid database path detected", db_path=db_path)
            return {"status": "error", "message": "Invalid database path"}

        # Sanitize query (basic check for dangerous keywords)
        query_lower = query.lower()
        if any(keyword in query_lower for keyword in ["drop", "delete", "update", "insert"]):
            logger.error("Potentially dangerous query detected", query=query)
            return {"status": "error", "message": "Query contains unsupported operations"}

        async def run_query() -> Dict[str, Any]:
            with duckdb.connect(db_path, read_only=True) as conn:
                result = conn.execute(query).fetchall()
                columns = [desc[0] for desc in conn.description]
                return {
                    "status": "success",
                    "data": [dict(zip(columns, row)) for row in result],
                    "row_count": len(result)
                }

        # Run synchronous DuckDB query in thread pool
        return await asyncio.wait_for(
            asyncio.to_thread(run_query),
            timeout=QUERY_TIMEOUT_SECONDS
        )

    except asyncio.TimeoutError:
        logger.error("DuckDB query timed out", query=query, timeout=QUERY_TIMEOUT_SECONDS)
        return {"status": "error", "message": f"Query timed out after {QUERY_TIMEOUT_SECONDS} seconds"}
    except duckdb.Error as e:
        logger.error("DuckDB query execution failed", query=query, error=str(e))
        return {"status": "error", "message": f"Query failed: {str(e)}"}
    except Exception as e:
        logger.error("Unexpected error during DuckDB query", query=query, error=str(e), exc_info=True)
        return {"status": "error", "message": f"Unexpected error: {str(e)}"}

class TokenUsageCallback(BaseCallbackHandler):
    """Callback handler to track OpenAI LLM token usage."""
    def __init__(self) -> None:
        self.token_usage: Dict[str, int] = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0
        }

    def reset(self) -> None:
        """Reset token usage counts to zero."""
        self.token_usage = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0
        }
        logger.debug("Reset token usage", callback_id=id(self))

    def on_llm_end(self, response: Any, **kwargs: Any) -> None:
        """Update token usage from LLM response.

        Args:
            response: LLM response object containing token usage data.
            **kwargs: Additional keyword arguments (ignored).
        """
        usage = response.llm_output.get("token_usage", {}) if response.llm_output else {}
        self.token_usage["prompt_tokens"] += usage.get("prompt_tokens", 0)
        self.token_usage["completion_tokens"] += usage.get("completion_tokens", 0)
        self.token_usage["total_tokens"] += usage.get("total_tokens", 0)

class TelemetryAgent:
    """Agent for processing user queries about MAVLink telemetry data stored in DuckDB.

    Uses a language model to interpret queries, identify relevant tables,
    generate SQL queries, and execute them asynchronously against a DuckDB database.

    Attributes:
        session_id: Unique identifier for the session.
        db_path: Path to the DuckDB database file.
        openai_api_key: OpenAI API key for LLM.
        llm_model: Name of the LLM model (e.g., 'gpt-4o').
        token_encoder: TokenEncoder instance for counting tokens.
        max_tokens: Maximum tokens for LLM context window.
        max_context_tokens: Maximum tokens for context (excludes reserved response tokens).
        fallback_token_limit: Token limit for fallback memory strategy.
        embeddings: OpenAIEmbeddings instance for vector store.
        llm: ChatOpenAI instance for language model interactions.
        prompt: ChatPromptTemplate for structuring LLM input.
        conversation_memory_manager: Manager for conversation history and memory strategies.
        vector_store_manager: Manager for FAISS vector store operations.
        token_callback: Callback to track token usage.
        agent_chain: Runnable chain for processing queries with memory.
    """

    def __init__(
        self,
        session_id: str,
        db_path: str,
        openai_api_key: str,
        llm_model: str,
        token_encoder: Any,
        vector_store_manager: VectorStoreManager,
        embeddings: OpenAIEmbeddings,
        max_tokens: Optional[int] = None
    ) -> None:
        """Initialize TelemetryAgent with session and configuration details.

        Args:
            session_id: Unique identifier for the session.
            db_path: Path to the DuckDB database file.
            openai_api_key: OpenAI API key for LLM.
            llm_model: Name of the LLM model (e.g., 'gpt-4o').
            token_encoder: TokenEncoder instance for counting tokens.
            vector_store_manager: Manager for FAISS vector store operations.
            embeddings: Pre-initialized OpenAIEmbeddings instance.
            max_tokens: Maximum tokens for LLM context window (defaults to None).

        Note:
            Call async_initialize to complete initialization of dependencies.
        """
        self.session_id: str = session_id
        self.db_path: str = db_path
        self.openai_api_key: str = openai_api_key
        self.llm_model: str = llm_model
        self.token_encoder: Any = token_encoder
        self.vector_store_manager: VectorStoreManager = vector_store_manager
        self.embeddings: OpenAIEmbeddings = embeddings
        self.max_tokens: Optional[int] = max_tokens
        self.max_context_tokens: Optional[int] = None
        self.fallback_token_limit: Optional[int] = None
        self.llm: Optional[ChatOpenAI] = None
        self.prompt: Optional[ChatPromptTemplate] = None
        self.conversation_memory_manager: Optional[ConversationMemoryManager] = None
        self.token_callback: Optional[TokenUsageCallback] = None
        self.agent_chain: Optional[Any] = None
        self.logger = logger.bind(session_id=session_id)

    async def async_initialize(self) -> None:
        """Asynchronously initialize TelemetryAgent dependencies.

        Initializes token limits, LLM, prompt, conversation memory manager,
        token callback, and agent chain.

        Raises:
            ValueError: If initialization fails due to invalid configuration.
        """
        # Set token limits
        self.max_tokens = self.max_tokens if self.max_tokens is not None else MAX_MODEL_TOKENS
        self.max_context_tokens = self.max_tokens - RESERVED_RESPONSE_TOKENS
        self.fallback_token_limit = self.max_context_tokens // FALLBACK_TOKEN_FACTOR

        # Validate token configurations
        await asyncio.to_thread(self._validate_token_limits)

        # Initialize token usage callback
        self.token_callback = TokenUsageCallback()

        # Initialize LLM with configured settings
        self.llm = ChatOpenAI(
            model=self.llm_model,
            api_key=self.openai_api_key,
            temperature=LLM_TEMPERATURE,
            max_tokens=self.max_tokens,
            callbacks=[self.token_callback]
        )

        # Define system prompt for LLM
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}")
        ])

        # Initialize conversation memory manager
        self.conversation_memory_manager = ConversationMemoryManager()
        await self.conversation_memory_manager.async_initialize(
            llm=self.llm,
            model_name=self.llm_model,
            llm_token_encoder=self.token_encoder,
            max_context_tokens=self.max_context_tokens,
            fallback_token_limit=self.fallback_token_limit,
            embeddings=self.embeddings
        )

        # Initialize agent chain with memory
        memory, _ = await self.conversation_memory_manager.aget_memory()
        self.agent_chain = RunnableMap({
            "input": lambda x: x["input"],
            "chat_history": lambda _: memory.load_memory_variables({})["chat_history"],
            "tables": lambda _: self.get_tables_as_string(),
        }) | self.prompt | self.llm

    def _validate_token_limits(self) -> None:
        """Validate token-related configurations.

        Raises:
            ValueError: If token limits are invalid or unsafe.
        """
        if self.max_tokens <= 0:
            self.logger.error("max_tokens must be positive", max_tokens=self.max_tokens)
            raise ValueError(f"Invalid max_tokens: {self.max_tokens} must be positive")
        if self.max_context_tokens <= 0:
            self.logger.error("max_context_tokens must be positive", max_context_tokens=self.max_context_tokens)
            raise ValueError(f"Invalid max_context_tokens: {self.max_context_tokens} must be positive")
        if self.fallback_token_limit <= 0:
            self.logger.error("fallback_token_limit must be positive", fallback_token_limit=self.fallback_token_limit)
            raise ValueError(f"Invalid fallback_token_limit: {self.fallback_token_limit} must be positive")
        if self.max_tokens > MAX_TOKEN_SAFETY_LIMIT:
            self.logger.error(
                "max_tokens exceeds safety limit",
                max_tokens=self.max_tokens,
                safety_limit=MAX_TOKEN_SAFETY_LIMIT
            )
            raise ValueError(f"max_tokens {self.max_tokens} exceeds safety limit {MAX_TOKEN_SAFETY_LIMIT}")
        if self.max_tokens <= RESERVED_RESPONSE_TOKENS:
            self.logger.error(
                "max_tokens must be greater than RESERVED_RESPONSE_TOKENS",
                max_tokens=self.max_tokens,
                reserved=RESERVED_RESPONSE_TOKENS
            )
            raise ValueError(
                f"max_tokens {self.max_tokens} must be greater than RESERVED_RESPONSE_TOKENS {RESERVED_RESPONSE_TOKENS}"
            )

    async def get_tables_as_string(self) -> str:
        """Generate a comma-separated string of available table names asynchronously.

        Returns:
            str: Comma-separated list of table names from TELEMETRY_SCHEMA.
        """
        # Lightweight operation, yield to event loop for consistency
        await asyncio.sleep(0)
        return ", ".join(meta["table"] for meta in TELEMETRY_SCHEMA)

    async def process_message(
        self, message: str, max_tokens: Optional[int] = None
    ) -> Tuple[str, Optional[Dict[str, Any]]]:
        """Process a user message asynchronously and generate a response with metadata.

        Args:
            message: User input message to process.
            max_tokens: Optional maximum tokens for the LLM response (defaults to self.max_tokens).

        Returns:
            Tuple[str, Optional[Dict[str, Any]]]: Response message and metadata dictionary containing:
                - relevant_tables: List of relevant table names.
                - query: SQL query executed (if any).
                - token_usage: Token usage statistics.
                - memory_strategy: Current memory strategy used.
                - is_clarification: Whether the response requests clarification.

        Raises:
            ValueError: If message processing fails, token limits are invalid, or agent is not initialized.
            asyncio.TimeoutError: If processing exceeds CHAT_TIMEOUT_SECONDS.
        """
        if not message.strip():
            self.logger.warning("Empty message provided, skipping processing")
            return "", None

        try:
            # Determine effective token limit
            effective_max_tokens: int = max_tokens if max_tokens is not None else self.max_tokens

            # Validate token limits
            if effective_max_tokens <= 0:
                self.logger.error("max_tokens must be positive", max_tokens=effective_max_tokens)
                raise ValueError(f"Invalid max_tokens: {effective_max_tokens} must be positive")
            if effective_max_tokens > MAX_TOKEN_SAFETY_LIMIT:
                self.logger.error(
                    "max_tokens exceeds safety limit",
                    max_tokens=effective_max_tokens,
                    safety_limit=MAX_TOKEN_SAFETY_LIMIT
                )
                raise ValueError(f"max_tokens {effective_max_tokens} exceeds safety limit {MAX_TOKEN_SAFETY_LIMIT}")
            if effective_max_tokens <= RESERVED_RESPONSE_TOKENS:
                self.logger.error(
                    "max_tokens must be greater than RESERVED_RESPONSE_TOKENS",
                    max_tokens=effective_max_tokens,
                    reserved=RESERVED_RESPONSE_TOKENS
                )
                raise ValueError(
                    f"max_tokens {effective_max_tokens} must be greater than "
                    f"RESERVED_RESPONSE_TOKENS {RESERVED_RESPONSE_TOKENS}"
                )

            # Reset token usage tracking
            self.token_callback.reset()

            # Retrieve current memory and strategy asynchronously
            memory, memory_strategy = await self.conversation_memory_manager.aget_memory()

            # Perform vector store similarity search asynchronously
            search_results = await self.vector_store_manager.asimilarity_search(
                message, k=VECTOR_RETRIEVER_K
            )
            relevant_tables: List[str] = [result.metadata.get("table", "") for result in search_results if result.metadata.get("table")]
            self.logger.info("Identified relevant tables", tables=relevant_tables)

            # Initialize agent with tools and memory
            tools = [query_duckdb]
            agent = create_openai_tools_agent(
                llm=self.llm,
                tools=tools,
                prompt=self.prompt
            )
            executor = AgentExecutor(
                agent=agent,
                tools=tools,
                memory=memory,
                verbose=False
            )

            # Prepare input data for agent execution
            input_data = {
                "input": f"{message}\nRelevant tables: {', '.join(relevant_tables)}\nDatabase path: {self.db_path}",
            }

            # Execute agent asynchronously
            async def run_executor() -> Dict[str, Any]:
                return await executor.ainvoke({
                    "input": input_data["input"],
                    "chat_history": memory.load_memory_variables({})["chat_history"],
                    "tables": await self.get_tables_as_string()
                })

            result = await asyncio.wait_for(
                run_executor(),
                timeout=CHAT_TIMEOUT_SECONDS
            )

            # Process response and check for clarification needs
            response: str = result["output"]
            response_lower = response.strip().lower()
            is_clarification: bool = (
                response.strip().endswith("?") or
                any(phrase in response_lower for phrase in CLARIFICATION_PHRASES)
            )
            if is_clarification:
                self.logger.info("Response requests clarification", response=response)
                response += "\nPlease provide more details to proceed."

            # Update memory with the message and response asynchronously
            await self.conversation_memory_manager.add_message((message, response))

            # Compile metadata for the response
            metadata: Dict[str, Any] = {
                "relevant_tables": relevant_tables,
                "query": (
                    result.get("intermediate_steps", [{}])[-1].get("query", "")
                    if result.get("intermediate_steps") else ""
                ),
                "token_usage": self.token_callback.token_usage,
                "memory_strategy": memory_strategy.value,
                "is_clarification": is_clarification
            }

            self.logger.info(
                "Generated response",
                response_length=len(response),
                metadata=metadata
            )
            return response, metadata

        except asyncio.TimeoutError:
            self.logger.error(
                "Message processing timed out",
                message=message,
                timeout=CHAT_TIMEOUT_SECONDS
            )
            raise ValueError(f"Message processing timed out after {CHAT_TIMEOUT_SECONDS} seconds")
        except Exception as e:
            self.logger.error(
                "Failed to process message",
                message=message,
                error=str(e),
                exc_info=True
            )
            raise ValueError(f"Failed to process message: {str(e)}") from e