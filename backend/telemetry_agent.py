from typing import Dict, Any, Optional, List, Tuple, Union
import os
import structlog
import asyncio
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
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
MAX_MODEL_TOKENS: int = 8192  # Maximum tokens for LLM input/output
RESERVED_RESPONSE_TOKENS: int = 1024  # Tokens reserved for LLM response
MAX_TOKEN_SAFETY_LIMIT: int = 16384  # Absolute token limit for safety
FALLBACK_TOKEN_FACTOR: int = 2  # Factor to reduce context tokens if needed
LLM_TEMPERATURE: float = 0.0  # LLM temperature for deterministic responses
VECTOR_RETRIEVER_K: int = 4  # Number of vector store search results
QUERY_TIMEOUT_SECONDS: float = 30.0  # Timeout for DuckDB queries
CHAT_TIMEOUT_SECONDS: float = 60.0  # Timeout for agent processing, including ML
SYSTEM_PROMPT: str = """You are a telemetry data analysis assistant specialized in detecting flight anomalies in MAVLink data. Use the provided tools to query telemetry data from DuckDB tables and analyze it for anomalies.
Tables available: {tables}.

For each question:
1. Identify relevant table(s) using the vector store, which includes anomaly detection hints (e.g., "check sudden altitude changes").
2. Generate a precise SQL query using the query_duckdb tool to fetch relevant data.
3. Use the analyze_stats tool to compute statistical metrics (e.g., Z-scores, differences) for anomaly detection.
4. Use the detect_anomalies_ml tool for unsupervised anomaly detection on numerical columns.
5. Reason about patterns, thresholds, and inconsistencies dynamically, avoiding hardcoded rules.
6. For high-level questions (e.g., "Are there any anomalies in this flight?"), query multiple tables, analyze key columns (e.g., roll, altitude, battery voltage), and summarize findings.
7. Format the response clearly, explaining detected anomalies and their potential impact.

Anomaly detection strategies:
- Look for sudden changes in values (e.g., altitude drops, battery voltage decreases).
- Identify outliers using Z-scores or machine learning (e.g., unusual roll/pitch combinations).
- Check for inconsistencies (e.g., poor GPS fix, high EKF variances).
- Correlate anomalies across tables (e.g., battery issues coinciding with attitude changes).

If the question is ambiguous, ask for clarification and suggest possible interpretations.
If data is missing, inform the user and suggest alternatives."""
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

# Validate token-related constants at module load
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

async def query_duckdb(db_path: str, query: str) -> Dict[str, Any]:
    """Execute a SQL query asynchronously on a DuckDB database.

    Args:
        db_path (str): Path to the DuckDB database file.
        query (str): SQL query to execute.

    Returns:
        Dict[str, Any]: Query results or error details.
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

        # Block dangerous SQL operations
        query_lower: str = query.lower()
        if any(keyword in query_lower for keyword in ["drop", "delete", "update", "insert"]):
            logger.error("Potentially dangerous query detected", query=query)
            return {"status": "error", "message": "Query contains unsupported operations"}

        async def run_query() -> Dict[str, Any]:
            # Use read-only connection to ensure data safety
            with duckdb.connect(db_path, read_only=True) as conn:
                result: List[Tuple] = conn.execute(query).fetchall()
                columns: List[str] = [desc[0] for desc in conn.description]
                return {
                    "status": "success",
                    "data": [dict(zip(columns, row)) for row in result],
                    "row_count": len(result)
                }

        # Run query in a thread to avoid blocking the event loop
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
        logger.error("Unexpected error during DuckDB query", query=query, error=str(e))
        return {"status": "error", "message": f"Unexpected error: {str(e)}"}

# Register query_duckdb as a LangChain tool
query_duckdb_tool: Tool = Tool.from_function(
    func=query_duckdb,
    name="query_duckdb",
    description="Execute a SQL query on a DuckDB database and return the results."
)

async def analyze_stats(db_path: str, table: str, columns: List[str], metrics: List[str]) -> Dict[str, Any]:
    """Compute statistical metrics for anomaly detection on telemetry data.

    Args:
        db_path (str): Path to the DuckDB database file.
        table (str): Name of the telemetry table (e.g., 'telemetry_attitude').
        columns (List[str]): Columns to analyze (e.g., ['roll', 'pitch']).
        metrics (List[str]): Metrics to compute ('z_score', 'difference', 'mean', 'std').

    Returns:
        Dict[str, Any]: Statistical results or error details.
            - status: 'success' or 'error'.
            - data: List of dictionaries with metric results per column.
            - row_count: Number of rows analyzed (if success).
            - message: Error message (if error).

    Raises:
        asyncio.TimeoutError: If analysis exceeds QUERY_TIMEOUT_SECONDS.
        ValueError: If db_path, table, or columns are invalid.
    """
    try:
        # Prevent path traversal attacks
        if ".." in os.path.relpath(db_path):
            logger.error("Invalid database path detected", db_path=db_path)
            return {"status": "error", "message": "Invalid database path"}

        async def compute_stats() -> Dict[str, Any]:
            with duckdb.connect(db_path, read_only=True) as conn:
                # Validate table and columns
                available_columns: List[str] = [col[0] for col in conn.execute(f"DESCRIBE {table}").fetchall()]
                valid_columns: List[str] = [col for col in columns if col in available_columns]
                if not valid_columns:
                    return {"status": "error", "message": f"No valid columns in {table}: {columns}"}

                # Fetch data, including timestamp if available
                query_columns: List[str] = valid_columns + ['timestamp'] if 'timestamp' in available_columns else valid_columns
                query: str = f"SELECT {', '.join(query_columns)} FROM {table}"
                df: pd.DataFrame = conn.execute(query).fetchdf()

                # Compute metrics for each column
                results: List[Dict[str, Any]] = []
                for col in valid_columns:
                    result: Dict[str, Any] = {"column": col}
                    if 'mean' in metrics:
                        result["mean"] = float(df[col].mean()) if pd.api.types.is_numeric_dtype(df[col]) else None
                    if 'std' in metrics:
                        result["std"] = float(df[col].std()) if pd.api.types.is_numeric_dtype(df[col]) else None
                    if 'z_score' in metrics and pd.api.types.is_numeric_dtype(df[col]):
                        mean: float = df[col].mean()
                        std: float = df[col].std()
                        if std > 0:
                            result["z_scores"] = [float(z) for z in (df[col] - mean) / std]
                    if 'difference' in metrics and pd.api.types.is_numeric_dtype(df[col]):
                        result["differences"] = [float(d) for d in df[col].diff().abs()]
                    results.append(result)

                return {"status": "success", "data": results, "row_count": len(df)}

        return await asyncio.wait_for(
            asyncio.to_thread(compute_stats),
            timeout=QUERY_TIMEOUT_SECONDS
        )

    except asyncio.TimeoutError:
        logger.error("Statistical analysis timed out", table=table, timeout=QUERY_TIMEOUT_SECONDS)
        return {"status": "error", "message": f"Analysis timed out after {QUERY_TIMEOUT_SECONDS} seconds"}
    except Exception as e:
        logger.error("Statistical analysis failed", table=table, error=str(e))
        return {"status": "error", "message": f"Analysis failed: {str(e)}"}

# Register analyze_stats as a LangChain tool
analyze_stats_tool: Tool = Tool.from_function(
    func=analyze_stats,
    name="analyze_stats",
    description="Compute statistical metrics (e.g., Z-scores, differences) on telemetry data for anomaly detection."
)

async def detect_anomalies_ml(db_path: str, table: str, columns: List[str]) -> Dict[str, Any]:
    """Detect anomalies in telemetry data using Isolation Forest.

    Args:
        db_path (str): Path to the DuckDB database file.
        table (str): Name of the telemetry table.
        columns (List[str]): Numerical columns to analyze.

    Returns:
        Dict[str, Any]: Anomaly detection results or error details.
            - status: 'success' or 'error'.
            - data: List of dictionaries with row indices and anomaly flags (True for anomalies).
            - row_count: Number of rows analyzed (if success).
            - message: Error message (if error).

    Raises:
        asyncio.TimeoutError: If detection exceeds QUERY_TIMEOUT_SECONDS.
        ValueError: If db_path, table, or columns are invalid.
    """
    try:
        # Prevent path traversal attacks
        if ".." in os.path.relpath(db_path):
            logger.error("Invalid database path detected", db_path=db_path)
            return {"status": "error", "message": "Invalid database path"}

        async def run_ml() -> Dict[str, Any]:
            with duckdb.connect(db_path, read_only=True) as conn:
                # Validate numerical columns
                available_columns: List[str] = [col[0] for col in conn.execute(f"DESCRIBE {table}").fetchall()]
                valid_columns: List[str] = [
                    col for col in columns
                    if col in available_columns and pd.api.types.is_numeric_dtype(
                        conn.execute(f"SELECT {col} FROM {table} LIMIT 1").fetchdf()[col]
                    )
                ]
                if not valid_columns:
                    return {"status": "error", "message": f"No valid numerical columns in {table}: {columns}"}

                # Fetch data
                query: str = f"SELECT {', '.join(valid_columns)} FROM {table}"
                df: pd.DataFrame = conn.execute(query).fetchdf()
                if df.empty:
                    return {"status": "success", "data": [], "row_count": 0}

                # Fit Isolation Forest model
                model: IsolationForest = IsolationForest(contamination=0.1, random_state=42)
                predictions: np.ndarray = model.fit_predict(df[valid_columns])
                anomalies: np.ndarray = predictions == -1  # -1 indicates anomaly

                # Format results
                results: List[Dict[str, Any]] = [
                    {"row_index": i, "is_anomaly": bool(a)} for i, a in enumerate(anomalies)
                ]
                return {
                    "status": "success",
                    "data": results,
                    "row_count": len(df)
                }

        return await asyncio.wait_for(
            asyncio.to_thread(run_ml),
            timeout=QUERY_TIMEOUT_SECONDS
        )

    except asyncio.TimeoutError:
        logger.error("ML anomaly detection timed out", table=table, timeout=QUERY_TIMEOUT_SECONDS)
        return {"status": "error", "message": f"Detection timed out after {QUERY_TIMEOUT_SECONDS} seconds"}
    except Exception as e:
        logger.error("ML anomaly detection failed", table=table, error=str(e))
        return {"status": "error", "message": f"Detection failed: {str(e)}"}

# Register detect_anomalies_ml as a LangChain tool
detect_anomalies_ml_tool: Tool = Tool.from_function(
    func=detect_anomalies_ml,
    name="detect_anomalies_ml",
    description="Detect anomalies in telemetry data using an unsupervised machine learning model (Isolation Forest)."
)

class TokenUsageCallback(BaseCallbackHandler):
    """Callback handler to track OpenAI LLM token usage."""

    def __init__(self) -> None:
        """Initialize the callback with zeroed token counts."""
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
            response (Any): LLM response object containing token usage data.
            kwargs (Any): Additional keyword arguments (ignored).
        """
        usage: Dict[str, int] = response.llm_output.get("token_usage", {}) if response.llm_output else {}
        self.token_usage["prompt_tokens"] += usage.get("prompt_tokens", 0)
        self.token_usage["completion_tokens"] += usage.get("completion_tokens", 0)
        self.token_usage["total_tokens"] += usage.get("total_tokens", 0)

class TelemetryAgent:
    """Agent for analyzing MAVLink telemetry data and detecting flight anomalies."""

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
        """Initialize the TelemetryAgent with required dependencies.

        Args:
            session_id (str): Unique identifier for the session.
            db_path (str): Path to the DuckDB database file.
            openai_api_key (str): API key for OpenAI services.
            llm_model (str): Name of the LLM model to use.
            token_encoder (Any): Token encoder for context management.
            vector_store_manager (VectorStoreManager): Manager for FAISS vector store.
            embeddings (OpenAIEmbeddings): Embeddings for vector store.
            max_tokens (Optional[int]): Maximum tokens for LLM responses (default: None).
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
        self.agent_chain: Optional[RunnableMap] = None
        self.logger: structlog.stdlib.BoundLogger = logger.bind(session_id=session_id)

    async def async_initialize(self) -> None:
        """Initialize TelemetryAgent dependencies asynchronously.

        Args:
            None

        Raises:
            ValueError: If token limits or configurations are invalid.
        """
        # Set and validate token limits
        self.max_tokens = self.max_tokens if self.max_tokens is not None else MAX_MODEL_TOKENS
        self.max_context_tokens = self.max_tokens - RESERVED_RESPONSE_TOKENS
        self.fallback_token_limit = self.max_context_tokens // FALLBACK_TOKEN_FACTOR
        await asyncio.to_thread(self._validate_token_limits)

        # Initialize token usage tracking
        self.token_callback = TokenUsageCallback()

        # Initialize LLM with deterministic settings
        self.llm = ChatOpenAI(
            model=self.llm_model,
            api_key=self.openai_api_key,
            temperature=LLM_TEMPERATURE,
            max_tokens=self.max_tokens,
            callbacks=[self.token_callback]
        )

        # Define prompt with system instructions and chat history
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

        # Set up agent chain with tools and memory
        memory, _ = await self.conversation_memory_manager.aget_memory()
        tools: List[Tool] = [query_duckdb_tool, analyze_stats_tool, detect_anomalies_ml_tool]
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
        self.agent_chain = RunnableMap({
            "input": lambda x: x["input"],
            "chat_history": lambda _: memory.load_memory_variables({})["chat_history"],
            "tables": lambda _: self.get_tables_as_string(),
        }) | self.prompt | executor

    def _validate_token_limits(self) -> None:
        """Validate token-related configurations.

        Args:
            None

        Raises:
            ValueError: If token limits are invalid or exceed safety thresholds.
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
        """Generate a comma-separated string of available table names.

        Args:
            None

        Returns:
            str: Comma-separated list of table names from TELEMETRY_SCHEMA.
        """
        # Yield to event loop to maintain async responsiveness
        await asyncio.sleep(0)
        return ", ".join(meta["table"] for meta in TELEMETRY_SCHEMA)

    async def process_message(
        self, message: str, max_tokens: Optional[int] = None
    ) -> Tuple[str, Optional[Dict[str, Any]]]:
        """Process a user message and generate a response with anomaly detection results.

        Args:
            message (str): User query about telemetry data (e.g., "Are there any GPS anomalies?").
            max_tokens (Optional[int]): Maximum tokens for LLM response (defaults to self.max_tokens).

        Returns:
            Tuple[str, Optional[Dict[str, Any]]]: LLM response and optional metadata.

        Raises:
            ValueError: If message is empty, token limits are invalid, or agent is not initialized.
            asyncio.TimeoutError: If processing exceeds CHAT_TIMEOUT_SECONDS.
        """
        if not message.strip():
            self.logger.warning("Empty message provided, skipping processing")
            return "", None

        try:
            # Use provided max_tokens or default
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

            # Reset token usage for this query
            self.token_callback.reset()

            # Retrieve conversation memory
            memory, memory_strategy = await self.conversation_memory_manager.aget_memory()

            # Identify relevant tables and hints using vector store
            search_results = await self.vector_store_manager.asimilarity_search(
                message, k=VECTOR_RETRIEVER_K
            )
            relevant_tables: List[str] = [
                result.metadata.get("table", "") for result in search_results if result.metadata.get("table")
            ]
            anomaly_hints: List[str] = [
                result.metadata.get("anomaly_hint", "") for result in search_results if result.metadata.get("anomaly_hint")
            ]
            self.logger.info("Identified relevant tables and hints", tables=relevant_tables, hints=anomaly_hints)

            # Prepare input with context for LLM
            input_data: Dict[str, str] = {
                "input": f"{message}\nRelevant tables: {', '.join(relevant_tables)}\nAnomaly hints: {', '.join(anomaly_hints)}\nDatabase path: {self.db_path}",
            }

            # Execute agent chain
            async def run_executor() -> Dict[str, Any]:
                return await self.agent_chain.ainvoke({
                    "input": input_data["input"],
                    "chat_history": memory.load_memory_variables({})["chat_history"],
                    "tables": await self.get_tables_as_string()
                })

            result: Dict[str, Any] = await asyncio.wait_for(
                run_executor(),
                timeout=CHAT_TIMEOUT_SECONDS
            )

            # Process response and check for clarification
            response: str = result["output"]
            response_lower: str = response.strip().lower()
            is_clarification: bool = (
                response.strip().endswith("?") or
                any(phrase in response_lower for phrase in CLARIFICATION_PHRASES)
            )
            if is_clarification:
                self.logger.info("Response requests clarification", response=response)
                response += "\nPlease provide more details to proceed."

            # Update conversation memory
            await self.conversation_memory_manager.add_message((message, response))

            # Compile metadata
            metadata: Dict[str, Any] = {
                "relevant_tables": relevant_tables,
                "anomaly_hints": anomaly_hints,
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
                error=str(e)
            )
            raise ValueError(f"Failed to process message: {str(e)}") from e