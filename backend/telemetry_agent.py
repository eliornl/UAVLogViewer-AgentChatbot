from typing import Dict, Any, Optional, List, Tuple
import os
from pydantic import BaseModel, Field, model_validator
import structlog
import asyncio
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate
from langchain_core.tools import Tool, StructuredTool
from langchain.agents import create_openai_tools_agent, AgentExecutor, create_react_agent
from langchain.callbacks.base import BaseCallbackHandler
from langchain_core.runnables import RunnableLambda
import duckdb
import json
from backend.agent_scratchpad import AgentScratchpad
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

REACT_SYSTEM_PROMPT = """You are a telemetry data analysis assistant specialized in detecting flight anomalies in MAVLink data. 

Tables available (including column names in parentheses): {tables}

You have access to the following tools:
{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action (must be valid JSON)
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

IMPORTANT: When using tools, the Action Input must be valid JSON format. Examples:

For query_duckdb:
Action Input: {{"db_path": "/path/to/database.duckdb", "query": "SELECT AVG(altitude) FROM table_name"}}

For analyze_stats:
Action Input: {{"db_path": "/path/to/database.duckdb", "table": "table_name", "columns": ["col1", "col2"], "metrics": ["mean", "std"]}}

For detect_anomalies_ml:
Action Input: {{"db_path": "/path/to/database.duckdb", "table": "table_name", "columns": ["col1", "col2"]}}

For each question:
1. Consult the 'Tables available' list to identify relevant table(s) AND THEIR EXACT COLUMN NAMES. The format is table_name(column1, column2, ...).
2. CRITICAL: Use ONLY the column names explicitly listed for a table in your SQL queries. DO NOT assume common names like 'timestamp' if not listed for that specific table. Verify against the list before every query.
3. Generate a precise SQL query using the query_duckdb tool to fetch relevant data.
4. Use the analyze_stats tool to compute statistical metrics (e.g., Z-scores, differences) for anomaly detection.
5. Use the detect_anomalies_ml tool for unsupervised anomaly detection on numerical columns.
6. Reason about patterns, thresholds, and inconsistencies dynamically, avoiding hardcoded rules.
7. For high-level questions (e.g., "Are there any anomalies in this flight?"), query multiple tables, analyze key columns (e.g., roll, altitude, battery voltage), and summarize findings.
8. Format the response clearly, explaining detected anomalies and their potential impact.

Anomaly detection strategies:
- Look for sudden changes in values (e.g., altitude drops, battery voltage decreases).
- Identify outliers using Z-scores or machine learning (e.g., unusual roll/pitch combinations).
- Check for inconsistencies (e.g., poor GPS fix, high EKF variances).
- Correlate anomalies across tables (e.g., battery issues coinciding with attitude changes).

If the question is ambiguous, ask for clarification and suggest possible interpretations.
If data is missing, inform the user and suggest alternatives.

Scratchpad (previous analysis context):
{agent_scratchpad_content}

Begin!

Question: {input}
Thought: {agent_scratchpad}"""

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

# Pydantic models for the tools' input arguments
class QueryDuckDBInput(BaseModel):
    db_path: str = Field(description="Path to the DuckDB database file")
    query: str = Field(description="SQL query to execute")

    @model_validator(mode='before')
    @classmethod
    def _handle_potentially_nested_json_input(cls, data: Any) -> Any:
        if isinstance(data, dict):
            # Check if 'query' is missing and 'db_path' holds a string,
            # which matches the observed error pattern.
            if 'query' not in data and 'db_path' in data and isinstance(data['db_path'], str):
                db_path_value = data['db_path']
                try:
                    # Attempt to parse the string in 'db_path' as JSON
                    parsed_json_from_db_path = json.loads(db_path_value)
                    # Check if the parsed JSON is a dict and contains both 'db_path' and 'query'
                    if isinstance(parsed_json_from_db_path, dict) and \
                       'db_path' in parsed_json_from_db_path and \
                       'query' in parsed_json_from_db_path:
                        logger.info(
                            f"{cls.__name__}: Input data appeared to have JSON string in 'db_path'. Expanding.",
                            original_data_keys=list(data.keys()),
                            parsed_json_keys=list(parsed_json_from_db_path.keys())
                        )
                        # Return the parsed JSON dictionary for Pydantic to use
                        return parsed_json_from_db_path
                except json.JSONDecodeError:
                    # 'db_path' value was a string but not valid JSON, or not the structure we expected.
                    # Log this, but let Pydantic handle the original 'data'.
                    # This will likely lead to the original validation error if 'query' is truly missing.
                    logger.warning(
                        f"{cls.__name__}: 'db_path' contained a string that was not parsable " +
                        "into the expected {{'db_path': ..., 'query': ...}} structure.",
                        db_path_value_snippet=db_path_value[:100] # Log a snippet for diagnosis
                    )
                    # Fall through to return original data, allowing standard Pydantic validation to proceed
                    pass
        # If not the specific scenario identified, or if correction failed, return data as is.
        return data

def query_duckdb(tool_input: Any) -> Dict[str, Any]:
    """Execute a SQL query on a DuckDB database.

    Args:
        tool_input (Any): Raw input from the agent, expected to be a JSON string or a dictionary
                          containing db_path and query.

    Returns:
        Dict[str, Any]: Query results or error details.
            - status: "success" or "error".
            - data: List of dictionaries mapping column names to row values (if success).
            - row_count: Number of rows returned (if success).
            - message: Error message (if error).
    """
    logger.info(
        "query_duckdb received raw tool_input", 
        input_type=str(type(tool_input)), 
        input_value_snippet=str(tool_input)[:200]
    )

    args_dict: Optional[Dict[str, Any]] = None
    if isinstance(tool_input, str):
        try:
            args_dict = json.loads(tool_input)
        except json.JSONDecodeError as e:
            logger.error("Failed to decode tool_input string as JSON", error_str=str(e), tool_input_str=tool_input)
            return {"status": "error", "message": f"Invalid JSON input string: {str(e)}"}
    elif isinstance(tool_input, dict):
        args_dict = tool_input
    else:
        logger.error("tool_input is not a string or dict", received_type=str(type(tool_input)))
        return {"status": "error", "message": "Tool input must be a JSON string or a dictionary."}

    if args_dict is None: # Should not happen if logic above is correct, but as a safeguard
        logger.error("args_dict is None after input processing, which is unexpected.")
        return {"status": "error", "message": "Internal error: Failed to process tool input."}

    try:
        # Validate the dictionary using Pydantic. 
        # The model_validator in QueryDuckDBInput will handle nested JSON if necessary.
        parsed_args = QueryDuckDBInput.model_validate(args_dict)
        db_path = parsed_args.db_path
        query = parsed_args.query
    except Exception as pydantic_exc: # Catch Pydantic ValidationError or other issues
        logger.error(
            "Pydantic validation failed for derived args_dict", 
            args_dict_val=str(args_dict)[:500], # Log potentially large dict snippet
            error_str=str(pydantic_exc)
        )
        return {"status": "error", "message": f"Input validation failed: {str(pydantic_exc)}"}

    try:
        # Prevent path traversal attacks
        if ".." in os.path.relpath(db_path):
            logger.error("Invalid database path detected", db_path=db_path)
            return {"status": "error", "message": "Invalid database path"}

        # Block dangerous SQL operations
        query_lower: str = query.lower().strip()
        dangerous_keywords = ["drop", "delete", "update", "insert", "alter", "create", "truncate"]
        if any(keyword in query_lower for keyword in dangerous_keywords):
            logger.error("Potentially dangerous query detected", query=query)
            return {"status": "error", "message": "Query contains unsupported operations"}
        
        with duckdb.connect(db_path, read_only=True) as conn:
            result: List[tuple] = conn.execute(query).fetchall()
            columns: List[str] = [desc[0] for desc in conn.description] if conn.description else []
            
            return {
                "status": "success",
                "data": [dict(zip(columns, row)) for row in result],
                "row_count": len(result)
            }
    except KeyError as e: # Should ideally be caught by Pydantic validation now
        logger.error(
            "query_duckdb missing expected keys after Pydantic validation (should not happen)", 
            parsed_args_dict=parsed_args.model_dump() if parsed_args else "parsed_args_is_None",
            missing_key=str(e)
        )
        return {"status": "error", "message": f"Internal error: Missing expected argument {str(e)} after validation"}
    except duckdb.Error as e:
        logger.error("DuckDB query execution failed", query=query, error_str=str(e))
        return {"status": "error", "message": f"Query failed: {str(e)}"}
    except Exception as e:
        logger.error("Unexpected error during DuckDB query execution", query=query, error_str=str(e), error_type=str(type(e)))
        return {"status": "error", "message": f"Unexpected error: {str(e)}"}

class AnalyzeStatsInput(BaseModel):
    db_path: str = Field(description="Path to the DuckDB database file")
    table: str = Field(description="Name of the telemetry table")
    columns: List[str] = Field(description="Columns to analyze")
    metrics: List[str] = Field(description="Metrics to compute: 'z_score', 'difference', 'mean', 'std'")

def analyze_stats(tool_input: Any) -> Dict[str, Any]:
    """Compute statistical metrics for anomaly detection on telemetry data.

    Args:
        tool_input (Any): Raw input from the agent, expected to be a JSON string or a dictionary
                          containing db_path, table, columns, and metrics.

    Returns:
        Dict[str, Any]: Statistical results or error details.
            - status: 'success' or 'error'.
            - data: List of dictionaries with metric results per column.
            - row_count: Number of rows analyzed (if success).
            - message: Error message (if error).
    """
    logger.info(
        "analyze_stats received raw tool_input", 
        input_type=str(type(tool_input)), 
        input_value_snippet=str(tool_input)[:200]
    )

    args_dict: Optional[Dict[str, Any]] = None
    if isinstance(tool_input, str):
        try:
            args_dict = json.loads(tool_input)
        except json.JSONDecodeError as e:
            logger.error("Failed to decode tool_input string as JSON", error_str=str(e), tool_input_str=tool_input)
            return {"status": "error", "message": f"Invalid JSON input string: {str(e)}"}
    elif isinstance(tool_input, dict):
        args_dict = tool_input
    else:
        logger.error("tool_input is not a string or dict", received_type=str(type(tool_input)))
        return {"status": "error", "message": "Tool input must be a JSON string or a dictionary."}

    if args_dict is None: # Should not happen if logic above is correct, but as a safeguard
        logger.error("args_dict is None after input processing, which is unexpected.")
        return {"status": "error", "message": "Internal error: Failed to process tool input."}

    try:
        # Validate the dictionary using Pydantic. 
        # The model_validator in AnalyzeStatsInput will handle nested JSON if necessary.
        parsed_args = AnalyzeStatsInput.model_validate(args_dict)
        db_path = parsed_args.db_path
        table = parsed_args.table
        columns = parsed_args.columns
        metrics = parsed_args.metrics
    except Exception as pydantic_exc: # Catch Pydantic ValidationError or other issues
        logger.error(
            "Pydantic validation failed for derived args_dict", 
            args_dict_val=str(args_dict)[:500], # Log potentially large dict snippet
            error_str=str(pydantic_exc)
        )
        return {"status": "error", "message": f"Input validation failed: {str(pydantic_exc)}"}

    try:
        # Prevent path traversal attacks
        if ".." in os.path.relpath(db_path):
            logger.error("Invalid database path detected", db_path=db_path)
            return {"status": "error", "message": "Invalid database path"}

        with duckdb.connect(db_path, read_only=True) as conn:
            # Validate table exists
            try:
                conn.execute(f"SELECT 1 FROM {table} LIMIT 1")
            except duckdb.Error:
                return {"status": "error", "message": f"Table '{table}' does not exist"}
            
            # Validate table and columns
            try:
                available_columns: List[str] = [col[0] for col in conn.execute(f"DESCRIBE {table}").fetchall()]
            except duckdb.Error as e:
                return {"status": "error", "message": f"Failed to describe table '{table}': {str(e)}"}
            
            valid_columns: List[str] = [col for col in columns if col in available_columns]
            if not valid_columns:
                return {"status": "error", "message": f"No valid columns in {table}. Available: {available_columns}, Requested: {columns}"}

            # Fetch data, including timestamp if available
            query_columns: List[str] = valid_columns.copy()
            if 'timestamp' in available_columns and 'timestamp' not in query_columns:
                query_columns.append('timestamp')
            
            query: str = f"SELECT {', '.join(query_columns)} FROM {table}"
            df: pd.DataFrame = conn.execute(query).fetchdf()
            
            if df.empty:
                return {"status": "success", "data": [], "row_count": 0}

            # Compute metrics for each column
            results: List[Dict[str, Any]] = []
            for col in valid_columns:
                result: Dict[str, Any] = {"column": col}
                
                # Check if column is numeric
                is_numeric = pd.api.types.is_numeric_dtype(df[col])
                result["is_numeric"] = is_numeric
                
                if is_numeric:
                    col_data = df[col].dropna()  # Remove NaN values
                    if len(col_data) == 0:
                        result["error"] = "No valid numeric data"
                        results.append(result)
                        continue
                    
                    if 'mean' in metrics:
                        result["mean"] = float(col_data.mean())
                    if 'std' in metrics:
                        result["std"] = float(col_data.std())
                    if 'min' in metrics:
                        result["min"] = float(col_data.min())
                    if 'max' in metrics:
                        result["max"] = float(col_data.max())
                    
                    if 'z_score' in metrics:
                        mean_val: float = col_data.mean()
                        std_val: float = col_data.std()
                        if std_val > 0:
                            z_scores = (col_data - mean_val) / std_val
                            result["z_scores"] = z_scores.tolist()
                            result["z_score_outliers"] = (abs(z_scores) > 3).sum()  # Count outliers
                        else:
                            result["z_scores"] = []
                            result["z_score_outliers"] = 0
                    
                    if 'difference' in metrics:
                        differences = col_data.diff().abs().dropna()
                        result["differences"] = differences.tolist()
                        result["max_difference"] = float(differences.max()) if len(differences) > 0 else 0
                else:
                    result["error"] = "Column is not numeric"
                
                results.append(result)

            return {"status": "success", "data": results, "row_count": len(df)}

    except Exception as e:
        logger.error("Statistical analysis failed", table=table, error=str(e))
        return {"status": "error", "message": f"Analysis failed: {str(e)}"}

class DetectAnomaliesMLInput(BaseModel):
    db_path: str = Field(description="Path to the DuckDB database file")
    table: str = Field(description="Name of the telemetry table")
    columns: List[str] = Field(description="Numerical columns to analyze")

def detect_anomalies_ml(tool_input: Any) -> Dict[str, Any]:
    """Detect anomalies in telemetry data using Isolation Forest.

    Args:
        tool_input (Any): Raw input from the agent, expected to be a JSON string or a dictionary
                          containing db_path, table, and columns.

    Returns:
        Dict[str, Any]: Anomaly detection results or error details.
            - status: 'success' or 'error'.
            - data: List of dictionaries with row indices and anomaly flags (True for anomalies).
            - anomaly_count: Number of anomalies detected.
            - row_count: Number of rows analyzed (if success).
            - message: Error message (if error).
    """
    logger.info(
        "detect_anomalies_ml received raw tool_input", 
        input_type=str(type(tool_input)), 
        input_value_snippet=str(tool_input)[:200]
    )

    args_dict: Optional[Dict[str, Any]] = None
    if isinstance(tool_input, str):
        try:
            args_dict = json.loads(tool_input)
        except json.JSONDecodeError as e:
            logger.error("Failed to decode tool_input string as JSON", error_str=str(e), tool_input_str=tool_input)
            return {"status": "error", "message": f"Invalid JSON input string: {str(e)}"}
    elif isinstance(tool_input, dict):
        args_dict = tool_input
    else:
        logger.error("tool_input is not a string or dict", received_type=str(type(tool_input)))
        return {"status": "error", "message": "Tool input must be a JSON string or a dictionary."}

    if args_dict is None: # Should not happen if logic above is correct, but as a safeguard
        logger.error("args_dict is None after input processing, which is unexpected.")
        return {"status": "error", "message": "Internal error: Failed to process tool input."}

    try:
        # Validate the dictionary using Pydantic. 
        # The model_validator in DetectAnomaliesMLInput will handle nested JSON if necessary.
        parsed_args = DetectAnomaliesMLInput.model_validate(args_dict)
        db_path = parsed_args.db_path
        table = parsed_args.table
        columns = parsed_args.columns
    except Exception as pydantic_exc: # Catch Pydantic ValidationError or other issues
        logger.error(
            "Pydantic validation failed for derived args_dict", 
            args_dict_val=str(args_dict)[:500], # Log potentially large dict snippet
            error_str=str(pydantic_exc)
        )
        return {"status": "error", "message": f"Input validation failed: {str(pydantic_exc)}"}

    try:
        # Prevent path traversal attacks
        if ".." in os.path.relpath(db_path):
            logger.error("Invalid database path detected", db_path=db_path)
            return {"status": "error", "message": "Invalid database path"}

        with duckdb.connect(db_path, read_only=True) as conn:
            # Validate table exists
            try:
                conn.execute(f"SELECT 1 FROM {table} LIMIT 1")
            except duckdb.Error:
                return {"status": "error", "message": f"Table '{table}' does not exist"}
            
            # Get available columns
            try:
                available_columns: List[str] = [col[0] for col in conn.execute(f"DESCRIBE {table}").fetchall()]
            except duckdb.Error as e:
                return {"status": "error", "message": f"Failed to describe table '{table}': {str(e)}"}
            
            # Filter for valid columns that exist in the table
            valid_columns: List[str] = [col for col in columns if col in available_columns]
            if not valid_columns:
                return {"status": "error", "message": f"No valid columns in {table}. Available: {available_columns}, Requested: {columns}"}

            # Fetch data
            query: str = f"SELECT {', '.join(valid_columns)} FROM {table}"
            df: pd.DataFrame = conn.execute(query).fetchdf()
            
            if df.empty:
                return {"status": "success", "data": [], "anomaly_count": 0, "row_count": 0}

            # Filter for numerical columns only
            numerical_columns = []
            for col in valid_columns:
                if pd.api.types.is_numeric_dtype(df[col]):
                    numerical_columns.append(col)
            
            if not numerical_columns:
                return {"status": "error", "message": f"No numerical columns found. Available columns: {valid_columns}"}

            # Remove rows with NaN values in the selected columns
            df_clean = df[numerical_columns].dropna()
            if df_clean.empty:
                return {"status": "error", "message": "No valid data after removing NaN values"}

            # Fit Isolation Forest model
            model: IsolationForest = IsolationForest(
                contamination=0.1,  # Expect 10% anomalies
                random_state=42,
                n_estimators=100
            )
            predictions: np.ndarray = model.fit_predict(df_clean)
            anomaly_scores: np.ndarray = model.decision_function(df_clean)
            
            # -1 indicates anomaly, 1 indicates normal
            anomalies: np.ndarray = predictions == -1
            
            # Format results - map back to original dataframe indices
            results: List[Dict[str, Any]] = []
            anomaly_count = 0
            
            for i, (idx, is_anomaly) in enumerate(zip(df_clean.index, anomalies)):
                result = {
                    "row_index": int(idx),  # Original dataframe index
                    "is_anomaly": bool(is_anomaly),
                    "anomaly_score": float(anomaly_scores[i])
                }
                results.append(result)
                if is_anomaly:
                    anomaly_count += 1

            return {
                "status": "success",
                "data": results,
                "anomaly_count": anomaly_count,
                "row_count": len(df_clean),
                "columns_used": numerical_columns
            }

    except Exception as e:
        logger.error("ML anomaly detection failed", table=table, error=str(e))
        return {"status": "error", "message": f"Detection failed: {str(e)}"}

# Register tools for LangChain - all functions are now synchronous
query_duckdb_tool: StructuredTool = StructuredTool.from_function(
    func=query_duckdb,
    name="query_duckdb",
    description=(
        "Execute a SQL query on a DuckDB database. Use this to fetch telemetry data. "
        "Parameters: db_path (string, path to database), query (string, SQL query). "
        "Example: query_duckdb(db_path='/path/to/db.duckdb', query='SELECT AVG(altitude) FROM telemetry_global_position'). "
        "Returns query results with status, data, and row_count."
    ),
    args_schema=QueryDuckDBInput
)

analyze_stats_tool: StructuredTool = StructuredTool.from_function(
    func=analyze_stats,
    name="analyze_stats",
    description=(
        "Compute statistical metrics on telemetry data for anomaly detection. "
        "Parameters: db_path (string), table (string), columns (list of strings), metrics (list of strings). "
        "Available metrics: 'mean', 'std', 'min', 'max', 'z_score', 'difference'. "
        "Example: analyze_stats(db_path='/path/to/db.duckdb', table='telemetry_attitude', columns=['roll', 'pitch'], metrics=['mean', 'std', 'z_score']). "
        "Use z_score to find outliers (>3 indicates anomaly), difference to find sudden changes."
    ),
    args_schema=AnalyzeStatsInput
)

detect_anomalies_ml_tool: StructuredTool = StructuredTool.from_function(
    func=detect_anomalies_ml,
    name="detect_anomalies_ml",
    description=(
        "Detect anomalies in telemetry data using machine learning (Isolation Forest). "
        "Parameters: db_path (string), table (string), columns (list of numerical column names). "
        "Example: detect_anomalies_ml(db_path='/path/to/db.duckdb', table='telemetry_attitude', columns=['roll', 'pitch', 'yaw']). "
        "Returns anomaly flags and scores for each row. Use this for unsupervised anomaly detection."
    ),
    args_schema=DetectAnomaliesMLInput
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
        self.prompt: Optional[PromptTemplate] = None
        self.conversation_memory_manager: Optional[ConversationMemoryManager] = None
        self.token_callback: Optional[TokenUsageCallback] = None
        self.agent_executor: Optional[AgentExecutor] = None
        self.scratchpad: AgentScratchpad = AgentScratchpad(session_id=session_id)
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

        # Get tables info for the prompt
        tables_info = await self.get_tables_as_string()

        # Define ReAct prompt template
        self.prompt = PromptTemplate(
            template=REACT_SYSTEM_PROMPT,
            input_variables=["input", "agent_scratchpad", "agent_scratchpad_content"],
            partial_variables={
                "tables": tables_info,
                "tools": "{tools}",
                "tool_names": "{tool_names}"
            }
        )

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

        # Set up ReAct agent with tools
        tools: List[Tool] = [query_duckdb_tool, analyze_stats_tool, detect_anomalies_ml_tool]

         # Create ReAct agent
        agent = create_react_agent(
            llm=self.llm,
            tools=tools,
            prompt=self.prompt
        )

        # Create agent executor without memory (we'll handle memory manually)
        self.agent_executor = AgentExecutor(
            agent=agent,
            tools=tools,
            verbose=True, 
            handle_parsing_errors="Check your tool input format. It should be valid JSON with proper key-value pairs. For example: {\"db_path\": \"/path/to/file.duckdb\", \"query\": \"SELECT * FROM table\"}",
            max_iterations=10,  # Prevent infinite loops
            max_execution_time=120  # 2 minute timeout per execution
        )

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

    def _fetch_actual_schema_for_table(self, table_name: str) -> List[str]:
        """Connects to DB and fetches actual column names for a given table.
        Returns a list of column names, or an empty list if fetching fails or table not found.
        """
        actual_column_names: List[str] = []
        try:
            if not self.db_path or not os.path.exists(self.db_path):
                self.logger.error(f"Database path is invalid or file does not exist for table {table_name}", db_path=self.db_path)
                # No fallback to static schema here; if DB path is bad, we can't confirm anything.
                return []

            with duckdb.connect(self.db_path, read_only=True) as conn:
                tables_result = conn.execute("SELECT table_name FROM information_schema.tables WHERE table_schema = 'main';").fetchall()
                available_tables = [t[0].lower() for t in tables_result]

                if table_name.lower() in available_tables:
                    columns_data = conn.execute(f"DESCRIBE {table_name};").fetchall()
                    actual_column_names = [col_data[0] for col_data in columns_data]
                    self.logger.info(f"Dynamically fetched schema for {table_name}", columns=actual_column_names, db_path=self.db_path)
                else:
                    self.logger.debug(f"Table {table_name} from TELEMETRY_SCHEMA not found in DB {self.db_path}. It will NOT be included in the agent's list of available tables.")
                    actual_column_names = [] # Explicitly empty, no static fallback here
        except duckdb.Error as e:
            self.logger.error(f"DuckDB error when trying to determine schema for {table_name} from {self.db_path}", error=str(e))
            actual_column_names = [] # No static fallback on error
            self.logger.warning(f"Could not determine schema for {table_name} due to DuckDB error. It will be treated as unavailable for the agent's direct table list.")
        except Exception as e:
            self.logger.error(f"Unexpected error fetching schema for {table_name} from {self.db_path}", error=str(e), error_type=str(type(e)))
            actual_column_names = [] # No static fallback on error
            self.logger.warning(f"Could not determine schema for {table_name} due to unexpected error. It will be treated as unavailable for the agent's direct table list.")
        
        if not actual_column_names and os.path.exists(self.db_path or ""): # Log only if db_path was valid
             # This log might be redundant if already logged that table was not found or error occurred.
             # Consider if this specific log "Could not determine columns..." is still needed if the cases above are clear.
             pass # Keeping it for now, but can be removed if too noisy.
             # self.logger.warning(f"Final check: Columns for table {table_name} are empty.", db_path=self.db_path)

        return actual_column_names

    async def get_tables_as_string(self) -> str:
        """Generate a comma-separated string of available table names with their columns.
        Columns are fetched dynamically from the DB, with fallback to static TELEMETRY_SCHEMA.
        """
        table_strings: List[str] = []
        # Iterate through the tables defined in the static TELEMETRY_SCHEMA
        # as it also contains useful metadata like descriptions and anomaly hints.
        for table_meta_static in TELEMETRY_SCHEMA:
            table_name = table_meta_static["table"]
            
            # Fetch actual column names dynamically using the helper method
            # Run the synchronous DB operation in a separate thread
            actual_columns = await asyncio.to_thread(self._fetch_actual_schema_for_table, table_name)
            
            if actual_columns: # If columns were found (either dynamic or fallback)
                column_names_str = ", ".join(actual_columns)
                table_strings.append(f"{table_name}({column_names_str})")
            else:
                # If no columns could be determined for this table from TELEMETRY_SCHEMA, log and omit.
                # Changed from warning to debug as this is expected if table not in DB
                self.logger.debug(
                    f"No columns determined for table '{table_name}' from TELEMETRY_SCHEMA " +
                    f"(via dynamic fetch from {self.db_path}). Omitting from prompt schema string."
                )
        
        if not table_strings:
            self.logger.error("No tables could be processed to form the schema string for the prompt. Agent might not have table info.", db_path=self.db_path)
            return "Error: No table schema information could be loaded."

        final_schema_string = ", ".join(table_strings)
        self.logger.info("Generated table schema string for prompt using dynamic fetching (with fallback)", schema_string_snippet=final_schema_string[:500])
        return final_schema_string

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
            chat_history = memory.load_memory_variables({}).get("chat_history", [])

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

            # Prepare enhanced input with context
            enhanced_message = f"{message}\n\nContext:\n- Relevant tables: {', '.join(relevant_tables)}\n- Anomaly detection hints: {', '.join(anomaly_hints)}\n- Database path: {self.db_path}"
            
            # Add chat history context if available
            if chat_history:
                history_summary = "\n- Previous conversation context available"
                enhanced_message += history_summary

            # Prepare input data for ReAct agent
            input_data: Dict[str, str] = {
                "input": enhanced_message,
                "agent_scratchpad_content": self.scratchpad.to_string()
            }

            # Execute agent chain
            async def run_executor() -> Dict[str, Any]:
                self.logger.debug("ReAct agent input", input_data=input_data)
                # Ensure that the agent_scratchpad is passed correctly
                current_scratchpad_content = self.scratchpad.to_string()
                input_data_with_scratchpad = {
                    "input": enhanced_message, # This already includes 'enhanced_message' from above
                    "agent_scratchpad": current_scratchpad_content # Pass the current scratchpad content directly
                }
                # The REACT_SYSTEM_PROMPT template uses {agent_scratchpad} which is where LangChain injects history/intermediate steps.
                # My manual `agent_scratchpad_content` was for the separate variable in the prompt.
                # The actual intermediate steps are handled by AgentExecutor.
                # So, the agent_scratchpad argument to .ainvoke should be a string representation of previous steps.
                # This is typically handled internally by Langchain if memory is part of the agent.
                # For ReAct, the intermediate steps are built up and passed in the 'agent_scratchpad' input variable.
                
                # Let's ensure the input to ainvoke is what create_react_agent expects for the {agent_scratchpad} variable.
                # The agent_executor will fill "agent_scratchpad" with the history of thoughts and actions.
                # My current `input_data` has "agent_scratchpad_content" for my custom prompt part,
                # and "input" for the main question.
                # The `create_react_agent`'s prompt has `{agent_scratchpad}` which is where it expects the sequence of
                # Action, Action Input, Observation.

                # Correct input for react agent:
                react_input = {"input": enhanced_message, "agent_scratchpad": self.scratchpad.intermediate_steps}


                # The AgentExecutor will internally manage the agent_scratchpad based on intermediate_steps.
                # The input to agent_executor.ainvoke should primarily be "input".
                # The prompt template has: Question: {input} Thought: {agent_scratchpad}

                # So, the AgentExecutor expects 'input' and will populate 'agent_scratchpad'
                # The 'agent_scratchpad_content' is my own addition for extra context.
                
                # Let's revert to simpler input for AgentExecutor if `scratchpad.intermediate_steps` is handled by it.
                # The `create_react_agent` prompt expects "input" and "agent_scratchpad" (for thoughts/actions).
                # `AgentExecutor` creates the `agent_scratchpad` string from `intermediate_steps`.
                # My `REACT_SYSTEM_PROMPT` also includes `{agent_scratchpad_content}` which I manually populate.

                # The input to `agent_executor.ainvoke` should be a dictionary containing
                # all keys expected by the underlying agent's prompt, excluding 'agent_scratchpad'
                # if it's internally managed, or including it if explicitly built.
                # For `create_react_agent`, `agent_scratchpad` is the string of thoughts/actions.

                final_input_for_executor = {
                    "input": enhanced_message, # For {input} in prompt
                    "agent_scratchpad_content": self.scratchpad.to_string() # For {agent_scratchpad_content}
                }
                # `agent_scratchpad` (for thought/action/observation chain) is managed by AgentExecutor

                self.logger.debug("ReAct agent executor input", final_input_for_executor=final_input_for_executor)
                result = await self.agent_executor.ainvoke(final_input_for_executor)


                # Add intermediate steps to scratchpad
                # This is important for the next turn if the agent is stateful across multiple calls to process_message
                # For ReAct, the scratchpad is built per call, based on iterations.
                # My self.scratchpad is for multi-turn context, if I were to implement that more deeply.
                # For now, intermediate_steps from the result are key for *this* call's analysis.
                current_call_intermediate_steps = result.get("intermediate_steps", [])
                # self.scratchpad.add_steps(current_call_intermediate_steps) # If we want to persist across calls
                # self.logger.debug("Added steps from current call to persistent scratchpad", steps_count=len(current_call_intermediate_steps))

                return result

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

            # Estimate tokens for scratchpad content from the current agent run
            # The 'agent_scratchpad' in the result is the final scratchpad string used by the agent
            agent_internal_scratchpad_str = result.get("agent_scratchpad", "")
            if not agent_internal_scratchpad_str and isinstance(result.get("intermediate_steps"), list):
                # Reconstruct scratchpad string if not directly available (though usually it is for react)
                # This is complex, assume 'agent_scratchpad' or 'intermediate_steps' is primary.
                pass # Prefer direct 'agent_scratchpad' or 'intermediate_steps'

            # Let's use the `self.scratchpad` which is being updated for the custom context
            # For overall context length management, consider all inputs to the LLM.
            scratchpad_for_token_check = self.scratchpad.to_string() # Custom scratchpad
            scratchpad_tokens = len(self.token_encoder.encode(scratchpad_for_token_check))

            if scratchpad_tokens > self.max_context_tokens // 2:
                self.logger.warning(
                    "Custom scratchpad content approaching token limit",
                    tokens=scratchpad_tokens,
                    max_context_tokens=self.max_context_tokens
                )
                self.scratchpad.intermediate_steps = self.scratchpad.intermediate_steps[-10:] # Truncate custom
                self.logger.info("Truncated custom scratchpad to last 10 steps to manage token limit")

            # Sanitize intermediate_steps for logging to remove time_boot_ms from query_duckdb data
            raw_intermediate_steps = result.get("intermediate_steps", [])
            logged_intermediate_steps = []
            for action, observation in raw_intermediate_steps:
                if (action.tool == "query_duckdb" and
                        isinstance(observation, dict) and
                        observation.get("status") == "success" and
                        isinstance(observation.get("data"), list)):
                    
                    new_observation_data = []
                    for item in observation["data"]:
                        if isinstance(item, dict):
                            logged_item = item.copy()
                            logged_item.pop('time_boot_ms', None)
                            new_observation_data.append(logged_item)
                        else:
                            new_observation_data.append(item) # Should not happen with current query_duckdb
                    
                    logged_observation = observation.copy()
                    logged_observation["data"] = new_observation_data
                    logged_intermediate_steps.append((action, logged_observation))
                else:
                    logged_intermediate_steps.append((action, observation))
            
            # Compile metadata
            metadata: Dict[str, Any] = {
                "relevant_tables": relevant_tables,
                "anomaly_hints": anomaly_hints,
                "intermediate_steps": logged_intermediate_steps, # Use sanitized version for logging
                "token_usage": self.token_callback.token_usage,
                "memory_strategy": memory_strategy.value,
                "is_clarification": is_clarification,
                "agent_scratchpad_custom_context": self.scratchpad.to_string(), # Log the custom context scratchpad
                "agent_internal_scratchpad_final": agent_internal_scratchpad_str, # Log agent's final scratchpad string
                "agent_type": "react"
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