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
import re # Import re module

logger = structlog.get_logger(__name__)

# Load environment variables from .env file
load_dotenv()

# Constants for tool output truncation
MAX_TOOL_OUTPUT_ROWS: int = 50  # Max rows for query_duckdb and detect_anomalies_ml data
MAX_TOOL_OUTPUT_ITEMS: int = 10 # Max items in the main 'data' list for analyze_stats
MAX_LIST_ITEMS_IN_TOOL_OUTPUT: int = 5 # Max items for lists within tool output dicts (e.g., z_scores)

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

**Critical Instructions & Workflow:**

1.  **PRIORITY ONE: Check Your Memory & Scratchpad First!**
    *   Before doing ANYTHING else, review your **Chat History** (provided below) and your **Scratchpad** (previous analysis context: {agent_scratchpad_content}).
    *   Chat History:
        {chat_history}
    *   Is the answer to the current question "{input}" already available from your previous work (in Scratchpad) or the Chat History?
    *   **IF YES:** Use that information directly to formulate your Final Answer. Do NOT proceed to use tools or the ReAct loop below.
    *   **IF NO (or if the information is insufficient):** Then, and only then, proceed to the analysis steps below using the specified format.

Now, if you determined the answer was NOT in your memory, proceed with the following:

Use the following format:

Question: the input question you must answer
Thought: I need to understand the question and devise a plan. I will first determine if this is a broad anomaly/error query or a specific one.
Action: query_duckdb (or another appropriate tool based on my plan)
Action Input: {{...}}
Observation: [Result of tool]
Thought: Based on this observation, I will decide the next step.
...
Final Answer: the final answer to the original input question, grounded in the data obtained from tool use.

**Core Instructions for Answering Questions:**
1.  **Data-First & Tool-Driven:** Always start by using tools to get data. Conclusions MUST be based on data observed from tool outputs.
2.  **Use `{tables}` Schema:** Consult the `{tables}` list for available tables and their exact column names. Only use existing columns.
3.  **Verify Column Existence:** If a general strategy suggests a column, verify it exists in the `{tables}` schema for the specific table before querying it. If not, adapt your query or note its absence.

**Main Analysis Strategy Decision:**

*   **Is the query a broad request for "critical errors", "anomalies", "flight issues", "health check", or similar general problems?**
    *   **YES (Broad Query):** 
        *   Thought: The user's query is a broad request for general anomalies or errors. For this type of query, I will ONLY use the `detect_anomalies_ml` tool on a predefined list of key telemetry tables (if they exist in the current log). I will iterate through these tables, check for their existence, and if present, run `detect_anomalies_ml` on their relevant numerical columns. I will NOT use other tools or strategies from the 'Detailed Multi-Category Investigation' section for this broad query.
        *   **Step 1: Perform ML Anomaly Detection on Predefined Key Tables.**
            *   The predefined key tables to check with `detect_anomalies_ml` are: `telemetry_attitude`, `telemetry_global_position_int`, `telemetry_vfr_hud`, `telemetry_gps_raw_int`, `telemetry_ekf_status_report`, `telemetry_battery_status`, `telemetry_rc_channels`.
            *   For each of these predefined tables:
                *   Thought: I need to check if `[predefined_table_name]` exists in the current log's available `{tables}`. If it does, I will identify its numerical columns from the schema and run `detect_anomalies_ml` on them.
                *   Action: (Conditional) If `[predefined_table_name]` is in `{tables}`: `detect_anomalies_ml` (Input: {{"db_path": "[path_to_db]", "table": "[predefined_table_name]", "columns": ["list_of_ALL_relevant_NUMERICAL_columns_from_that_table_as_per_schema"]}} )
                *   Observation: [Results from `detect_anomalies_ml` for `[predefined_table_name]`, or note if table was not available/had no suitable numerical columns]. Note any detected anomalies.
        *   **Step 2: Synthesize and Conclude.**
            *   Thought: I have gathered results from `detect_anomalies_ml` for the available and applicable predefined tables. I will now synthesize these to answer the user's broad query.
            *   Final Answer: Summarize all significant anomalies found by `detect_anomalies_ml` across the analyzed tables. If no significant issues or anomalies are found after checking the predefined tables with `detect_anomalies_ml`, state that clearly.

    *   **NO (Specific Query):**
        *   Thought: The user's query is more specific and not a general request for anomalies. I will use the "Detailed Multi-Category Investigation" below, focusing on the categories and tables most relevant to the specific question.


**Detailed Multi-Category Investigation (for Specific Queries):**
When the user's query is specific (e.g., "What was the battery voltage?", "Were there GPS glitches?", "Analyze roll and pitch stability"), use the following categories to guide your investigation. Focus only on what's relevant.

*   **Category 1: Explicitly Logged Events & Messages**
    *   **Goal:** Identify any system-logged errors, warnings, or critical status messages relevant to the query.
    *   **Strategy:** Query tables like `telemetry_statustext` (if available and relevant). Look for low `severity` values or textual content suggesting problems.

*   **Category 2: System & Power Health**
    *   **Goal:** Assess system integrity relevant to the query (sensor health, power, comms).
    *   **Strategy:** Query tables like `telemetry_sys_status` (if available and relevant). Investigate columns like `onboard_control_sensors_health`, `voltage_battery`, `current_battery`, `battery_remaining`, `load`, `drop_rate_comm`, `errors_comm`. Use `analyze_stats` for trends/outliers if needed.

*   **Category 3: Navigation System Integrity (EKF, GPS, Position)**
    *   **Goal:** Determine if the navigation system is reliable, if relevant to the query.
    *   **Strategy:** Query `telemetry_ekf_status_report` (analyze `flags`, `variances`), `telemetry_gps_raw_int` (check `fix_type`, `satellites_visible`, `eph`, `epv`), `telemetry_global_position_int` (analyze `alt`, `relative_alt`, `vx`, `vy`, `vz`, `hdg`). Use `analyze_stats` or `detect_anomalies_ml` if detailed numerical analysis is needed for specific columns.

*   **Category 4: Flight Performance & Stability Analysis**
    *   **Goal:** Detect abnormal flight characteristics or control issues, if relevant to a specific query about performance.
    *   **Strategy:** 
        *   Identify the specific flight dynamics table(s) and column(s) mentioned or implied by the user's query from `{tables}` (e.g., `telemetry_attitude` for roll/pitch, `telemetry_vfr_hud` for airspeed/altitude).
        *   Thought: I will query relevant numerical columns from the specified table(s) for analysis.
        *   Action: `query_duckdb` (to select the specific numerical columns needed).
        *   Observation: [Data from query]
        *   Thought: Now I have the raw numerical data. I will use `detect_anomalies_ml` if complex unknown patterns are suspected or interpret the raw data directly if the query asks for simple statistics (e.g. min, max, or an approximate trend).
        *   Action: (Conditional) `detect_anomalies_ml` (if appropriate for the specific query and data).
        *   Observation: (Conditional) [Results of `detect_anomalies_ml`].
        *   Thought: Note findings relevant to the user's specific query based on raw data and/or ML analysis.

*   **Synthesize & Conclude (for Specific Queries):** After investigating relevant categories, review all observations. Formulate a final answer that directly addresses the user's specific question based on the data.

IMPORTANT: When using tools, the Action Input must be valid JSON format. Examples:
For query_duckdb: Action Input: {{"db_path": "/path/to/database.duckdb", "query": "SELECT * FROM table_name"}}
For detect_anomalies_ml: Action Input: {{"db_path": "/path/to/database.duckdb", "table": "table_name", "columns": ["col1", "col2"]}}

Scratchpad (previous analysis context):
{agent_scratchpad_content}

Begin!

Question: {input}
Thought: {agent_scratchpad}
"""

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

        # Block dangerous SQL operations using regex for whole word matching
        query_lower: str = query.lower().strip()
        # Ensure keywords are treated as whole words and handle cases like `drop table` vs `drop_rate_comm`
        dangerous_keywords = ["drop", "delete", "update", "insert", "alter", "create", "truncate"]
        for keyword in dangerous_keywords:
            # denotes a word boundary
            if re.search(r"\b" + re.escape(keyword) + r"\b", query_lower):
                logger.error("Potentially dangerous query detected due to keyword: ", query=query, keyword=keyword)
                return {"status": "error", "message": f"Query contains prohibited keyword: {keyword}"}
        
        with duckdb.connect(db_path, read_only=True) as conn:
            result: List[tuple] = conn.execute(query).fetchall()
            columns: List[str] = [desc[0] for desc in conn.description] if conn.description else []
            
            data_to_return = [dict(zip(columns, row)) for row in result]
            row_count = len(data_to_return)

            if row_count > MAX_TOOL_OUTPUT_ROWS:
                logger.debug(
                    f"query_duckdb output truncated for agent scratchpad from {row_count} to {MAX_TOOL_OUTPUT_ROWS} rows.",
                    query=query
                )
                # Return a subset and a message indicating truncation
                return {
                    "status": "success_truncated",
                    "data": data_to_return[:MAX_TOOL_OUTPUT_ROWS],
                    "message": f"Output truncated to first {MAX_TOOL_OUTPUT_ROWS} rows. Original row count: {row_count}. Consider refining your query.",
                    "row_count": row_count, # Still report original row count
                    "displayed_row_count": MAX_TOOL_OUTPUT_ROWS
                }
            
            return {
                "status": "success",
                "data": data_to_return,
                "row_count": row_count
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

class DetectAnomaliesMLInput(BaseModel):
    db_path: str = Field(description="Path to the DuckDB database file")
    table: str = Field(description="Name of the telemetry table")
    columns: List[str] = Field(description="Numerical columns to analyze")

    @model_validator(mode='before')
    @classmethod
    def _handle_potentially_nested_json_input(cls, data: Any) -> Any:
        if isinstance(data, dict):
            # Check for the specific nested JSON pattern for this model
            # i.e. table and columns are missing, and db_path is a string that might be JSON
            if ('table' not in data and 
                'columns' not in data and 
                'db_path' in data and 
                isinstance(data['db_path'], str)):
                
                potential_json_str = data['db_path']
                try:
                    parsed_json = json.loads(potential_json_str)
                    # Check if the parsed JSON has the expected keys for DetectAnomaliesMLInput
                    if (isinstance(parsed_json, dict) and
                        'db_path' in parsed_json and
                        'table' in parsed_json and
                        'columns' in parsed_json):
                        logger.info(
                            f"{cls.__name__}: Input data appeared to have JSON string in 'db_path'. Expanding.",
                            original_data_keys=list(data.keys()),
                            parsed_json_keys=list(parsed_json.keys())
                        )
                        return parsed_json # Return the parsed dict for Pydantic to validate
                except json.JSONDecodeError:
                    logger.warning(
                        f"{cls.__name__}: 'db_path' contained a string that was not parsable " +
                        "into the expected structure for DetectAnomaliesMLInput.",
                        db_path_value_snippet=potential_json_str[:100]
                    )
                    # Fall through to return original data, Pydantic will likely raise validation error
                    pass
        return data

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

            # Truncate the results if too long
            final_row_count = len(df_clean)
            if len(results) > MAX_TOOL_OUTPUT_ROWS:
                logger.warning(
                    f"detect_anomalies_ml output truncated from {len(results)} to {MAX_TOOL_OUTPUT_ROWS} rows.",
                    table=table
                )
                # Include a message about truncation in the response
                return {
                    "status": "success_truncated",
                    "data": results[:MAX_TOOL_OUTPUT_ROWS],
                    "message": f"Output truncated to first {MAX_TOOL_OUTPUT_ROWS} anomaly detection results. Original result count: {len(results)}.",
                    "anomaly_count": (pd.Series([r['is_anomaly'] for r in results[:MAX_TOOL_OUTPUT_ROWS]])).sum(), # Recalculate anomaly_count for truncated data
                    "row_count": final_row_count,
                    "displayed_row_count": MAX_TOOL_OUTPUT_ROWS,
                    "columns_used": numerical_columns
                }

            return {
                "status": "success",
                "data": results,
                "anomaly_count": anomaly_count,
                "row_count": final_row_count,
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
            input_variables=["input", "agent_scratchpad", "agent_scratchpad_content", "chat_history"],
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
        tools: List[Tool] = [query_duckdb_tool, detect_anomalies_ml_tool]

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
            max_iterations=20,  # Prevent infinite loops (Increased from 10)
            max_execution_time=200,  # Increased from 120s, less than CHAT_TIMEOUT_SECONDS (240s)
            return_intermediate_steps=True # Ensure agent's thought process is returned
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
            chat_history_messages = memory.load_memory_variables({}).get("history", [])
            
            # Format chat_history_messages into a string for the prompt
            chat_history_string = ""
            if isinstance(chat_history_messages, list):
                for msg in chat_history_messages:
                    if hasattr(msg, 'type') and hasattr(msg, 'content'): # LangChain BaseMessage structure
                        chat_history_string += f"{msg.type.upper()}: {msg.content}\\n"
                    else: # Fallback for other potential structures
                        chat_history_string += f"{str(msg)}\\n"
            if not chat_history_string:
                chat_history_string = "No prior conversation history."

            # Identify relevant tables and hints using vector store
            search_results = await self.vector_store_manager.asimilarity_search(
                message, k=VECTOR_RETRIEVER_K
            )
            relevant_tables: List[str] = list(set( # Use set to ensure uniqueness
                result.metadata.get("table", "") for result in search_results if result.metadata.get("table")
            ))
            anomaly_hints: List[str] = list(set( # Use set to ensure uniqueness
                result.metadata.get("anomaly_hint", "") for result in search_results if result.metadata.get("anomaly_hint")
            ))

            # Prepare enhanced input with context
            # The prompt already has {tables} so we don't need to add relevant_tables to enhanced_message explicitly
            # Anomaly hints are general guidance; the main prompt now emphasizes data-driven investigation.
            enhanced_message = f"{message}\n\nContext provided by system (use for general guidance but prioritize direct data querying):\n- Anomaly detection hints related to your query: {', '.join(anomaly_hints)}\n- Database path for tools: {self.db_path}"
            
            # The custom scratchpad content is passed separately
            current_custom_scratchpad = self.scratchpad.to_string()

            final_input_for_executor = {
                "input": enhanced_message,
                "agent_scratchpad_content": current_custom_scratchpad,
                "chat_history": chat_history_string # Pass the formatted chat history string
                # agent_scratchpad for ReAct internal steps is handled by AgentExecutor
            }

            self.logger.debug("ReAct agent executor input", final_input_for_executor_keys=list(final_input_for_executor.keys()), chat_history_snippet=chat_history_string[:200], custom_scratchpad_snippet=current_custom_scratchpad[:200])
            
            result = await asyncio.wait_for(
                self.agent_executor.ainvoke(final_input_for_executor),
                timeout=CHAT_TIMEOUT_SECONDS
            )

            # Log all keys from the agent executor result for debugging
            self.logger.info("Full keys from AgentExecutor result", keys=list(result.keys()))
            
            # Populate our custom multi-turn scratchpad
            intermediate_steps_raw = result.get("intermediate_steps", [])
            if intermediate_steps_raw:
                self.logger.info(f"Populating custom scratchpad with {len(intermediate_steps_raw)} steps from current turn.")
                for agent_action_obj, observation_str in intermediate_steps_raw:
                    # agent_action_obj is of type langchain_core.agents.AgentAction
                    # observation_str is the string output of the tool
                    action_details_for_scratchpad = {
                        "tool": agent_action_obj.tool,
                        "tool_input": agent_action_obj.tool_input,
                        # The 'log' in AgentAction contains the thought process leading to this action.
                        "thought_process_for_this_action": agent_action_obj.log.strip() 
                    }
                    self.scratchpad.add_step(action_details_for_scratchpad, str(observation_str))
            else:
                self.logger.warning("No intermediate_steps from AgentExecutor for this turn to add to custom scratchpad.")


            response: str = result.get("output", "Error: No output from agent.") # Ensure there's a default
            response_lower: str = response.strip().lower()
            is_clarification: bool = (
                response.strip().endswith("?") or
                any(phrase in response_lower for phrase in CLARIFICATION_PHRASES)
            )
            if is_clarification and not response.strip().lower().startswith("thought"): # Avoid adding to thoughts
                self.logger.info("Response requests clarification", response=response)
                response += "\nPlease provide more details to proceed."

            await self.conversation_memory_manager.add_message((message, response))

            # Get intermediate steps for metadata
            # This is crucial for debugging if the agent is taking steps
            intermediate_steps = result.get("intermediate_steps", [])
            if not intermediate_steps:
                self.logger.warning("No intermediate steps were recorded by the agent for this call.", agent_output=response)


            # Sanitize intermediate_steps for logging in metadata
            logged_intermediate_steps_for_metadata = []
            for action, observation in intermediate_steps_raw: # Use intermediate_steps_raw here
                logged_observation_str = str(observation)
                if len(logged_observation_str) > 500: # Arbitrary limit for logging
                    logged_observation_str = logged_observation_str[:500] + " ... (truncated in log)"
                
                # action is an AgentAction object. Access its attributes.
                tool_name = action.tool if hasattr(action, 'tool') else 'UnknownTool'
                tool_input_val = action.tool_input if hasattr(action, 'tool_input') else 'UnknownInput'
                
                logged_intermediate_steps_for_metadata.append(
                    (f"Tool: {tool_name}, Input: {tool_input_val}", logged_observation_str)
                )

            metadata: Dict[str, Any] = {
                "relevant_tables": relevant_tables, # These are from vector search, not agent's direct use confirmation
                "anomaly_hints": anomaly_hints,
                "intermediate_steps": logged_intermediate_steps_for_metadata, # Use the correctly formatted one 
                "token_usage": self.token_callback.token_usage,
                "memory_strategy": memory_strategy.value,
                "is_clarification": is_clarification,
                "agent_scratchpad_custom_context": current_custom_scratchpad, 
                "agent_internal_scratchpad_final": result.get("agent_scratchpad", "") or result.get("log", ""), # ReAct often uses 'log'
                "agent_type": "react",
                "raw_agent_output": result.get("output", "") # Add raw output for easier debugging
            }
            
            # Manage custom scratchpad size (Token check was here, keep it)
            scratchpad_tokens = len(self.token_encoder.encode(current_custom_scratchpad))
            if scratchpad_tokens > self.max_context_tokens // 2: # type: ignore
                self.logger.warning(
                    "Custom scratchpad content approaching token limit",
                    tokens=scratchpad_tokens,
                    max_context_tokens=self.max_context_tokens
                )
                self.scratchpad.intermediate_steps = self.scratchpad.intermediate_steps[-10:] 
                self.logger.info("Truncated custom scratchpad to last 10 steps to manage token limit")


            self.logger.info(
                "Generated response",
                response_length=len(response),
                # Full metadata can be very verbose if intermediate steps are complex.
                # Consider logging only key parts or a summary of metadata here.
                metadata_intermediate_steps_count=len(logged_intermediate_steps_for_metadata),
                metadata_token_usage=metadata["token_usage"],
                metadata_is_clarification=metadata["is_clarification"]
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
                error_str=str(e), # Use error_str for consistency with other logs
                error_type=str(type(e))
            )
            raise # Re-raise the caught exception