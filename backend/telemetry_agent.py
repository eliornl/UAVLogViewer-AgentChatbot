"""Telemetry Agent module for UAV Log Viewer application.

This module implements an intelligent agent for analyzing UAV telemetry data,
detecting flight anomalies, and answering user queries about flight data. It uses
LangChain's ReAct agent framework with custom tools for SQL querying and anomaly
detection, along with memory management for multi-turn conversations.

The agent can process natural language queries about telemetry data, execute
appropriate SQL queries or anomaly detection algorithms, and provide insights
about flight performance, system status, and potential issues.
"""

from typing import Dict, Any, Optional, List, Tuple
import os
import time
from pydantic import BaseModel, Field, model_validator, ValidationError
import structlog
import asyncio
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import Tool, StructuredTool
from langchain.agents import AgentExecutor, create_react_agent
from langchain.callbacks.base import BaseCallbackHandler
import duckdb
import json
from backend.agent_scratchpad import AgentScratchpad
from backend.conversation_memory_manager import ConversationMemoryManager
from backend.vector_store_manager import VectorStoreManager
from backend.telemetry_schema import TELEMETRY_SCHEMA
from backend.anomaly_detector import AnomalyDetector
import re

logger = structlog.get_logger(__name__)

# Load environment variables from .env file
load_dotenv()

# Token management constants
MAX_MODEL_TOKENS: int = 8192  # Maximum tokens for LLM input/output
RESERVED_RESPONSE_TOKENS: int = 1024  # Tokens reserved for LLM response
MAX_TOKEN_SAFETY_LIMIT: int = 16384  # Absolute token limit for safety
FALLBACK_TOKEN_FACTOR: int = 2  # Factor to reduce context tokens if needed

# LLM configuration constants
LLM_TEMPERATURE: float = 0.0  # LLM temperature for deterministic responses
VECTOR_RETRIEVER_K: int = 4  # Number of vector store search results

# Timeout constants
QUERY_TIMEOUT_SECONDS: float = 30.0  # Timeout for DuckDB queries
CHAT_TIMEOUT_SECONDS: float = 60.0  # Timeout for agent processing, including ML

# Output formatting constants
MAX_TOOL_OUTPUT_ROWS: int = 50  # Max rows for query_duckdb and detect_anomalies output

# Error message constants
ERROR_EMPTY_MESSAGE: str = "Empty message provided, skipping processing"
ERROR_AGENT_NOT_INITIALIZED: str = (
    "Agent executor is not initialized. Call async_initialize() first."
)
ERROR_MEMORY_NOT_INITIALIZED: str = (
    "Conversation memory manager is not initialized. Call async_initialize() first."
)
ERROR_TOKEN_CALLBACK_NOT_INITIALIZED: str = (
    "Token callback is not initialized. Call async_initialize() first."
)
ERROR_INVALID_MAX_TOKENS: str = "Invalid max_tokens: {} must be positive"
ERROR_MAX_TOKENS_EXCEEDS_LIMIT: str = "max_tokens {} exceeds safety limit {}"
ERROR_MAX_TOKENS_TOO_SMALL: str = (
    "max_tokens {} must be greater than RESERVED_RESPONSE_TOKENS {}"
)
ERROR_PROCESSING_TIMEOUT: str = "Message processing timed out after {} seconds"
ERROR_SCRATCHPAD_TOKEN_LIMIT: str = "Custom scratchpad content approaching token limit"

# Clarification detection phrases
CLARIFICATION_PHRASES: List[str] = [
    "could you clarify",
    "can you clarify",
    "can you specify",
    "what do you mean",
    "please clarify",
    "could you specify",
    "more details needed",
    "can you provide more",
]

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
4.  **CRITICAL TOOL SELECTION RULES:**
    * ALWAYS use `detect_anomalies` (NOT query_duckdb) when analyzing issues, anomalies, problems, or quality of ANY system (GPS, attitude, position, etc.)
    * Use `query_duckdb` ONLY for direct data retrieval like getting specific values, statistics, or filtered data
    * When the query contains words like "issues", "anomalies", "problems", "glitches", "errors", "quality" - ALWAYS use detect_anomalies
5.  **IMPORTANT COLUMN NAME CONVENTIONS:**
    * Use `timestamp` (not `time_boot_ms`) for time-related queries in all tables
    * Always check the exact column names in the `{tables}` list before constructing queries
    * If you get a 'column not found' error, immediately check the schema and adjust your query

**Main Analysis Strategy Decision:**

*   **Is the query about a specific flight phase (takeoff, landing, mid-flight)?**
    *   **YES:**
        *   Thought: The user is asking about a specific flight phase. I need to determine the timestamp range for this phase and analyze data within that range.
        *   **Step 1: Determine flight phase timestamp range.**
            *   Action: `query_duckdb` to get min and max timestamps
            *   Calculate phase boundaries based on percentages: Takeoff (0-25%), Mid-flight (25-75%), Landing (75-100%)
        *   **Step 2: Analyze data within that phase.**
            *   If the user mentioned a specific table, query only that table with the timestamp range filter
            *   If no specific table was mentioned, query each of the PRIORITY_TABLES (telemetry_attitude, telemetry_vfr_hud, telemetry_global_position_int, telemetry_gps_raw_int) with the timestamp range
            *   Use `query_duckdb` with appropriate WHERE clauses to filter by the timestamp range
            *   Final Answer: Provide insights specific to the requested flight phase

*   **Is the query a broad request for "critical errors", "anomalies", "flight issues", "problems", "warnings", "failures", "malfunctions", "incidents", "troubles", "concerns", "errors", "faults", "defects", "issues", or similar general problems?**
    *   **YES (Broad Query):** 
        *   Thought: The user's query is a broad request for general anomalies or errors. For this type of query, I will use the `detect_anomalies` tool which comprehensively analyzes all data in the most relevant tables.
        *   **Step 1: Use Anomaly Detection.**
            *   Action: `detect_anomalies` (Input: {{"db_path": "[path_to_db]"}} )
            *   Observation: [Results from `detect_anomalies` showing tables processed and anomalies found across all tables]
        *   **Step 2: Synthesize and Conclude.**
            *   Thought: I have received comprehensive anomaly detection results across the most important tables. These results include analysis of all available data and are highly reliable.
            *   Final Answer: Summarize all significant anomalies found across the analyzed tables. If no significant issues or anomalies are found, state that clearly.
            
        *   **IMPORTANT: Trust the model results completely.** The anomaly detection model has been trained on all available data and focuses on the most relevant tables where anomalies are likely to occur. DO NOT perform additional queries to verify the model's findings. The model's results are sufficient and comprehensive.

    *   **YES, BUT SPECIFIC TO ONE SYSTEM (e.g., "GPS issues", "attitude problems"):**
        *   Thought: The user is asking about anomalies or issues in a specific system. I will use the `detect_anomalies` tool but focus on the relevant tables for that system.
        *   **Step 1: Use Targeted Anomaly Detection.**
            *   Action: `detect_anomalies` (Input: {{"db_path": "[path_to_db]", "tables": ["relevant_table_name"]}} )
            *   Observation: [Results from `detect_anomalies` showing anomalies found in the specific table]
        *   **Step 2: Synthesize and Conclude.**
            *   Thought: I have received anomaly detection results for the specific system the user asked about.
            *   Final Answer: Summarize ONLY the anomalies related to the specific system the user asked about. If no issues are found in that system, state that clearly.

    *   **NO (Other Specific Query):**
        *   Thought: The user's query is specific but not about anomalies or issues. I will use the "Detailed Multi-Category Investigation" below, focusing on the categories and tables most relevant to the specific question.


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
For detect_anomalies: Action Input: {{"db_path": "/path/to/database.duckdb"}} or {{"db_path": "/path/to/database.duckdb", "tables": ["table1", "table2"]}}

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
    "can you provide more",
]

# Validate token-related constants at module load
if MAX_MODEL_TOKENS <= 0 or RESERVED_RESPONSE_TOKENS < 0:
    logger.error(
        "Invalid token limits",
        max_model_tokens=MAX_MODEL_TOKENS,
        reserved_response_tokens=RESERVED_RESPONSE_TOKENS,
    )
    raise ValueError(
        "MAX_MODEL_TOKENS must be positive and RESERVED_RESPONSE_TOKENS non-negative"
    )
if MAX_MODEL_TOKENS <= RESERVED_RESPONSE_TOKENS:
    logger.error(
        "MAX_MODEL_TOKENS must be greater than RESERVED_RESPONSE_TOKENS",
        max_model_tokens=MAX_MODEL_TOKENS,
        reserved_response_tokens=RESERVED_RESPONSE_TOKENS,
    )
    raise ValueError(
        f"MAX_MODEL_TOKENS ({MAX_MODEL_TOKENS}) must be greater than "
        f"RESERVED_RESPONSE_TOKENS ({RESERVED_RESPONSE_TOKENS})"
    )
if MAX_TOKEN_SAFETY_LIMIT <= MAX_MODEL_TOKENS:
    logger.error(
        "MAX_TOKEN_SAFETY_LIMIT must be greater than MAX_MODEL_TOKENS",
        max_token_safety_limit=MAX_TOKEN_SAFETY_LIMIT,
        max_model_tokens=MAX_MODEL_TOKENS,
    )
    raise ValueError(
        f"MAX_TOKEN_SAFETY_LIMIT ({MAX_TOKEN_SAFETY_LIMIT}) must be greater than "
        f"MAX_MODEL_TOKENS ({MAX_MODEL_TOKENS})"
    )


# Pydantic models for the tools' input arguments
class QueryDuckDBInput(BaseModel):
    db_path: str = Field(description="Path to the DuckDB database file")
    query: str = Field(description="SQL query to execute")

    @model_validator(mode="before")
    @classmethod
    def _handle_potentially_nested_json_input(cls, data: Any) -> Any:
        # Handle the case when the agent sends the input as a JSON string
        if isinstance(data, str):
            try:
                parsed_data = json.loads(data)
                if isinstance(parsed_data, dict):
                    logger.info(
                        f"{cls.__name__}: Input was a JSON string, parsed successfully"
                    )
                    data = parsed_data
            except json.JSONDecodeError:
                logger.warning(f"{cls.__name__}: Input was a string but not valid JSON")
                # Continue with original data

        if isinstance(data, dict):
            # Case 1: Check if 'query' is missing and 'db_path' holds a string that might be JSON
            if (
                "query" not in data
                and "db_path" in data
                and isinstance(data["db_path"], str)
            ):
                db_path_value = data["db_path"]

                # Check if db_path looks like it might contain JSON (has curly braces)
                if "{" in db_path_value and "}" in db_path_value:
                    try:
                        # Try to extract a complete JSON object
                        first_brace = db_path_value.find("{")
                        last_brace = db_path_value.rfind("}")
                        if first_brace >= 0 and last_brace > first_brace:
                            json_part = db_path_value[first_brace : last_brace + 1]
                            parsed_json = json.loads(json_part)

                            if isinstance(parsed_json, dict):
                                # Check if the parsed JSON has the expected structure
                                if "db_path" in parsed_json and "query" in parsed_json:
                                    logger.info(
                                        f"{cls.__name__}: Successfully extracted JSON from db_path"
                                    )
                                    return parsed_json
                                # If it has query but not db_path, try to combine them
                                elif (
                                    "query" in parsed_json
                                    and "db_path" not in parsed_json
                                ):
                                    # Extract the actual db_path from the beginning of the string
                                    actual_db_path = db_path_value.split("{")[0].strip()
                                    if actual_db_path.endswith(".duckdb"):
                                        parsed_json["db_path"] = actual_db_path
                                        logger.info(
                                            f"{cls.__name__}: Combined db_path with query from JSON"
                                        )
                                        return parsed_json
                    except json.JSONDecodeError:
                        # Not valid JSON, try other approaches
                        pass

                    # If JSON parsing failed, try to extract query directly if it's in a multi-line format
                    if '"query":' in db_path_value or "'query':" in db_path_value:
                        try:
                            # Extract the query part
                            query_start = max(
                                db_path_value.find('"query":'),
                                db_path_value.find("'query':"),
                            )
                            if query_start > 0:
                                # Find the actual query content
                                content_start = db_path_value.find('"""', query_start)
                                if content_start > 0:
                                    content_end = db_path_value.find(
                                        '"""', content_start + 3
                                    )
                                    if content_end > 0:
                                        query_content = db_path_value[
                                            content_start + 3 : content_end
                                        ].strip()
                                        # Extract the actual db_path from the beginning of the string
                                        actual_db_path = db_path_value.split("{")[
                                            0
                                        ].strip()
                                        if actual_db_path.endswith(".duckdb"):
                                            logger.info(
                                                f"{cls.__name__}: Extracted multi-line query and db_path"
                                            )
                                            return {
                                                "db_path": actual_db_path,
                                                "query": query_content,
                                            }
                        except Exception as e:
                            logger.warning(
                                f"{cls.__name__}: Failed to extract multi-line query",
                                error=str(e),
                            )
                            pass

                    # Log the failure
                    logger.warning(
                        f"{cls.__name__}: 'db_path' contained a string that was not parsable "
                        + "into the expected {{'db_path': ..., 'query': ...}} structure.",
                        db_path_value_snippet=db_path_value[
                            :100
                        ],  # Log a snippet for diagnosis
                    )
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
        input_value_snippet=str(tool_input)[:200],
    )

    args_dict: Optional[Dict[str, Any]] = None
    if isinstance(tool_input, str):
        try:
            args_dict = json.loads(tool_input)
        except json.JSONDecodeError as e:
            logger.error(
                "Failed to decode tool_input string as JSON",
                error_str=str(e),
                tool_input_str=tool_input,
            )
            return {
                "status": "error",
                "message": f"Invalid JSON input string: {str(e)}",
            }
    elif isinstance(tool_input, dict):
        args_dict = tool_input
    else:
        logger.error(
            "tool_input is not a string or dict", received_type=str(type(tool_input))
        )
        return {
            "status": "error",
            "message": "Tool input must be a JSON string or a dictionary.",
        }

    if (
        args_dict is None
    ):  # Should not happen if logic above is correct, but as a safeguard
        logger.error("args_dict is None after input processing, which is unexpected.")
        return {
            "status": "error",
            "message": "Internal error: Failed to process tool input.",
        }

    try:
        # Validate the dictionary using Pydantic.
        # The model_validator in QueryDuckDBInput will handle nested JSON if necessary.
        parsed_args = QueryDuckDBInput.model_validate(args_dict)
        db_path = parsed_args.db_path
        query = parsed_args.query
    except Exception as pydantic_exc:  # Catch Pydantic ValidationError or other issues
        logger.error(
            "Pydantic validation failed for derived args_dict",
            args_dict_val=str(args_dict)[:500],  # Log potentially large dict snippet
            error_str=str(pydantic_exc),
        )
        return {
            "status": "error",
            "message": f"Input validation failed: {str(pydantic_exc)}",
        }

    try:
        # Prevent path traversal attacks
        if ".." in os.path.relpath(db_path):
            logger.error("Invalid database path detected", db_path=db_path)
            return {"status": "error", "message": "Invalid database path"}

        # Block dangerous SQL operations using regex for whole word matching
        query_lower: str = query.lower().strip()
        # Ensure keywords are treated as whole words and handle cases like `drop table` vs `drop_rate_comm`
        dangerous_keywords = [
            "drop",
            "delete",
            "update",
            "insert",
            "alter",
            "create",
            "truncate",
        ]
        for keyword in dangerous_keywords:
            # denotes a word boundary
            if re.search(r"\b" + re.escape(keyword) + r"\b", query_lower):
                logger.error(
                    "Potentially dangerous query detected due to keyword: ",
                    query=query,
                    keyword=keyword,
                )
                return {
                    "status": "error",
                    "message": f"Query contains prohibited keyword: {keyword}",
                }

        with duckdb.connect(db_path, read_only=True) as conn:
            result: List[tuple] = conn.execute(query).fetchall()
            columns: List[str] = (
                [desc[0] for desc in conn.description] if conn.description else []
            )

            data_to_return = [dict(zip(columns, row)) for row in result]
            row_count = len(data_to_return)

            # Return all results without truncation
            return {"status": "success", "data": data_to_return, "row_count": row_count}
    except KeyError as e:  # Should ideally be caught by Pydantic validation now
        logger.error(
            "query_duckdb missing expected keys after Pydantic validation (should not happen)",
            parsed_args_dict=(
                parsed_args.model_dump() if parsed_args else "parsed_args_is_None"
            ),
            missing_key=str(e),
        )
        return {
            "status": "error",
            "message": f"Internal error: Missing expected argument {str(e)} after validation",
        }
    except duckdb.Error as e:
        logger.error("DuckDB query execution failed", query=query, error_str=str(e))
        return {"status": "error", "message": f"Query failed: {str(e)}"}
    except Exception as e:
        logger.error(
            "Unexpected error during DuckDB query execution",
            query=query,
            error_str=str(e),
            error_type=str(type(e)),
        )
        return {"status": "error", "message": f"Unexpected error: {str(e)}"}


class DetectAnomaliesMLInput(BaseModel):
    db_path: str = Field(description="Path to the DuckDB database file")
    table: str = Field(description="Name of the telemetry table")
    columns: List[str] = Field(description="Numerical columns to analyze")


class DetectAnomaliesBatchInput(BaseModel):
    db_path: str = Field(description="Path to the DuckDB database file")
    tables: Optional[List[str]] = Field(
        default=None,
        description="List of tables to analyze. If None, will use priority tables",
    )

    @model_validator(mode="before")
    @classmethod
    def check_values(cls, data: Any) -> Any:
        # Handle the case when the agent sends the input as a JSON string
        if isinstance(data, str):
            try:
                parsed_data = json.loads(data)
                if isinstance(parsed_data, dict):
                    logger.info(
                        f"{cls.__name__}: Input was a JSON string, parsed successfully"
                    )
                    data = parsed_data
            except json.JSONDecodeError:
                logger.warning(f"{cls.__name__}: Input was a string but not valid JSON")
                # Continue with original data

        if isinstance(data, dict):
            # If db_path is a JSON string (common error from agents), try to parse it
            if "db_path" in data and isinstance(data["db_path"], str):
                db_path = data["db_path"]

                # Check if db_path looks like it might contain JSON (has curly braces)
                if "{" in db_path and "}" in db_path:
                    try:
                        # Try to extract a complete JSON object
                        first_brace = db_path.find("{")
                        last_brace = db_path.rfind("}")
                        if first_brace >= 0 and last_brace > first_brace:
                            json_part = db_path[first_brace : last_brace + 1]
                            parsed_json = json.loads(json_part)

                            if (
                                isinstance(parsed_json, dict)
                                and "db_path" in parsed_json
                            ):
                                logger.info(
                                    f"{cls.__name__}: Successfully extracted JSON from db_path"
                                )
                                # Merge the parsed JSON with the original data
                                # but keep the original db_path if the parsed one doesn't look like a path
                                if not parsed_json["db_path"].endswith(".duckdb"):
                                    parsed_json["db_path"] = db_path.split("{")[
                                        0
                                    ].strip()
                                return parsed_json
                    except json.JSONDecodeError:
                        # Not valid JSON, continue with original data
                        pass
        return data


# Global instance of AnomalyDetector for the anomaly detection functions
anomalies_detector_instance: Dict[str, Optional[AnomalyDetector]] = {"instance": None}


def detect_anomalies(tool_input: Any) -> Dict[str, Any]:
    """Detect anomalies across multiple tables efficiently.

    This function processes the most important tables where anomalies are likely to occur,
    analyzing all available data for comprehensive anomaly detection. It's ideal for
    broad queries about anomalies across the entire dataset.

    Args:
        tool_input (Any): Raw input from the agent, expected to be a JSON string or a dictionary
                          containing db_path and optionally tables.

    Returns:
        Dict[str, Any]: Combined anomaly detection results across multiple tables.
            - status: 'success', 'partial_success', or 'error'.
            - tables_processed: List of tables successfully analyzed.
            - tables_skipped: List of tables that were skipped.
            - anomalies_found: Total number of anomalies found across all tables.
            - results: Dictionary mapping table names to their individual results.
    """
    logger.info("detect_anomalies received request", input_type=str(type(tool_input)))
    start_time = time.time()

    args_dict: Optional[Dict[str, Any]] = None
    if isinstance(tool_input, str):
        try:
            # Try to clean up the input if it contains multiple JSON objects
            if tool_input.count("{") > 1 and tool_input.count("}") > 1:
                # Find the first and last braces for a complete JSON object
                first_open = tool_input.find("{")
                last_close = tool_input.rfind("}")
                if first_open >= 0 and last_close > first_open:
                    # Extract the entire string between first { and last }
                    clean_input = tool_input[first_open : last_close + 1]
                    logger.warning(
                        f"Detected multiple JSON objects in input, attempting to parse complete JSON: {clean_input}"
                    )
                    try:
                        args_dict = json.loads(clean_input)
                    except json.JSONDecodeError:
                        # If that fails, try finding just the first complete JSON object
                        first_close = tool_input.find("}")
                        if first_close > first_open:
                            clean_input = tool_input[first_open : first_close + 1]
                            logger.warning(
                                f"Trying first JSON object only: {clean_input}"
                            )
                            args_dict = json.loads(clean_input)
                        else:
                            # Last resort: try the original input
                            args_dict = json.loads(tool_input)
                else:
                    args_dict = json.loads(tool_input)
            else:
                args_dict = json.loads(tool_input)
        except json.JSONDecodeError as e:
            logger.error("Failed to decode tool_input string as JSON", error_str=str(e))
            return {
                "status": "error",
                "message": f"Invalid JSON input string: {str(e)}",
            }
    elif isinstance(tool_input, dict):
        args_dict = tool_input
    else:
        logger.error(
            "tool_input is not a string or dict", received_type=str(type(tool_input))
        )
        return {
            "status": "error",
            "message": "Tool input must be a JSON string or a dictionary.",
        }

    if args_dict is None:
        logger.error("args_dict is None after input processing, which is unexpected.")
        return {
            "status": "error",
            "message": "Internal error: Failed to process tool input.",
        }

    try:
        # Validate the dictionary using Pydantic
        parsed_args = DetectAnomaliesBatchInput.model_validate(args_dict)

        # Get or create the global anomaly detector instance
        global anomalies_detector_instance
        detector = anomalies_detector_instance["instance"]
        if detector is None or detector.db_path != parsed_args.db_path:
            detector = AnomalyDetector(parsed_args.db_path)
            anomalies_detector_instance["instance"] = detector
            # Initialize the detector (loads schemas, starts background model training)
            # Run in a new event loop since this is now a synchronous function
            loop = asyncio.new_event_loop()
            try:
                loop.run_until_complete(detector.initialize())
            finally:
                loop.close()

        # Run batch detection with a time limit
        # Run in a new event loop since this is now a synchronous function
        loop = asyncio.new_event_loop()
        try:
            result = loop.run_until_complete(
                detector.detect_anomalies_batch(  # Using the batch method internally
                    tables=parsed_args.tables,
                    max_rows_per_table=0,  # 0 means use all available data
                    time_limit_seconds=45.0,  # Leave buffer for the 60s timeout
                )
            )
        finally:
            loop.close()

        # Truncate results to avoid excessive token usage
        if "results" in result:
            for table_name, table_result in result["results"].items():
                if (
                    "data" in table_result
                    and len(table_result["data"]) > MAX_TOOL_OUTPUT_ROWS
                ):
                    table_result["data"] = table_result["data"][:MAX_TOOL_OUTPUT_ROWS]
                    table_result["displayed_row_count"] = MAX_TOOL_OUTPUT_ROWS
                    table_result["message"] = (
                        f"Output truncated to first {MAX_TOOL_OUTPUT_ROWS} anomaly detection results. Original result count: {table_result.get('row_count', 0)}."
                    )

        logger.info(
            "detect_anomalies completed",
            time_taken=time.time() - start_time,
            tables_processed=len(result.get("tables_processed", [])),
            tables_skipped=len(result.get("tables_skipped", [])),
            anomalies_found=result.get("anomalies_found", 0),
        )

        # Extract essential metrics for structured logging
        tables_processed = len(result.get("tables_processed", []))
        anomalies_found = result.get("anomalies_found", 0)
        total_rows = result.get("total_rows_analyzed", 0)

        # Log the detailed results for debugging purposes, but don't return them
        logger.debug(
            "Full anomaly detection result",
            result=json.dumps(result, default=str),
            tables_processed=tables_processed,
            anomalies_found=anomalies_found,
            total_rows=total_rows,
        )

        # Create an ultra-minimal version with only what the agent needs
        # This preserves just enough structure for the agent while drastically reducing size
        simplified_result = {
            "status": result.get("status", ""),
            "summary": f"Anomaly detection completed on {tables_processed} tables. Found {anomalies_found} anomalies across {total_rows} rows.",
            "anomalies_found": anomalies_found,
            "tables_summary": [],
        }

        # Add only the most essential details for each table
        for table_name, table_result in result.get("results", {}).items():
            table_anomalies = table_result.get("anomaly_count", 0)
            # Only include tables with anomalies to further reduce size
            if table_anomalies > 0:
                simplified_result["tables_summary"].append(
                    {"table": table_name, "anomalies": table_anomalies}
                )

        # Return the simplified result - still structured but much smaller
        return simplified_result

    except ValidationError as e:
        logger.error("Validation error in detect_anomalies", error=str(e))
        return {"status": "error", "message": f"Invalid input: {str(e)}"}
    except Exception as e:
        logger.error("Error in detect_anomalies", error=str(e), exc_info=True)
        return {"status": "error", "message": f"Unexpected error: {str(e)}"}


# The detect_anomalies_ml function has been removed as it's been replaced by the more efficient detect_anomalies function

# The _fast_statistical_anomaly_detection function has been removed as it was only used by the detect_anomalies_ml function


# Register tools for LangChain - all functions are now synchronous
query_duckdb_tool: StructuredTool = StructuredTool.from_function(
    func=query_duckdb,
    name="query_duckdb",
    description=(
        "Execute a SQL query on a DuckDB database. Use this to fetch telemetry data. "
        "Parameters: db_path (string, path to database), query (string, SQL query). "
        "Example: query_duckdb(db_path='/path/to/db.duckdb', query='SELECT AVG(altitude) FROM telemetry_global_position'). "
        "Returns query results with status, data, and row_count. "
        "\n\nWHEN TO USE THIS TOOL:\n"
        "- Use for DIRECT DATA RETRIEVAL when you need specific values, statistics, or filtered data\n"
        "- DO NOT use for anomaly detection, critical errors, or identifying unusual patterns - use detect_anomalies tool instead\n"
        "- Good for: altitude measurements, speed calculations, position tracking, etc.\n"
        "\n\nWhen discussing query results, always provide context about tables and columns: "
        "\n- When mentioning a specific table (e.g., 'telemetry_attitude'), explain what kind of data this table contains (e.g., 'the telemetry_attitude table contains vehicle orientation data including roll, pitch, and yaw angles'). "
        "\n- When mentioning specific columns (e.g., 'roll'), explain what this column represents and its unit of measurement (e.g., 'the roll column represents the aircraft's roll angle measured in degrees'). "
        "\n- For numerical values, include appropriate units and normal ranges when possible. "
        "\n- When presenting query results, translate technical column names into user-friendly descriptions. "
        "\n- Always include the number of rows returned to give context about the data volume."
    ),
    args_schema=QueryDuckDBInput,
)

# The detect_anomalies_ml_tool has been removed as it's been replaced by the more efficient detect_anomalies_tool

detect_anomalies_tool: StructuredTool = StructuredTool.from_function(
    func=detect_anomalies,
    name="detect_anomalies",
    description=(
        "Detect anomalies across multiple tables efficiently. "
        "Parameters: db_path (string), tables (optional list of table names). "
        "Example: detect_anomalies(db_path='/path/to/db.duckdb') or detect_anomalies(db_path='/path/to/db.duckdb', tables=['telemetry_attitude', 'telemetry_gps_raw_int']). "
        "If tables parameter is omitted, will use priority tables (attitude, position, GPS, etc). "
        "Returns combined results across all processed tables. "
        "\n\nWHEN TO USE THIS TOOL:\n"
        "1. For SPECIFIC ANOMALIES in a particular system: Use when the query mentions anomalies, issues, problems, or errors with a specific component (e.g., 'GPS anomalies', 'attitude issues', 'battery problems'). Specify the relevant table in the 'tables' parameter. When reporting results, ONLY include anomalies from the specific system the user asked about.\n"
        "2. For BROAD FLIGHT ANOMALIES: Use when the query asks about overall flight anomalies, critical errors, unusual patterns, or issues across the entire flight (e.g., 'list all critical errors', 'find anomalies during mid-flight', 'what went wrong during the flight'). In this case, don't specify tables to analyze all priority tables.\n"
        "3. ALWAYS USE THIS TOOL (not query_duckdb) when looking for critical errors, warnings, or unusual patterns in the flight data.\n"
        "4. IMPORTANT: If the user asks about issues in a specific system (e.g., GPS), even if you run the tool on all tables, your response should ONLY discuss the anomalies relevant to the system they asked about.\n"
        "5. CRITICAL GPS EXAMPLES: ALWAYS use this tool (NEVER query_duckdb) for ANY of these queries:\n"
        "   - 'Analyze GPS data for issues' → Use detect_anomalies with tables=['telemetry_gps_raw_int']\n"
        "   - 'Check GPS quality' → Use detect_anomalies with tables=['telemetry_gps_raw_int']\n"
        "   - 'Find GPS fix problems' → Use detect_anomalies with tables=['telemetry_gps_raw_int']\n"
        "\n\nWhen discussing results, always provide context about tables and columns: "
        "\n- When mentioning a specific table (e.g., 'telemetry_attitude'), explain what kind of data this table contains (e.g., 'the telemetry_attitude table contains vehicle orientation data including roll, pitch, and yaw angles'). "
        "\n- When mentioning specific columns (e.g., 'roll'), explain what this column represents and its unit of measurement (e.g., 'the roll column represents the aircraft's roll angle measured in degrees'). "
        "\n- For numerical values, include appropriate units and normal ranges when possible. "
        "\n- Prioritize explaining tables and columns in a way that helps users understand the telemetry data without requiring technical knowledge."
    ),
    args_schema=DetectAnomaliesBatchInput,
)


class TokenUsageCallback(BaseCallbackHandler):
    """Callback handler to track OpenAI LLM token usage."""

    def __init__(self) -> None:
        """Initialize the callback with zeroed token counts."""
        self.token_usage: Dict[str, int] = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }

    def reset(self) -> None:
        """Reset token usage counts to zero."""
        self.token_usage = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }
        logger.debug("Reset token usage", callback_id=id(self))

    def on_llm_end(self, response: Any, **kwargs: Any) -> None:
        """Update token usage from LLM response.

        Args:
            response (Any): LLM response object containing token usage data.
            kwargs (Any): Additional keyword arguments (ignored).
        """
        usage: Dict[str, int] = (
            response.llm_output.get("token_usage", {}) if response.llm_output else {}
        )
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
        max_tokens: Optional[int] = None,
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
        self.anomaly_detector: Optional[AnomalyDetector] = None

    async def async_initialize(self) -> None:
        """Initialize TelemetryAgent dependencies asynchronously.

        Args:
            None

        Raises:
            ValueError: If token limits or configurations are invalid.
        """
        # Set and validate token limits
        self.max_tokens = (
            self.max_tokens if self.max_tokens is not None else MAX_MODEL_TOKENS
        )
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
            callbacks=[self.token_callback],
        )

        # Initialize the anomaly detector in the background
        # This will return quickly but start training models in the background
        self.logger.info("Starting anomaly detector initialization")
        self.anomaly_detector = AnomalyDetector(db_path=self.db_path)
        # Start initialization but don't wait for model training
        await self.anomaly_detector.initialize()
        self.logger.info("Anomaly detector initialized successfully")

        # Get tables info for the prompt
        tables_info = await self.get_tables_as_string()

        # Define ReAct prompt template
        self.prompt = PromptTemplate(
            template=REACT_SYSTEM_PROMPT,
            input_variables=[
                "input",
                "agent_scratchpad",
                "agent_scratchpad_content",
                "chat_history",
            ],
            partial_variables={
                "tables": tables_info,
                "tools": "{tools}",
                "tool_names": "{tool_names}",
            },
        )

        # Initialize conversation memory manager
        self.conversation_memory_manager = ConversationMemoryManager()
        await self.conversation_memory_manager.async_initialize(
            llm=self.llm,
            model_name=self.llm_model,
            llm_token_encoder=self.token_encoder,
            max_context_tokens=self.max_context_tokens,
            fallback_token_limit=self.fallback_token_limit,
            embeddings=self.embeddings,
        )

        # Set up ReAct agent with tools
        tools: List[Tool] = [query_duckdb_tool, detect_anomalies_tool]

        # Create ReAct agent
        agent = create_react_agent(llm=self.llm, tools=tools, prompt=self.prompt)

        # Create agent executor without memory (we'll handle memory manually)
        self.agent_executor = AgentExecutor(
            agent=agent,
            tools=tools,
            verbose=True,
            handle_parsing_errors='Check your tool input format. It should be valid JSON with proper key-value pairs. For example: {"db_path": "/path/to/file.duckdb", "query": "SELECT * FROM table"}',
            max_iterations=20,  # Prevent infinite loops (Increased from 10)
            max_execution_time=200,  # Increased from 120s, less than CHAT_TIMEOUT_SECONDS (240s)
            return_intermediate_steps=True,  # Ensure agent's thought process is returned
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
            self.logger.error(
                "max_context_tokens must be positive",
                max_context_tokens=self.max_context_tokens,
            )
            raise ValueError(
                f"Invalid max_context_tokens: {self.max_context_tokens} must be positive"
            )
        if self.fallback_token_limit <= 0:
            self.logger.error(
                "fallback_token_limit must be positive",
                fallback_token_limit=self.fallback_token_limit,
            )
            raise ValueError(
                f"Invalid fallback_token_limit: {self.fallback_token_limit} must be positive"
            )
        if self.max_tokens > MAX_TOKEN_SAFETY_LIMIT:
            self.logger.error(
                "max_tokens exceeds safety limit",
                max_tokens=self.max_tokens,
                safety_limit=MAX_TOKEN_SAFETY_LIMIT,
            )
            raise ValueError(
                f"max_tokens {self.max_tokens} exceeds safety limit {MAX_TOKEN_SAFETY_LIMIT}"
            )
        if self.max_tokens <= RESERVED_RESPONSE_TOKENS:
            self.logger.error(
                "max_tokens must be greater than RESERVED_RESPONSE_TOKENS",
                max_tokens=self.max_tokens,
                reserved=RESERVED_RESPONSE_TOKENS,
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
                self.logger.error(
                    f"Database path is invalid or file does not exist for table {table_name}",
                    db_path=self.db_path,
                )
                # No fallback to static schema here; if DB path is bad, we can't confirm anything.
                return []

            with duckdb.connect(self.db_path, read_only=True) as conn:
                tables_result = conn.execute(
                    "SELECT table_name FROM information_schema.tables WHERE table_schema = 'main';"
                ).fetchall()
                available_tables = [t[0].lower() for t in tables_result]

                if table_name.lower() in available_tables:
                    columns_data = conn.execute(f"DESCRIBE {table_name};").fetchall()
                    actual_column_names = [col_data[0] for col_data in columns_data]
                    self.logger.info(
                        f"Dynamically fetched schema for {table_name}",
                        columns=actual_column_names,
                        db_path=self.db_path,
                    )
                else:
                    self.logger.debug(
                        f"Table {table_name} from TELEMETRY_SCHEMA not found in DB {self.db_path}. It will NOT be included in the agent's list of available tables."
                    )
                    actual_column_names = (
                        []
                    )  # Explicitly empty, no static fallback here
        except duckdb.Error as e:
            self.logger.error(
                f"DuckDB error when trying to determine schema for {table_name} from {self.db_path}",
                error=str(e),
            )
            actual_column_names = []  # No static fallback on error
            self.logger.warning(
                f"Could not determine schema for {table_name} due to DuckDB error. It will be treated as unavailable for the agent's direct table list."
            )
        except Exception as e:
            self.logger.error(
                f"Unexpected error fetching schema for {table_name} from {self.db_path}",
                error=str(e),
                error_type=str(type(e)),
            )
            actual_column_names = []  # No static fallback on error
            self.logger.warning(
                f"Could not determine schema for {table_name} due to unexpected error. It will be treated as unavailable for the agent's direct table list."
            )

        if not actual_column_names and os.path.exists(
            self.db_path or ""
        ):  # Log only if db_path was valid
            # This log might be redundant if already logged that table was not found or error occurred.
            # Consider if this specific log "Could not determine columns..." is still needed if the cases above are clear.
            pass  # Keeping it for now, but can be removed if too noisy.
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
            actual_columns = await asyncio.to_thread(
                self._fetch_actual_schema_for_table, table_name
            )

            if actual_columns:  # If columns were found (either dynamic or fallback)
                column_names_str = ", ".join(actual_columns)
                table_strings.append(f"{table_name}({column_names_str})")
            else:
                # If no columns could be determined for this table from TELEMETRY_SCHEMA, log and omit.
                # Changed from warning to debug as this is expected if table not in DB
                self.logger.debug(
                    f"No columns determined for table '{table_name}' from TELEMETRY_SCHEMA "
                    + f"(via dynamic fetch from {self.db_path}). Omitting from prompt schema string."
                )

        if not table_strings:
            self.logger.error(
                "No tables could be processed to form the schema string for the prompt. Agent might not have table info.",
                db_path=self.db_path,
            )
            return "Error: No table schema information could be loaded."

        final_schema_string = ", ".join(table_strings)
        self.logger.info(
            "Generated table schema string for prompt using dynamic fetching (with fallback)",
            schema_string_snippet=final_schema_string[:500],
        )
        return final_schema_string

    async def process_message(
        self, message: str, max_tokens: Optional[int] = None
    ) -> Tuple[str, Optional[Dict[str, Any]]]:
        """Process a user message and generate a response with anomaly detection results.

        This method handles the complete workflow of processing a user query about telemetry data:
        1. Validates input parameters and token limits
        2. Retrieves conversation history from memory
        3. Performs vector search to identify relevant tables and anomaly hints
        4. Enhances the user query with context information
        5. Invokes the ReAct agent to process the query using available tools
        6. Updates the agent's scratchpad with intermediate reasoning steps
        7. Adds the conversation to memory for future reference
        8. Returns the response with detailed metadata for debugging

        Args:
            message (str): User query about telemetry data (e.g., "Are there any GPS anomalies?").
            max_tokens (Optional[int]): Maximum tokens for LLM response (defaults to self.max_tokens).

        Returns:
            Tuple[str, Optional[Dict[str, Any]]]:
                - LLM response as a string
                - Metadata dictionary containing token usage, intermediate steps, and other debug info

        Raises:
            ValueError: If message is empty, token limits are invalid, or agent is not initialized.
            asyncio.TimeoutError: If processing exceeds CHAT_TIMEOUT_SECONDS.
            RuntimeError: If agent components are not properly initialized.
        """
        # Validate input message
        if not message.strip():
            self.logger.warning(ERROR_EMPTY_MESSAGE)
            return "", None

        # Validate agent components are initialized
        if self.agent_executor is None:
            self.logger.error(ERROR_AGENT_NOT_INITIALIZED)
            raise RuntimeError(ERROR_AGENT_NOT_INITIALIZED)

        if self.conversation_memory_manager is None:
            self.logger.error(ERROR_MEMORY_NOT_INITIALIZED)
            raise RuntimeError(ERROR_MEMORY_NOT_INITIALIZED)

        if self.token_callback is None:
            self.logger.error(ERROR_TOKEN_CALLBACK_NOT_INITIALIZED)
            raise RuntimeError(ERROR_TOKEN_CALLBACK_NOT_INITIALIZED)

        try:
            # Use provided max_tokens or default
            effective_max_tokens: int = (
                max_tokens if max_tokens is not None else self.max_tokens
            )

            # Validate token limits
            if effective_max_tokens <= 0:
                self.logger.error(
                    "max_tokens must be positive", max_tokens=effective_max_tokens
                )
                raise ValueError(ERROR_INVALID_MAX_TOKENS.format(effective_max_tokens))

            if effective_max_tokens > MAX_TOKEN_SAFETY_LIMIT:
                self.logger.error(
                    "max_tokens exceeds safety limit",
                    max_tokens=effective_max_tokens,
                    safety_limit=MAX_TOKEN_SAFETY_LIMIT,
                )
                raise ValueError(
                    ERROR_MAX_TOKENS_EXCEEDS_LIMIT.format(
                        effective_max_tokens, MAX_TOKEN_SAFETY_LIMIT
                    )
                )

            if effective_max_tokens <= RESERVED_RESPONSE_TOKENS:
                self.logger.error(
                    "max_tokens must be greater than RESERVED_RESPONSE_TOKENS",
                    max_tokens=effective_max_tokens,
                    reserved=RESERVED_RESPONSE_TOKENS,
                )
                raise ValueError(
                    ERROR_MAX_TOKENS_TOO_SMALL.format(
                        effective_max_tokens, RESERVED_RESPONSE_TOKENS
                    )
                )

            # Reset token usage for this query
            self.token_callback.reset()

            # Retrieve conversation memory
            memory, memory_strategy = (
                await self.conversation_memory_manager.aget_memory()
            )
            chat_history_messages = memory.load_memory_variables({}).get("history", [])

            # Format chat_history_messages into a string for the prompt
            chat_history_string = ""
            if isinstance(chat_history_messages, list):
                for msg in chat_history_messages:
                    if hasattr(msg, "type") and hasattr(
                        msg, "content"
                    ):  # LangChain BaseMessage structure
                        chat_history_string += f"{msg.type.upper()}: {msg.content}\\n"
                    else:  # Fallback for other potential structures
                        chat_history_string += f"{str(msg)}\\n"
            if not chat_history_string:
                chat_history_string = "No prior conversation history."

            # Identify relevant tables and hints using vector store
            search_results = await self.vector_store_manager.async_similarity_search(
                message, k=VECTOR_RETRIEVER_K
            )
            relevant_tables: List[str] = list(
                set(  # Use set to ensure uniqueness
                    result.metadata.get("table", "")
                    for result in search_results
                    if result.metadata.get("table")
                )
            )
            anomaly_hints: List[str] = list(
                set(  # Use set to ensure uniqueness
                    result.metadata.get("anomaly_hint", "")
                    for result in search_results
                    if result.metadata.get("anomaly_hint")
                )
            )

            # Prepare enhanced input with context
            # The prompt already has {tables} so we don't need to add relevant_tables to enhanced_message explicitly
            # Anomaly hints are general guidance; the main prompt now emphasizes data-driven investigation.
            enhanced_message = f"{message}\n\nContext provided by system (use for general guidance but prioritize direct data querying):\n- Anomaly detection hints related to your query: {', '.join(anomaly_hints)}\n- Database path for tools: {self.db_path}"

            # The custom scratchpad content is passed separately
            current_custom_scratchpad = self.scratchpad.to_string()

            final_input_for_executor = {
                "input": enhanced_message,
                "agent_scratchpad_content": current_custom_scratchpad,
                "chat_history": chat_history_string,  # Pass the formatted chat history string
                # agent_scratchpad for ReAct internal steps is handled by AgentExecutor
            }

            self.logger.debug(
                "ReAct agent executor input",
                final_input_for_executor_keys=list(final_input_for_executor.keys()),
                chat_history_snippet=chat_history_string[:200],
                custom_scratchpad_snippet=current_custom_scratchpad[:200],
            )

            result = await asyncio.wait_for(
                self.agent_executor.ainvoke(final_input_for_executor),
                timeout=CHAT_TIMEOUT_SECONDS,
            )

            # Log all keys from the agent executor result for debugging
            self.logger.info(
                "Full keys from AgentExecutor result", keys=list(result.keys())
            )

            # Populate our custom multi-turn scratchpad
            intermediate_steps_raw = result.get("intermediate_steps", [])
            if intermediate_steps_raw:
                self.logger.info(
                    f"Populating custom scratchpad with {len(intermediate_steps_raw)} steps from current turn."
                )
                for agent_action_obj, observation_str in intermediate_steps_raw:
                    # agent_action_obj is of type langchain_core.agents.AgentAction
                    # observation_str is the string output of the tool
                    action_details_for_scratchpad = {
                        "tool": agent_action_obj.tool,
                        "tool_input": agent_action_obj.tool_input,
                        # The 'log' in AgentAction contains the thought process leading to this action.
                        "thought_process_for_this_action": agent_action_obj.log.strip(),
                    }
                    self.scratchpad.add_step(
                        action_details_for_scratchpad, str(observation_str)
                    )
            else:
                self.logger.warning(
                    "No intermediate_steps from AgentExecutor for this turn to add to custom scratchpad."
                )

            response: str = result.get(
                "output", "Error: No output from agent."
            )  # Ensure there's a default
            response_lower: str = response.strip().lower()
            is_clarification: bool = response.strip().endswith("?") or any(
                phrase in response_lower for phrase in CLARIFICATION_PHRASES
            )
            if is_clarification and not response.strip().lower().startswith(
                "thought"
            ):  # Avoid adding to thoughts
                self.logger.info("Response requests clarification", response=response)
                response += "\nPlease provide more details to proceed."

            # We now handle anomaly detection queries earlier in the method
            # by returning a direct message if models are still training

            await self.conversation_memory_manager.add_message((message, response))

            # Get intermediate steps for metadata
            # This is crucial for debugging if the agent is taking steps
            intermediate_steps = result.get("intermediate_steps", [])
            if not intermediate_steps:
                self.logger.warning(
                    "No intermediate steps were recorded by the agent for this call.",
                    agent_output=response,
                )

            # Sanitize intermediate_steps for logging in metadata
            logged_intermediate_steps_for_metadata = []
            for (
                action,
                observation,
            ) in intermediate_steps_raw:  # Use intermediate_steps_raw here
                logged_observation_str = str(observation)
                if len(logged_observation_str) > 500:  # Arbitrary limit for logging
                    logged_observation_str = (
                        logged_observation_str[:500] + " ... (truncated in log)"
                    )

                # action is an AgentAction object. Access its attributes.
                tool_name = action.tool if hasattr(action, "tool") else "UnknownTool"
                tool_input_val = (
                    action.tool_input
                    if hasattr(action, "tool_input")
                    else "UnknownInput"
                )

                logged_intermediate_steps_for_metadata.append(
                    (
                        f"Tool: {tool_name}, Input: {tool_input_val}",
                        logged_observation_str,
                    )
                )

            metadata: Dict[str, Any] = {
                "relevant_tables": relevant_tables,  # These are from vector search, not agent's direct use confirmation
                "anomaly_hints": anomaly_hints,
                "intermediate_steps": logged_intermediate_steps_for_metadata,  # Use the correctly formatted one
                "token_usage": self.token_callback.token_usage,
                "memory_strategy": memory_strategy.value,
                "is_clarification": is_clarification,
                "agent_scratchpad_custom_context": current_custom_scratchpad,
                "agent_internal_scratchpad_final": result.get("agent_scratchpad", "")
                or result.get("log", ""),  # ReAct often uses 'log'
                "agent_type": "react",
                "raw_agent_output": result.get(
                    "output", ""
                ),  # Add raw output for easier debugging
            }

            # Manage custom scratchpad size to prevent token limit issues
            scratchpad_tokens = len(
                self.token_encoder.encode(current_custom_scratchpad)
            )
            if scratchpad_tokens > self.max_context_tokens // 2:  # type: ignore
                self.logger.warning(
                    ERROR_SCRATCHPAD_TOKEN_LIMIT,
                    tokens=scratchpad_tokens,
                    max_context_tokens=self.max_context_tokens,
                    token_percentage=f"{(scratchpad_tokens / self.max_context_tokens) * 100:.1f}%",
                )
                # Keep only the most recent steps to reduce token usage
                self.scratchpad.intermediate_steps = self.scratchpad.intermediate_steps[
                    -10:
                ]
                self.logger.info(
                    "Truncated custom scratchpad to last 10 steps to manage token limit"
                )

            self.logger.info(
                "Generated response",
                response_length=len(response),
                # Full metadata can be very verbose if intermediate steps are complex.
                # Consider logging only key parts or a summary of metadata here.
                metadata_intermediate_steps_count=len(
                    logged_intermediate_steps_for_metadata
                ),
                metadata_token_usage=metadata["token_usage"],
                metadata_is_clarification=metadata["is_clarification"],
            )
            return response, metadata

        except asyncio.TimeoutError:
            self.logger.error(
                "Message processing timed out",
                message=message,
                timeout=CHAT_TIMEOUT_SECONDS,
                session_id=self.session_id,
            )
            raise ValueError(ERROR_PROCESSING_TIMEOUT.format(CHAT_TIMEOUT_SECONDS))
        except Exception as e:
            self.logger.error(
                "Failed to process message",
                message=message,
                error_str=str(e),  # Use error_str for consistency with other logs
                error_type=str(type(e)),
            )
            raise  # Re-raise the caught exception
