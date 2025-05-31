"""
Agent scratchpad module for tracking agent actions and observations.

This module provides a scratchpad for tracking agent actions and observations
during multi-turn conversations. It helps maintain context across turns and
provides a formatted history for the agent to reference.
"""

import json
import re
import time
from typing import Any, Dict, List, Tuple, Union
import structlog
from langchain_core.agents import AgentAction, AgentFinish

# Constants for scratchpad management
MAX_SCRATCHPAD_STEPS = 1  # AGGRESSIVE RETENTION: Keep only last 1 step for performance
SCRATCHPAD_RETENTION_POLICY = "aggressive_last_1_step"  # Policy description
MAX_OBSERVATION_STORAGE_LENGTH = 5000  # Maximum length of observation to store in full
MAX_OBSERVATION_SNIPPET_LENGTH = (
    100  # Maximum length of observation snippet for logging
)
DEFAULT_EMPTY_SCRATCHPAD_MESSAGE = "No previous actions recorded."

logger = structlog.get_logger(__name__)


class AgentScratchpad:
    """
    Scratchpad for tracking agent actions and observations during multi-turn conversations.

    This class maintains a history of agent actions and their results, providing
    context for the agent across multiple turns. It formats this history in a way
    that can be included in the agent's prompt, helping it understand what actions
    have already been taken and their outcomes.
    
    PERFORMANCE OPTIMIZATION: Implements aggressive retention policy - keeps only the
    last step to prevent memory buildup and reduce token usage across conversations.
    """

    def __init__(self, session_id: str) -> None:
        """
        Initialize the AgentScratchpad.

        Args:
            session_id (str): Unique identifier for the session
        """
        self.session_id = session_id
        self.intermediate_steps: List[Tuple[Dict[str, Any], str]] = []
        self.agent_actions: List[Dict[str, Any]] = []
        self.logger = logger.bind(session_id=session_id)
        self.logger.info("Initialized agent scratchpad with aggressive retention policy")

    def process_agent_action(
        self, next_action: Union[AgentAction, AgentFinish]
    ) -> None:
        """
        Process an agent action and add it to the scratchpad.

        This method takes an agent action (or finish), processes it to extract
        relevant details, and adds it to the scratchpad history. It also handles
        logging and formatting of the action and its results.

        Args:
            next_action (Union[AgentAction, AgentFinish]): The agent action to process
        """
        # Skip AgentFinish objects as they don't contain actions to execute
        if isinstance(next_action, AgentFinish):
            self.logger.debug("Received AgentFinish, skipping")
            return

        # Process the action and create a structured log
        agent_action_log = self._process_agent_action_full(next_action)

        # Get full observation
        observation = self._get_observation_for_agent_action(agent_action_log)
        observation_str = str(observation)

        # Determine if this is a large observation that needs special handling
        large_observation = len(observation_str) > MAX_OBSERVATION_STORAGE_LENGTH

        # Clean up observation for storage
        clean_observation = self._clean_observation_string(observation)

        # For especially large observations, use a more aggressive cleaning approach
        # and don't store the raw observation at all
        if large_observation:
            self.logger.info(
                "Large observation detected",
                observation_length=len(observation_str),
                clean_observation=clean_observation,
            )

            # Save the action log with observation but without raw data
            self.agent_actions.append(
                {
                    "action": agent_action_log,
                    "observation": clean_observation,
                    # Don't store raw observation for large outputs
                    "observation_raw": f"[Large output ({len(observation_str)} chars) - not stored in history]",
                }
            )
        else:
            # Normal case - save with raw observation
            self.agent_actions.append(
                {
                    "action": agent_action_log,
                    "observation": clean_observation,
                    "observation_raw": observation,
                }
            )

        # Add to the scratchpad with potentially summarized observation
        self._add_to_scratchpad(agent_action_log, observation_str)

    def _process_agent_action_full(self, next_action: AgentAction) -> Dict[str, Any]:
        """Process an agent action and produce full structured log.

        Args:
            next_action (AgentAction): The agent action to process

        Returns:
            Dict[str, Any]: Full structured log of the processed action
        """
        action_type = next_action.tool if hasattr(next_action, "tool") else "thinking"
        action_input = (
            next_action.tool_input if hasattr(next_action, "tool_input") else ""
        )

        # For certain tools that may return large responses, sanitize the input
        # to prevent large logs
        if action_type == "detect_anomalies":
            # Log just what's necessary for the detect_anomalies tool
            try:
                if isinstance(action_input, str) and len(action_input) > 100:
                    # For long string inputs, just log a summary
                    action_input = (
                        f"[Anomaly detection request - {len(action_input)} chars]"
                    )
                elif isinstance(action_input, dict):
                    # For dict inputs, just log that it's a dict with db_path
                    if "db_path" in action_input:
                        action_input = {
                            "db_path": action_input["db_path"],
                            "...": "[details omitted]",
                        }
            except Exception:
                # If anything fails, log a generic summary
                action_input = "[Anomaly detection request]"

        action_details = {"tool": action_type, "input": action_input}

        # Extract thought process if available
        if hasattr(next_action, "log"):
            action_details["thought_process_for_this_action"] = next_action.log

        # Add timestamp for debugging
        action_details["timestamp"] = time.time()

        return action_details

    def _get_observation_for_agent_action(self, action_details: Dict[str, Any]) -> Any:
        """Get the observation for an agent action.

        Args:
            action_details (Dict[str, Any]): Details of the agent action

        Returns:
            Any: The observation for the agent action
        """
        # In the current implementation, the observation is added later
        # This is a placeholder for future implementations
        return None

    def _clean_observation_string(self, observation: Any) -> str:
        """Clean up observation strings for better readability.

        This method creates clean, readable summaries of tool outputs and
        adds appropriate context markers to help the agent understand
        which context (time ranges, etc.) applies to each observation.

        Args:
            observation (Any): The observation to clean

        Returns:
            str: A cleaned string representation of the observation
        """
        # Convert to string first
        obs_str = str(observation)

        # If it's a JSON response with status and data, format it more cleanly
        if isinstance(observation, dict):
            if "status" in observation and observation.get("status") == "success":
                # For query results
                if "data" in observation and isinstance(observation["data"], list):
                    row_count = len(observation["data"])
                    if row_count == 0:
                        return "Query returned no results."
                    elif row_count <= 3:
                        # For small result sets, show a simplified version
                        return f"Query successful: {row_count} results found."
                    else:
                        return f"Query successful: {row_count} results found."

                # For anomaly detection results - updated for ultra-minimal format
                if "anomalies_found" in observation and "summary" in observation:
                    # Check if we have the enhanced time summary from telemetry_agent
                    if "time_summary" in observation:
                        return observation.get("summary", "Anomaly detection completed.")
                    
                    # Fall back to extracting time info ourselves if needed
                    time_boot_info = ""
                    
                    # Check if time range filter was applied
                    if "time_boot_ms_range" in observation:
                        time_range = observation["time_boot_ms_range"]
                        start = time_range.get("start", "?")
                        end = time_range.get("end", "?")
                        time_boot_info += f" [CONTEXT BOUNDARY MARKER - QUERY-SPECIFIC TIME FILTER] CRITICAL: Results filtered to time range: {start}-{end}ms. Only consider anomalies within this range for THIS QUERY ONLY. This filter MUST BE DISCARDED for any new user question."
                    else:
                        time_boot_info += " [CONTEXT: FULL DATASET - This result includes ALL anomalies across the ENTIRE flight with NO time filtering]"
                    
                    if observation.get("includes_time_boot_ms") and "tables_summary" in observation:
                        # Check for time insights first
                        if "time_insights" in observation and observation["time_insights"]:
                            insights = observation["time_insights"]
                            
                            # Add flight phase information if available
                            if "most_problematic_phase" in insights and "flight_phases" in insights:
                                phase = insights["most_problematic_phase"]
                                phase_data = insights["flight_phases"][phase]
                                time_boot_info += f" Most anomalies during {phase} phase ({phase_data['start_time']}-{phase_data['end_time']}ms)."
                            
                            # Add correlated anomalies info if available
                            elif "correlated_anomalies" in insights and insights["correlated_anomalies"]:
                                correlated = insights["correlated_anomalies"][0]
                                time_boot_info += f" Found related anomalies between {' and '.join(correlated['tables'])} at {correlated['time_range'][0]}ms."
                        
                        # Fall back to simple time ranges if no insights
                        if not time_boot_info or "time range" in time_boot_info:
                            time_ranges = []
                            for table in observation.get("tables_summary", []):
                                if "time_range" in table:
                                    time_ranges.append(f"{table['table']}: {table['time_range']['min']}-{table['time_range']['max']}ms")
                                elif "anomaly_details" in table:
                                    times = [detail.get("time_boot_ms") for detail in table.get("anomaly_details", []) 
                                            if detail.get("time_boot_ms") is not None]
                                    if times:
                                        time_ranges.append(f"{table['table']}: {min(times)}-{max(times)}ms")
                            
                            if time_ranges:
                                phase_info = f" Time ranges: {'; '.join(time_ranges[:3])}"
                                if len(time_ranges) > 3:
                                    phase_info += f" and {len(time_ranges) - 3} more tables"
                                time_boot_info += phase_info
                    
                    # Return the summary with time_boot_ms info if available
                    summary = observation.get("summary", "Anomaly detection completed.")
                    return f"{summary}{time_boot_info}"

                # Legacy format handling
                elif (
                    "tables_processed" in observation
                    and "anomalies_found" in observation
                ):
                    tables = observation.get("tables_processed", [])
                    anomalies = observation.get("anomalies_found", 0)
                    tables_str = (
                        ", ".join(tables)
                        if len(tables) <= 3
                        else f"{len(tables)} tables"
                    )
                    return f"Anomaly detection completed on {tables_str}. Found {anomalies} anomalies across {observation.get('total_rows_analyzed', '?')} rows."

            elif "status" in observation and observation.get("status") == "error":
                # For error responses
                return f"Error: {observation.get('message', 'Unknown error')}"

        # For other string observations, try to detect and clean up JSON-like content
        if obs_str.startswith("{") and (
            "'status': 'success'" in obs_str or "'data':" in obs_str
        ):
            # This looks like a raw JSON dictionary in string form
            if "'data': []" in obs_str or "'row_count': 0" in obs_str:
                return "Query returned no results."
            elif "'row_count':" in obs_str:
                # Try to extract row count
                import re

                row_match = re.search(r"'row_count':\s*(\d+)", obs_str)
                if row_match:
                    row_count = row_match.group(1)
                    return f"Query successful: {row_count} results found."

            # Default cleaner response for JSON strings
            return "Query completed successfully."

        # Check if this looks like an anomaly detection result
        if (
            isinstance(obs_str, str)
            and "anomalies_found" in obs_str
            and "tables_summary" in obs_str
        ):
            # This is likely a stringified anomaly detection result
            try:
                # Try to extract the summary from the string
                import re

                # Use a simpler regex pattern with single quotes to avoid escaping issues
                summary_match = re.search(
                    r'summary[\'|"]?:\s*[\'|"]([^\'|"]+)[\'|"]?', obs_str
                )
                
                # Try to extract time-related information if available
                time_boot_info = ""
                
                # Check for time range filter
                time_range_match = re.search(r'time_boot_ms_range[\'|"]?:\s*{[\'|"]?start[\'|"]?:\s*(\d+),\s*[\'|"]?end[\'|"]?:\s*(\d+)}', obs_str)
                if time_range_match:
                    start = time_range_match.group(1)
                    end = time_range_match.group(2)
                    time_boot_info = f" [STRICT CONTEXT BOUNDARY - TIME FILTER: {start}-{end}ms] (CRITICAL: Only anomalies between {start}-{end}ms are relevant for THIS QUERY ONLY. ANY NEW QUESTION MUST DISCARD THIS FILTER COMPLETELY)"
                else:
                    time_boot_info = " [FULL DATASET CONTEXT: Complete flight data with NO time filtering - ALL anomalies included]"
                
                # Look for time summary first
                time_summary_match = re.search(r'time_summary[\'|"]?:\s*[\'|"]([^\'|"]+)[\'|"]', obs_str)
                if time_summary_match:
                    time_boot_info += f" {time_summary_match.group(1)}"
                else:
                    # Fall back to checking for time_boot_ms directly
                    time_ms_match = re.search(r'time_boot_ms[\'|"]?:\s*(\d+)', obs_str)
                    if time_ms_match and not time_boot_info:
                        time_boot_info = f" (time_boot_ms: {time_ms_match.group(1)})"
                    
                    # Check for phase information
                    phase_match = re.search(r'most_problematic_phase[\'|"]?:\s*[\'|"]([^\'|"]+)[\'|"]', obs_str)
                    if phase_match:
                        phase_info = f" (Most anomalies during {phase_match.group(1)} phase)"
                        time_boot_info = time_boot_info + phase_info if time_boot_info else phase_info
                
                if summary_match:
                    return f"{summary_match.group(1)}{time_boot_info}"

                # If we can't extract the summary, return a generic one
                anomalies_match = re.search(r'anomalies_found[\'|"]?:\s*(\d+)', obs_str)
                if anomalies_match:
                    anomalies = anomalies_match.group(1)
                    return f"Anomaly detection completed. Found {anomalies} anomalies.{time_boot_info}"
            except Exception:
                # If anything fails, fall back to generic message
                return "Anomaly detection completed."

        # Return the original string for other cases
        return obs_str

    def _handle_large_observation(
        self, observation_str: str, action_details: Dict[str, Any]
    ) -> str:
        """Handle large observation strings by creating a compact summary.

        Args:
            observation_str: The full observation string
            action_details: Details about the action that produced this observation

        Returns:
            str: A compact summary of the observation
        """
        original_length = len(observation_str)
        if original_length > MAX_OBSERVATION_STORAGE_LENGTH:
            # Create a summarized version of the observation
            tool_name = action_details.get("tool", "unknown_tool")

            # With a 100 char limit, we need extremely concise summaries
            if (
                "query" in tool_name.lower()
                or "search" in tool_name.lower()
                or "duckdb" in tool_name.lower()
            ):
                try:
                    # First check if this is an error response
                    error_match = re.search(
                        r"error[_str]*['\"]?\s*:\s*['\"]([^'\"]+)['\"]?",
                        observation_str,
                    )
                    column_error_match = re.search(
                        r"Referenced column \"([^\"]+)\" not found", observation_str
                    )
                    if error_match or "'status': 'error'" in observation_str:
                        error_msg = (
                            error_match.group(1) if error_match else "Unknown error"
                        )
                        if column_error_match:
                            column_name = column_error_match.group(1)
                            return f"Query failed: Column '{column_name}' not found in table. Try using a different column name."
                        return f"Query failed: {error_msg[:50]}..."

                    # Extract status and row count
                    status = "unknown"
                    row_count = "?"
                    has_data = False

                    # Extract status
                    status_match = re.search(r"status['\"]?\s*:\s*['\"]?([^,}'\"]*)")
                    if status_match:
                        status = status_match.group(1).strip()

                    # Extract row count
                    row_match = re.search(r"row_count['\"]?\s*:\s*(\d+)")
                    if row_match:
                        row_count = row_match.group(1)

                    # Check for data presence
                    has_data = (
                        "data_length" in observation_str
                        or "sample" in observation_str
                        or "data_present" in observation_str
                    )

                    # Create an ultra-compact summary
                    if "query" in observation_str:
                        # Try to extract table name from query string
                        table_name = ""
                        table_match = re.search(
                            r"FROM\s+([\w_]+)", observation_str, re.IGNORECASE
                        )
                        if table_match:
                            table_name = table_match.group(1)

                        # Create minimal summary with just essential info
                        if row_count == "0" or not has_data:
                            return f"No results found in {table_name if table_name else 'query'}."
                        else:
                            return f"{status}: {row_count} rows{' from ' + table_name if table_name else ''}. Has data."
                    else:
                        return f"Query result: {status}, {row_count} rows"

                except Exception as e:
                    self.logger.warning(
                        "Failed to parse query observation", error=str(e)
                    )
                    return f"Query parsing failed: {str(e)[:50]}..."
            else:
                # For non-query tools, create an ultra-compact summary
                # Just capture the tool type and a hint of the content
                first_line = (
                    observation_str.split("\n")[0]
                    if "\n" in observation_str
                    else observation_str
                )
                first_line = (
                    first_line[:50] + "..." if len(first_line) > 50 else first_line
                )
                observation_str = f"{tool_name} result: {first_line}"

            self.logger.info(
                "Created summary of large observation in scratchpad",
                original_size=original_length,
                new_size=len(observation_str),
                action_tool=action_details.get("tool"),
            )

        # Add the step to our history
        self.intermediate_steps.append((action_details, observation_str))

        # If we have too many steps, convert older ones to ultra-compact summaries
        if len(self.intermediate_steps) > MAX_SCRATCHPAD_STEPS:
            # Keep only the MAX_SCRATCHPAD_STEPS most recent steps
            keep_recent = MAX_SCRATCHPAD_STEPS
            steps_to_summarize = len(self.intermediate_steps) - keep_recent

            if steps_to_summarize > 0:
                summarized_steps = []
                for i in range(steps_to_summarize):
                    old_action, old_observation = self.intermediate_steps[i]
                    tool_name = old_action.get("tool", "unknown")

                    # Create an ultra-compact summary based on tool type
                    if "query" in tool_name.lower() or "duckdb" in tool_name.lower():
                        # For database queries, try to extract table name
                        table = ""
                        if isinstance(old_action.get("tool_input"), dict):
                            query = old_action.get("tool_input", {}).get("query", "")
                            table_match = re.search(
                                r"FROM\s+([\w_]+)", query, re.IGNORECASE
                            )
                            if table_match:
                                table = f" on {table_match.group(1)}"

                        # Check if it had results
                        has_results = (
                            "No results" not in old_observation
                            and "'data': []" not in old_observation
                        )
                        result_indicator = "✓" if has_results else "∅"

                        summary = f"[{i+1}] {tool_name}{table}: {result_indicator}"
                    else:
                        # For other tools, just note the tool was used
                        summary = f"[{i+1}] {tool_name}"

                    summarized_steps.append((old_action, summary))

                # Replace older steps with their summaries
                self.intermediate_steps = (
                    summarized_steps + self.intermediate_steps[steps_to_summarize:]
                )
                self.logger.info(
                    f"Created ultra-compact summaries for {steps_to_summarize} older steps"
                )

        # Log the action with a truncated observation snippet
        self.logger.debug(
            "Added step to multi-turn scratchpad",
            action_tool=action_details.get("tool"),
            observation_snippet=observation_str[:MAX_OBSERVATION_SNIPPET_LENGTH],
        )

        return observation_str

    def _add_to_scratchpad(
        self, action_details: Dict[str, Any], observation_str: str
    ) -> None:
        """Add an action and its observation to the scratchpad.

        This method adds actions and observations to the scratchpad, ensuring
        that context boundaries are clearly marked to prevent context contamination
        between different queries. Implements aggressive retention - keeps only the 
        last step to optimize performance.

        Args:
            action_details (Dict[str, Any]): Details of the agent action
            observation_str (str): String representation of the observation
        """
        # Handle large observations by creating a compact summary
        compact_observation = self._handle_large_observation(
            observation_str, action_details
        )

        # Mark time-specific queries to prevent context contamination
        if "time_boot_ms_range" in observation_str:
            compact_observation = f"[TIME-SPECIFIC CONTEXT - Applies only to current query] {compact_observation}"
            # Add timestamp to make the context boundary clearer
            compact_observation += f"\n[Context Timestamp: {time.time()}] THIS TIME FILTER IS VALID ONLY FOR THE CURRENT QUESTION"
        else:
            compact_observation = f"[FULL DATASET CONTEXT - COMPLETE FLIGHT DATA] {compact_observation}"

        # Add the action and observation to the intermediate steps
        self.intermediate_steps.append((action_details, compact_observation))

        # AGGRESSIVE RETENTION: Keep only the last N steps to optimize performance
        # This prevents memory buildup and context confusion across conversations
        if len(self.intermediate_steps) > MAX_SCRATCHPAD_STEPS:
            self.intermediate_steps = self.intermediate_steps[-MAX_SCRATCHPAD_STEPS:]
            self.logger.debug(
                "Truncated scratchpad for performance optimization",
                retained_steps=len(self.intermediate_steps),
                max_steps=MAX_SCRATCHPAD_STEPS
            )

        # Log the addition
        self.logger.debug(
            "Added step to scratchpad",
            action_tool=action_details.get("tool"),
            observation_snippet=(
                compact_observation[:100] if compact_observation else "None"
            ),
            total_steps=len(self.intermediate_steps)
        )

    def get_scratchpad_content(self) -> List[Tuple[Dict[str, Any], str]]:
        """
        Retrieve the current scratchpad content as a list of action-observation pairs.

        This method provides direct access to the underlying data structure for
        programmatic use, while to_string() provides a formatted string representation
        for inclusion in prompts.

        Returns:
            List[Tuple[Dict[str, Any], str]]: List of (action_details, observation_string) tuples
                representing the history of agent actions and their results.
        """
        return self.intermediate_steps

    def to_string(self) -> str:
        """
        Convert the scratchpad to a string representation suitable for the LLM prompt.

        This method formats the action-observation history into a human-readable string
        that will be injected into the LLM prompt as {agent_scratchpad_content}. The format
        preserves the agent's thought process and observations in a structured way that
        helps the LLM understand the context of previous interactions.

        Returns:
            str: Formatted string representation of the scratchpad history, or a default
                 message if no steps have been recorded yet.
        """
        # Return default message if no steps have been recorded
        if not self.intermediate_steps:
            return DEFAULT_EMPTY_SCRATCHPAD_MESSAGE

        formatted_steps = []
        for i, (action_dict, observation_str) in enumerate(self.intermediate_steps, 1):
            # Create header for each step
            step_entry = f"Summary of Previous Turn - Step {i}:\n"

            # Extract thought process if available
            thought_process = action_dict.get("thought_process_for_this_action", "")
            if thought_process:
                # The 'thought_process_for_this_action' should already contain the ReAct style
                # "Thought: ... Action: ... Action Input: ..."
                step_entry += f"Thought process leading to action:\n{thought_process}\n"
            else:
                # Fallback if thought_process is not captured as expected
                step_entry += f"  Tool Used: {action_dict.get('tool', 'N/A')}\n"

                # Handle tool input serialization safely
                try:
                    tool_input_str = json.dumps(action_dict.get("tool_input", {}))
                except TypeError:
                    # Handle non-JSON-serializable objects
                    tool_input_str = str(action_dict.get("tool_input", {}))

                # Truncate long tool inputs
                if len(tool_input_str) > 100:
                    tool_input_str = tool_input_str[:100] + "..."

                step_entry += f"  Tool Input: {tool_input_str}\n"

            # Add the observation
            step_entry += f"Result:\n{observation_str}\n"

            # Add a separator between steps
            step_entry += "-" * 40 + "\n"

            formatted_steps.append(step_entry)

        # Join all steps into a single string
        return "\n".join(formatted_steps)

    def add_step(self, action_details: Dict[str, Any], observation_str: str) -> None:
        """Add a step to the scratchpad.

        This is a public method that can be called by external code to add a step to the scratchpad.
        It's a wrapper around _add_to_scratchpad that provides a simpler interface.

        Args:
            action_details (Dict[str, Any]): Details of the agent action
            observation_str (str): String representation of the observation
        """
        self._add_to_scratchpad(action_details, observation_str)

    def get_agent_actions(self) -> List[Dict[str, Any]]:
        """
        Get the list of agent actions with their observations.

        Returns:
            List[Dict[str, Any]]: List of dictionaries containing action details and observations
        """
        return self.agent_actions

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the scratchpad.
        
        Returns:
            Dict[str, Any]: Statistics including step count, total size, etc.
        """
        total_text = self.to_string()
        return {
            "total_steps": len(self.intermediate_steps),
            "total_characters": len(total_text),
            "session_id": self.session_id,
            "retention_policy": SCRATCHPAD_RETENTION_POLICY,
            "max_steps_configured": MAX_SCRATCHPAD_STEPS
        }

    def clear(self) -> None:
        """Clear all steps from the scratchpad."""
        steps_cleared = len(self.intermediate_steps)
        self.intermediate_steps.clear()
        self.agent_actions.clear()
        self.logger.info(
            "Cleared scratchpad",
            steps_cleared=steps_cleared
        )

    def get_step_count(self) -> int:
        """Get the current number of steps in the scratchpad."""
        return len(self.intermediate_steps)
