"""
Agent scratchpad module for tracking agent actions and observations.

This module provides a scratchpad for tracking agent actions and observations
during multi-turn conversations. It helps maintain context across turns and
provides a formatted history for the agent to reference.
"""

import json
import re
import time
from typing import Any, Dict, List, Optional, Tuple, Union
import structlog
from langchain_core.agents import AgentAction, AgentFinish

# Constants for scratchpad management
MAX_SCRATCHPAD_STEPS = 5  # Maximum number of steps to keep in the scratchpad
MAX_OBSERVATION_STORAGE_LENGTH = 5000  # Maximum length of observation to store in full
MAX_OBSERVATION_SNIPPET_LENGTH = 100  # Maximum length of observation snippet for logging
DEFAULT_EMPTY_SCRATCHPAD_MESSAGE = "No previous actions recorded."

logger = structlog.get_logger(__name__)

class AgentScratchpad:
    """
    Scratchpad for tracking agent actions and observations during multi-turn conversations.
    
    This class maintains a history of agent actions and their results, providing
    context for the agent across multiple turns. It formats this history in a way
    that can be included in the agent's prompt, helping it understand what actions
    have already been taken and their outcomes.
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
        self.logger.info("Initialized agent scratchpad")
        
    def process_agent_action(self, next_action: Union[AgentAction, AgentFinish]) -> None:
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
            self.logger.info("Large observation detected", 
                            observation_length=len(observation_str),
                            clean_observation=clean_observation)
            
            # Save the action log with observation but without raw data
            self.agent_actions.append({
                "action": agent_action_log,
                "observation": clean_observation,
                # Don't store raw observation for large outputs
                "observation_raw": f"[Large output ({len(observation_str)} chars) - not stored in history]"
            })
        else:
            # Normal case - save with raw observation
            self.agent_actions.append({
                "action": agent_action_log,
                "observation": clean_observation,
                "observation_raw": observation
            })
        
        # Add to the scratchpad with potentially summarized observation
        self._add_to_scratchpad(agent_action_log, observation_str)
    
    def _process_agent_action_full(self, next_action: AgentAction) -> Dict[str, Any]:
        """Process an agent action and produce full structured log.
        
        Args:
            next_action (AgentAction): The agent action to process
            
        Returns:
            Dict[str, Any]: Full structured log of the processed action
        """
        action_type = next_action.tool if hasattr(next_action, 'tool') else "thinking"
        action_input = next_action.tool_input if hasattr(next_action, 'tool_input') else ""
        
        # For certain tools that may return large responses, sanitize the input
        # to prevent large logs
        if action_type == "detect_anomalies":
            # Log just what's necessary for the detect_anomalies tool
            try:
                if isinstance(action_input, str) and len(action_input) > 100:
                    # For long string inputs, just log a summary
                    action_input = f"[Anomaly detection request - {len(action_input)} chars]"
                elif isinstance(action_input, dict):
                    # For dict inputs, just log that it's a dict with db_path
                    if "db_path" in action_input:
                        action_input = {"db_path": action_input["db_path"], "...":"[details omitted]"}
            except Exception:
                # If anything fails, log a generic summary
                action_input = "[Anomaly detection request]"  
                
        action_details = {
            "tool": action_type,
            "input": action_input
        }
        
        # Extract thought process if available
        if hasattr(next_action, 'log'):
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
        """Clean up an observation for storage and display.
        
        Args:
            observation (Any): The observation to clean up
            
        Returns:
            str: A cleaned string representation of the observation
        """
        # Convert to string first
        obs_str = str(observation)
        
        # If it's a JSON response with status and data, format it more cleanly
        if isinstance(observation, dict):
            if 'status' in observation and observation.get('status') == 'success':
                # For query results
                if 'data' in observation and isinstance(observation['data'], list):
                    row_count = len(observation['data'])
                    if row_count == 0:
                        return "Query returned no results."
                    elif row_count <= 3:
                        # For small result sets, show a simplified version
                        return f"Query successful: {row_count} results found."
                    else:
                        return f"Query successful: {row_count} results found."
                        
                # For anomaly detection results - updated for ultra-minimal format
                if 'anomalies_found' in observation and 'summary' in observation:
                    # Just return the summary directly, no need to reconstruct it
                    return observation.get('summary', 'Anomaly detection completed.')
                    
                # Legacy format handling
                elif 'tables_processed' in observation and 'anomalies_found' in observation:
                    tables = observation.get('tables_processed', [])
                    anomalies = observation.get('anomalies_found', 0)
                    tables_str = ', '.join(tables) if len(tables) <= 3 else f"{len(tables)} tables"
                    return f"Anomaly detection completed on {tables_str}. Found {anomalies} anomalies across {observation.get('total_rows_analyzed', '?')} rows."
                    
            elif 'status' in observation and observation.get('status') == 'error':
                # For error responses
                return f"Error: {observation.get('message', 'Unknown error')}"
        
        # For other string observations, try to detect and clean up JSON-like content
        if obs_str.startswith("{") and ("'status': 'success'" in obs_str or "'data':" in obs_str):
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
        if isinstance(obs_str, str) and 'anomalies_found' in obs_str and 'tables_summary' in obs_str:
            # This is likely a stringified anomaly detection result
            try:
                # Try to extract the summary from the string
                import re
                # Use a simpler regex pattern with single quotes to avoid escaping issues
                summary_match = re.search(r'summary[\'|"]?:\s*[\'|"]([^\'|"]+)[\'|"]?', obs_str)
                if summary_match:
                    return summary_match.group(1)
                    
                # If we can't extract the summary, return a generic one
                anomalies_match = re.search(r'anomalies_found[\'|"]?:\s*(\d+)', obs_str)
                if anomalies_match:
                    anomalies = anomalies_match.group(1)
                    return f"Anomaly detection completed. Found {anomalies} anomalies."
            except Exception:
                # If anything fails, fall back to generic message
                return "Anomaly detection completed."
        
        # Return the original string for other cases
        return obs_str
    
    def _handle_large_observation(self, observation_str: str, action_details: Dict[str, Any]) -> str:
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
            if "query" in tool_name.lower() or "search" in tool_name.lower() or "duckdb" in tool_name.lower():
                try:
                    # First check if this is an error response
                    error_match = re.search(r"error[_str]*['\"]?\s*:\s*['\"]([^'\"]+)['\"]?", observation_str)
                    column_error_match = re.search(r"Referenced column \"([^\"]+)\" not found", observation_str)
                    if error_match or "'status': 'error'" in observation_str:
                        error_msg = error_match.group(1) if error_match else "Unknown error"
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
                    has_data = "data_length" in observation_str or "sample" in observation_str or "data_present" in observation_str
                    
                    # Create an ultra-compact summary
                    if 'query' in observation_str:
                        # Try to extract table name from query string
                        table_name = ""
                        table_match = re.search(r'FROM\s+([\w_]+)', observation_str, re.IGNORECASE)
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
                    self.logger.warning("Failed to parse query observation", error=str(e))
                    return f"Query parsing failed: {str(e)[:50]}..."
            else:
                # For non-query tools, create an ultra-compact summary
                # Just capture the tool type and a hint of the content
                first_line = observation_str.split('\n')[0] if '\n' in observation_str else observation_str
                first_line = first_line[:50] + "..." if len(first_line) > 50 else first_line
                observation_str = f"{tool_name} result: {first_line}"
            
            self.logger.info(
                "Created summary of large observation in scratchpad", 
                original_size=original_length,
                new_size=len(observation_str),
                action_tool=action_details.get("tool")
            )
        
        # Add the step to our history
        self.intermediate_steps.append((action_details, observation_str))
        
        # If we have too many steps, convert older ones to ultra-compact summaries
        if len(self.intermediate_steps) > MAX_SCRATCHPAD_STEPS:
            # Keep the most recent steps intact, summarize older ones
            keep_recent = 10  # Keep the 10 most recent steps intact
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
                            table_match = re.search(r'FROM\s+([\w_]+)', query, re.IGNORECASE)
                            if table_match:
                                table = f" on {table_match.group(1)}"
                        
                        # Check if it had results
                        has_results = "No results" not in old_observation and "'data': []" not in old_observation
                        result_indicator = "✓" if has_results else "∅"
                        
                        summary = f"[{i+1}] {tool_name}{table}: {result_indicator}"
                    else:
                        # For other tools, just note the tool was used
                        summary = f"[{i+1}] {tool_name}"
                    
                    summarized_steps.append((old_action, summary))
                
                # Replace older steps with their summaries
                self.intermediate_steps = summarized_steps + self.intermediate_steps[steps_to_summarize:]
                self.logger.info(f"Created ultra-compact summaries for {steps_to_summarize} older steps")
        
        # Log the action with a truncated observation snippet
        self.logger.debug(
            "Added step to multi-turn scratchpad", 
            action_tool=action_details.get("tool"), 
            observation_snippet=observation_str[:MAX_OBSERVATION_SNIPPET_LENGTH]
        )
        
        return observation_str
    
    def _add_to_scratchpad(self, action_details: Dict[str, Any], observation_str: str) -> None:
        """Add an action and its observation to the scratchpad.
        
        Args:
            action_details (Dict[str, Any]): Details of the agent action
            observation_str (str): String representation of the observation
        """
        # Handle large observations by creating a compact summary
        compact_observation = self._handle_large_observation(observation_str, action_details)
        
        # Add the action and observation to the intermediate steps
        self.intermediate_steps.append((action_details, compact_observation))
        
        # Log the addition
        self.logger.debug(
            "Added step to scratchpad", 
            action_tool=action_details.get("tool"), 
            observation_snippet=compact_observation[:100] if compact_observation else "None"
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
            thought_process = action_dict.get('thought_process_for_this_action', '')
            if thought_process:
                # The 'thought_process_for_this_action' should already contain the ReAct style 
                # "Thought: ... Action: ... Action Input: ..."
                step_entry += f"Thought process leading to action:\n{thought_process}\n"
            else: 
                # Fallback if thought_process is not captured as expected
                step_entry += f"  Tool Used: {action_dict.get('tool', 'N/A')}\n"
                
                # Handle tool input serialization safely
                try:
                    tool_input_str = json.dumps(action_dict.get('tool_input', {}))
                except TypeError:
                    # Handle non-JSON-serializable objects
                    tool_input_str = str(action_dict.get('tool_input', {}))
                
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
