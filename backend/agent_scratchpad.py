from typing import List, Dict, Any, Tuple, Optional
import structlog
import json

# Constants for scratchpad formatting and logging
MAX_OBSERVATION_SNIPPET_LENGTH: int = 100  # Maximum length of observation snippet for logging
DEFAULT_EMPTY_SCRATCHPAD_MESSAGE: str = "No intermediate reasoning steps recorded from previous turns yet."
STEP_SEPARATOR: str = "\n---\n"  # Separator between steps in the string representation

logger = structlog.get_logger(__name__)

class AgentScratchpad:
    """Manages the agent's multi-turn scratchpad state for a session."""

    def __init__(self, session_id: str):
        """
        Initialize the AgentScratchpad.

        Args:
            session_id (str): Unique identifier for the session.
        """
        # Stores (action_details_dict, observation_string)
        self.intermediate_steps: List[Tuple[Dict[str, Any], str]] = []
        self.logger = logger.bind(session_id=session_id)

    def add_step(self, action_details: Dict[str, Any], observation: Any) -> None:
        """
        Add an intermediate step (action and its observation) to the scratchpad.

        Args:
            action_details (Dict[str, Any]): Dictionary containing details of the agent's action,
                                             including 'tool', 'tool_input', and 'thought_process_for_this_action'.
            observation (Any): The result/observation from the action. Will be converted to string if not already.
        """
        # Ensure observation is a string to maintain consistent typing
        observation_str = str(observation)
        
        self.intermediate_steps.append((action_details, observation_str))
        
        # Log the action with a truncated observation snippet
        self.logger.debug(
            "Added step to multi-turn scratchpad", 
            action_tool=action_details.get("tool"), 
            observation_snippet=observation_str[:MAX_OBSERVATION_SNIPPET_LENGTH]
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
                    
                step_entry += f"  Tool Input: {tool_input_str}\n"
            
            # Add the observation result
            step_entry += f"Observation: {observation_str}\n"
            formatted_steps.append(step_entry)
            
        # Join all steps with the defined separator
        return STEP_SEPARATOR.join(formatted_steps)

    def clear(self) -> None:
        """
        Clear all steps from the scratchpad.
        
        This method resets the scratchpad by removing all recorded action-observation pairs.
        Useful when starting a new conversation or when the context needs to be reset.
        """
        self.intermediate_steps = []
        self.logger.info("Cleared multi-turn scratchpad.")