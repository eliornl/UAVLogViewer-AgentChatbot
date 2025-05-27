from typing import List, Dict, Any, Tuple
import structlog
import json

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

    def add_step(self, action_details: Dict[str, Any], observation: str) -> None:
        """
        Add an intermediate step (action and its observation) to the scratchpad.

        Args:
            action_details (Dict[str, Any]): Dictionary containing details of the agent's action,
                                             including 'tool', 'tool_input', and 'thought_process_for_this_action'.
            observation (str): The string result/observation from the action.
        """
        self.intermediate_steps.append((action_details, str(observation))) # Ensure observation is a string
        self.logger.debug("Added step to multi-turn scratchpad", action_tool=action_details.get("tool"), observation_snippet=str(observation)[:100])

    def get_scratchpad_content(self) -> List[Tuple[Dict[str, Any], str]]: # Renamed for clarity
        """
        Retrieve the current scratchpad content.

        Returns:
            List[Tuple[Dict[str, Any], str]]: List of (action_details, observation_string) tuples.
        """
        return self.intermediate_steps

    def to_string(self) -> str:
        """
        Convert the scratchpad to a string representation suitable for the LLM prompt.
        This string will be injected as {agent_scratchpad_content}.

        Returns:
            str: String representation of the scratchpad.
        """
        if not self.intermediate_steps:
            return "No intermediate reasoning steps recorded from previous turns yet."

        formatted_steps = []
        for i, (action_dict, observation_str) in enumerate(self.intermediate_steps, 1):
            step_entry = f"Summary of Previous Turn - Step {i}:\n"
            
            thought_process = action_dict.get('thought_process_for_this_action', '')
            if thought_process:
                # The 'thought_process_for_this_action' should already contain the ReAct style "Thought: ... Action: ... Action Input: ..."
                step_entry += f"Thought process leading to action:\n{thought_process}\n"
            else: # Fallback if thought_process is not captured as expected
                step_entry += f"  Tool Used: {action_dict.get('tool', 'N/A')}\n"
                try:
                    tool_input_str = json.dumps(action_dict.get('tool_input', {}))
                except TypeError:
                    tool_input_str = str(action_dict.get('tool_input', {}))
                step_entry += f"  Tool Input: {tool_input_str}\n"
            
            step_entry += f"Observation: {observation_str}\n"
            formatted_steps.append(step_entry)
            
        return "\n---\n".join(formatted_steps)

    def clear(self) -> None:
        """Clear all steps from the scratchpad."""
        self.intermediate_steps = []
        self.logger.info("Cleared multi-turn scratchpad.")