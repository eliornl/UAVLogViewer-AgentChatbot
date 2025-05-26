from typing import List, Dict, Any
import structlog
import json

logger = structlog.get_logger(__name__)

class AgentScratchpad:
    """Manages the agent's scratchpad state for a session."""

    def __init__(self, session_id: str):
        """
        Initialize the AgentScratchpad.

        Args:
            session_id (str): Unique identifier for the session.
        """
        self.intermediate_steps: List[Dict[str, Any]] = []
        self.logger = logger.bind(session_id=session_id)

    def add_step(self, step: Dict[str, Any]) -> None:
        """
        Add an intermediate step to the scratchpad.

        Args:
            step (Dict[str, Any]): The intermediate step to store.
        """
        self.intermediate_steps.append(step)
        self.logger.debug("Added intermediate step to scratchpad", step=step)

    def get_scratchpad(self) -> List[Dict[str, Any]]:
        """
        Retrieve the current scratchpad state.

        Returns:
            List[Dict[str, Any]]: List of intermediate steps.
        """
        return self.intermediate_steps

    def to_string(self) -> str:
        """
        Convert the scratchpad to a string representation suitable for the LLM.

        Returns:
            str: String representation of the scratchpad.
        """
        if not self.intermediate_steps:
            return "No intermediate reasoning steps recorded yet."

        formatted_steps = []
        for i, step in enumerate(self.intermediate_steps, 1):
            formatted_steps.append(f"Step {i}:\n{json.dumps(step, indent=2)}")

        return "\n\n".join(formatted_steps)