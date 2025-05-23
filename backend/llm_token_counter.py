from typing import List, Tuple
import tiktoken
import structlog

logger: structlog.stdlib.BoundLogger = structlog.get_logger(__name__)

class LLMTokenCounter:
    """Handles token counting for text inputs using tiktoken for LLM models.

    Provides robust and efficient token counting for message pairs, with support
    for various model encodings and comprehensive error handling.
    """

    def __init__(self, model_name: str) -> None:
        """Initialize the LLMTokenCounter with a specific model.

        Args:
            model_name: Name of the model for token encoding.

        Raises:
            ValueError: If the model_name is invalid or unsupported by tiktoken.
        """
        try:
            self.model_name: str = model_name
            self.encoding = tiktoken.encoding_for_model(model_name)
            logger.debug("Initialized LLMTokenCounter", model_name=model_name)
        except KeyError as e:
            logger.error("Invalid model name provided", model_name=model_name, error=str(e))
            raise ValueError(f"Unsupported model name: {model_name}") from e

    def count_tokens(self, messages: List[Tuple[str, str]]) -> int:
        """Count the total number of tokens in a list of message pairs.

        Args:
            messages: List of (user_content, assistant_content) tuples representing
                user and assistant messages.

        Returns:
            int: Total token count across all messages.

        Raises:
            ValueError: If messages is not a list of tuples or contains invalid content.
            TypeError: If message content is not a string.
        """
        if not isinstance(messages, list):
            logger.error("Messages must be a list", type=type(messages).__name__)
            raise ValueError("Messages must be a list")

        total_tokens: int = 0
        for index, message in enumerate(messages):
            # Validate message tuple
            if not isinstance(message, tuple) or len(message) != 2:
                logger.error(
                    "Invalid message format at index",
                    index=index,
                    expected="tuple[str, str]",
                    got=type(message).__name__
                )
                raise ValueError(f"Message at index {index} must be a tuple of two strings")

            user_content, assistant_content = message

            # Validate user content
            if user_content is not None and not isinstance(user_content, str):
                logger.error(
                    "Invalid user content type at index",
                    index=index,
                    expected="str",
                    got=type(user_content).__name__
                )
                raise TypeError(f"User content at index {index} must be a string")

            # Validate assistant content
            if assistant_content is not None and not isinstance(assistant_content, str):
                logger.error(
                    "Invalid assistant content type at index",
                    index=index,
                    expected="str",
                    got=type(assistant_content).__name__
                )
                raise TypeError(f"Assistant content at index {index} must be a string")

            # Count tokens for user content
            if user_content:
                try:
                    total_tokens += len(self.encoding.encode(user_content))
                except Exception as e:
                    logger.error(
                        "Failed to encode user content at index",
                        index=index,
                        error=str(e)
                    )
                    raise ValueError(f"Failed to encode user content at index {index}: {str(e)}") from e

            # Count tokens for assistant content
            if assistant_content:
                try:
                    total_tokens += len(self.encoding.encode(assistant_content))
                except Exception as e:
                    logger.error(
                        "Failed to encode assistant content at index",
                        index=index,
                        error=str(e)
                    )
                    raise ValueError(f"Failed to encode assistant content at index {index}: {str(e)}") from e

        logger.debug(
            "Counted tokens for messages",
            model_name=self.model_name,
            message_count=len(messages),
            total_tokens=total_tokens
        )
        return total_tokens

    def encode_text(self, text: str) -> List[int]:
        """Encode a single text string into tokens.

        Args:
            text: Text to encode.

        Returns:
            List[int]: List of token IDs.

        Raises:
            TypeError: If text is not a string.
            ValueError: If encoding fails.
        """
        if not isinstance(text, str):
            logger.error("Text must be a string", type=type(text).__name__)
            raise TypeError("Text must be a string")

        try:
            tokens = self.encoding.encode(text)
            logger.debug("Encoded text", model_name=self.model_name, token_count=len(tokens))
            return tokens
        except Exception as e:
            logger.error("Failed to encode text", model_name=self.model_name, error=str(e))
            raise ValueError(f"Failed to encode text: {str(e)}") from e