from typing import List, Tuple, Any, Callable
from langchain_core.messages import BaseMessage
import tiktoken
import structlog
from functools import lru_cache

logger: structlog.stdlib.BoundLogger = structlog.get_logger(__name__)

# Token cache size - adjust based on memory constraints
TOKEN_CACHE_SIZE = 100


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
            logger.error(
                "Invalid model name provided", model_name=model_name, error=str(e)
            )
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
                    got=type(message).__name__,
                )
                raise ValueError(
                    f"Message at index {index} must be a tuple of two strings"
                )

            user_content, assistant_content = message

            # Validate user content
            if user_content is not None and not isinstance(user_content, str):
                logger.error(
                    "Invalid user content type at index",
                    index=index,
                    expected="str",
                    got=type(user_content).__name__,
                )
                raise TypeError(f"User content at index {index} must be a string")

            # Validate assistant content
            if assistant_content is not None and not isinstance(assistant_content, str):
                logger.error(
                    "Invalid assistant content type at index",
                    index=index,
                    expected="str",
                    got=type(assistant_content).__name__,
                )
                raise TypeError(f"Assistant content at index {index} must be a string")

            # Count tokens for user content
            if user_content:
                try:
                    total_tokens += len(self._cached_encode(user_content))
                except Exception as e:
                    logger.error(
                        "Failed to encode user content at index",
                        index=index,
                        error=str(e),
                    )
                    raise ValueError(
                        f"Failed to encode user content at index {index}: {str(e)}"
                    ) from e

            # Count tokens for assistant content
            if assistant_content:
                try:
                    total_tokens += len(self._cached_encode(assistant_content))
                except Exception as e:
                    logger.error(
                        "Failed to encode assistant content at index",
                        index=index,
                        error=str(e),
                    )
                    raise ValueError(
                        f"Failed to encode assistant content at index {index}: {str(e)}"
                    ) from e

        logger.debug(
            "Counted tokens for messages",
            model_name=self.model_name,
            message_count=len(messages),
            total_tokens=total_tokens,
        )
        return total_tokens

    def count_message_tokens(self, message: BaseMessage) -> int:
        """
        Count tokens in a LangChain BaseMessage object.

        This method is used by the CustomBufferWindowMemory class to count tokens
        in individual messages for the token-based windowing functionality.

        Args:
            message: A LangChain BaseMessage object (HumanMessage, AIMessage, etc.)

        Returns:
            int: Number of tokens in the message content

        Raises:
            ValueError: If message content cannot be encoded
        """
        try:
            # Extract the content from the message
            content = message.content

            # Handle different content types
            if isinstance(content, str):
                # Use cached encoding for better performance
                return len(self._cached_encode(content))
            elif isinstance(content, list):
                # For multi-modal content, only count text parts
                total = 0
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "text":
                        text = item.get("text", "")
                        # Use cached encoding for better performance
                        total += len(self._cached_encode(text))
                return total
            else:
                # For other types, convert to string and count
                return len(self._cached_encode(str(content)))
        except Exception as e:
            logger.error("Failed to count tokens in message", error=str(e))
            # Return a safe default value instead of raising an exception
            # This ensures the memory system continues to function
            return 1  # Minimum token count to avoid division by zero

    def encode(self, text: str) -> List[int]:
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
            # Use cached encoding for common inputs
            tokens = self._cached_encode(text)
            logger.debug(
                "Encoded text", model_name=self.model_name, token_count=len(tokens), 
                cached=True
            )
            return tokens
        except Exception as e:
            logger.error(
                "Failed to encode text", model_name=self.model_name, error=str(e)
            )
            raise ValueError(f"Failed to encode text: {str(e)}") from e
            
    @lru_cache(maxsize=TOKEN_CACHE_SIZE)
    def _cached_encode(self, text: str) -> List[int]:
        """Cached version of token encoding for performance.
        
        Args:
            text: Text to encode.
            
        Returns:
            List[int]: List of token IDs.
        """
        return self.encoding.encode(text)
