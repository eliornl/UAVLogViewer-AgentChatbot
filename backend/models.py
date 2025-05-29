from enum import Enum
from datetime import datetime
import re
from typing import List, Dict, Optional, Any, Literal, TypeAlias
from pydantic import BaseModel, Field, field_validator, ConfigDict, computed_field

# Type alias for metadata to ensure consistent usage
MetadataDict: TypeAlias = Dict[str, Any]

# Constants for validation
MAX_ID_LENGTH: int = 36  # UUID length for session_id, message_id
MAX_FILE_NAME_LENGTH: int = 255  # Standard filesystem limit
MAX_CONTENT_LENGTH: int = 10000  # Limit for message content
MAX_MESSAGE_LENGTH: int = 5000  # Limit for user message in ChatRequest
MAX_FILE_SIZE: int = 1024 * 1024 * 100  # 100 MB max file size
MAX_MESSAGES: int = 1000  # Max messages per session
MAX_METADATA_SIZE: int = 100  # Max key-value pairs in metadata
MAX_TOKEN_SAFETY_LIMIT: int = 16384  # From telemetry_agent.py
ID_PATTERN: re.Pattern = re.compile(r"^[a-zA-Z0-9\-_]+$")  # Alphanumeric, hyphens, underscores
# UUID pattern for session IDs and message IDs
UUID_PATTERN: re.Pattern = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")  # UUID format
FILE_NAME_PATTERN: re.Pattern = re.compile(r"^[a-zA-Z0-9\-_\.]+$")  # Alphanumeric, hyphens, underscores, dots
ROLE_TYPES: tuple[str, ...] = ("user", "assistant")

class ResponseStatus(str, Enum):
    """Status codes for API responses."""
    SUCCESS = "SUCCESS"
    ERROR = "ERROR"

class SessionStatus(str, Enum):
    """Status codes for session processing state."""
    PENDING = "PENDING"
    READY = "READY"
    ERROR = "ERROR"

class Metadata(BaseModel):
    """Structured metadata for messages and responses."""
    model_config = ConfigDict(extra="allow")  # Allow extra fields for flexibility
    relevant_tables: Optional[List[str]] = Field(default=None, description="List of relevant telemetry tables")
    query: Optional[str] = Field(default=None, description="SQL query executed, if any")
    token_usage: Optional[Dict[str, int]] = Field(default=None, description="Token usage statistics")
    memory_strategy: Optional[str] = Field(default=None, description="Memory strategy used")
    is_clarification: Optional[bool] = Field(default=None, description="Whether response requests clarification")

    @field_validator("relevant_tables", mode="before")
    @classmethod
    def validate_relevant_tables(cls, v: Optional[List[str]]) -> Optional[List[str]]:
        """Ensure relevant_tables contains valid table names."""
        if v is None:
            return None
        if len(v) > 50:  # Arbitrary limit to prevent abuse
            raise ValueError("Too many relevant tables, maximum is 50")
        for table in v:
            if not table or len(table) > 100 or not re.match(r"^[a-zA-Z0-9_]+$", table):
                raise ValueError(f"Invalid table name: {table}")
        return v

    @field_validator("query", mode="before")
    @classmethod
    def validate_query(cls, v: Optional[str]) -> Optional[str]:
        """Ensure query is a valid SQL string."""
        if v is None:
            return None
        if len(v) > 1000:
            raise ValueError("SQL query too long, maximum length is 1000 characters")
        return v

    @field_validator("token_usage", mode="before")
    @classmethod
    def validate_token_usage(cls, v: Optional[Dict[str, int]]) -> Optional[Dict[str, int]]:
        """Ensure token_usage contains valid keys and values."""
        if v is None:
            return None
        expected_keys = {"prompt_tokens", "completion_tokens", "total_tokens"}
        if not all(key in expected_keys for key in v):
            raise ValueError(f"Invalid token_usage keys, expected {expected_keys}")
        if any(not isinstance(val, int) or val < 0 for val in v.values()):
            raise ValueError("token_usage values must be non-negative integers")
        return v

class Message(BaseModel):
    """Chat message details for a session.

    Represents a single message in a session, including user or assistant content,
    with metadata for context like SQL queries or token usage.

    Attributes:
        message_id: Unique identifier for the message (e.g., UUID).
        role: Sender role, either 'user' or 'assistant'.
        content: The message text.
        timestamp: Creation timestamp in UTC.
        metadata: Optional structured metadata (e.g., query, token usage).

    Raises:
        ValueError: If message_id, role, content, or timestamp are invalid.
    """
    model_config = ConfigDict(str_strip_whitespace=True)  # Strip whitespace from strings
    message_id: str = Field(
        ...,
        description="Unique message identifier (e.g., UUID)",
        max_length=MAX_ID_LENGTH
    )
    role: Literal["user", "assistant"] = Field(
        ...,
        description="Role of the message sender ('user' or 'assistant')"
    )
    content: str = Field(
        ...,
        description="Message content",
        max_length=MAX_CONTENT_LENGTH,
        min_length=1
    )
    timestamp: datetime = Field(
        ...,
        description="Message creation timestamp in UTC"
    )
    metadata: Optional[Metadata] = Field(
        default_factory=Metadata,  # Initialize empty Metadata
        description="Structured metadata (e.g., SQL query, token usage)"
    )

    @field_validator("message_id")
    @classmethod
    def validate_message_id(cls, v: str) -> str:
        """Ensure message_id is valid and safe."""
        if not v or not ID_PATTERN.match(v):
            raise ValueError(f"Invalid message_id: {v}, must be alphanumeric with hyphens/underscores")
        return v

    @field_validator("content")
    @classmethod
    def validate_content(cls, v: str) -> str:
        """Ensure content is non-empty and within length limits."""
        if not v.strip():
            raise ValueError("Message content cannot be empty")
        return v

    @field_validator("timestamp")
    @classmethod
    def validate_timestamp(cls, v: datetime) -> datetime:
        """Ensure timestamp is UTC."""
        if v.tzinfo is None or v.tzinfo.utcoffset(v) is None:
            raise ValueError("Timestamp must be timezone-aware (UTC)")
        return v

    @computed_field
    @property
    def timestamp_str(self) -> str:
        """Formatted timestamp for JSON export (YYYY-MM-DD HH:MM:SS).

        Returns:
            str: Timestamp in 'YYYY-MM-DD HH:MM:SS' format.
        """
        return self.timestamp.strftime("%Y-%m-%d %H:%M:%S")

    def to_export_dict(self) -> Dict[str, Any]:
        """Convert message to dictionary for JSON export.

        Returns:
            Dict[str, Any]: Dictionary with timestamp, role, and message for export.
        """
        return {
            "timestamp": self.timestamp_str,
            "role": self.role,
            "message": self.content
        }

class Session(BaseModel):
    """Session data for a flight log.

    Represents a session associated with an uploaded telemetry log, including
    processing status and chat messages.

    Attributes:
        session_id: Unique identifier for the session (e.g., UUID).
        created_at: Session creation timestamp in UTC.
        file_name: Sanitized filename of the uploaded log.
        file_size: Size of the uploaded file in bytes.
        status: Session processing status (PENDING, READY, ERROR).
        status_message: Optional descriptive message for the status.
        messages: List of chat messages (up to MAX_MESSAGES).

    Raises:
        ValueError: If session_id, file_name, file_size, or created_at are invalid.
    """
    model_config = ConfigDict(str_strip_whitespace=True)
    session_id: str = Field(
        ...,
        description="Unique session identifier (e.g., UUID)",
        max_length=MAX_ID_LENGTH
    )
    created_at: datetime = Field(
        ...,
        description="Session creation timestamp in UTC"
    )
    file_name: str = Field(
        ...,
        description="Sanitized filename of the uploaded log",
        max_length=MAX_FILE_NAME_LENGTH
    )
    file_size: int = Field(
        ...,
        description="Size of the uploaded file in bytes",
        ge=0,
        le=MAX_FILE_SIZE
    )
    status: SessionStatus = Field(
        default=SessionStatus.PENDING,
        description="Session processing status"
    )
    status_message: Optional[str] = Field(
        default=None,
        description="Descriptive message for the session status",
        max_length=1000
    )
    messages: List[Message] = Field(
        default_factory=list,
        description=f"List of chat messages (max {MAX_MESSAGES})"
    )

    @field_validator("session_id")
    @classmethod
    def validate_session_id(cls, v: str) -> str:
        """Ensure session_id is valid and safe."""
        if not v or not ID_PATTERN.match(v):
            raise ValueError(f"Invalid session_id: {v}, must be alphanumeric with hyphens/underscores")
        return v

    @field_validator("file_name")
    @classmethod
    def validate_file_name(cls, v: str) -> str:
        """Ensure file_name is valid and safe."""
        if not v or not FILE_NAME_PATTERN.match(v):
            raise ValueError(f"Invalid file_name: {v}, must be alphanumeric with hyphens/underscores/dots")
        if v.startswith(".") or v.endswith("."):
            raise ValueError(f"Invalid file_name: {v}, cannot start or end with a dot")
        return v

    @field_validator("created_at")
    @classmethod
    def validate_created_at(cls, v: datetime) -> datetime:
        """Ensure created_at is UTC."""
        if v.tzinfo is None or v.tzinfo.utcoffset(v) is None:
            raise ValueError("Created_at timestamp must be timezone-aware (UTC)")
        return v

    @field_validator("messages")
    @classmethod
    def validate_messages(cls, v: List[Message]) -> List[Message]:
        """Ensure messages list does not exceed MAX_MESSAGES."""
        if len(v) > MAX_MESSAGES:
            raise ValueError(f"Too many messages, maximum is {MAX_MESSAGES}")
        return v

class UploadResponse(BaseModel):
    """Response for file upload and session creation.

    Provides details about the uploaded file and session creation result.

    Attributes:
        timestamp: Response timestamp in UTC.
        file_name: Sanitized filename of the uploaded file.
        file_size: Size of the uploaded file in bytes.
        session_id: Unique session identifier.
        status: Upload status (SUCCESS or ERROR).
        message: Descriptive result message.
        request_duration: Processing duration in seconds.

    Raises:
        ValueError: If fields are invalid (e.g., negative request_duration).
    """
    model_config = ConfigDict(str_strip_whitespace=True)
    timestamp: datetime = Field(
        ...,
        description="Response timestamp in UTC"
    )
    file_name: str = Field(
        ...,
        description="Sanitized filename of the uploaded file",
        max_length=MAX_FILE_NAME_LENGTH
    )
    file_size: int = Field(
        ...,
        description="Size of the uploaded file in bytes",
        ge=0,
        le=MAX_FILE_SIZE
    )
    session_id: str = Field(
        ...,
        description="Unique session identifier",
        max_length=MAX_ID_LENGTH
    )
    status: ResponseStatus = Field(
        ...,
        description="Upload status (SUCCESS or ERROR)"
    )
    message: str = Field(
        ...,
        description="Descriptive result message",
        max_length=1000,
        min_length=1
    )
    request_duration: float = Field(
        ...,
        description="Request processing duration in seconds",
        ge=0.0
    )

    @field_validator("timestamp")
    @classmethod
    def validate_timestamp(cls, v: datetime) -> datetime:
        """Ensure timestamp is UTC."""
        if v.tzinfo is None or v.tzinfo.utcoffset(v) is None:
            raise ValueError("Timestamp must be timezone-aware (UTC)")
        return v

    @field_validator("session_id")
    @classmethod
    def validate_session_id(cls, v: str) -> str:
        """Ensure session_id is valid and safe."""
        if not v or not ID_PATTERN.match(v):
            raise ValueError(f"Invalid session_id: {v}, must be alphanumeric with hyphens/underscores")
        return v

    @field_validator("file_name")
    @classmethod
    def validate_file_name(cls, v: str) -> str:
        """Ensure file_name is valid and safe."""
        if not v or not FILE_NAME_PATTERN.match(v):
            raise ValueError(f"Invalid file_name: {v}, must be alphanumeric with hyphens/underscores/dots")
        return v

class ChatRequest(BaseModel):
    """Payload for sending chat messages to the TelemetryAgent.

    Defines the structure for user chat requests, including the session and message details.

    Attributes:
        session_id: Unique session identifier.
        message: User message content.
        message_id: Optional message ID for context.
        max_tokens: Optional maximum tokens for the assistant response.

    Raises:
        ValueError: If session_id, message, message_id, or max_tokens are invalid.

    Example:
        ```python
        chat_request = ChatRequest(
            session_id="123e4567-e89b-12d3-a456-426614174000",
            message="What is the average altitude?",
            max_tokens=1000
        )
        ```
    """
    model_config = ConfigDict(str_strip_whitespace=True)
    session_id: str = Field(
        ...,
        description="Unique session identifier",
        max_length=MAX_ID_LENGTH
    )
    message: str = Field(
        ...,
        description="User message content",
        max_length=MAX_MESSAGE_LENGTH,
        min_length=1
    )
    message_id: Optional[str] = Field(
        default=None,
        description="Optional message ID for context",
        max_length=MAX_ID_LENGTH
    )
    max_tokens: Optional[int] = Field(
        default=None,
        description=f"Maximum tokens for assistant response (max {MAX_TOKEN_SAFETY_LIMIT})",
        ge=1,
        le=MAX_TOKEN_SAFETY_LIMIT
    )

    @field_validator("session_id")
    @classmethod
    def validate_session_id(cls, v: str) -> str:
        """Ensure session_id is valid and safe."""
        if not v or not ID_PATTERN.match(v):
            raise ValueError(f"Invalid session_id: {v}, must be alphanumeric with hyphens/underscores")
        return v

    @field_validator("message")
    @classmethod
    def validate_message(cls, v: str) -> str:
        """Ensure message is non-empty."""
        if not v.strip():
            raise ValueError("Message content cannot be empty")
        return v

    @field_validator("message_id")
    @classmethod
    def validate_message_id(cls, v: Optional[str]) -> Optional[str]:
        """Ensure message_id is valid if provided."""
        if v is None:
            return None
        if not v:
            raise ValueError("Message ID cannot be empty")
        # Accept standard ID pattern, UUID pattern, or numeric timestamp
        if not (ID_PATTERN.match(v) or UUID_PATTERN.match(v) or v.isdigit()):
            raise ValueError(f"Invalid message_id: {v}, must be alphanumeric with hyphens/underscores, a UUID, or a numeric timestamp")
        return v

class ChatResponse(BaseModel):
    """Response for chat endpoint.

    Provides the assistant's response or error details for a chat request.

    Attributes:
        timestamp: Response timestamp in UTC.
        session_id: Unique session identifier.
        message_id: Unique message identifier.
        status: Chat processing status (SUCCESS or ERROR).
        message: Assistant response or error message.
        metadata: Optional structured metadata (e.g., SQL query, token usage).
        request_duration: Processing duration in seconds.

    Raises:
        ValueError: If fields are invalid (e.g., invalid session_id).
    """
    model_config = ConfigDict(str_strip_whitespace=True)
    timestamp: datetime = Field(
        ...,
        description="Response timestamp in UTC"
    )
    session_id: str = Field(
        ...,
        description="Unique session identifier",
        max_length=MAX_ID_LENGTH
    )
    message_id: str = Field(
        ...,
        description="Unique message identifier",
        max_length=MAX_ID_LENGTH
    )
    status: ResponseStatus = Field(
        ...,
        description="Chat processing status (SUCCESS or ERROR)"
    )
    message: str = Field(
        ...,
        description="Assistant response or error message",
        max_length=MAX_CONTENT_LENGTH,
        min_length=1
    )
    metadata: Optional[Metadata] = Field(
        default_factory=Metadata,
        description="Structured metadata (e.g., SQL query, token usage)"
    )
    request_duration: float = Field(
        ...,
        description="Request processing duration in seconds",
        ge=0.0
    )

    @field_validator("timestamp")
    @classmethod
    def validate_timestamp(cls, v: datetime) -> datetime:
        """Ensure timestamp is UTC."""
        if v.tzinfo is None or v.tzinfo.utcoffset(v) is None:
            raise ValueError("Timestamp must be timezone-aware (UTC)")
        return v

    @field_validator("session_id")
    @classmethod
    def validate_session_id(cls, v: str) -> str:
        """Ensure session_id is valid and safe."""
        if not v or not ID_PATTERN.match(v):
            raise ValueError(f"Invalid session_id: {v}, must be alphanumeric with hyphens/underscores")
        return v

    @field_validator("message_id")
    @classmethod
    def validate_message_id(cls, v: str) -> str:
        """Ensure message_id is valid and safe."""
        if not v or not ID_PATTERN.match(v):
            raise ValueError(f"Invalid message_id: {v}, must be alphanumeric with hyphens/underscores")
        return v

    @field_validator("message")
    @classmethod
    def validate_message(cls, v: str) -> str:
        """Ensure message is non-empty."""
        if not v.strip():
            raise ValueError("Message content cannot be empty")
        return v

class DeleteSessionResponse(BaseModel):
    """Response for session deletion.

    Provides details about the session deletion result.

    Attributes:
        timestamp: Response timestamp in UTC.
        session_id: Unique session identifier.
        status: Deletion status (SUCCESS or ERROR).
        message: Descriptive result message.
        request_duration: Processing duration in seconds.

    Raises:
        ValueError: If fields are invalid (e.g., invalid session_id).
    """
    model_config = ConfigDict(str_strip_whitespace=True)
    timestamp: datetime = Field(
        ...,
        description="Response timestamp in UTC"
    )
    session_id: str = Field(
        ...,
        description="Unique session identifier",
        max_length=MAX_ID_LENGTH
    )
    status: ResponseStatus = Field(
        ...,
        description="Deletion status (SUCCESS or ERROR)"
    )
    message: str = Field(
        ...,
        description="Descriptive result message",
        max_length=1000,
        min_length=1
    )
    request_duration: float = Field(
        ...,
        description="Request processing duration in seconds",
        ge=0.0
    )

    @field_validator("timestamp")
    @classmethod
    def validate_timestamp(cls, v: datetime) -> datetime:
        """Ensure timestamp is UTC."""
        if v.tzinfo is None or v.tzinfo.utcoffset(v) is None:
            raise ValueError("Timestamp must be timezone-aware (UTC)")
        return v

    @field_validator("session_id")
    @classmethod
    def validate_session_id(cls, v: str) -> str:
        """Ensure session_id is valid and safe."""
        if not v or not ID_PATTERN.match(v):
            raise ValueError(f"Invalid session_id: {v}, must be alphanumeric with hyphens/underscores")
        return v

    @field_validator("message")
    @classmethod
    def validate_message(cls, v: str) -> str:
        """Ensure message is non-empty."""
        if not v.strip():
            raise ValueError("Message content cannot be empty")
        return v