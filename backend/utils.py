"""Utility functions for the UAV Log Viewer application.

This module contains utility functions used across different modules in the UAV Log Viewer
application. It provides common functionality like UUID validation that is needed in
multiple places, following the DRY (Don't Repeat Yourself) principle.
"""

import uuid
import structlog
from fastapi import HTTPException


# Constants for error messages
INVALID_UUID_ERROR_MSG: str = "Invalid UUID format"
INVALID_SESSION_ID_ERROR_MSG: str = "Invalid session ID format: {}"

# HTTP status code
BAD_REQUEST_STATUS_CODE: int = 400  # Corresponds to status.HTTP_400_BAD_REQUEST


def validate_uuid(uuid_str: str, raise_http_exception: bool = False) -> None:
    """Validate that a string is a valid UUID.

    This function checks if a given string is a valid UUID according to RFC 4122.
    It's used throughout the application to validate session IDs and other UUID-based
    identifiers. The function can either raise a standard ValueError or a FastAPI
    HTTPException depending on the context in which it's used.

    Args:
        uuid_str: String to validate as UUID (session ID or other identifier)
        raise_http_exception: If True, raises HTTPException instead of ValueError

    Raises:
        ValueError: If string is not a valid UUID and raise_http_exception is False
        HTTPException: If string is not a valid UUID and raise_http_exception is True

    Example:
        >>> validate_uuid("123e4567-e89b-12d3-a456-426614174000")  # Valid UUID
        >>> validate_uuid("invalid-uuid")  # Raises ValueError
    """
    try:
        # More lenient validation - just try to parse it as a UUID
        _ = uuid.UUID(uuid_str)
        # No need to check if str(uuid_obj) == uuid_str as this can vary by case
    except ValueError:
        if raise_http_exception:
            logger = structlog.get_logger("utils")
            logger.error(INVALID_UUID_ERROR_MSG, session_id=uuid_str)
            raise HTTPException(
                status_code=BAD_REQUEST_STATUS_CODE,
                detail=INVALID_SESSION_ID_ERROR_MSG.format(uuid_str),
            )
        else:
            raise ValueError(f"{INVALID_UUID_ERROR_MSG}: {uuid_str}")
