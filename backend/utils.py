"""Utility functions for the UAV Log Viewer application.

This module contains utility functions used across different modules.
"""

import uuid
from typing import Optional

def validate_uuid(uuid_str: str, raise_http_exception: bool = False) -> None:
    """Validate that a string is a valid UUID.
    
    Args:
        uuid_str: String to validate as UUID.
        raise_http_exception: If True, raises HTTPException instead of ValueError.
        
    Raises:
        ValueError: If string is not a valid UUID and raise_http_exception is False.
        HTTPException: If string is not a valid UUID and raise_http_exception is True.
    """
    try:
        # More lenient validation - just try to parse it as a UUID
        uuid_obj = uuid.UUID(uuid_str)
        # No need to check if str(uuid_obj) == uuid_str as this can vary by case
    except ValueError:
        if raise_http_exception:
            from fastapi import HTTPException, status
            import structlog
            logger = structlog.get_logger("utils")
            logger.error("Invalid UUID format", session_id=uuid_str)
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid session ID format: {uuid_str}"
            )
        else:
            raise ValueError(f"Invalid UUID format: {uuid_str}")
