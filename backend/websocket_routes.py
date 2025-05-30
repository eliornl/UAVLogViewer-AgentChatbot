"""WebSocket routes for the FastAPI application.

This module defines the WebSocket endpoints for real-time chat communication.
It handles connection establishment, message processing, and proper cleanup
when connections are closed.
"""

import asyncio
import structlog
from typing import Dict, Any, Optional
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, status
from starlette.websockets import WebSocketState
from backend.websocket_handler import (
    connect_websocket,
    disconnect_websocket,
    handle_websocket_message,
)
from backend.utils import validate_uuid

# Constants for WebSocket configuration
WS_RECEIVE_TIMEOUT_SECONDS: int = 3600  # 1 hour timeout for receiving messages
WS_INVALID_SESSION_CODE: int = 1008  # WebSocket close code for invalid session
WS_NORMAL_CLOSURE_CODE: int = 1000  # Normal closure code
WS_INTERNAL_ERROR_CODE: int = 1011  # Internal error code

# Create a dedicated logger for WebSocket routes
logger = structlog.get_logger("websocket_routes")

# Create router for WebSocket endpoints
router = APIRouter()


@router.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str) -> None:
    """WebSocket endpoint for real-time chat communication.

    Establishes a WebSocket connection for the specified session, handles incoming
    messages, and manages the connection lifecycle. The connection supports token
    streaming for real-time chat responses and maintains session isolation.

    Args:
        websocket: The WebSocket connection object
        session_id: The unique session ID to associate with this connection

    Note:
        The connection will automatically close after WS_RECEIVE_TIMEOUT_SECONDS
        of inactivity to prevent resource leaks.
    """
    # Create session-specific logger
    session_logger = logger.bind(session_id=session_id)

    # Validate session ID format
    try:
        validate_uuid(session_id)
    except Exception:
        await websocket.close(
            code=WS_INVALID_SESSION_CODE, reason="Invalid session ID format"
        )
        session_logger.warning("Invalid session ID format")
        return

    # Connect WebSocket
    connection_success = await connect_websocket(websocket, session_id)
    if not connection_success:
        session_logger.info("WebSocket connection rejected")
        return  # Connection was rejected

    session_logger.info("WebSocket connection established")

    # Handle messages
    try:
        while True:
            try:
                # Receive JSON message with timeout to prevent hanging connections
                data = await asyncio.wait_for(
                    websocket.receive_json(), timeout=WS_RECEIVE_TIMEOUT_SECONDS
                )

                # Process message through the websocket handler
                await handle_websocket_message(websocket, session_id, data)

            except asyncio.TimeoutError:
                # Connection has been idle for too long
                session_logger.info(
                    "WebSocket connection timed out after inactivity",
                    timeout_seconds=WS_RECEIVE_TIMEOUT_SECONDS,
                )
                if websocket.client_state == WebSocketState.CONNECTED:
                    await websocket.close(
                        code=WS_NORMAL_CLOSURE_CODE,
                        reason="Connection timeout due to inactivity",
                    )
                break

    except WebSocketDisconnect as e:
        # Handle client disconnect with reason
        disconnect_reason = "Client initiated disconnect"
        if hasattr(e, "code"):
            disconnect_reason = (
                f"Code: {e.code}, Reason: {getattr(e, 'reason', 'Unknown')}"
            )

        session_logger.info("WebSocket client disconnected", reason=disconnect_reason)
        await disconnect_websocket(websocket, session_id)

    except Exception as e:
        # Handle other errors
        session_logger.error("WebSocket error", error=str(e), exc_info=True)
        await disconnect_websocket(websocket, session_id)

        # Only try to close if still connected
        if websocket.client_state == WebSocketState.CONNECTED:
            try:
                await websocket.close(
                    code=WS_INTERNAL_ERROR_CODE, reason="Server error"
                )
            except Exception:
                session_logger.debug(
                    "Failed to close WebSocket connection that was already closed"
                )
                pass  # Connection might already be closed
