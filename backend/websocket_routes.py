"""WebSocket routes for the FastAPI application.

This module defines the WebSocket endpoints for real-time chat communication.
"""

import asyncio
import structlog
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, status
from starlette.websockets import WebSocketState
from backend.websocket_handler import (
    connect_websocket,
    disconnect_websocket,
    handle_websocket_message
)

from backend.utils import validate_uuid

# Create a dedicated logger for WebSocket routes
logger = structlog.get_logger("websocket_routes")

# Create router for WebSocket endpoints
router = APIRouter()

@router.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    """WebSocket endpoint for real-time chat communication.
    
    Args:
        websocket: The WebSocket connection
        session_id: The session ID to associate with this connection
    """
    # Create session-specific logger
    session_logger = logger.bind(session_id=session_id)
    
    # Validate session ID format
    try:
        validate_uuid(session_id)
    except Exception:
        await websocket.close(code=1008, reason="Invalid session ID format")
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
        # Set receive timeout to prevent hanging connections
        receive_timeout = 3600  # 1 hour timeout
        
        while True:
            try:
                # Receive JSON message with timeout
                data = await asyncio.wait_for(
                    websocket.receive_json(),
                    timeout=receive_timeout
                )
                
                # Process message
                await handle_websocket_message(websocket, session_id, data)
                
            except asyncio.TimeoutError:
                # Connection has been idle for too long
                session_logger.info("WebSocket connection timed out after inactivity", timeout_seconds=receive_timeout)
                if websocket.client_state == WebSocketState.CONNECTED:
                    await websocket.close(code=status.WS_1000_NORMAL_CLOSURE, reason="Connection timeout due to inactivity")
                break
            
    except WebSocketDisconnect as e:
        # Handle client disconnect with reason
        disconnect_reason = "Client initiated disconnect"
        if hasattr(e, 'code'):
            disconnect_reason = f"Code: {e.code}, Reason: {getattr(e, 'reason', 'Unknown')}"
        
        session_logger.info("WebSocket client disconnected", reason=disconnect_reason)
        await disconnect_websocket(websocket, session_id)
        
    except Exception as e:
        # Handle other errors
        session_logger.error("WebSocket error", error=str(e), exc_info=True)
        await disconnect_websocket(websocket, session_id)
        
        # Only try to close if still connected
        if websocket.client_state == WebSocketState.CONNECTED:
            try:
                await websocket.close(code=status.WS_1011_INTERNAL_ERROR, reason="Server error")
            except Exception:
                pass  # Connection might already be closed
