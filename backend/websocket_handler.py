"""WebSocket handler for real-time chat communication.

This module implements WebSocket endpoints for the UAV Log Viewer chat application.
It handles real-time message streaming and chat interactions.
"""

import asyncio
import json
import structlog
from typing import Dict, List, Any, Optional
from fastapi import WebSocket, status
from datetime import datetime, timezone
from backend.models import (
    Session,
    Message,
    ChatRequest,
    ChatResponse,
    ResponseStatus,
    Metadata,
)
from backend.shared_state import sessions, agents, per_session_locks


# Custom JSON encoder to handle datetime objects
class DateTimeEncoder(json.JSONEncoder):
    """Custom JSON encoder that converts datetime objects to ISO format strings."""

    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)


# Constants for WebSocket handler configuration
WS_TOKEN_DELAY_SECONDS: float = 0.05  # Delay between token streaming
WS_INVALID_SESSION_CODE: int = 1008  # WebSocket close code for invalid session
WS_ERROR_MESSAGE_TYPE: str = "error"  # Message type for error responses
WS_STREAM_START_TYPE: str = "stream_start"  # Message type for stream start
WS_STREAM_TOKEN_TYPE: str = "stream_token"  # Message type for token streaming
WS_STREAM_END_TYPE: str = "stream_end"  # Message type for stream end

# Create a dedicated logger for WebSocket handler
logger = structlog.get_logger("websocket_handler")

# Store active WebSocket connections
active_connections: Dict[str, List[WebSocket]] = {}


async def send_json_safe(websocket: WebSocket, data: Any) -> None:
    """Send JSON data through a WebSocket with datetime handling.

    Args:
        websocket: The WebSocket connection to send data through
        data: The data to send (will be JSON serialized with datetime handling)
    """
    json_str = json.dumps(data, cls=DateTimeEncoder)
    await websocket.send_text(json_str)


async def connect_websocket(websocket: WebSocket, session_id: str) -> bool:
    """Accept a WebSocket connection and register it with a session.

    This function validates the session, accepts the WebSocket connection if valid,
    and registers it in the active_connections dictionary for future communication.

    Args:
        websocket: The WebSocket connection to register
        session_id: The session ID to associate with this connection

    Returns:
        bool: True if connection was successful, False otherwise
    """
    # Create session-specific logger
    session_logger = logger.bind(session_id=session_id)

    # Validate session exists
    if session_id not in sessions:
        await websocket.close(code=WS_INVALID_SESSION_CODE, reason="Invalid session")
        session_logger.warning("WebSocket connection rejected - invalid session")
        return False

    # Accept the connection
    await websocket.accept()

    # Register the connection
    if session_id not in active_connections:
        active_connections[session_id] = []
    active_connections[session_id].append(websocket)

    session_logger.info("WebSocket connection established")
    return True


async def disconnect_websocket(websocket: WebSocket, session_id: str) -> None:
    """Remove a WebSocket connection from the active connections.

    Args:
        websocket: The WebSocket connection to remove
        session_id: The session ID associated with this connection
    """
    # Create session-specific logger
    session_logger = logger.bind(session_id=session_id)

    if session_id in active_connections:
        if websocket in active_connections[session_id]:
            active_connections[session_id].remove(websocket)
        if not active_connections[session_id]:
            del active_connections[session_id]
    session_logger.info("WebSocket connection closed")


async def broadcast_to_session(session_id: str, message: Dict[str, Any]) -> None:
    """Broadcast a message to all WebSocket connections for a session.

    Args:
        session_id: The session ID to broadcast to
        message: The message to broadcast
    """
    # Create session-specific logger
    session_logger = logger.bind(session_id=session_id)

    if session_id in active_connections:
        disconnected_websockets = []
        active_count = len(active_connections[session_id])

        for websocket in active_connections[session_id]:
            try:
                await send_json_safe(websocket, message)
            except RuntimeError:
                # Connection is closed or in an error state
                disconnected_websockets.append(websocket)

        # Clean up disconnected websockets
        for websocket in disconnected_websockets:
            if websocket in active_connections[session_id]:
                active_connections[session_id].remove(websocket)

        if disconnected_websockets:
            session_logger.info(
                "Cleaned up disconnected WebSockets",
                disconnected_count=len(disconnected_websockets),
                remaining_count=len(active_connections[session_id]),
            )


async def handle_websocket_message(
    websocket: WebSocket, session_id: str, data: Dict[str, Any]
) -> None:
    """Handle a message received from a WebSocket connection.

    Args:
        websocket: The WebSocket connection that sent the message
        session_id: The session ID associated with this connection
        data: The message data
    """
    start_time = asyncio.get_event_loop().time()
    # Create session-specific logger
    session_logger = logger.bind(session_id=session_id)

    # Validate message format
    if "message" not in data:
        await send_json_safe(
            websocket,
            {
                "type": WS_ERROR_MESSAGE_TYPE,
                "message": "Invalid message format: 'message' field is required",
            },
        )
        session_logger.warning(
            "Invalid message format received - missing message field"
        )
        return

    message_text = data["message"]
    # Support both camelCase and snake_case for backward compatibility
    message_id = data.get(
        "message_id", data.get("messageId", str(datetime.now(timezone.utc).timestamp()))
    )

    # Create a ChatRequest object
    try:
        chat_request = ChatRequest(
            session_id=session_id,
            message=message_text,
            message_id=message_id,
            # Note: We don't include max_tokens as per your requirement
        )
    except ValueError as e:
        await send_json_safe(
            websocket,
            {"type": WS_ERROR_MESSAGE_TYPE, "message": f"Invalid request: {str(e)}"},
        )
        session_logger.warning("Invalid chat request", error=str(e))
        return

    # Get session lock
    async with per_session_locks[session_id]:
        # Validate session exists
        if session_id not in sessions:
            await send_json_safe(
                websocket,
                {"type": WS_ERROR_MESSAGE_TYPE, "message": "Invalid session ID"},
            )
            session_logger.warning("Message received for invalid session")
            return

        session: Session = sessions[session_id]

        # Check session status
        if session.status != "READY":
            error_message = "Session is not ready for chat"
            if session.status == "PENDING":
                error_message = "Session is still being processed. Please wait until processing is complete."
                session_logger.info("Rejected message - session still processing")
            elif session.status == "ERROR":
                error_message = (
                    session.status_message or "Session is in an error state."
                )
                session_logger.info("Rejected message - session in error state")
            else:
                error_message = f"Session is in an invalid state: {session.status}"
                session_logger.warning(
                    f"Rejected message - invalid session state: {session.status}"
                )

            await send_json_safe(
                websocket, {"type": WS_ERROR_MESSAGE_TYPE, "message": error_message}
            )
            return

        # Add user message to session
        user_message = Message(
            message_id=message_id,
            role="user",
            content=message_text,
            timestamp=datetime.now(timezone.utc),
        )
        session.messages.append(user_message)
        session_logger.info("Added user message to session", message_id=message_id)

        # Get agent for this session
        if session_id not in agents:
            try:
                # Import here to avoid circular imports
                from backend.main import (
                    initialize_agent,
                    STORAGE_DIR,
                    DUCKDB_FILE_PREFIX,
                )
                import os
                import time

                # Initialize agent
                db_path = os.path.join(
                    STORAGE_DIR, f"{DUCKDB_FILE_PREFIX}{session_id}.duckdb"
                )
                start_time = time.perf_counter()
                agent = await initialize_agent(session_id, db_path, None, start_time)

                if not agent:
                    await send_json_safe(
                        websocket,
                        {
                            "type": WS_ERROR_MESSAGE_TYPE,
                            "message": "Failed to initialize agent for this session",
                        },
                    )
                    session_logger.error("Failed to initialize agent for session")
                    return

                session_logger.info("Agent initialized for WebSocket session")
            except Exception as e:
                await send_json_safe(
                    websocket,
                    {
                        "type": WS_ERROR_MESSAGE_TYPE,
                        "message": f"Error initializing agent: {str(e)}",
                    },
                )
                session_logger.error(
                    "Error initializing agent", error=str(e), exc_info=True
                )
                return

        agent = agents[session_id]

        # Process message with streaming
        try:
            # Start streaming
            await send_json_safe(
                websocket, {"type": WS_STREAM_START_TYPE, "message_id": message_id}
            )

            # Process message and stream tokens
            response_text = ""
            metadata: Optional[Metadata] = None
            token_count = 0

            session_logger.info("Starting message processing with agent")

            # Process the message using the agent
            async def token_generator():
                nonlocal response_text, metadata, token_count
                try:
                    response, meta = await agent.process_message(message_text)
                    response_text = response
                    metadata = meta

                    # Simulate token-by-token streaming
                    words = response.split()
                    for word in words:
                        token_count += 1
                        yield word + " "
                        await asyncio.sleep(
                            WS_TOKEN_DELAY_SECONDS
                        )  # Small delay between tokens
                except Exception as e:
                    session_logger.error(
                        "Error during token generation", error=str(e), exc_info=True
                    )
                    raise

            # Stream tokens
            try:
                async for token in token_generator():
                    await send_json_safe(
                        websocket, {"type": WS_STREAM_TOKEN_TYPE, "token": token}
                    )
                session_logger.info(
                    "Completed token streaming", token_count=token_count
                )
            except Exception as e:
                session_logger.error(
                    "Error during token streaming", error=str(e), exc_info=True
                )
                raise

            # Create a ChatResponse object
            request_duration = asyncio.get_event_loop().time() - start_time
            chat_response = ChatResponse(
                timestamp=datetime.now(timezone.utc),
                session_id=session_id,
                message_id=message_id,
                status=ResponseStatus.SUCCESS,
                message=response_text,
                metadata=metadata,
                request_duration=request_duration,
            )

            # End streaming
            await send_json_safe(
                websocket,
                {
                    "type": WS_STREAM_END_TYPE,
                    "message_id": message_id,
                    "response": chat_response.model_dump(),
                },
            )

            # Add assistant message to session
            assistant_message = Message(
                message_id=message_id,
                role="assistant",
                content=response_text,
                timestamp=datetime.now(timezone.utc),
                metadata=metadata,
            )
            session.messages.append(assistant_message)

            session_logger.info(
                "Processed WebSocket message",
                message_id=message_id,
                response_length=len(response_text),
                request_duration=request_duration,
            )

        except Exception as e:
            # Create error response
            request_duration = asyncio.get_event_loop().time() - start_time
            error_response = ChatResponse(
                timestamp=datetime.now(timezone.utc),
                session_id=session_id,
                message_id=message_id,
                status=ResponseStatus.ERROR,
                message=f"Error processing message: {str(e)}",
                request_duration=request_duration,
            )

            session_logger.error(
                "Error processing WebSocket message",
                message_id=message_id,
                error=str(e),
                exc_info=True,
            )

            await send_json_safe(
                websocket,
                {
                    "type": WS_ERROR_MESSAGE_TYPE,
                    "message": f"Error processing message: {str(e)}",
                    "response": error_response.model_dump(),
                },
            )
