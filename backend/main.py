import os
import uuid
import asyncio
import duckdb
import structlog
import time
import aiofiles
import json
from json import JSONEncoder
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Dict, List, Set, Optional, AsyncGenerator, Tuple, Any
from fastapi import FastAPI, UploadFile, File, HTTPException, status, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from tenacity import retry, stop_after_attempt, wait_exponential
from dotenv import load_dotenv
from backend.telemetry_agent import TelemetryAgent
from backend.telemetry_processor import TelemetryProcessor
from backend.llm_token_counter import LLMTokenCounter
from backend.vector_store_manager import VectorStoreManager
from langchain_openai import OpenAIEmbeddings
from backend.models import (
    ResponseStatus,
    ChatRequest,
    UploadResponse,
    ChatResponse,
    DeleteSessionResponse,
    Session,
    Message,
    SessionStatus,
)


# Custom JSON Encoder for Metadata objects
class CustomJSONEncoder(JSONEncoder):
    """Custom JSON encoder that handles Metadata objects and other special types."""

    def default(self, obj):
        if hasattr(obj, "model_dump"):
            # Handle Pydantic models (like Metadata)
            return obj.model_dump()
        # Let the base class default method handle other types
        return super().default(obj)


# Constants
# File handling constants
ALLOWED_EXTENSIONS: Set[str] = {".bin", ".tlog"}  # Supported flight log file extensions
MAX_FILE_SIZE: int = 100 * 1024 * 1024  # 100 MB - Maximum allowed file size for uploads
CHUNK_SIZE: int = 1048576  # 1 MB - Chunk size for file reading/writing operations

# Media types for responses
JSON_MEDIA_TYPE: str = "application/json"
TEXT_MEDIA_TYPE: str = "text/plain"
CSV_MEDIA_TYPE: str = "text/csv"

# Session management constants
MAX_MESSAGES: int = 10_000  # Limit messages per session to prevent memory issues
MAX_SESSIONS: int = 1000  # Maximum number of concurrent sessions allowed
MAX_SESSION_ATTEMPTS: int = 5  # Maximum attempts to generate a unique session ID
CHAT_TIMEOUT_SECONDS: float = 30.0  # Timeout for chat message processing

# File naming constants
FILENAME_SAFE_CHARS: str = (
    "abcdefghijklmnopqrstuvwxyz0123456789-_"  # Safe characters for filenames
)
DUCKDB_FILE_PREFIX: str = ""  # Prefix for DuckDB database files

# Configure structlog for structured logging
structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.stdlib.add_log_level,
        structlog.processors.JSONRenderer(),
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)
logger: structlog.stdlib.BoundLogger = structlog.get_logger()

# Load environment variables
load_dotenv()

# Environment variables with type hints
# Keep only the essential environment variables
STORAGE_DIR: str = os.path.abspath(os.getenv("STORAGE_DIR", "./storage"))
LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
LLM_MODEL: str = os.getenv("LLM_MODEL", "gpt-4o")
MAX_MODEL_TOKENS: int = int(os.getenv("MAX_MODEL_TOKENS", 8192))

# These constants are now defined at the top of the file


# Validate environment variables
def validate_env_vars() -> None:
    """Validate essential environment variables and raise ValueError if invalid.

    Checks:
        - OPENAI_API_KEY: Must be provided for LLM functionality
        - STORAGE_DIR: Must exist and be writable
        - LLM_MODEL: Not validated but used for LLM initialization
        - LOG_LEVEL: Not validated but used for logging configuration
    """
    if not OPENAI_API_KEY:
        logger.error("OPENAI_API_KEY is not set")
        raise ValueError("OPENAI_API_KEY environment variable is required")

    # Ensure storage directory exists and is writable
    if not os.path.exists(STORAGE_DIR):
        os.makedirs(STORAGE_DIR, exist_ok=True)
        logger.info("Created storage directory", storage_dir=STORAGE_DIR)

    if not os.path.isdir(STORAGE_DIR) or not os.access(STORAGE_DIR, os.W_OK):
        logger.error("STORAGE_DIR is not a writable directory", storage_dir=STORAGE_DIR)
        raise ValueError(
            f"Storage directory '{STORAGE_DIR}' is not a writable directory"
        )


validate_env_vars()

# Initialize LLMTokenCounter
try:
    token_counter: LLMTokenCounter = LLMTokenCounter(model_name=LLM_MODEL)
except ValueError as e:
    logger.error("Failed to initialize LLMTokenCounter", error=str(e))
    raise ValueError(f"Failed to initialize LLMTokenCounter: {str(e)}") from e

# Initialize OpenAIEmbeddings
try:
    embeddings: OpenAIEmbeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)
except Exception as e:
    logger.error("Failed to initialize OpenAIEmbeddings", error=str(e), exc_info=True)
    raise ValueError(f"Failed to initialize OpenAIEmbeddings: {str(e)}") from e


# Lifespan event handler for startup and shutdown
@asynccontextmanager
async def lifespan(app: FastAPI):
    global vector_store_manager
    # Startup: Initialize VectorStoreManager
    try:
        vector_store_manager = VectorStoreManager(embeddings=embeddings)
        await vector_store_manager.initialize()
        logger.info("VectorStoreManager initialized successfully")
    except Exception as e:
        logger.error(
            "Failed to initialize VectorStoreManager", error=str(e), exc_info=True
        )
        raise ValueError(f"Failed to initialize VectorStoreManager: {str(e)}") from e
    yield
    # Shutdown: Clean up resources (if needed)
    logger.info("Shutting down application")


# Initialize FastAPI app
app = FastAPI(title="AeroTelemetry AI Analysis API", lifespan=lifespan)

# Apply CORS middleware with more permissive settings
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for now to debug the issue
    allow_credentials=False,  # Must be False when using wildcard origins
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],  # Allow all headers
)


# Global OPTIONS endpoint to handle all CORS preflight requests
@app.options("/{full_path:path}")
async def options_handler(full_path: str):
    """Handle OPTIONS preflight requests for any endpoint.

    Args:
        full_path: The full path of the request.

    Returns:
        Response with CORS headers.
    """
    headers = {
        "Access-Control-Allow-Origin": "*",  # Allow all origins
        "Access-Control-Allow-Methods": "*",  # Allow all methods
        "Access-Control-Allow-Headers": "*",  # Allow all headers
        "Access-Control-Max-Age": "86400",  # Cache preflight response for 24 hours
    }
    return Response(status_code=200, content=b"", headers=headers)


# Import shared state to use the same instances across modules
from backend.shared_state import sessions, agents, per_session_locks

# Global lock for operations that modify the sessions dictionary
global_sessions_lock = asyncio.Lock()

# Import WebSocket routes - must be after shared state initialization
from backend.websocket_routes import router as websocket_router

# Include WebSocket routes
app.include_router(websocket_router)


async def validate_file(file: UploadFile) -> Tuple[str, str, int]:
    """Validate uploaded file and return sanitized filename, extension, and size.

    Args:
        file: Uploaded file to validate.

    Returns:
        Tuple[str, str, int]: Sanitized filename, file extension, and file size.

    Raises:
        HTTPException: If file is invalid, empty, too large, or has invalid type/extension.
    """
    if not file or not file.filename:
        logger.error("No file provided")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="No file provided"
        )

    sanitized_filename: str = "".join(
        c if c.isalnum() or c in ".-_" else "_" for c in os.path.basename(file.filename)
    )
    file_ext: str = os.path.splitext(sanitized_filename)[1].lower()
    if file_ext not in ALLOWED_EXTENSIONS:
        logger.error(
            "Invalid file extension", file_name=sanitized_filename, ext=file_ext
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid file extension. Allowed: {ALLOWED_EXTENSIONS}",
        )

    file_size: int = file.size or 0
    if file_size == 0:
        await file.seek(0)
        async for chunk in file:
            file_size += len(chunk)
        await file.seek(0)

    if file_size == 0:
        logger.error("Empty file provided", file_name=sanitized_filename)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Empty file provided"
        )
    if file_size > MAX_FILE_SIZE:
        logger.error(
            "File size exceeds maximum",
            file_name=sanitized_filename,
            size=file_size,
            max_size=MAX_FILE_SIZE,
        )
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File size exceeds maximum {MAX_FILE_SIZE/1024/1024}MB",
        )

    return sanitized_filename, file_ext, file_size


async def process_file(file: UploadFile, tmp_file_path: str) -> None:
    """Write uploaded file to temporary file using async I/O.

    Args:
        file: Uploaded file to write.
        tmp_file_path: Path to temporary file.

    Raises:
        HTTPException: If file write fails due to I/O error.
    """
    try:
        async with aiofiles.open(tmp_file_path, "wb") as tmp_file:
            await file.seek(0)
            while chunk := await file.read(CHUNK_SIZE):  # type: bytes
                await tmp_file.write(chunk)
    except OSError as e:
        logger.error("File write error", tmp_file_path=tmp_file_path, error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to process file",
        )


# Import utility functions
from backend.utils import validate_uuid as _validate_uuid


def validate_uuid(session_id: str) -> None:
    """Validate that session_id is a valid UUID.

    Args:
        session_id: Session ID to validate.

    Raises:
        HTTPException: If session_id is not a valid UUID.
    """
    _validate_uuid(session_id, raise_http_exception=True)


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=5))
async def cleanup_temp_file(
    tmp_file_path: str,
    logger: structlog.stdlib.BoundLogger,
    file_name: Optional[str] = None,
) -> None:
    """Clean up a temporary file with retries.

    Args:
        tmp_file_path: Path to temporary file.
        logger: Logger instance for logging cleanup events.
        file_name: Optional filename for logging context.

    Raises:
        OSError: If cleanup fails after retries, re-raises the specific OSError.
    """
    try:
        if os.path.exists(tmp_file_path):
            await asyncio.to_thread(os.unlink, tmp_file_path)
            logger.info(
                "Cleaned up temporary file",
                file_name=file_name or "",
                tmp_file_path=tmp_file_path,
            )
    except OSError as e:
        logger.warning(
            "Failed to clean up temporary file",
            file_name=file_name or "",
            tmp_file_path=tmp_file_path,
            error=str(e),
        )
        # Explicitly re-raise OSError for tenacity retry
        raise OSError(
            f"Failed to delete temporary file {tmp_file_path}: {str(e)}"
        ) from e


def generate_filename(session_id: str, extension: str) -> str:
    """Generate a safe filename for export.

    Args:
        session_id: Session ID to include in filename.
        extension: File extension (e.g., '.json').

    Returns:
        str: Safe filename like 'chat_{session_id}_{timestamp}{extension}'.
    """
    export_timestamp: str = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    safe_session_id: str = "".join(
        c if c.lower() in FILENAME_SAFE_CHARS else "_" for c in session_id
    )
    return f"chat_{safe_session_id}_{export_timestamp}{extension}"


async def create_session(
    sanitized_filename: str, file_size: int, session_id: str
) -> Session:
    """Create a new session with PENDING status.

    Args:
        sanitized_filename: Sanitized name of uploaded file.
        file_size: Size of uploaded file in bytes.
        session_id: Unique session ID.

    Returns:
        Session: New session object.
    """
    session: Session = Session(
        session_id=session_id,
        created_at=datetime.now(timezone.utc),
        file_name=sanitized_filename,
        file_size=file_size,
        status=SessionStatus.PENDING,
        status_message="Processing telemetry file",
    )
    sessions[session_id] = session
    per_session_locks[session_id] = asyncio.Lock()
    logger.info(
        "Created session with PENDING status",
        session_id=session_id,
        file_name=sanitized_filename,
        file_size=file_size,
    )
    return session


@app.post("/upload", response_model=UploadResponse, status_code=status.HTTP_201_CREATED)
async def upload_file(file: UploadFile = File(...)) -> UploadResponse:
    """Upload a flight log file, parse it, and create a new session.

    Args:
        file: Flight log file (.bin or .tlog) to upload.

    Returns:
        UploadResponse: Response with session ID, file details, and processing status.

    Raises:
        HTTPException: If file is invalid, too large, or processing fails.
    """
    start_time: float = time.perf_counter()
    sanitized_filename: str
    file_ext: str
    file_size: int
    sanitized_filename, file_ext, file_size = await validate_file(file)

    async with aiofiles.tempfile.NamedTemporaryFile(
        suffix=file_ext, delete=False
    ) as tmp_file:
        tmp_file_path: str = tmp_file.name
        try:
            await process_file(file, tmp_file_path)
            logger.info(
                "Created temporary file",
                file_name=sanitized_filename,
                tmp_file_path=tmp_file_path,
                file_size=file_size,
            )

            # Create unique session ID
            async with global_sessions_lock:
                if len(sessions) >= MAX_SESSIONS:
                    logger.error(
                        "Maximum session limit reached", max_sessions=MAX_SESSIONS
                    )
                    raise HTTPException(
                        status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                        detail="Maximum session limit reached",
                    )

                session_id: str = str(uuid.uuid4())
                attempt: int = 0
                while session_id in sessions and attempt < MAX_SESSION_ATTEMPTS:
                    logger.warning(
                        "Session ID collision", session_id=session_id, attempt=attempt
                    )
                    session_id = str(uuid.uuid4())
                    attempt += 1
                if session_id in sessions:
                    logger.error(
                        "Failed to generate unique session ID",
                        max_attempts=MAX_SESSION_ATTEMPTS,
                    )
                    raise HTTPException(
                        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                        detail="Failed to generate unique session ID",
                    )

                session: Session = await create_session(
                    sanitized_filename, file_size, session_id
                )
                session_logger = logger.bind(session_id=session_id)

                db_path: str = os.path.join(
                    STORAGE_DIR, f"{DUCKDB_FILE_PREFIX}{session_id}.duckdb"
                )
                if os.path.exists(db_path):
                    session_logger.error("DuckDB file already exists", db_path=db_path)
                    raise HTTPException(
                        status_code=status.HTTP_409_CONFLICT,
                        detail="Session ID conflict",
                    )

            try:
                await TelemetryProcessor.parse_and_save(
                    tmp_file_path, session, STORAGE_DIR, DUCKDB_FILE_PREFIX
                )
                session_logger.info(
                    "Session processing completed",
                    session_status=session.status,
                    status_message=session.status_message,
                )
                return UploadResponse(
                    timestamp=session.created_at,
                    file_name=sanitized_filename,
                    file_size=file_size,
                    session_id=session_id,
                    status=ResponseStatus.SUCCESS,
                    message=session.status_message
                    or "File uploaded and processed successfully",
                    request_duration=time.perf_counter() - start_time,
                )
            except ValueError as e:
                session_logger.error("Parsing failed", error=str(e), exc_info=True)
                return UploadResponse(
                    timestamp=session.created_at,
                    file_name=sanitized_filename,
                    file_size=file_size,
                    session_id=session_id,
                    status=ResponseStatus.ERROR,
                    message=session.status_message or f"File parsing failed: {str(e)}",
                    request_duration=time.perf_counter() - start_time,
                )
            except (OSError, duckdb.IOException) as e:
                session_logger.error(
                    "File or database error", error=str(e), exc_info=True
                )
                return UploadResponse(
                    timestamp=session.created_at,
                    file_name=sanitized_filename,
                    file_size=file_size,
                    session_id=session_id,
                    status=ResponseStatus.ERROR,
                    message=session.status_message
                    or f"File or database error: {str(e)}",
                    request_duration=time.perf_counter() - start_time,
                )
        finally:
            await cleanup_temp_file(
                tmp_file_path,
                session_logger if "session_logger" in locals() else logger,
                sanitized_filename,
            )


async def initialize_agent(
    session_id: str, db_path: str, max_tokens: int, max_start_time: float
) -> Optional[TelemetryAgent]:
    """Initialize or retrieve a TelemetryAgent for a session.

    Args:
        session_id: Session ID for the agent.
        db_path: Path to DuckDB file for the session.
        max_tokens: Maximum number of tokens for the agent.
        max_start_time: Maximum start time for the agent.

    Returns:
        Optional[TelemetryAgent]: Initialized or existing agent, or None if initialization fails.
    """
    if session_id not in agents:
        try:
            agent: TelemetryAgent = TelemetryAgent(
                session_id=session_id,
                db_path=db_path,
                openai_api_key=OPENAI_API_KEY,
                llm_model=LLM_MODEL,
                token_encoder=token_counter,
                vector_store_manager=vector_store_manager,
                embeddings=embeddings,
                max_tokens=max_tokens,
            )
            await agent.async_initialize()
            agents[session_id] = agent
            logger.info(
                "Initialized new TelemetryAgent for session", session_id=session_id
            )
            return agent
        except Exception as e:
            logger.error(
                "Failed to initialize TelemetryAgent",
                session_id=session_id,
                error=str(e),
                exc_info=True,
            )
            return None
    logger.debug("Retrieved existing TelemetryAgent for session", session_id=session_id)
    return agents[session_id]


async def handle_session_error(
    session: Session, session_id: str, message_id: str, start_time: float
) -> Optional[ChatResponse]:
    """Handle session error states (ERROR or PENDING).

    Args:
        session: Session object to check.
        session_id: Session ID for response.
        message_id: Message ID for response.
        start_time: Request start time for duration.

    Returns:
        Optional[ChatResponse]: Error response if session is in error state, else None.
    """
    if session.status == SessionStatus.ERROR:
        logger.info(
            "Session has an error",
            session_id=session_id,
            status_message=session.status_message,
        )
        return ChatResponse(
            timestamp=datetime.now(timezone.utc),
            session_id=session_id,
            message_id=message_id,
            status=ResponseStatus.ERROR,
            message=session.status_message
            or "Session cannot be used due to processing error",
            metadata=None,
            request_duration=time.perf_counter() - start_time,
        )
    if session.status == SessionStatus.PENDING:
        logger.info("Session is still processing", session_id=session_id)
        return ChatResponse(
            timestamp=datetime.now(timezone.utc),
            session_id=session_id,
            message_id=message_id,
            status=ResponseStatus.ERROR,
            message=session.status_message
            or "Session is still being processed. Please wait until processing is complete.",
            metadata=None,
            request_duration=time.perf_counter() - start_time,
        )
    return None


@app.post("/chat", response_model=ChatResponse)
async def chat_message(request: ChatRequest) -> ChatResponse:
    """Send a user message to the TelemetryAgent and receive a response.

    Args:
        request: Chat request with session_id, message, optional message_id, and max_tokens.

    Returns:
        ChatResponse: Assistant response with metadata and request duration.

    Raises:
        HTTPException: If session_id is invalid or session is not found.
    """
    start_time: float = time.perf_counter()
    # Try to validate UUID but don't fail if it's not a perfect UUID
    try:
        validate_uuid(request.session_id)
    except ValueError:
        # Log the issue but continue processing
        logger.warning(
            f"Non-standard session ID format: {request.session_id}, but continuing"
        )
    session_logger = logger.bind(session_id=request.session_id)

    # Check session existence
    session_lock: Optional[asyncio.Lock] = None
    async with global_sessions_lock:
        if request.session_id not in sessions:
            session_logger.error("Session not found", session_id=request.session_id)
            return ChatResponse(
                timestamp=datetime.now(timezone.utc),
                session_id=request.session_id,
                message_id=request.message_id or str(uuid.uuid4()),
                status=ResponseStatus.ERROR,
                message="Session not found",
                metadata=None,
                request_duration=time.perf_counter() - start_time,
            )
        session_lock = per_session_locks[request.session_id]

    # Process message under session lock
    async with session_lock:
        session: Session = sessions[request.session_id]
        if len(session.messages) >= MAX_MESSAGES:
            session_logger.error(
                "Maximum messages reached",
                session_id=request.session_id,
                max_messages=MAX_MESSAGES,
            )
            return ChatResponse(
                timestamp=datetime.now(timezone.utc),
                session_id=request.session_id,
                message_id=request.message_id or str(uuid.uuid4()),
                status=ResponseStatus.ERROR,
                message=f"Maximum message limit ({MAX_MESSAGES}) reached",
                metadata=None,
                request_duration=time.perf_counter() - start_time,
            )

        # Handle session status errors
        error_response: Optional[ChatResponse] = await handle_session_error(
            session,
            request.session_id,
            request.message_id or str(uuid.uuid4()),
            start_time,
        )
        if error_response:
            return error_response

        # Initialize agent
        db_path: str = os.path.join(
            STORAGE_DIR, f"{DUCKDB_FILE_PREFIX}{request.session_id}.duckdb"
        )
        agent: Optional[TelemetryAgent] = await initialize_agent(
            request.session_id, db_path, request.max_tokens, start_time
        )
        if not agent:
            return ChatResponse(
                timestamp=datetime.now(timezone.utc),
                session_id=request.session_id,
                message_id=request.message_id or str(uuid.uuid4()),
                status=ResponseStatus.ERROR,
                message="Failed to initialize chat agent",
                metadata=None,
                request_duration=time.perf_counter() - start_time,
            )

        # Create and save user message
        message_id: str = request.message_id or str(uuid.uuid4())
        user_message: Message = Message(
            message_id=message_id,
            role="user",
            content=request.message,
            timestamp=datetime.now(timezone.utc),
        )
        session.messages.append(user_message)
        session_logger.debug(
            "Saved user message to session",
            session_id=request.session_id,
            message_id=message_id,
        )

        # Process message
        try:
            response: str
            metadata: Optional[Dict]
            response, metadata = await asyncio.wait_for(
                agent.process_message(request.message, max_tokens=request.max_tokens),
                timeout=CHAT_TIMEOUT_SECONDS,
            )

            assistant_message: Message = Message(
                message_id=message_id,
                role="assistant",
                content=response,
                timestamp=datetime.now(timezone.utc),
                metadata=metadata,
            )
            session.messages.append(assistant_message)
            session_logger.info(
                "Processed chat message",
                session_id=request.session_id,
                message_id=message_id,
                response_length=len(response),
                max_tokens=request.max_tokens,
            )

            return ChatResponse(
                timestamp=assistant_message.timestamp,
                session_id=request.session_id,
                message_id=message_id,
                status=ResponseStatus.SUCCESS,
                message=response,
                metadata=metadata,
                request_duration=time.perf_counter() - start_time,
            )
        except asyncio.TimeoutError:
            session_logger.error(
                "Chat processing timed out",
                session_id=request.session_id,
                timeout=CHAT_TIMEOUT_SECONDS,
            )
            return ChatResponse(
                timestamp=datetime.now(timezone.utc),
                session_id=request.session_id,
                message_id=message_id,
                status=ResponseStatus.ERROR,
                message="Chat processing timed out",
                metadata=None,
                request_duration=time.perf_counter() - start_time,
            )
        except ValueError as e:
            session_logger.error(
                "Invalid chat request", session_id=request.session_id, error=str(e)
            )
            return ChatResponse(
                timestamp=datetime.now(timezone.utc),
                session_id=request.session_id,
                message_id=message_id,
                status=ResponseStatus.ERROR,
                message=str(e),
                metadata=None,
                request_duration=time.perf_counter() - start_time,
            )
        except Exception as e:
            session_logger.error(
                "Failed to process chat message",
                session_id=request.session_id,
                error=str(e),
                exc_info=True,
            )
            return ChatResponse(
                timestamp=datetime.now(timezone.utc),
                session_id=request.session_id,
                message_id=message_id,
                status=ResponseStatus.ERROR,
                message=f"Failed to process chat message: {type(e).__name__}",
                metadata=None,
                request_duration=time.perf_counter() - start_time,
            )


async def stream_json_messages(messages: List[Message]) -> AsyncGenerator[bytes, None]:
    """Stream JSON messages to reduce memory usage for large exports.

    Formats each message as a JSON object with sender, timestamp, and message fields.
    Yields chunks of the JSON array to avoid loading the entire dataset into memory.

    Args:
        messages: List of Message objects to serialize to JSON.

    Yields:
        bytes: Chunks of UTF-8 encoded JSON data.

    Raises:
        ValueError: If JSON serialization fails for any message.
    """
    yield b"["
    for i, message in enumerate(messages):
        # Only include sender, timestamp, and message as requested
        message_dict: Dict = {
            "sender": message.role,
            "timestamp": message.timestamp.isoformat(),
            "message": message.content,
        }
        # Metadata is intentionally excluded as per user request
        try:
            # Use the custom JSON encoder to handle Metadata objects
            chunk: str = json.dumps(
                message_dict, ensure_ascii=False, cls=CustomJSONEncoder
            )
            if i > 0:
                chunk = f",{chunk}"
            yield chunk.encode("utf-8")
            await asyncio.sleep(0)  # Yield control to event loop
        except TypeError as e:
            logger.error(
                "JSON serialization failed", message_id=message.message_id, error=str(e)
            )
            raise ValueError(f"Failed to serialize message to JSON: {str(e)}") from e
    yield b"]"


async def stream_text_messages(messages: List[Message]) -> AsyncGenerator[bytes, None]:
    """Stream text messages to reduce memory usage for large exports.

    Formats each message in the format: [timestamp] Role: Message
    with double line breaks between messages for readability.

    Args:
        messages: List of Message objects to format as text.

    Yields:
        bytes: Chunks of UTF-8 encoded text data.
    """
    for message in messages:
        timestamp = message.timestamp.strftime("%Y-%m-%d %H:%M:%S")
        sender = message.role
        content = message.content

        # Format: [timestamp] Role: Message
        line = f"[{timestamp}] {sender.capitalize()}: {content}\n\n"
        yield line.encode("utf-8")
        await asyncio.sleep(0)  # Yield control to event loop


async def stream_csv_messages(messages: List[Message]) -> AsyncGenerator[bytes, None]:
    """Stream CSV messages to reduce memory usage for large exports.

    Creates a CSV file with columns: sender, timestamp, message.
    Properly escapes and quotes message content according to CSV standards.

    Args:
        messages: List of Message objects to format as CSV rows.

    Yields:
        bytes: Chunks of UTF-8 encoded CSV data, starting with a header row.
    """
    # CSV header
    yield b"sender,timestamp,message\n"

    for message in messages:
        sender = message.role
        timestamp = message.timestamp.isoformat()
        # Escape quotes in the message content and wrap in quotes
        escaped_content = message.content.replace('"', '""')
        content = f'"{escaped_content}"'

        line = f"{sender},{timestamp},{content}\n"
        yield line.encode("utf-8")
        await asyncio.sleep(0)  # Yield control to event loop


# The specific OPTIONS handler for export_chat has been removed
# We're now using the global OPTIONS handler defined earlier in the file
# This avoids conflicts and ensures consistent handling of all OPTIONS requests


@app.get("/export_chat/{session_id}")
async def export_chat(session_id: str, format: str = "json") -> StreamingResponse:
    """Export chat messages for a session as a downloadable file in the specified format.

    This endpoint provides a streaming response to efficiently handle large chat histories
    without loading the entire conversation into memory. It supports three export formats:

    Supported formats:
    - json: A JSON array of objects with sender, timestamp, and message fields
    - text: Plain text format with [timestamp] Sender: Message format and double line breaks
    - csv: CSV format with sender, timestamp, and message columns (message content is properly escaped)

    The response includes appropriate Content-Disposition headers for browser download
    and CORS headers to ensure cross-origin compatibility.

    Args:
        session_id: Unique session identifier (UUID).
        format: Export format (json, text, or csv). Defaults to json.

    Returns:
        StreamingResponse: Streaming response with appropriate Content-Type and
                          Content-Disposition headers for file download.

    Raises:
        HTTPException: If session_id is invalid, session not found, or no messages exist.
                      Also if an unsupported format is requested.
    """
    start_time: float = time.perf_counter()
    validate_uuid(session_id)
    session_logger = logger.bind(session_id=session_id)

    # Check session existence
    session_lock: Optional[asyncio.Lock] = None
    async with global_sessions_lock:
        if session_id not in sessions:
            session_logger.error("Session not found", session_id=session_id)
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Session not found",
            )
        session_lock = per_session_locks[session_id]

    # Export messages
    async with session_lock:
        session: Session = sessions[session_id]
        if not session.messages:
            session_logger.info("No messages to export", session_id=session_id)
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="No messages found for this session",
            )

        # Normalize format to lowercase
        format = format.lower()

        # Validate format
        if format not in ["json", "text", "csv"]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unsupported format: {format}. Supported formats are: json, text, csv",
            )

        # Set file extension and media type based on format
        file_extension = "." + format
        if format == "text":
            file_extension = ".txt"
            media_type = TEXT_MEDIA_TYPE
        elif format == "csv":
            media_type = CSV_MEDIA_TYPE
        else:  # json
            media_type = JSON_MEDIA_TYPE

        filename: str = generate_filename(session_id, file_extension)
        session_logger.info(
            "Exporting chat messages",
            session_id=session_id,
            format=format,
            message_count=len(session.messages),
            filename=filename,
        )

        try:
            # Select the appropriate streaming function based on format
            if format == "text":
                content_stream = stream_text_messages(session.messages)
            elif format == "csv":
                content_stream = stream_csv_messages(session.messages)
            else:  # json
                content_stream = stream_json_messages(session.messages)

            # Add CORS headers to the response
            headers = {
                "Content-Disposition": f'attachment; filename="{filename}"',
                "Cache-Control": "no-cache",
                "Access-Control-Allow-Origin": "*",  # Allow all origins to match middleware
                "Access-Control-Allow-Methods": "*",  # Allow all methods
                "Access-Control-Allow-Headers": "*",  # Allow all headers
            }

            return StreamingResponse(
                content=content_stream,
                media_type=media_type,
                headers=headers,
            )
        except ValueError as e:
            session_logger.error(
                "Failed to export chat messages",
                session_id=session_id,
                error=str(e),
                exc_info=True,
            )
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to export chat messages: {str(e)}",
            )
        finally:
            session_logger.info(
                "Completed export request",
                session_id=session_id,
                request_duration=time.perf_counter() - start_time,
            )


@app.get("/get_messages/{session_id}")
async def get_messages(session_id: str) -> List[Dict[str, Any]]:
    """Get all messages for a session in a format suitable for the frontend.

    This endpoint retrieves the complete chat history for a specific session,
    including user and assistant messages with their timestamps and metadata.
    It's used by the frontend to display the chat interface when a user returns
    to a previous session.

    The messages are returned in chronological order and include:
    - message_id: Unique identifier for the message
    - role: Either 'user' or 'assistant'
    - content: The actual message text
    - timestamp: ISO-formatted timestamp
    - metadata: Optional metadata from assistant messages (may include tool usage info)

    Args:
        session_id: Unique session identifier (UUID).

    Returns:
        List[Dict[str, Any]]: List of message objects with all necessary fields for display.

    Raises:
        HTTPException: If session_id is invalid or session not found (404).
    """
    validate_uuid(session_id)
    session_logger = logger.bind(session_id=session_id)  # Consistent variable naming

    # Check session existence
    session_lock: Optional[asyncio.Lock] = None
    async with global_sessions_lock:
        if session_id not in sessions:
            session_logger.error("Session not found", session_id=session_id)
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Session not found",
            )
        session_lock = per_session_locks[session_id]

    # Get messages
    async with session_lock:
        session: Session = sessions[session_id]

        # Convert messages to a format suitable for the frontend
        messages = [
            {
                "message_id": msg.message_id,
                "role": msg.role,
                "content": msg.content,
                "timestamp": msg.timestamp.isoformat(),
                "metadata": msg.metadata,
            }
            for msg in session.messages
        ]

        session_logger.info(
            "Retrieved messages for session",
            session_id=session_id,
            message_count=len(messages),
        )

        return messages


@app.post("/session/{session_id}/delete", response_model=DeleteSessionResponse)  # Endpoint for session deletion via beacon
async def delete_session(session_id: str) -> DeleteSessionResponse:
    """Delete a session, its agent, and associated DuckDB file.

    Args:
        session_id: Unique session identifier.

    Returns:
        DeleteSessionResponse: Confirmation of deletion with request duration.

    Raises:
        HTTPException: If session_id is invalid or session not found.
    """
    start_time: float = time.perf_counter()
    validate_uuid(session_id)
    session_logger = logger.bind(session_id=session_id)

    # Check session existence
    session_lock: Optional[asyncio.Lock] = None
    async with global_sessions_lock:
        if session_id not in sessions:
            session_logger.error("Session not found", session_id=session_id)
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Session not found",
            )
        session_lock = per_session_locks[session_id]

    # Delete session data
    async with session_lock:
        db_path: str = os.path.join(
            STORAGE_DIR, f"{DUCKDB_FILE_PREFIX}{session_id}.duckdb"
        )
        try:
            # Remove agent
            if session_id in agents:
                del agents[session_id]
                session_logger.info("Deleted TelemetryAgent", session_id=session_id)

            # Remove DuckDB file
            if os.path.exists(db_path):
                await asyncio.to_thread(os.unlink, db_path)
                session_logger.info("Deleted DuckDB file", db_path=db_path)

            # Remove session and lock
            del sessions[session_id]
            del per_session_locks[session_id]
            session_logger.info(
                "Deleted session",
                session_id=session_id,
                db_path=db_path,
            )

            return DeleteSessionResponse(
                timestamp=datetime.now(timezone.utc),
                session_id=session_id,
                status=ResponseStatus.SUCCESS,
                message="Session deleted successfully",
                request_duration=time.perf_counter() - start_time,
            )

        except OSError as e:
            session_logger.error(
                "Failed to delete DuckDB file",
                db_path=db_path,
                error=str(e),
                exc_info=True,
            )
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to delete session data: {str(e)}",
            )
        except Exception as e:
            session_logger.error(
                "Failed to delete session",
                session_id=session_id,
                error=str(e),
                exc_info=True,
            )
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to delete session: {str(e)}",
            )
