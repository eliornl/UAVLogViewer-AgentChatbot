"""FastAPI application for uploading, parsing, and analyzing flight log files.

Supports uploading .bin/.tlog files, storing telemetry in DuckDB, and managing chat sessions.
"""

import os
import uuid
import enum
import asyncio
import duckdb
import structlog
import pandas as pd
import re
import tempfile
import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional, Set, Union, Any, Tuple
from fastapi import FastAPI, UploadFile, File, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from pymavlink import mavutil
import mimetypes

# Configure structlog for structured logging with session_id context
structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.stdlib.add_log_level,
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)
logger: structlog.stdlib.BoundLogger = structlog.get_logger()

# Load environment variables from .env file for configuration
load_dotenv()

# Environment variables:
# - ALLOWED_ORIGINS: Comma-separated CORS origins (e.g., "http://localhost:8080")
# - ALLOWED_METHODS: Allowed HTTP methods (e.g., "GET,POST,DELETE,OPTIONS")
# - ALLOWED_HEADERS: Allowed headers (e.g., "Content-Type,Authorization")
# - ALLOW_CREDENTIALS: Enable credentials (true/false)
# - MAX_FILE_SIZE_MB: Max upload size in MB (e.g., 100)
# - STORAGE_DIR: Directory for DuckDB files (e.g., "./storage")
# - CHUNK_SIZE: File read chunk size in bytes (e.g., 1048576)
# - LOG_LEVEL: Logging level (e.g., "INFO", "DEBUG")
# - MAX_SESSION_ATTEMPTS: Max session ID generation attempts (e.g., 5)
# - DUCKDB_FILE_PREFIX: Prefix for DuckDB files (e.g., "" or "telemetry_")
# - ALLOWED_MIME_TYPES: Comma-separated MIME types (e.g., "application/octet-stream,text/plain")
# - SESSION_TTL: Session expiration time in seconds (e.g., 86400 for 24 hours)

# Initialize FastAPI application
app: FastAPI = FastAPI()

# Configure application settings from environment variables
ALLOWED_ORIGINS: List[str] = os.getenv("ALLOWED_ORIGINS", "http://localhost:8080").split(",")
ALLOWED_METHODS: List[str] = os.getenv("ALLOWED_METHODS", "GET,POST,DELETE,OPTIONS").split(",")
ALLOWED_HEADERS: List[str] = os.getenv("ALLOWED_HEADERS", "Content-Type,Authorization").split(",")
ALLOW_CREDENTIALS: bool = os.getenv("ALLOW_CREDENTIALS", "false").lower() == "true"
MAX_FILE_SIZE: int = int(float(os.getenv("MAX_FILE_SIZE_MB", "100")) * 1024 * 1024)
STORAGE_DIR: str = os.path.abspath(os.getenv("STORAGE_DIR", "./storage"))
CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "1048576"))
LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
MAX_SESSION_ATTEMPTS: int = int(os.getenv("MAX_SESSION_ATTEMPTS", "5"))
DUCKDB_FILE_PREFIX: str = os.getenv("DUCKDB_FILE_PREFIX", "")
ALLOWED_MIME_TYPES: Set[str] = set(os.getenv("ALLOWED_MIME_TYPES", "application/octet-stream,text/plain").split(","))
SESSION_TTL: int = int(os.getenv("SESSION_TTL", "86400"))

# Configure logging level
logging.getLogger().setLevel(LOG_LEVEL)

# Validate STORAGE_DIR
if ".." in os.path.relpath(STORAGE_DIR, os.getcwd()):
    raise ValueError("STORAGE_DIR cannot contain parent directory references")
if not os.path.exists(STORAGE_DIR):
    os.makedirs(STORAGE_DIR, exist_ok=True)
if not os.path.isdir(STORAGE_DIR):
    raise ValueError(f"STORAGE_DIR '{STORAGE_DIR}' is not a directory")
if not os.access(STORAGE_DIR, os.W_OK):
    raise ValueError(f"STORAGE_DIR '{STORAGE_DIR}' is not writable")

# Validate CORS settings for security
if ALLOW_CREDENTIALS and "*" in ALLOWED_ORIGINS:
    raise ValueError("Credentials cannot be used with wildcard origins")
if not any(ALLOWED_ORIGINS) or all(not origin.strip() for origin in ALLOWED_ORIGINS):
    raise ValueError("ALLOWED_ORIGINS must specify valid origins")

# Apply CORS middleware to enable cross-origin requests from frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=ALLOW_CREDENTIALS,
    allow_methods=ALLOWED_METHODS,
    allow_headers=ALLOWED_HEADERS,
)

# Lock for thread-safe access to sessions
sessions_lock: asyncio.Lock = asyncio.Lock()

# Relevant telemetry message types for parsing
RELEVANT_TELEMETRY_MESSAGES: List[str] = [
    'ATTITUDE',
    'GLOBAL_POSITION_INT',
    'VFR_HUD',
    'SYS_STATUS',
    'STATUSTEXT',
    'RC_CHANNELS',
    'GPS_RAW_INT',
    'BATTERY_STATUS',
    'EKF_STATUS_REPORT'
]

# Enum for API response status
class Status(str, enum.Enum):
    """Status codes for API responses."""
    SUCCESS = "SUCCESS"
    ERROR = "ERROR"

# Pydantic models for request and response validation
class Message(BaseModel):
    """Chat message details for a session."""
    message_id: str
    user_message: str
    chat_response: str
    timestamp: datetime

class Session(BaseModel):
    """Session data for a flight log, including messages."""
    session_id: str
    created_at: datetime
    file_name: str
    file_size: int
    messages: List[Message] = Field(default_factory=list)

class ChatRequest(BaseModel):
    """Payload for sending chat messages to the FlightAgent."""
    session_id: str
    message: str

class UploadResponse(BaseModel):
    """Response for file upload and session creation."""
    timestamp: datetime
    file_name: str
    file_size: int
    session_id: str
    status: Status
    message: str

class DeleteSessionResponse(BaseModel):
    """Response for session deletion."""
    timestamp: datetime
    session_id: str
    status: Status
    message: str

class ExportChatResponse(BaseModel):
    """Response for exporting chat history."""
    timestamp: datetime
    session_id: str
    status: Status
    messages: List[Message]

class HealthResponse(BaseModel):
    """Response for health check."""
    timestamp: datetime
    status: Status
    message: str
    details: Dict[str, bool]

# In-memory session storage: mapping session_id to Session
sessions: Dict[str, Session] = {}

# Utility functions
def validate_uuid(session_id: str) -> None:
    """Validate that session_id is a valid UUID.

    Args:
        session_id: Session ID to validate.

    Raises:
        HTTPException: If session_id is not a valid UUID.
    """
    try:
        uuid_obj = uuid.UUID(session_id)
        if str(uuid_obj) != session_id:
            raise ValueError
    except ValueError:
        logger.error("Invalid UUID format", session_id=session_id)
        raise HTTPException(status_code=400, detail="Invalid session_id format; must be a valid UUID")

async def cleanup_temp_file(tmp_file_path: str, logger: structlog.stdlib.BoundLogger, file_name: str = "") -> None:
    """Clean up a temporary file and log the result.

    Args:
        tmp_file_path: Path to the temporary file.
        logger: Logger instance for logging cleanup status.
        file_name: Optional filename for logging context.
    """
    try:
        os.unlink(tmp_file_path)
        logger.info("Cleaned up temporary file", file_name=file_name, tmp_file_path=tmp_file_path)
    except OSError as e:
        logger.warning("Failed to clean up temporary file", file_name=file_name, tmp_file_path=tmp_file_path, error=str(e))

# Placeholder class for telemetry parsing and DuckDB storage
class TelemetryProcessor:
    """Processes flight log files and saves telemetry data to DuckDB."""
    
    @staticmethod
    def map_dtype(dtype: Any) -> str:
        """Map pandas dtype to DuckDB column type.

        Args:
            dtype: Pandas data type to map.

        Returns:
            str: Corresponding DuckDB column type (e.g., 'BIGINT', 'VARCHAR').
        """
        if pd.api.types.is_integer_dtype(dtype):
            return "BIGINT"
        elif pd.api.types.is_float_dtype(dtype):
            return "DOUBLE"
        elif pd.api.types.is_bool_dtype(dtype):
            return "BOOLEAN"
        elif pd.api.types.is_datetime64_any_dtype(dtype):
            return "TIMESTAMP"
        elif pd.api.types.is_timedelta64_dtype(dtype):
            return "INTERVAL"
        else:
            return "VARCHAR"
    
    @staticmethod
    async def parse_and_save(file_path: str, session_id: str, sanitized_filename: str) -> None:
        """Parse flight log file and save telemetry data to DuckDB.

        Creates per-session DuckDB file in STORAGE_DIR with tables for each telemetry message type.

        Args:
            file_path: Path to the temporary file containing the flight log.
            session_id: Unique session ID for associating telemetry data.
            sanitized_filename: Sanitized filename for logging.

        Raises:
            ValueError: If file parsing fails (e.g., invalid telemetry file).
            OSError: If file read fails.
            duckdb.Error: If DuckDB operation fails.

        Returns:
            None
        """
        try:
            # Parse .bin file using pymavlink
            try:
                mav = mavutil.mavlogfile(file_path, zero_time_base=True)
            except Exception as e:
                logger.error("Failed to initialize MAVLink file", file_name=sanitized_filename, session_id=session_id, error=str(e))
                raise ValueError(f"Invalid telemetry file: {str(e)}")
            
            # Initialize dictionaries for each message type
            telemetry_data: Dict[str, List[Dict[str, Any]]] = {msg_type: [] for msg_type in RELEVANT_TELEMETRY_MESSAGES}
            
            # Parse messages
            while True:
                try:
                    msg = mav.recv_msg()
                    if msg is None:
                        break
                    msg_type = msg.get_type()
                    if msg_type in RELEVANT_TELEMETRY_MESSAGES:
                        msg_dict = msg.to_dict()
                        # Add timestamp (MAVLink timestamp or current time)
                        msg_dict['timestamp'] = getattr(msg, '_timestamp', datetime.now(timezone.utc).timestamp())
                        telemetry_data[msg_type].append(msg_dict)
                except Exception as e:
                    logger.warning("Error parsing message", file_name=sanitized_filename, session_id=session_id, error=str(e))
                    continue
            
            # Create storage directory and connect to per-session DuckDB
            os.makedirs(STORAGE_DIR, exist_ok=True)
            db_path = os.path.join(STORAGE_DIR, f"{DUCKDB_FILE_PREFIX}{session_id}.duckdb")
            with duckdb.connect(db_path) as conn:
                # Store data in DuckDB tables
                for msg_type, messages in telemetry_data.items():
                    if not messages:
                        logger.info("No messages found for type", msg_type=msg_type, file_name=sanitized_filename, session_id=session_id)
                        continue
                    
                    # Convert to pandas DataFrame
                    messages: List[Dict[str, Any]]
                    df: pd.DataFrame = pd.DataFrame(messages)
                    
                    # Create sanitized table name
                    table_name = f"telemetry_{re.sub(r'[^a-z0-9_]', '_', msg_type.lower())}"
                    
                    # Prepare columns, excluding mavpackettype
                    if 'mavpackettype' in df.columns:
                        df = df.drop(columns=['mavpackettype'])
                    columns = [f"{col} {TelemetryProcessor.map_dtype(df[col].dtype)}" for col in df.columns]
                    
                    # Create table
                    create_table_query = f"""
                    CREATE TABLE {table_name} (
                        {', '.join(columns)}
                    )
                    """
                    conn.execute(create_table_query)
                    
                    # Insert data into DuckDB
                    conn.execute(f"INSERT INTO {table_name} SELECT * FROM df", {'df': df})
                    
                    logger.info("Saved telemetry data", table_name=table_name, row_count=len(df), file_name=sanitized_filename, session_id=session_id)
            
        except OSError as e:
            logger.error("File read error", file_name=sanitized_filename, session_id=session_id, error=str(e))
            raise
        except duckdb.Error as e:
            logger.error("DuckDB error", file_name=sanitized_filename, session_id=session_id, error=str(e))
            raise
        except ValueError as e:
            logger.error("Parsing failed", file_name=sanitized_filename, session_id=session_id, error=str(e))
            raise
        except Exception as e:
            logger.exception("Unexpected error", file_name=sanitized_filename, session_id=session_id, error_type=type(e).__name__)
            raise HTTPException(status_code=500, detail="Internal server error")

# API Endpoints
@app.get("/health")
async def health_check() -> HealthResponse:
    """Check the health of the API, including storage and DuckDB connectivity.

    Returns:
        HealthResponse: Response with timestamp, status, message, and check details (storage_writable, duckdb_writable, duckdb_connection).
    """
    timestamp = datetime.now(timezone.utc)
    details: Dict[str, bool] = {}
    status = Status.SUCCESS
    message = "API is healthy"

    # Check storage directory accessibility
    try:
        if not os.path.exists(STORAGE_DIR):
            await asyncio.to_thread(os.makedirs, STORAGE_DIR, exist_ok=True)
        storage_writable = await asyncio.to_thread(os.access, STORAGE_DIR, os.W_OK)
        details["storage_writable"] = storage_writable
        if not storage_writable:
            logger.error("Storage directory is not writable", storage_dir=STORAGE_DIR)
            status = Status.ERROR
            message = "Health check failed"
    except Exception as e:
        logger.error("Storage check failed", storage_dir=STORAGE_DIR, error=str(e))
        details["storage_writable"] = False
        status = Status.ERROR
        message = "Health check failed"

    # Check DuckDB connectivity and file writability
    with tempfile.NamedTemporaryFile(suffix=".duckdb", dir=STORAGE_DIR, delete=False) as temp_file:
        temp_db_path = temp_file.name
        try:
            # Validate DuckDB file writability
            duckdb_writable = await asyncio.to_thread(os.access, temp_db_path, os.W_OK)
            details["duckdb_writable"] = duckdb_writable
            if not duckdb_writable:
                logger.error("DuckDB file is not writable", db_path=temp_db_path)
                status = Status.ERROR
                message = "Health check failed"

            # Only attempt connection if file is writable
            if duckdb_writable:
                with duckdb.connect(temp_db_path) as conn:
                    conn.execute("SELECT 1")
                details["duckdb_connection"] = True
            else:
                details["duckdb_connection"] = False
                status = Status.ERROR
                message = "Health check failed"
        except duckdb.Error as e:
            logger.error("DuckDB connection failed", db_path=temp_db_path, error=str(e))
            details["duckdb_connection"] = False
            status = Status.ERROR
            message = "Health check failed"
        finally:
            await cleanup_temp_file(temp_db_path, logger)

    if status == Status.SUCCESS:
        logger.info("Health check passed", details=details)
    else:
        logger.error("Health check failed", details=details)

    return HealthResponse(
        timestamp=timestamp,
        status=status,
        message=message,
        details=details
    )

async def validate_file(file: UploadFile) -> Tuple[str, str, int]:
    """Validate uploaded file and return sanitized filename, extension, and size.

    Args:
        file: Uploaded file to validate.

    Returns:
        Tuple[str, str, int]: Sanitized filename, file extension, and file size.

    Raises:
        HTTPException: If validation fails.
    """
    if not file or not file.filename:
        logger.error("No file provided")
        raise HTTPException(status_code=400, detail="No file provided")
    
    sanitized_filename = "".join(c if c.isalnum() or c in ".-_" else "_" for c in os.path.basename(file.filename))
    allowed_extensions = {".bin", ".tlog"}
    file_ext = os.path.splitext(sanitized_filename)[1].lower()
    if file_ext not in allowed_extensions:
        logger.error("Invalid file extension", file_name=sanitized_filename, ext=file_ext)
        raise HTTPException(status_code=400, detail=f"Invalid file extension. Allowed: {allowed_extensions}")

    mime_type, _ = mimetypes.guess_type(file.filename)
    if mime_type is None or mime_type not in ALLOWED_MIME_TYPES:
        logger.error("Invalid MIME type", file_name=sanitized_filename, mime_type=mime_type)
        raise HTTPException(status_code=400, detail=f"Invalid MIME type. Allowed: {ALLOWED_MIME_TYPES}")

    file_size = file.size
    if file_size is None:
        try:
            file_size = 0
            await file.seek(0)
            async for chunk in file:
                file_size += len(chunk)
            await file.seek(0)
        except OSError as e:
            logger.error("File streaming error", file_name=sanitized_filename, error=str(e))
            raise HTTPException(status_code=400, detail=f"File streaming error: {str(e)}")
    
    if file_size == 0:
        logger.error("Empty file provided", file_name=sanitized_filename)
        raise HTTPException(status_code=400, detail="Empty file provided")
    if file_size > MAX_FILE_SIZE:
        logger.error("File size exceeds maximum", file_name=sanitized_filename, size=file_size, max_size=MAX_FILE_SIZE)
        raise HTTPException(status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, detail=f"File size exceeds maximum {MAX_FILE_SIZE/1024/1024}MB")

    return sanitized_filename, file_ext, file_size

async def process_file(file: UploadFile, tmp_file_path: str) -> None:
    """Write uploaded file to temporary file using chunked reading.

    Args:
        file: Uploaded file to process.
        tmp_file_path: Path to temporary file.

    Raises:
        OSError: If file writing fails.
    """
    with open(tmp_file_path, "wb") as tmp_file:
        await file.seek(0)
        while chunk := await file.read(CHUNK_SIZE):
            tmp_file.write(chunk)

@app.post("/upload", response_model=UploadResponse, status_code=status.HTTP_201_CREATED)
async def upload_file(file: UploadFile = File(...)) -> UploadResponse:
    """Upload a flight log file, parse it, and create a new session with a UUID.

    Args:
        file: Flight log file uploaded by the client (.bin or .tlog).

    Returns:
        UploadResponse: Response with timestamp, file_name, file_size, session_id, status, and message.

    Raises:
        HTTPException: If file is invalid, exceeds size limit, or processing fails.
    """
    sanitized_filename, file_ext, file_size = await validate_file(file)

    with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp_file:
        tmp_file_path = tmp_file.name
        try:
            await process_file(file, tmp_file_path)
            logger.info("Created temporary file", file_name=sanitized_filename, tmp_file_path=tmp_file_path, file_size=file_size)
            
            async with sessions_lock:
                session_id: str = str(uuid.uuid4())
                attempt = 0
                while session_id in sessions and attempt < MAX_SESSION_ATTEMPTS:
                    logger.warning("Session ID already exists, generating new ID", session_id=session_id, attempt=attempt)
                    session_id = str(uuid.uuid4())
                    attempt += 1
                if session_id in sessions:
                    logger.error("Failed to generate unique session ID", session_id=session_id, max_attempts=MAX_SESSION_ATTEMPTS)
                    raise HTTPException(status_code=500, detail="Failed to generate unique session ID")

                db_path = os.path.join(STORAGE_DIR, f"{DUCKDB_FILE_PREFIX}{session_id}.duckdb")
                if os.path.exists(db_path):
                    logger.error("DuckDB file already exists", session_id=session_id, db_path=db_path)
                    raise HTTPException(status_code=409, detail="Session ID conflict")

                logger = logger.bind(session_id=session_id)

                try:
                    logger.info("Starting file parsing", file_name=sanitized_filename, session_id=session_id, file_size=file_size)
                    await TelemetryProcessor.parse_and_save(tmp_file_path, session_id, sanitized_filename)
                    
                    session = Session(
                        session_id=session_id,
                        created_at=datetime.now(timezone.utc),
                        file_name=sanitized_filename,
                        file_size=file_size,
                        messages=[]
                    )
                    sessions[session_id] = session
                    logger.info("Created session", file_name=sanitized_filename, file_size=file_size)
                    
                    return UploadResponse(
                        timestamp=session.created_at,
                        file_name=sanitized_filename,
                        file_size=file_size,
                        session_id=session_id,
                        status=Status.SUCCESS,
                        message="File uploaded and processed successfully"
                    )
                except ValueError as e:
                    logger.error("Parsing failed, session not created", file_name=sanitized_filename, session_id=session_id, error=str(e))
                    raise HTTPException(status_code=400, detail=f"File parsing failed: {str(e)}")
                except OSError as e:
                    logger.error("File read error, session not created", file_name=sanitized_filename, session_id=session_id, error=str(e))
                    raise HTTPException(status_code=400, detail=f"File read error: {str(e)}")
                except duckdb.Error as e:
                    logger.error("DuckDB error, session not created", file_name=sanitized_filename, session_id=session_id, error=str(e))
                    raise HTTPException(status_code=500, detail="Database error")
                except Exception as e:
                    logger.exception("Unexpected error, session not created", file_name=sanitized_filename, session_id=session_id, error_type=type(e).__name__)
                    raise HTTPException(status_code=500, detail="Internal server error")
        finally:
            await cleanup_temp_file(tmp_file_path, logger, sanitized_filename)

@app.post("/chat")
async def chat_message(request: ChatRequest) -> Dict[str, Union[str, dict]]:
    """Send a user message to the FlightAgent for a session (to be implemented)."""
    raise HTTPException(status_code=501, detail="Chat functionality not implemented")

@app.delete("/session/{session_id}")
async def delete_session(session_id: str) -> DeleteSessionResponse:
    """Delete a session and free its resources, including associated DuckDB files.

    Args:
        session_id: Unique UUID of the session to delete.

    Returns:
        DeleteSessionResponse: Response with timestamp, session_id, status, and message.

    Raises:
        HTTPException: If session_id is invalid or not found.
    """
    validate_uuid(session_id)
    logger = logger.bind(session_id=session_id)
    
    async with sessions_lock:
        if session_id not in sessions:
            logger.error("Session not found")
            raise HTTPException(status_code=404, detail="Session not found")
        
        del sessions[session_id]
        logger.info("Deleted session from memory")
        
        db_path = os.path.join(STORAGE_DIR, f"{DUCKDB_FILE_PREFIX}{session_id}.duckdb")
        if not os.path.exists(db_path):
            logger.info("No DuckDB file found to delete")
            return DeleteSessionResponse(
                timestamp=datetime.now(timezone.utc),
                session_id=session_id,
                status=Status.SUCCESS,
                message=f"Session {session_id} deleted"
            )
        try:
            await asyncio.to_thread(os.remove, db_path)
            logger.info("Deleted DuckDB file", db_file=db_path)
        except OSError as e:
            logger.warning("Failed to delete DuckDB file", db_file=db_path, error=str(e))
        
        return DeleteSessionResponse(
            timestamp=datetime.now(timezone.utc),
            session_id=session_id,
            status=Status.SUCCESS,
            message=f"Session {session_id} deleted"
        )

@app.get("/session/{session_id}/export-chat")
async def export_chat(session_id: str) -> ExportChatResponse:
    """Export chat history for a session as JSON.

    Args:
        session_id: Unique UUID of the session.

    Returns:
        ExportChatResponse: Response with timestamp, session_id, status, and messages.

    Raises:
        HTTPException: If session_id is invalid or not found.
    """
    validate_uuid(session_id)
    logger = logger.bind(session_id=session_id)
    
    async with sessions_lock:
        if session_id not in sessions:
            logger.error("Session not found")
            raise HTTPException(status_code=404, detail="Session not found")
        
        messages = sessions[session_id].messages
        logger.info("Exported chat history", message_count=len(messages))
        
        return ExportChatResponse(
            timestamp=datetime.now(timezone.utc),
            session_id=session_id,
            status=Status.SUCCESS,
            messages=messages
        )