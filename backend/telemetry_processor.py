"""Telemetry processing module for UAV Log Viewer application.

This module handles the parsing and processing of MAVLink telemetry log files,
converting binary data into structured tables stored in DuckDB for efficient
querying and analysis. It supports asynchronous processing with timeouts and
retries to handle large log files reliably.
"""

from typing import Dict, List, Any, Optional, Tuple, Union, Callable
import os
import re
import asyncio
import duckdb
import numpy as np
import pandas as pd
import structlog
from pymavlink import mavutil
from tenacity import retry, stop_after_attempt, wait_exponential
from backend.models import Session, SessionStatus

# Configure logger
logger: structlog.stdlib.BoundLogger = structlog.get_logger(__name__)


class TelemetryProcessor:
    """Processes MAVLink telemetry log files and saves data to DuckDB asynchronously.

    This class handles the complete workflow of processing UAV telemetry data:
    1. Parsing binary MAVLink log files (.bin)
    2. Extracting relevant message types (attitude, position, status, etc.)
    3. Converting messages to pandas DataFrames with appropriate data types
    4. Storing data in session-specific DuckDB tables for efficient querying
    5. Handling errors, timeouts, and retries for robust processing
    6. Creating indexes on time_boot_ms columns when requested (separate operation)

    The process is fully asynchronous and designed to handle large log files with
    configurable timeouts. Each message type is stored in a separate table with
    appropriate column types derived from the data.

    Note: The `timestamp` column is included in tables only if `_timestamp` is present in the
    MAVLink messages; otherwise, it is omitted or may contain NULL values.
    """

    # List of telemetry message types to process
    TELEMETRY_MESSAGE_TYPES: List[str] = [
        "ATTITUDE",
        "GLOBAL_POSITION_INT",
        "VFR_HUD",
        "SYS_STATUS",
        "STATUSTEXT",
        "RC_CHANNELS",
        "GPS_RAW_INT",
        "BATTERY_STATUS",
        "EKF_STATUS_REPORT",
    ]

    # Timeouts for I/O operations
    PARSE_TIMEOUT_SECONDS: float = 300.0  # 5 minutes for large log files
    SAVE_TIMEOUT_SECONDS: float = 120.0  # 2 minutes for saving all tables

    # Retry configuration
    MAX_RETRY_ATTEMPTS: int = 3
    RETRY_MULTIPLIER: float = 1.0
    RETRY_MIN_SECONDS: float = 1.0
    RETRY_MAX_SECONDS: float = 5.0

    # Error messages
    INVALID_FILE_PATH_ERROR: str = "Invalid file path: {}"
    INVALID_TELEMETRY_FILE_ERROR: str = "Invalid telemetry file: {}"
    INVALID_DB_PATH_ERROR: str = "Invalid database path: {}"
    DATAFRAME_CREATION_ERROR: str = "Failed to create DataFrame for {}: {}"
    TABLE_CREATION_ERROR: str = "Failed to create table {} in {}: {}"
    SAVE_TIMEOUT_ERROR: str = "DuckDB save timed out after {} seconds"
    OPERATION_TIMEOUT_ERROR: str = "Operation timed out: {}"
    FILE_DB_ERROR: str = "File or database operation failed: {}"
    PARSING_ERROR: str = "Telemetry parsing failed: {}"
    UNEXPECTED_ERROR: str = "Unexpected telemetry processing error: {}"
    INDEX_CREATION_ERROR: str = "Failed to create time_boot_ms index in {}: {}"
    INDEX_TIMEOUT_ERROR: str = "Index creation timed out after {} seconds"

    @staticmethod
    def map_dtype(dtype: pd.api.extensions.ExtensionDtype | np.dtype | type) -> str:
        """Map pandas data types to DuckDB column types.

        This method converts pandas DataFrame column data types to appropriate
        DuckDB column types for table creation. It handles various numeric types,
        datetime types, and defaults to VARCHAR for text and other types.

        Args:
            dtype: Pandas data type to map (can be numpy dtype, pandas extension dtype, or Python type)

        Returns:
            str: Corresponding DuckDB column type (e.g., 'BIGINT', 'DOUBLE', 'VARCHAR')

        Examples:
            >>> TelemetryProcessor.map_dtype(np.dtype('int64'))
            'BIGINT'
            >>> TelemetryProcessor.map_dtype(np.dtype('float64'))
            'DOUBLE'
            >>> TelemetryProcessor.map_dtype(np.dtype('O'))
            'VARCHAR'
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
    @retry(
        stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=5)
    )
    async def parse_and_save(
        file_path: str, session: Session, storage_dir: str, db_prefix: str
    ) -> None:
        """Parse a telemetry log file asynchronously and save data to DuckDB with retries.

        Collects all messages, groups them by telemetry message type, creates one DataFrame
        per type, and saves them to session-specific DuckDB tables in a single connection.
        Updates session.status to READY on success or ERROR on failure, with a corresponding
        status_message.

        Args:
            file_path: Path to the telemetry log file (.bin).
            session: Session object containing session_id, file_name, status, and status_message.
            storage_dir: Directory to store DuckDB files.
            db_prefix: Prefix for DuckDB database filenames.

        Raises:
            ValueError: If the file path is invalid or the file is not a valid telemetry file.
            OSError: If file read operations fail.
            duckdb.IOException: If DuckDB database operations fail.
            asyncio.TimeoutError: If parsing or saving exceeds timeouts.

        """
        session_logger = logger.bind(
            session_id=session.session_id, file_name=session.file_name
        )
        try:
            # Validate file path to prevent path traversal and ensure file exists
            if not os.path.isfile(file_path):
                session.status = SessionStatus.ERROR
                session.status_message = "Invalid file path provided"
                session_logger.error("Invalid file path", file_path=file_path)
                raise ValueError(
                    TelemetryProcessor.INVALID_FILE_PATH_ERROR.format(file_path)
                )

            # Initialize MAVLink log file parser
            def init_mav() -> Any:
                return mavutil.mavlogfile(file_path)

            try:
                mav = await asyncio.wait_for(
                    asyncio.to_thread(init_mav),
                    timeout=TelemetryProcessor.PARSE_TIMEOUT_SECONDS,
                )
            except Exception as e:
                session.status = SessionStatus.ERROR
                session.status_message = (
                    TelemetryProcessor.INVALID_TELEMETRY_FILE_ERROR.format(str(e))
                )
                session_logger.error("Failed to initialize MAVLink file", error=str(e))
                raise ValueError(
                    TelemetryProcessor.INVALID_TELEMETRY_FILE_ERROR.format(str(e))
                ) from e

            # Initialize data storage for each message type
            telemetry_data: Dict[str, List[Dict[str, Any]]] = {
                msg_type: [] for msg_type in TelemetryProcessor.TELEMETRY_MESSAGE_TYPES
            }

            # Parse all messages from the log file
            def parse_messages() -> None:
                while True:
                    msg = mav.recv_msg()
                    if msg is None:
                        break
                    msg_type = msg.get_type()
                    if msg_type in TelemetryProcessor.TELEMETRY_MESSAGE_TYPES:
                        msg_dict = msg.to_dict()
                        telemetry_data[msg_type].append(msg_dict)

            await asyncio.wait_for(
                asyncio.to_thread(parse_messages),
                timeout=TelemetryProcessor.PARSE_TIMEOUT_SECONDS,
            )

            # Create and save DataFrames for each message type
            async def save_to_duckdb() -> None:
                # Prepare DuckDB database path
                os.makedirs(storage_dir, exist_ok=True)
                db_path = os.path.join(
                    storage_dir, f"{db_prefix}{session.session_id}.duckdb"
                )
                if ".." in os.path.relpath(db_path, storage_dir):
                    session_logger.error("Invalid database path", db_path=db_path)
                    raise ValueError(
                        TelemetryProcessor.INVALID_DB_PATH_ERROR.format(db_path)
                    )

                # Connect to DuckDB
                with duckdb.connect(db_path) as conn:
                    for msg_type in TelemetryProcessor.TELEMETRY_MESSAGE_TYPES:
                        messages = telemetry_data[msg_type]
                        if not messages:
                            session_logger.debug(
                                "No messages for type", msg_type=msg_type
                            )
                            continue

                        # Create DataFrame
                        try:
                            df = pd.DataFrame(messages)
                        except Exception as e:
                            session_logger.error(
                                "Failed to create DataFrame for message type",
                                msg_type=msg_type,
                                error=str(e),
                            )
                            raise ValueError(
                                TelemetryProcessor.DATAFRAME_CREATION_ERROR.format(
                                    msg_type, str(e)
                                )
                            ) from e

                        # Drop 'mavpackettype' column if present
                        if "mavpackettype" in df.columns:
                            df = df.drop(columns=["mavpackettype"])

                        # Generate sanitized table name
                        table_name = (
                            f"telemetry_{re.sub(r'[^a-z0-9_]', '_', msg_type.lower())}"
                        )

                        # Define table columns based on DataFrame dtypes
                        columns = [
                            f"{col} {TelemetryProcessor.map_dtype(df[col].dtype)}"
                            for col in df.columns
                        ]

                        # Check if the table exists
                        table_exists = (
                            await asyncio.to_thread(
                                lambda: conn.execute(
                                    "SELECT COUNT(*) FROM duckdb_tables() WHERE table_name = ?",
                                    (table_name,),
                                ).fetchone()[0]
                            )
                            > 0
                        )

                        # Create table if it doesn't exist
                        if not table_exists:
                            session_logger.debug(
                                "Creating table", table_name=table_name, db_path=db_path
                            )
                            quoted_table_name = f'"{table_name}"'
                            create_table_query = f"""
                            CREATE TABLE {quoted_table_name} (
                                {', '.join(columns)}
                            )
                            """
                            try:
                                await asyncio.to_thread(lambda: conn.execute(create_table_query))
                            except Exception as e:
                                session_logger.error(
                                    "Failed to create table",
                                    table_name=table_name,
                                    db_path=db_path,
                                    error=str(e),
                                )
                                raise duckdb.IOException(
                                    TelemetryProcessor.TABLE_CREATION_ERROR.format(
                                        table_name, db_path, str(e)
                                    )
                                ) from e
                        else:
                            session_logger.debug(
                                "Table exists, will append data", table_name=table_name
                            )

                        # Insert data into table via temporary view
                        await asyncio.to_thread(lambda: conn.register("df_view", df))
                        await asyncio.to_thread(
                            lambda: conn.execute(
                                f"INSERT INTO {quoted_table_name} SELECT * FROM df_view"
                            )
                        )
                        await asyncio.to_thread(lambda: conn.unregister("df_view"))
                        session_logger.info(
                            "Saved telemetry data",
                            table_name=table_name,
                            msg_type=msg_type,
                            row_count=len(df),
                        )
                        await asyncio.to_thread(lambda: conn.commit())
                        
                        # Note: Indexes for time_boot_ms are created separately via
                        # create_time_boot_ms_indexes() to avoid timeouts during initial data load

            try:
                await asyncio.wait_for(
                    save_to_duckdb(),
                    timeout=TelemetryProcessor.SAVE_TIMEOUT_SECONDS,
                )
            except asyncio.TimeoutError:
                session.status = SessionStatus.ERROR
                session.status_message = TelemetryProcessor.SAVE_TIMEOUT_ERROR.format(
                    TelemetryProcessor.SAVE_TIMEOUT_SECONDS
                )
                session_logger.error(
                    "DuckDB save timed out",
                    timeout=TelemetryProcessor.SAVE_TIMEOUT_SECONDS,
                )
                raise duckdb.IOException(
                    TelemetryProcessor.SAVE_TIMEOUT_ERROR.format(
                        TelemetryProcessor.SAVE_TIMEOUT_SECONDS
                    )
                )

            # Update session status to READY upon successful completion
            session.status = SessionStatus.READY
            session.status_message = "Ready for chat"
            session_logger.info("Session status updated to READY")

        except asyncio.TimeoutError as e:
            session.status = SessionStatus.ERROR
            session.status_message = TelemetryProcessor.OPERATION_TIMEOUT_ERROR.format(
                str(e)
            )
            session_logger.error("Operation timed out", error=str(e))
            raise ValueError(
                TelemetryProcessor.OPERATION_TIMEOUT_ERROR.format(str(e))
            ) from e
        except (OSError, duckdb.IOException) as e:
            session.status = SessionStatus.ERROR
            session.status_message = TelemetryProcessor.FILE_DB_ERROR.format(str(e))
            session_logger.error("File or database error", error=str(e))
            raise OSError(TelemetryProcessor.FILE_DB_ERROR.format(str(e))) from e
        except ValueError as e:
            session_logger.error("Parsing failed", error=str(e))
            raise ValueError(TelemetryProcessor.PARSING_ERROR.format(str(e))) from e
        except Exception as e:
            session.status = SessionStatus.ERROR
            session.status_message = TelemetryProcessor.UNEXPECTED_ERROR.format(str(e))
            session_logger.error(
                "Unexpected error during telemetry processing",
                error_type=type(e).__name__,
                error=str(e),
                exc_info=True,
            )
            raise ValueError(TelemetryProcessor.UNEXPECTED_ERROR.format(str(e))) from e
            
    @staticmethod
    async def create_time_boot_ms_indexes(session: Session, storage_dir: str, db_prefix: str) -> None:
        """Create indexes on time_boot_ms columns for all tables that contain this field.
        
        This method should be called separately from parse_and_save to avoid timeouts during
        initial data loading. It creates an index on the time_boot_ms column for each table
        that contains this field, improving query performance for time-based filtering.
        
        The method is designed to be non-blocking and fault-tolerant:
        - It will not raise exceptions if index creation fails
        - It logs errors but allows the application to continue
        - It uses a single database transaction for efficiency
        - It has a timeout to prevent hanging
        
        Args:
            session: Session object containing session_id and other metadata.
            storage_dir: Directory where DuckDB files are stored.
            db_prefix: Prefix for DuckDB database filenames.
        """
        session_logger = logger.bind(
            session_id=session.session_id, file_name=session.file_name
        )
        try:
            # Prepare DuckDB database path
            db_path = os.path.join(storage_dir, f"{db_prefix}{session.session_id}.duckdb")
            if ".." in os.path.relpath(db_path, storage_dir):
                session_logger.error("Invalid database path", db_path=db_path)
                raise ValueError(TelemetryProcessor.INVALID_DB_PATH_ERROR.format(db_path))
                
            # Define function to create indexes
            async def create_indexes() -> None:
                with duckdb.connect(db_path) as conn:
                    # Get list of all tables with time_boot_ms column in a single query
                    table_columns = await asyncio.to_thread(
                        lambda: conn.execute("""
                            SELECT t.table_name, c.column_name 
                            FROM duckdb_tables() t 
                            JOIN duckdb_columns() c ON t.table_name = c.table_name 
                            WHERE t.table_name LIKE 'telemetry_%' 
                            AND c.column_name = 'time_boot_ms'
                        """).fetchall()
                    )
                    
                    # Create a single transaction for all indexes
                    await asyncio.to_thread(lambda: conn.execute("BEGIN TRANSACTION"))
                    
                    created_count = 0
                    for table_name, _ in table_columns:
                        try:
                            # Create index if it doesn't exist
                            index_name = f"{table_name}_time_boot_ms_idx"
                            await asyncio.to_thread(
                                lambda: conn.execute(
                                    f"CREATE INDEX IF NOT EXISTS \"{index_name}\" ON \"{table_name}\"(time_boot_ms)"
                                )
                            )
                            created_count += 1
                            session_logger.debug(
                                "Created index on time_boot_ms column",
                                table_name=table_name,
                                index_name=index_name
                            )
                        except Exception as e:
                            session_logger.warning(
                                "Failed to create index",
                                table_name=table_name,
                                error=str(e)
                            )
                            # Continue with other tables even if this one fails
                    
                    # Commit all index creations at once
                    await asyncio.to_thread(lambda: conn.execute("COMMIT"))
                    
                    session_logger.info(
                        "Created time_boot_ms indexes",
                        created_count=created_count,
                        total_tables=len(table_columns)
                    )
            
            # Execute index creation with timeout
            await asyncio.wait_for(
                create_indexes(),
                timeout=120.0  # 2 minutes timeout for index creation
            )
            
        except asyncio.TimeoutError as e:
            session_logger.error(
                "Index creation timed out",
                error=str(e),
                timeout_seconds=120.0
            )
            # Don't raise the exception, just log it
            # This allows the application to continue even if index creation fails
            return
        except (OSError, duckdb.IOException) as e:
            session_logger.error(
                "Database error during index creation",
                error=str(e)
            )
            # Don't raise the exception, just log it
            return
        except Exception as e:
            session_logger.error(
                "Unexpected error during index creation",
                error_type=type(e).__name__,
                error=str(e),
                exc_info=True
            )
            # Don't raise the exception, just log it
            return
