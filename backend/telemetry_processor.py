from typing import Dict, List, Any
import os
import re
import asyncio
import duckdb
import pandas as pd
import structlog
from pymavlink import mavutil
from tenacity import async_retry, stop_after_attempt, wait_exponential
from datetime import datetime, timezone
from backend.models import Session, SessionStatus

logger: structlog.stdlib.BoundLogger = structlog.get_logger(__name__)

class TelemetryProcessor:
    """Processes MAVLink telemetry log files and saves data to DuckDB asynchronously.

    Parses binary telemetry logs, extracts relevant messages, and stores
    them in session-specific DuckDB tables for efficient querying.
    """

    # List of telemetry message types to process
    TELEMETRY_MESSAGE_TYPES: List[str] = [
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

    # Batch size for processing messages to optimize memory usage
    BATCH_SIZE: int = 1000

    # Timeouts for I/O operations
    PARSE_TIMEOUT_SECONDS: float = 300.0  # 5 minutes for large log files
    BATCH_SAVE_TIMEOUT_SECONDS: float = 60.0  # 1 minute per batch save

    @staticmethod
    def map_dtype(dtype: Any) -> str:
        """Map pandas data types to DuckDB column types.

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
    @async_retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=5))
    async def parse_and_save(
        file_path: str,
        session: Session,
        storage_dir: str,
        db_prefix: str
    ) -> None:
        """Parse a telemetry log file asynchronously and save data to DuckDB with retries.

        Creates a session-specific DuckDB file with tables for each telemetry
        message type. Updates session.status to READY on success or ERROR on
        failure, with a corresponding status_message.

        Args:
            file_path: Path to the telemetry log file (.bin).
            session: Session object containing session_id, file_name, status, and status_message.
            storage_dir: Directory to store DuckDB files.
            db_prefix: Prefix for DuckDB database filenames.

        Raises:
            ValueError: If the file path is invalid or the file is not a valid telemetry file.
            OSError: If file read operations fail.
            duckdb.IOException: If DuckDB database operations fail.
            asyncio.TimeoutError: If parsing exceeds PARSE_TIMEOUT_SECONDS.
        """
        logger = logger.bind(session_id=session.session_id, file_name=session.file_name)
        try:
            # Validate file path to prevent path traversal and ensure file exists
            if not os.path.isfile(file_path) or ".." in os.path.relpath(file_path):
                session.status = SessionStatus.ERROR
                session.status_message = "Invalid file path provided"
                logger.error("Invalid file path", file_path=file_path)
                raise ValueError(f"Invalid file path: {file_path}")

            # Initialize MAVLink log file parser
            async def init_mav() -> Any:
                return mavutil.mavlogfile(file_path, zero_time_base=True)

            try:
                mav = await asyncio.wait_for(
                    asyncio.to_thread(init_mav),
                    timeout=TelemetryProcessor.PARSE_TIMEOUT_SECONDS
                )
            except Exception as e:
                session.status = SessionStatus.ERROR
                session.status_message = f"Invalid telemetry file: {str(e)}"
                logger.error("Failed to initialize MAVLink file", error=str(e))
                raise ValueError(f"Invalid telemetry file: {str(e)}") from e

            # Initialize data storage for each message type
            telemetry_data: Dict[str, List[Dict[str, Any]]] = {
                msg_type: [] for msg_type in TelemetryProcessor.TELEMETRY_MESSAGE_TYPES
            }

            # Parse messages from the log file
            async def parse_messages() -> None:
                while True:
                    msg = mav.recv_msg()
                    if msg is None:
                        break
                    msg_type = msg.get_type()
                    if msg_type in TelemetryProcessor.TELEMETRY_MESSAGE_TYPES:
                        msg_dict = msg.to_dict()
                        # Add timestamp if not present
                        msg_dict['timestamp'] = getattr(
                            msg, '_timestamp', datetime.now(timezone.utc).timestamp()
                        )
                        telemetry_data[msg_type].append(msg_dict)

                        # Save batch if size limit reached
                        if len(telemetry_data[msg_type]) >= TelemetryProcessor.BATCH_SIZE:
                            await TelemetryProcessor._save_batch(
                                telemetry_data[msg_type],
                                msg_type,
                                session.session_id,
                                storage_dir,
                                db_prefix,
                                logger
                            )
                            telemetry_data[msg_type].clear()

            await asyncio.wait_for(
                asyncio.to_thread(parse_messages),
                timeout=TelemetryProcessor.PARSE_TIMEOUT_SECONDS
            )

            # Save any remaining messages
            for msg_type, messages in telemetry_data.items():
                if messages:
                    try:
                        await TelemetryProcessor._save_batch(
                            messages, msg_type, session.session_id, storage_dir, db_prefix, logger
                        )
                    except (ValueError, duckdb.IOException) as e:
                        session.status = SessionStatus.ERROR
                        session.status_message = f"Failed to save batch for {msg_type}: {str(e)}"
                        logger.error(
                            "Failed to save batch for message type",
                            msg_type=msg_type,
                            error=str(e)
                        )
                        raise ValueError(f"Failed to save batch for {msg_type}: {str(e)}") from e

            # Update session status to READY upon successful completion
            session.status = SessionStatus.READY
            session.status_message = "Ready for chat"
            logger.info("Session status updated to READY")

        except asyncio.TimeoutError:
            session.status = SessionStatus.ERROR
            session.status_message = f"Telemetry parsing timed out after {TelemetryProcessor.PARSE_TIMEOUT_SECONDS} seconds"
            logger.error("Telemetry parsing timed out", timeout=TelemetryProcessor.PARSE_TIMEOUT_SECONDS)
            raise ValueError(f"Telemetry parsing timed out after {TelemetryProcessor.PARSE_TIMEOUT_SECONDS} seconds")
        except (OSError, duckdb.IOException) as e:
            session.status = SessionStatus.ERROR
            session.status_message = f"File or database error: {str(e)}"
            logger.error("File or database error", error=str(e))
            raise OSError(f"File or database operation failed: {str(e)}") from e
        except ValueError as e:
            logger.error("Parsing failed", error=str(e))
            raise ValueError(f"Telemetry parsing failed: {str(e)}") from e
        except Exception as e:
            session.status = SessionStatus.ERROR
            session.status_message = f"Unexpected error during telemetry processing: {str(e)}"
            logger.error(
                "Unexpected error during telemetry processing",
                error_type=type(e).__name__,
                error=str(e),
                exc_info=True
            )
            raise ValueError(f"Unexpected telemetry processing error: {str(e)}") from e

    @staticmethod
    async def _save_batch(
        messages: List[Dict[str, Any]],
        msg_type: str,
        session_id: str,
        storage_dir: str,
        db_prefix: str,
        logger: structlog.stdlib.BoundLogger
    ) -> None:
        """Save a batch of telemetry messages to a DuckDB table asynchronously.

        Args:
            messages: List of telemetry message dictionaries to save.
            msg_type: Type of telemetry message (e.g., 'ATTITUDE').
            session_id: Unique session identifier.
            storage_dir: Directory for storing DuckDB files.
            db_prefix: Prefix for DuckDB database filenames.
            logger: Logger instance for logging operations.

        Raises:
            ValueError: If the database path is invalid or DataFrame creation fails.
            duckdb.IOException: If database operations fail.
            asyncio.TimeoutError: If batch save exceeds BATCH_SAVE_TIMEOUT_SECONDS.
        """
        if not messages:
            logger.debug("No messages to save", msg_type=msg_type)
            return

        # Convert messages to pandas DataFrame
        try:
            df: pd.DataFrame = pd.DataFrame(messages)
        except Exception as e:
            logger.error(
                "Failed to create DataFrame for message type",
                msg_type=msg_type,
                error=str(e)
            )
            raise ValueError(f"Failed to create DataFrame for {msg_type}: {str(e)}") from e

        # Generate sanitized table name
        table_name: str = f"telemetry_{re.sub(r'[^a-z0-9_]', '_', msg_type.lower())}"

        # Prepare DuckDB database path
        os.makedirs(storage_dir, exist_ok=True)
        db_path: str = os.path.join(storage_dir, f"{db_prefix}{session_id}.duckdb")

        # Validate database path
        if ".." in os.path.relpath(db_path, storage_dir):
            logger.error("Invalid database path", db_path=db_path, msg_type=msg_type)
            raise ValueError(f"Invalid database path for {msg_type} processing: {db_path}")

        # Connect to DuckDB and save data
        async def save_to_duckdb() -> None:
            with duckdb.connect(db_path) as conn:
                # Drop 'mavpackettype' column if present
                if 'mavpackettype' in df.columns:
                    df_no_mav = df.drop(columns=['mavpackettype'])
                else:
                    df_no_mav = df

                # Define table columns based on DataFrame dtypes
                columns = [
                    f"{col} {TelemetryProcessor.map_dtype(df_no_mav[col].dtype)}" for col in df_no_mav.columns
                ]

                # Create table if it doesn't exist
                create_table_query = f"""
                CREATE TABLE IF NOT EXISTS {table_name} (
                    {', '.join(columns)}
                )
                """
                conn.execute(create_table_query)

                # Insert data into table
                conn.execute(f"INSERT INTO {table_name} SELECT * FROM df_no_mav", {'df_no_mav': df_no_mav})

        try:
            await asyncio.wait_for(
                asyncio.to_thread(save_to_duckdb),
                timeout=TelemetryProcessor.BATCH_SAVE_TIMEOUT_SECONDS
            )
            logger.info(
                "Saved telemetry batch",
                table_name=table_name,
                msg_type=msg_type,
                row_count=len(df)
            )
        except asyncio.TimeoutError:
            logger.error(
                "DuckDB batch save timed out",
                table_name=table_name,
                msg_type=msg_type,
                timeout=TelemetryProcessor.BATCH_SAVE_TIMEOUT_SECONDS
            )
            raise duckdb.IOException(
                f"DuckDB batch save timed out for {msg_type} after {TelemetryProcessor.BATCH_SAVE_TIMEOUT_SECONDS} seconds"
            )
        except duckdb.IOException as e:
            logger.error(
                "DuckDB operation failed for message type",
                table_name=table_name,
                msg_type=msg_type,
                error=str(e)
            )
            raise duckdb.IOException(f"DuckDB operation failed for {msg_type}: {str(e)}") from e
        except Exception as e:
            logger.error(
                "Unexpected error during batch save for message type",
                table_name=table_name,
                msg_type=msg_type,
                error_type=type(e).__name__,
                error=str(e),
                exc_info=True
            )
            raise ValueError(f"Unexpected error during batch save for {msg_type}: {str(e)}") from e