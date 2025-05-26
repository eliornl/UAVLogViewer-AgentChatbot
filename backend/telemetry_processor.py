from typing import Dict, List, Any
import os
import re
import asyncio
import duckdb
import pandas as pd
import structlog
from pymavlink import mavutil
from tenacity import retry, stop_after_attempt, wait_exponential
from backend.models import Session, SessionStatus

logger: structlog.stdlib.BoundLogger = structlog.get_logger(__name__)

class TelemetryProcessor:
    """Processes MAVLink telemetry log files and saves data to DuckDB asynchronously.

    Parses binary telemetry logs, collects all messages by type, creates one DataFrame
    per message type, and stores them in session-specific DuckDB tables for efficient querying.
    Note: The `timestamp` column is included in tables only if `_timestamp` is present in the
    MAVLink messages; otherwise, it is omitted or may contain NULL values.
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

    # Timeouts for I/O operations
    PARSE_TIMEOUT_SECONDS: float = 300.0  # 5 minutes for large log files
    SAVE_TIMEOUT_SECONDS: float = 120.0  # 2 minutes for saving all tables

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
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=5))
    async def parse_and_save(
        file_path: str,
        session: Session,
        storage_dir: str,
        db_prefix: str
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
        session_logger = logger.bind(session_id=session.session_id, file_name=session.file_name)
        try:
            # Validate file path to prevent path traversal and ensure file exists
            if not os.path.isfile(file_path):
                session.status = SessionStatus.ERROR
                session.status_message = "Invalid file path provided"
                session_logger.error("Invalid file path", file_path=file_path)
                raise ValueError(f"Invalid file path: {file_path}")

            # Initialize MAVLink log file parser
            def init_mav() -> Any:
                return mavutil.mavlogfile(file_path)

            try:
                mav = await asyncio.wait_for(
                    asyncio.to_thread(init_mav),
                    timeout=TelemetryProcessor.PARSE_TIMEOUT_SECONDS
                )
            except Exception as e:
                session.status = SessionStatus.ERROR
                session.status_message = f"Invalid telemetry file: {str(e)}"
                session_logger.error("Failed to initialize MAVLink file", error=str(e))
                raise ValueError(f"Invalid telemetry file: {str(e)}") from e

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
                timeout=TelemetryProcessor.PARSE_TIMEOUT_SECONDS
            )

            # Create and save DataFrames for each message type
            def save_to_duckdb() -> None:
                # Prepare DuckDB database path
                os.makedirs(storage_dir, exist_ok=True)
                db_path = os.path.join(storage_dir, f"{db_prefix}{session.session_id}.duckdb")
                if ".." in os.path.relpath(db_path, storage_dir):
                    session_logger.error("Invalid database path", db_path=db_path)
                    raise ValueError(f"Invalid database path: {db_path}")

                # Connect to DuckDB
                with duckdb.connect(db_path) as conn:
                    for msg_type in TelemetryProcessor.TELEMETRY_MESSAGE_TYPES:
                        messages = telemetry_data[msg_type]
                        if not messages:
                            session_logger.debug("No messages for type", msg_type=msg_type)
                            continue

                        # Create DataFrame
                        try:
                            df = pd.DataFrame(messages)
                        except Exception as e:
                            session_logger.error(
                                "Failed to create DataFrame for message type",
                                msg_type=msg_type,
                                error=str(e)
                            )
                            raise ValueError(f"Failed to create DataFrame for {msg_type}: {str(e)}") from e

                        # Drop 'mavpackettype' column if present
                        if 'mavpackettype' in df.columns:
                            df = df.drop(columns=['mavpackettype'])

                        # Generate sanitized table name
                        table_name = f"telemetry_{re.sub(r'[^a-z0-9_]', '_', msg_type.lower())}"

                        # Define table columns based on DataFrame dtypes
                        columns = [
                            f"{col} {TelemetryProcessor.map_dtype(df[col].dtype)}" for col in df.columns
                        ]

                        # Check if the table exists
                        table_exists = conn.execute(
                            "SELECT COUNT(*) FROM duckdb_tables() WHERE table_name = ?", (table_name,)
                        ).fetchone()[0] > 0

                        # Create table if it doesn't exist
                        if not table_exists:
                            session_logger.debug("Creating table", table_name=table_name, db_path=db_path)
                            quoted_table_name = f'"{table_name}"'
                            create_table_query = f"""
                            CREATE TABLE {quoted_table_name} (
                                {', '.join(columns)}
                            )
                            """
                            try:
                                conn.execute(create_table_query)
                            except Exception as e:
                                session_logger.error(
                                    "Failed to create table",
                                    table_name=table_name,
                                    db_path=db_path,
                                    error=str(e)
                                )
                                raise duckdb.IOException(
                                    f"Failed to create table {table_name} in {db_path}: {str(e)}"
                                ) from e
                        else:
                            session_logger.debug("Table exists, will append data", table_name=table_name)

                        # Insert data into table via temporary view
                        conn.register('df_view', df)
                        conn.execute(f"INSERT INTO {quoted_table_name} SELECT * FROM df_view")
                        conn.unregister('df_view')
                        session_logger.info(
                            "Saved telemetry data",
                            table_name=table_name,
                            msg_type=msg_type,
                            row_count=len(df)
                        )
                        conn.commit()

            try:
                await asyncio.wait_for(
                    asyncio.to_thread(save_to_duckdb),
                    timeout=TelemetryProcessor.SAVE_TIMEOUT_SECONDS
                )
            except asyncio.TimeoutError:
                session.status = SessionStatus.ERROR
                session.status_message = f"DuckDB save timed out after {TelemetryProcessor.SAVE_TIMEOUT_SECONDS} seconds"
                session_logger.error("DuckDB save timed out", timeout=TelemetryProcessor.SAVE_TIMEOUT_SECONDS)
                raise duckdb.IOException(
                    f"DuckDB save timed out after {TelemetryProcessor.SAVE_TIMEOUT_SECONDS} seconds"
                )

            # Update session status to READY upon successful completion
            session.status = SessionStatus.READY
            session.status_message = "Ready for chat"
            session_logger.info("Session status updated to READY")

        except asyncio.TimeoutError as e:
            session.status = SessionStatus.ERROR
            session.status_message = f"Operation timed out: {str(e)}"
            session_logger.error("Operation timed out", error=str(e))
            raise ValueError(f"Operation timed out: {str(e)}") from e
        except (OSError, duckdb.IOException) as e:
            session.status = SessionStatus.ERROR
            session.status_message = f"File or database error: {str(e)}"
            session_logger.error("File or database error", error=str(e))
            raise OSError(f"File or database operation failed: {str(e)}") from e
        except ValueError as e:
            session_logger.error("Parsing failed", error=str(e))
            raise ValueError(f"Telemetry parsing failed: {str(e)}") from e
        except Exception as e:
            session.status = SessionStatus.ERROR
            session.status_message = f"Unexpected error during telemetry processing: {str(e)}"
            session_logger.error(
                "Unexpected error during telemetry processing",
                error_type=type(e).__name__,
                error=str(e),
                exc_info=True
            )
            raise ValueError(f"Unexpected telemetry processing error: {str(e)}") from e