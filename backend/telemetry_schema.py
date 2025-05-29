"""Telemetry schema definitions for the UAV Log Viewer application.

This module defines the structure of telemetry data tables used in the application,
including column definitions, data types, descriptions, and example data. It also
provides validation functions to ensure the schema is correctly formatted.

The schema is used for:
1. Creating database tables in DuckDB
2. Generating vector embeddings for semantic search
3. Providing context for anomaly detection
4. Supporting the telemetry agent's understanding of data structure
"""

from typing import List, Dict, Any, TypedDict, Literal, Set
import structlog

# Configure logger
logger: structlog.stdlib.BoundLogger = structlog.get_logger(__name__)

# Constants for schema validation
REQUIRED_TABLE_FIELDS: List[str] = ["table", "columns", "description", "anomaly_hint", "example"]
REQUIRED_COLUMN_FIELDS: List[str] = ["name", "data_type", "description", "nullable"]
ALLOWED_DATA_TYPES: Set[str] = {"int", "float", "str", "list[int]"}

# Error message templates
MISSING_FIELDS_ERROR: str = "Missing required fields {fields} in telemetry schema for table {table}"
EMPTY_ANOMALY_HINT_ERROR: str = "Invalid or empty anomaly_hint for table {table}"
NO_COLUMNS_ERROR: str = "No columns defined for table {table}"
COLUMN_MISMATCH_ERROR: str = "Example for {table} does not match columns"
MISSING_COLUMN_FIELDS_ERROR: str = "Missing required column fields for {table}.{column}"
INVALID_DATA_TYPE_ERROR: str = "Invalid data type {data_type} for {table}.{column}"
INVALID_NULLABLE_ERROR: str = "Invalid nullable field for {table}.{column}"
INVALID_EXAMPLE_TYPE_ERROR: str = "Invalid example value type for {table}.{column}: expected {expected}"

# Define type aliases for clarity
ColumnDataType = Literal["int", "float", "str", "list[int]"]

# Define TypedDict for column metadata
class ColumnMetadata(TypedDict):
    name: str
    data_type: ColumnDataType
    unit: str | None
    description: str
    nullable: bool

# Define TypedDict for table metadata
class TableMetadata(TypedDict):
    table: str
    columns: List[ColumnMetadata]
    description: str
    anomaly_hint: str
    example: Dict[str, Any]

# Telemetry schema for FAISS indexing
TELEMETRY_SCHEMA: List[TableMetadata] = [
    {
        "table": "telemetry_attitude",
        "columns": [
            {"name": "timestamp", "data_type": "int", "unit": None, "description": "Unix epoch time in seconds", "nullable": False},
            {"name": "roll", "data_type": "float", "unit": "degrees", "description": "Roll angle", "nullable": False},
            {"name": "pitch", "data_type": "float", "unit": "degrees", "description": "Pitch angle", "nullable": False},
            {"name": "yaw", "data_type": "float", "unit": "degrees", "description": "Yaw angle", "nullable": False},
            {"name": "rollspeed", "data_type": "float", "unit": "degrees/s", "description": "Roll angular velocity", "nullable": False},
            {"name": "pitchspeed", "data_type": "float", "unit": "degrees/s", "description": "Pitch angular velocity", "nullable": False},
            {"name": "yawspeed", "data_type": "float", "unit": "degrees/s", "description": "Yaw angular velocity", "nullable": False}
        ],
        "description": "Vehicle orientation (roll, pitch, yaw) and angular velocities.",
        "anomaly_hint": "Look for sudden changes in roll, pitch, or yaw; detect outliers in rollspeed, pitchspeed, or yawspeed indicating instability; use ML to identify unusual orientation patterns; correlate with battery or GPS issues.",
        "example": {
            "timestamp": 1678923456,
            "roll": 1.23,
            "pitch": -0.45,
            "yaw": 90.0,
            "rollspeed": 0.1,
            "pitchspeed": 0.2,
            "yawspeed": 0.15
        }
    },
    {
        "table": "telemetry_global_position_int",
        "columns": [
            {"name": "timestamp", "data_type": "int", "unit": None, "description": "Unix epoch time in seconds", "nullable": False},
            {"name": "lat", "data_type": "int", "unit": "degrees*1e7", "description": "Latitude", "nullable": False},
            {"name": "lon", "data_type": "int", "unit": "degrees*1e7", "description": "Longitude", "nullable": False},
            {"name": "alt", "data_type": "int", "unit": "millimeters", "description": "Absolute altitude above mean sea level", "nullable": False},
            {"name": "relative_alt", "data_type": "int", "unit": "millimeters", "description": "Altitude relative to takeoff", "nullable": False},
            {"name": "vx", "data_type": "int", "unit": "cm/s", "description": "Velocity in X direction", "nullable": False},
            {"name": "vy", "data_type": "int", "unit": "cm/s", "description": "Velocity in Y direction", "nullable": False},
            {"name": "vz", "data_type": "int", "unit": "cm/s", "description": "Velocity in Z direction", "nullable": False},
            {"name": "hdg", "data_type": "int", "unit": "cdeg", "description": "Heading in centidegrees", "nullable": False}
        ],
        "description": "Global position (latitude, longitude, altitude) and velocity.",
        "anomaly_hint": "Check for sudden changes in altitude or velocity; detect inconsistencies in lat/lon movement; use ML to flag unusual position or velocity patterns; correlate with GPS fix quality.",
        "example": {
            "timestamp": 1678923456,
            "lat": 473976543,
            "lon": -1223456789,
            "alt": 500000,
            "relative_alt": 10000,
            "vx": 100,
            "vy": -50,
            "vz": 20,
            "hdg": 9000
        }
    },
    {
        "table": "telemetry_vfr_hud",
        "columns": [
            {"name": "timestamp", "data_type": "int", "unit": None, "description": "Unix epoch time in seconds", "nullable": False},
            {"name": "airspeed", "data_type": "float", "unit": "m/s", "description": "Airspeed", "nullable": False},
            {"name": "groundspeed", "data_type": "float", "unit": "m/s", "description": "Ground speed", "nullable": False},
            {"name": "heading", "data_type": "int", "unit": "degrees", "description": "Heading", "nullable": False},
            {"name": "throttle", "data_type": "int", "unit": "percent", "description": "Throttle percentage", "nullable": False},
            {"name": "alt", "data_type": "float", "unit": "meters", "description": "Absolute altitude", "nullable": False},
            {"name": "climb", "data_type": "float", "unit": "m/s", "description": "Climb rate", "nullable": False}
        ],
        "description": "Heads-up display data (airspeed, ground speed, throttle, etc.).",
        "anomaly_hint": "Look for sudden changes in airspeed, groundspeed, or climb rate; detect inconsistencies between throttle and climb; use ML to identify unusual speed or altitude patterns; correlate with battery status.",
        "example": {
            "timestamp": 1678923456,
            "airspeed": 15.5,
            "groundspeed": 14.2,
            "heading": 90,
            "throttle": 75,
            "alt": 500.0,
            "climb": 1.2
        }
    },
    {
        "table": "telemetry_sys_status",
        "columns": [
            {"name": "timestamp", "data_type": "int", "unit": None, "description": "Unix epoch time in seconds", "nullable": False},
            {"name": "onboard_control_sensors_present", "data_type": "int", "unit": None, "description": "Bitmap of present sensors", "nullable": False},
            {"name": "onboard_control_sensors_enabled", "data_type": "int", "unit": None, "description": "Bitmap of enabled sensors", "nullable": False},
            {"name": "onboard_control_sensors_health", "data_type": "int", "unit": None, "description": "Bitmap of sensor health", "nullable": False},
            {"name": "load", "data_type": "int", "unit": "c%", "description": "System load in centipercent", "nullable": False},
            {"name": "voltage_battery", "data_type": "int", "unit": "mV", "description": "Battery voltage", "nullable": False},
            {"name": "current_battery", "data_type": "int", "unit": "cA", "description": "Battery current", "nullable": False},
            {"name": "battery_remaining", "data_type": "int", "unit": "percent", "description": "Remaining battery percentage", "nullable": False},
            {"name": "drop_rate_comm", "data_type": "int", "unit": "c%", "description": "Communication drop rate", "nullable": True},
            {"name": "errors_comm", "data_type": "int", "unit": None, "description": "Communication errors", "nullable": True}
        ],
        "description": "System status including battery and communication metrics.",
        "anomaly_hint": "Detect sudden drops or unusually low battery voltage or remaining capacity; check for high communication drop rates or errors; use ML to identify unusual sensor health or system load patterns; correlate with attitude or position anomalies.",
        "example": {
            "timestamp": 1678923456,
            "onboard_control_sensors_present": 255,
            "onboard_control_sensors_enabled": 255,
            "onboard_control_sensors_health": 254,
            "load": 4500,
            "voltage_battery": 12600,
            "current_battery": 500,
            "battery_remaining": 85,
            "drop_rate_comm": 10,
            "errors_comm": 2
        }
    },
    {
        "table": "telemetry_statustext",
        "columns": [
            {"name": "timestamp", "data_type": "int", "unit": None, "description": "Unix epoch time in seconds", "nullable": False},
            {"name": "severity", "data_type": "int", "unit": None, "description": "Message severity level", "nullable": False},
            {"name": "text", "data_type": "str", "unit": None, "description": "Status message text", "nullable": True}
        ],
        "description": "Free-form status messages (e.g., warnings, errors).",
        "anomaly_hint": "Look for high-severity messages indicating warnings or errors; check text for patterns like battery or GPS issues; correlate with other tables to confirm anomalies.",
        "example": {
            "timestamp": 1678923456,
            "severity": 3,
            "text": "Low battery warning"
        }
    },
    {
        "table": "telemetry_rc_channels",
        "columns": [
            {"name": "timestamp", "data_type": "int", "unit": None, "description": "Unix epoch time in seconds", "nullable": False},
            {"name": "chan1_raw", "data_type": "int", "unit": None, "description": "Raw value for channel 1", "nullable": False},
            {"name": "chan2_raw", "data_type": "int", "unit": None, "description": "Raw value for channel 2", "nullable": False},
            {"name": "chan3_raw", "data_type": "int", "unit": None, "description": "Raw value for channel 3", "nullable": False},
            {"name": "chan4_raw", "data_type": "int", "unit": None, "description": "Raw value for channel 4", "nullable": False},
            {"name": "chan5_raw", "data_type": "int", "unit": None, "description": "Raw value for channel 5", "nullable": True},
            {"name": "chan6_raw", "data_type": "int", "unit": None, "description": "Raw value for channel 6", "nullable": True},
            {"name": "chan7_raw", "data_type": "int", "unit": None, "description": "Raw value for channel 7", "nullable": True},
            {"name": "chan8_raw", "data_type": "int", "unit": None, "description": "Raw value for channel 8", "nullable": True},
            {"name": "rssi", "data_type": "int", "unit": None, "description": "Receiver signal strength indicator", "nullable": True}
        ],
        "description": "Remote control channel values and signal strength.",
        "anomaly_hint": "Check for sudden changes in channel values or low signal strength; use ML to detect unusual channel patterns; correlate with communication errors in sys_status.",
        "example": {
            "timestamp": 1678923456,
            "chan1_raw": 1500,
            "chan2_raw": 1450,
            "chan3_raw": 1600,
            "chan4_raw": 1550,
            "chan5_raw": 1400,
            "chan6_raw": 1300,
            "chan7_raw": 1200,
            "chan8_raw": 1100,
            "rssi": 90
        }
    },
    {
        "table": "telemetry_gps_raw_int",
        "columns": [
            {"name": "timestamp", "data_type": "int", "unit": None, "description": "Unix epoch time in seconds", "nullable": False},
            {"name": "fix_type", "data_type": "int", "unit": None, "description": "GPS fix type", "nullable": False},
            {"name": "lat", "data_type": "int", "unit": "degrees*1e7", "description": "Latitude", "nullable": False},
            {"name": "lon", "data_type": "int", "unit": "degrees*1e7", "description": "Longitude", "nullable": False},
            {"name": "alt", "data_type": "int", "unit": "millimeters", "description": "Altitude", "nullable": False},
            {"name": "eph", "data_type": "int", "unit": None, "description": "Horizontal position uncertainty", "nullable": True},
            {"name": "epv", "data_type": "int", "unit": None, "description": "Vertical position uncertainty", "nullable": True},
            {"name": "vel", "data_type": "int", "unit": "cm/s", "description": "GPS velocity", "nullable": True},
            {"name": "cog", "data_type": "int", "unit": "cdeg", "description": "Course over ground", "nullable": True},
            {"name": "satellites_visible", "data_type": "int", "unit": None, "description": "Number of visible satellites", "nullable": False}
        ],
        "description": "Raw GPS data (position, fix type, satellite count).",
        "anomaly_hint": "Check for poor GPS fix quality or low satellite counts; detect high position uncertainties or inconsistent position/velocity; use ML to flag unusual GPS patterns; correlate with position data.",
        "example": {
            "timestamp": 1678923456,
            "fix_type": 3,
            "lat": 473976543,
            "lon": -1223456789,
            "alt": 500000,
            "eph": 150,
            "epv": 200,
            "vel": 1000,
            "cog": 9000,
            "satellites_visible": 8
        }
    },
    {
        "table": "telemetry_battery_status",
        "columns": [
            {"name": "timestamp", "data_type": "int", "unit": None, "description": "Unix epoch time in seconds", "nullable": False},
            {"name": "current_consumed", "data_type": "int", "unit": "mAh", "description": "Consumed current", "nullable": False},
            {"name": "energy_consumed", "data_type": "int", "unit": "hJ", "description": "Consumed energy", "nullable": False},
            {"name": "temperature", "data_type": "int", "unit": "c°C", "description": "Battery temperature", "nullable": True},
            {"name": "voltages", "data_type": "list[int]", "unit": "mV", "description": "Cell voltages", "nullable": True},
            {"name": "current_battery", "data_type": "int", "unit": "cA", "description": "Current battery draw", "nullable": False},
            {"name": "battery_remaining", "data_type": "int", "unit": "percent", "description": "Remaining battery percentage", "nullable": False}
        ],
        "description": "Detailed battery status (cell voltages, current, energy).",
        "anomaly_hint": "Detect sudden changes in battery capacity, current, or temperature; check for unusually low cell voltages or capacity; use ML to flag unusual battery patterns; correlate with system status or attitude anomalies.",
        "example": {
            "timestamp": 1678923456,
            "current_consumed": 1200,
            "energy_consumed": 500,
            "temperature": 2500,
            "voltages": [4100, 4050, 4000],
            "current_battery": 600,
            "battery_remaining": 80
        }
    },
    {
        "table": "telemetry_ekf_status_report",
        "columns": [
            {"name": "timestamp", "data_type": "int", "unit": None, "description": "Unix epoch time in seconds", "nullable": False},
            {"name": "flags", "data_type": "int", "unit": None, "description": "EKF status flags", "nullable": False},
            {"name": "velocity_variance", "data_type": "float", "unit": None, "description": "Velocity variance", "nullable": False},
            {"name": "pos_horiz_variance", "data_type": "float", "unit": None, "description": "Horizontal position variance", "nullable": False},
            {"name": "pos_vert_variance", "data_type": "float", "unit": None, "description": "Vertical position variance", "nullable": False},
            {"name": "compass_variance", "data_type": "float", "unit": None, "description": "Compass variance", "nullable": False},
            {"name": "terrain_alt_variance", "data_type": "float", "unit": None, "description": "Terrain altitude variance", "nullable": False}
        ],
        "description": "Extended Kalman Filter status for navigation.",
        "anomaly_hint": "Check for high variance in velocity, position, or compass; detect unusual EKF flags; use ML to identify abnormal variance patterns; correlate with GPS or position data.",
        "example": {
            "timestamp": 1678923456,
            "flags": 1,
            "velocity_variance": 0.05,
            "pos_horiz_variance": 0.1,
            "pos_vert_variance": 0.15,
            "compass_variance": 0.02,
            "terrain_alt_variance": 0.3
        }
    }
]

def validate_telemetry_schema(metadata: List[TableMetadata]) -> None:
    """Validate the structure and content of telemetry schema.

    Ensures that each table entry has required fields, valid column metadata,
    and consistent example data matching the column definitions. This validation
    is performed at module load time to catch any schema issues early.

    Args:
        metadata: List of table metadata dictionaries to validate.

    Raises:
        ValueError: If any metadata entry or column is invalid, or if example data
            does not match column definitions.
            
    Example:
        >>> validate_telemetry_schema(TELEMETRY_SCHEMA)
        # No return value if validation passes
    """
    for meta in metadata:
        # Validate required fields
        table_name: str = meta.get("table", "")
        if not all(key in meta for key in REQUIRED_TABLE_FIELDS):
            missing_fields = set(REQUIRED_TABLE_FIELDS) - set(meta.keys())
            logger.error(
                "Missing required fields in telemetry schema", 
                table=table_name,
                missing_fields=missing_fields
            )
            raise ValueError(MISSING_FIELDS_ERROR.format(fields=missing_fields, table=table_name))
        
        # Validate anomaly_hint
        if not isinstance(meta["anomaly_hint"], str) or not meta["anomaly_hint"].strip():
            logger.error("Invalid or empty anomaly_hint", table=table_name)
            raise ValueError(EMPTY_ANOMALY_HINT_ERROR.format(table=table_name))
        
        # Validate columns
        if not meta["columns"]:
            logger.error("No columns defined", table=table_name)
            raise ValueError(NO_COLUMNS_ERROR.format(table=table_name))
            
        # Validate example matches columns
        column_names: set[str] = {col["name"] for col in meta["columns"]}
        example_keys: set[str] = set(meta["example"].keys())
        if column_names != example_keys:
            logger.error(
                "Example keys do not match column names",
                table=table_name,
                missing_columns=column_names - example_keys,
                extra_columns=example_keys - column_names
            )
            raise ValueError(COLUMN_MISMATCH_ERROR.format(table=table_name))

        # Validate each column
        for col in meta["columns"]:
            col_name: str = col.get("name", "")
            # Check required column fields
            if not all(key in col for key in REQUIRED_COLUMN_FIELDS):
                missing_fields = set(REQUIRED_COLUMN_FIELDS) - set(col.keys())
                logger.error(
                    "Missing required column fields", 
                    table=table_name, 
                    column=col_name,
                    missing_fields=missing_fields
                )
                raise ValueError(MISSING_COLUMN_FIELDS_ERROR.format(table=table_name, column=col_name))

            # Validate data type
            if col["data_type"] not in ALLOWED_DATA_TYPES:
                logger.error(
                    "Invalid data type in column metadata",
                    table=table_name,
                    column=col_name,
                    data_type=col["data_type"],
                    allowed_types=list(ALLOWED_DATA_TYPES)
                )
                raise ValueError(INVALID_DATA_TYPE_ERROR.format(
                    data_type=col["data_type"],
                    table=table_name,
                    column=col_name
                ))

            # Validate nullable field
            if not isinstance(col["nullable"], bool):
                logger.error(
                    "Invalid nullable field",
                    table=table_name,
                    column=col_name,
                    nullable=col["nullable"],
                    expected_type="bool"
                )
                raise ValueError(INVALID_NULLABLE_ERROR.format(
                    table=table_name,
                    column=col_name
                ))

            # Validate example value type
            example_value: Any = meta["example"][col_name]
            if col["nullable"] and example_value is None:
                continue

            if col["data_type"] == "int" and not isinstance(example_value, int):
                logger.error(
                    "Invalid example value type",
                    table=table_name,
                    column=col_name,
                    expected="int",
                    got=type(example_value).__name__
                )
                raise ValueError(INVALID_EXAMPLE_TYPE_ERROR.format(
                    table=table_name,
                    column=col_name,
                    expected="int"
                ))
            if col["data_type"] == "float" and not isinstance(example_value, float):
                logger.error(
                    "Invalid example value type",
                    table=table_name,
                    column=col_name,
                    expected="float",
                    got=type(example_value).__name__
                )
                raise ValueError(INVALID_EXAMPLE_TYPE_ERROR.format(
                    table=table_name,
                    column=col_name,
                    expected="float"
                ))
            if col["data_type"] == "str" and not isinstance(example_value, str):
                logger.error(
                    "Invalid example value type",
                    table=table_name,
                    column=col_name,
                    expected="str",
                    got=type(example_value).__name__
                )
                raise ValueError(INVALID_EXAMPLE_TYPE_ERROR.format(
                    table=table_name,
                    column=col_name,
                    expected="str"
                ))
            if col["data_type"] == "list[int]" and not (
                isinstance(example_value, list) and all(isinstance(x, int) for x in example_value)
            ):
                logger.error(
                    "Invalid example value type",
                    table=table_name,
                    column=col_name,
                    expected="list[int]",
                    got=type(example_value).__name__
                )
                raise ValueError(INVALID_EXAMPLE_TYPE_ERROR.format(
                    table=table_name,
                    column=col_name,
                    expected="list[int]"
                ))

# Validate schema at module load
validate_telemetry_schema(TELEMETRY_SCHEMA)