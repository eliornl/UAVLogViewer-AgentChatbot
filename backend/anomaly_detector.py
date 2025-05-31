from typing import Dict, List, Any, Optional, Tuple, Set
import time
import asyncio
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import duckdb
import structlog
from concurrent.futures import ThreadPoolExecutor

# Constants for anomaly detection configuration
MAX_RESULTS: int = 1000  # Maximum number of results to process
MAX_TOOL_OUTPUT_ROWS: int = 50  # Maximum number of rows to return in tool output
DEFAULT_TIME_LIMIT_SECONDS: float = 45.0  # Default time limit for batch processing
DEFAULT_CONTAMINATION: float = (
    0.1  # Default contamination parameter for IsolationForest
)
DEFAULT_MAX_CACHE_SIZE: int = 20  # Default maximum number of models to cache
DEFAULT_MAX_ROWS: int = 5000  # Default maximum number of rows to analyze
DEFAULT_N_ESTIMATORS_LARGE: int = (
    25  # Default number of estimators for datasets > 500 rows
)
DEFAULT_N_ESTIMATORS_SMALL: int = (
    50  # Default number of estimators for datasets <= 500 rows
)
MAX_COLUMNS: int = 5  # Maximum number of columns to use for anomaly detection
DEFAULT_RANDOM_STATE: int = 42  # Default random state for reproducibility
DEFAULT_THREAD_WORKERS: int = 4  # Default number of thread workers

logger = structlog.get_logger(__name__)


class AnomalyDetector:
    """Class responsible for detecting anomalies in telemetry data.

    This class handles model creation, caching, and efficient anomaly detection
    with optimizations for performance and responsiveness.
    """

    # Priority tables where anomalies are most likely to occur
    PRIORITY_TABLES = [
        "telemetry_attitude",  # Aircraft orientation (roll, pitch, yaw)
        "telemetry_global_position_int",  # Position and velocity
        "telemetry_gps_raw_int",  # GPS data quality and accuracy
    ]

    def __init__(self, db_path: str, max_cache_size: int = DEFAULT_MAX_CACHE_SIZE):
        """
        Initialize the AnomalyDetector with database connection and caching parameters.

        Args:
            db_path (str): Path to the DuckDB database file containing telemetry data
            max_cache_size (int): Maximum number of trained models to keep in cache
                                  (defaults to DEFAULT_MAX_CACHE_SIZE)
        """
        self.db_path = db_path
        self.max_cache_size = max_cache_size
        self.model_cache: Dict[str, Dict[str, Any]] = {}  # Cache for trained models
        self.cache_access_times: Dict[str, float] = (
            {}
        )  # Track when models were last accessed
        self.executor = ThreadPoolExecutor(max_workers=DEFAULT_THREAD_WORKERS)
        self.logger = logger.bind(component="AnomalyDetector")
        self.table_schemas: Dict[str, List[str]] = (
            {}
        )  # Maps table names to their column names
        self.numerical_columns_cache: Dict[str, List[str]] = (
            {}
        )  # Maps tables to numerical columns
        self.initialization_complete: bool = False
        self.background_tasks: List[asyncio.Task] = []
        self.training_in_progress: Set[str] = (
            set()
        )  # Track which models are currently training

    async def initialize(self) -> None:
        """Initialize the detector by preloading table schemas and preparing resources.

        This method returns quickly after preloading schemas but starts background
        tasks to train models for common tables.
        """
        try:
            # Preload table schemas asynchronously - this is fast
            await self._preload_table_schemas()
            self.logger.info("AnomalyDetector schemas loaded", db_path=self.db_path)

            # Start background tasks to train models for priority tables
            # This doesn't block the initialization
            self._start_background_model_training()

            self.initialization_complete = True
            self.logger.info(
                "AnomalyDetector initialized successfully", db_path=self.db_path
            )
        except Exception as e:
            self.logger.error("Failed to initialize AnomalyDetector", error=str(e))
            raise

    def _start_background_model_training(self) -> None:
        """Start background tasks to train models for priority tables."""
        # Find which priority tables exist in the database
        available_priority_tables = [
            table for table in self.PRIORITY_TABLES if table in self.table_schemas
        ]

        if not available_priority_tables:
            self.logger.info("No priority tables found in database")
            return

        self.logger.info(
            f"Starting background model training for {len(available_priority_tables)} priority tables"
        )

        # Create and start a background task for each priority table
        for table in available_priority_tables:
            if (
                table in self.numerical_columns_cache
                and self.numerical_columns_cache[table]
            ):
                columns = self.numerical_columns_cache[table]
                task = asyncio.create_task(self._train_model_for_table(table, columns))
                self.background_tasks.append(task)
                self.logger.info(f"Started background model training for {table}")
            else:
                self.logger.info(f"Skipping {table} - no numerical columns identified")

    async def _train_model_for_table(self, table: str, columns: List[str]) -> None:
        """Train a model for a specific table and its numerical columns.

        Args:
            table (str): Table name
            columns (List[str]): Numerical columns to use for training
        """
        cache_key = f"{table}:{','.join(sorted(columns))}"

        # Check if model is already cached or training is in progress
        if cache_key in self.model_cache or cache_key in self.training_in_progress:
            return

        # Mark this model as training in progress
        self.training_in_progress.add(cache_key)

        try:
            self.logger.info(f"Training model for {table} with {len(columns)} columns")

            # Get data with sampling
            df, sampling_applied = await self._get_table_data_with_sampling(
                table, columns, DEFAULT_MAX_ROWS
            )
            if df.empty:
                self.logger.warning(f"No data available for {table}")
                return

            # Clean data
            df_clean = df.dropna()
            if df_clean.empty:
                self.logger.warning(
                    f"No clean data available for {table} after removing NaN values"
                )
                return

            # Use all available numerical columns
            top_columns = list(df_clean.columns)
            df_subset = df_clean

            # Create and train model
            n_estimators = (
                DEFAULT_N_ESTIMATORS_LARGE
                if len(df_subset) > 500
                else DEFAULT_N_ESTIMATORS_SMALL
            )
            model = IsolationForest(
                contamination=DEFAULT_CONTAMINATION,
                random_state=42,
                n_estimators=n_estimators,
                max_samples="auto",
                n_jobs=-1,
            )

            # Train model
            await asyncio.to_thread(model.fit, df_subset)

            # Cache the model
            model_info = {
                "model": model,
                "columns": top_columns,
                "created_at": time.time(),
                "sampling_applied": sampling_applied,
            }
            self._cache_model(table, columns, model_info)

            self.logger.info(f"Successfully trained and cached model for {table}")

        except Exception as e:
            self.logger.error(f"Failed to train model for {table}", error=str(e))
        finally:
            # Remove from in-progress set
            self.training_in_progress.discard(cache_key)

    async def _preload_table_schemas(self) -> None:
        """Preload all table schemas from the database to avoid repeated queries."""
        try:
            with duckdb.connect(self.db_path, read_only=True) as conn:
                # Get all tables
                tables_result = await asyncio.to_thread(
                    lambda: conn.execute(
                        "SELECT table_name FROM information_schema.tables WHERE table_schema = 'main'"
                    ).fetchall()
                )
                tables = [table[0] for table in tables_result]

                # For each table, get its schema
                for table in tables:
                    try:
                        columns_result = await asyncio.to_thread(
                            lambda: conn.execute(f"DESCRIBE {table}").fetchall()
                        )
                        self.table_schemas[table] = [col[0] for col in columns_result]

                        # Pre-identify numerical columns
                        numerical_cols = []
                        for col in self.table_schemas[table]:
                            try:
                                # Test if column is numerical
                                await asyncio.to_thread(
                                    lambda: conn.execute(
                                        f"SELECT AVG({col}) FROM {table} LIMIT 1"
                                    )
                                )
                                numerical_cols.append(col)
                            except:
                                # Not a numerical column, skip
                                pass

                        if numerical_cols:
                            self.numerical_columns_cache[table] = numerical_cols

                    except Exception as e:
                        self.logger.warning(
                            f"Could not preload schema for table {table}", error=str(e)
                        )

                self.logger.info(
                    "Preloaded table schemas",
                    table_count=len(self.table_schemas),
                    tables_with_numerical_columns=len(self.numerical_columns_cache),
                )
        except Exception as e:
            self.logger.error("Failed to preload table schemas", error=str(e))

    def _get_cached_model(
        self, table: str, columns: List[str]
    ) -> Optional[Dict[str, Any]]:
        """Get a cached model for the given table and columns if available.

        Args:
            table (str): Table name
            columns (List[str]): List of columns used for the model

        Returns:
            Optional[Dict[str, Any]]: Cached model info or None if not found
        """
        # Create a cache key based on table and sorted columns
        cache_key = f"{table}:{','.join(sorted(columns))}"

        if cache_key in self.model_cache:
            # Update access time
            self.cache_access_times[cache_key] = time.time()
            return self.model_cache[cache_key]

        return None

    def _cache_model(
        self, table: str, columns: List[str], model_info: Dict[str, Any]
    ) -> None:
        """Cache a model for future use.

        Args:
            table (str): Table name
            columns (List[str]): Columns used for the model
            model_info (Dict[str, Any]): Model information
        """
        key = f"{table}:{','.join(sorted(columns))}"
        self.model_cache[key] = model_info
        self.cache_access_times[key] = time.time()

    async def _train_model_in_background(
        self, model, df_subset, table, columns, top_columns, sampling_applied
    ):
        """Train a model in the background and cache it for future use.

        This method is designed to be run as a background task using asyncio.create_task().
        It trains the model, caches it, and logs the completion status.

        Args:
            model: The machine learning model to train
            df_subset (pd.DataFrame): Data to train on
            table (str): Table name
            columns (List[str]): Original columns requested for analysis
            top_columns (List[str]): Actual columns used for training
            sampling_applied (bool): Whether sampling was applied to the data
        """
        cache_key = f"{table}:{','.join(sorted(columns))}"

        # Mark this model as training in progress
        self.training_in_progress.add(cache_key)

        try:
            self.logger.info(f"Background training started for table {table}")

            # Train model
            await asyncio.to_thread(model.fit, df_subset)

            # Cache the model for future use
            model_info = {
                "model": model,
                "columns": top_columns,
                "created_at": time.time(),
                "sampling_applied": sampling_applied,
            }
            self._cache_model(table, columns, model_info)

            self.logger.info(f"Background training completed for table {table}")
        except Exception as e:
            self.logger.error(
                f"Error in background training for table {table}", error=str(e)
            )
        finally:
            # Remove from in-progress set
            self.training_in_progress.discard(cache_key)

    # Cache the model method is already defined above

    async def get_numerical_columns(
        self, table: str, requested_columns: List[str]
    ) -> List[str]:
        """Get numerical columns for a table, using cache if available.

        Args:
            table (str): Table name
            requested_columns (List[str]): Requested columns to check

        Returns:
            List[str]: List of numerical columns
        """
        # If we have cached numerical columns for this table
        if table in self.numerical_columns_cache:
            # Return only the columns that were requested and are in the cache
            return [
                col
                for col in requested_columns
                if col in self.numerical_columns_cache[table]
            ]

        # Otherwise, identify numerical columns dynamically
        numerical_columns = []
        try:
            with duckdb.connect(self.db_path, read_only=True) as conn:
                for col in requested_columns:
                    try:
                        # Test if column is numerical
                        await asyncio.to_thread(
                            lambda: conn.execute(
                                f"SELECT AVG({col}) FROM {table} LIMIT 1"
                            )
                        )
                        numerical_columns.append(col)
                    except:
                        # Not a numerical column, skip
                        pass

            # Cache for future use
            if table not in self.numerical_columns_cache:
                self.numerical_columns_cache[table] = []
            self.numerical_columns_cache[table].extend(numerical_columns)
            # Remove duplicates
            self.numerical_columns_cache[table] = list(
                set(self.numerical_columns_cache[table])
            )

        except Exception as e:
            self.logger.error(
                f"Error identifying numerical columns for table {table}", error=str(e)
            )

        return numerical_columns

    async def detect_anomalies(
        self,
        table: str,
        columns: List[str],
        max_rows: int = DEFAULT_MAX_ROWS,
        contamination: float = DEFAULT_CONTAMINATION,
        wait_for_model: bool = True,
        time_boot_ms_range: Optional[Dict[str, int]] = None,
    ) -> Dict[str, Any]:
        """Detect anomalies in the specified table and columns.

        Args:
            table (str): Table name
            columns (List[str]): Columns to analyze
            max_rows (int): Maximum number of rows to analyze
            contamination (float): Expected proportion of anomalies
            wait_for_model (bool): Whether to wait for model training to complete before analyzing
            time_boot_ms_range (Dict[str, int], optional): Time range to filter results
                                                          with format {'start': start_ms, 'end': end_ms}

        Returns:
            Dict[str, Any]: Detection results
        """
        # Validate inputs
        if not table or not columns:
            return {"status": "error", "message": "Table name and columns are required"}

        # Check if table exists in our schema cache
        if table not in self.table_schemas:
            try:
                # Try to load schema on demand
                with duckdb.connect(self.db_path, read_only=True) as conn:
                    try:
                        columns_result = await asyncio.to_thread(
                            lambda: conn.execute(f"DESCRIBE {table}").fetchall()
                        )
                        self.table_schemas[table] = [col[0] for col in columns_result]
                    except duckdb.Error:
                        return {
                            "status": "error",
                            "message": f"Table '{table}' does not exist",
                        }
            except Exception as e:
                return {
                    "status": "error",
                    "message": f"Failed to access database: {str(e)}",
                }

        # Filter for valid columns that exist in the table
        valid_columns = [col for col in columns if col in self.table_schemas[table]]
        if not valid_columns:
            return {
                "status": "error",
                "message": f"No valid columns in {table}. Available: {self.table_schemas[table]}, Requested: {columns}",
            }

        # Get numerical columns
        numerical_columns = await self.get_numerical_columns(table, valid_columns)
        if not numerical_columns:
            return {
                "status": "error",
                "message": f"No numerical columns found among: {valid_columns}",
            }

        # Check for cached model
        cached_model_info = self._get_cached_model(table, numerical_columns)
        if cached_model_info:
            self.logger.info(
                f"Using cached model for table {table}", columns=numerical_columns
            )
            return await self._apply_cached_model(
                table,
                numerical_columns,
                cached_model_info,
                max_rows,
                time_boot_ms_range,
            )

        # No cached model, create a new one
        return await self._create_and_apply_model(
            table,
            numerical_columns,
            max_rows,
            contamination,
            wait_for_model,
            time_boot_ms_range,
        )

    async def _apply_cached_model(
        self,
        table: str,
        columns: List[str],
        model_info: Dict[str, Any],
        max_rows: int,
        time_boot_ms_range: Optional[Dict[str, int]] = None,
    ) -> Dict[str, Any]:
        """Apply a cached model to detect anomalies.

        Args:
            table (str): Table name
            columns (List[str]): Columns to analyze
            model_info (Dict[str, Any]): Cached model information
            max_rows (int): Maximum number of rows to analyze
            time_boot_ms_range (Dict[str, int], optional): Time range to filter results
                                                          with format {'start': start_ms, 'end': end_ms}

        Returns:
            Dict[str, Any]: Detection results
        """
        try:
            model = model_info["model"]
            top_columns = model_info["columns"]

            # Get fresh data with sampling
            df = await self._get_table_data(table, columns, max_rows)
            if df.empty:
                return {
                    "status": "success",
                    "data": [],
                    "anomaly_count": 0,
                    "row_count": 0,
                }

            # Clean data
            df_clean = df.dropna()
            if df_clean.empty:
                return {
                    "status": "error",
                    "message": "No valid data after removing NaN values",
                }

            # Use only the columns the model was trained on
            available_columns = [col for col in top_columns if col in df_clean.columns]
            if not available_columns:
                return {
                    "status": "error",
                    "message": "None of the model's training columns are available in the current data",
                }

            df_subset = df_clean[available_columns]

            # Apply the model
            predictions = await asyncio.to_thread(model.predict, df_subset)
            anomaly_scores = await asyncio.to_thread(model.decision_function, df_subset)

            # Process results
            return self._format_results(
                df_subset,
                predictions,
                anomaly_scores,
                sampling_applied=model_info.get("sampling_applied", False),
                table=table,
            )

        except Exception as e:
            self.logger.error(
                f"Error applying cached model for table {table}", error=str(e)
            )
            return {
                "status": "error",
                "message": f"Failed to apply cached model: {str(e)}",
            }

    async def _create_and_apply_model(
        self,
        table: str,
        columns: List[str],
        max_rows: int,
        contamination: float = DEFAULT_CONTAMINATION,
        wait_for_model: bool = True,
        time_boot_ms_range: Optional[Dict[str, int]] = None,
    ) -> Dict[str, Any]:
        """
        Create a new anomaly detection model, apply it to the data, and cache it for future use.

        This method handles the full lifecycle of model creation, training, and application:
        1. Fetches data from the specified table with appropriate sampling
        2. Cleans the data by removing NaN values
        3. Trains an Isolation Forest model on the data
        4. Applies the model to detect anomalies
        5. Caches the model for future use

        Args:
            table (str): Name of the telemetry table to analyze
            columns (List[str]): List of numerical columns to include in the analysis
            max_rows (int): Maximum number of rows to analyze (for performance)
            contamination (float): Expected proportion of anomalies in the dataset
            wait_for_model (bool): Whether to wait for model training to complete before returning
                                  If False, returns a placeholder result while model trains in background

        Returns:
            Dict[str, Any]: Detection results including anomaly flags and scores
        """
        try:
            # Get data with sampling
            df, sampling_applied = await self._get_table_data_with_sampling(
                table, columns, max_rows
            )
            if df.empty:
                return {
                    "status": "success",
                    "data": [],
                    "anomaly_count": 0,
                    "row_count": 0,
                }

            # Clean data
            df_clean = df.dropna()
            if df_clean.empty:
                return {
                    "status": "error",
                    "message": "No valid data after removing NaN values",
                }

            # Limit columns if too many
            if len(df_clean.columns) > MAX_COLUMNS:
                self.logger.info(
                    f"Limiting anomaly detection to {MAX_COLUMNS} columns out of {len(df_clean.columns)}"
                )
                # Select columns with highest variance
                variances = df_clean.var()
                top_columns = variances.nlargest(MAX_COLUMNS).index.tolist()
                df_subset = df_clean[top_columns]
            else:
                top_columns = list(df_clean.columns)
                df_subset = df_clean

            # Create model with appropriate number of estimators based on dataset size
            n_estimators = (
                DEFAULT_N_ESTIMATORS_LARGE
                if len(df_subset) > 500
                else DEFAULT_N_ESTIMATORS_SMALL
            )
            model = IsolationForest(
                contamination=contamination,
                random_state=DEFAULT_RANDOM_STATE,
                n_estimators=n_estimators,
                max_samples="auto",
                n_jobs=-1,  # Use all available CPU cores
            )

            # If wait_for_model is False, return a placeholder result and train in background
            if not wait_for_model:
                # Start model training in background and store the task
                task = asyncio.create_task(
                    self._train_model_in_background(
                        model, df_subset, table, columns, top_columns, sampling_applied
                    )
                )
                self.background_tasks.append(
                    task
                )  # Store the task to prevent garbage collection

                # Return a placeholder result
                self.logger.info(
                    f"Model training started in background for table {table}"
                )
                return {
                    "status": "success",
                    "message": "Model training started in background",
                    "data": [],
                    "anomaly_count": 0,
                    "row_count": len(df_subset),
                    "background_training": True,
                }

            # Train model synchronously if wait_for_model is True
            await asyncio.to_thread(model.fit, df_subset)

            # Cache the model for future use
            model_info = {
                "model": model,
                "columns": top_columns,
                "created_at": time.time(),
                "sampling_applied": sampling_applied,
            }
            self._cache_model(table, columns, model_info)

            # Apply model
            predictions = await asyncio.to_thread(model.predict, df_subset)
            anomaly_scores = await asyncio.to_thread(model.decision_function, df_subset)

            # Process results
            return self._format_results(
                df_subset, predictions, anomaly_scores, sampling_applied, table=table
            )

        except Exception as e:
            self.logger.error(f"Error creating model for table {table}", error=str(e))
            return {"status": "error", "message": f"Failed to create model: {str(e)}"}

    async def _get_table_data(
        self,
        table: str,
        columns: List[str],
        max_rows: int,
        time_boot_ms_range: Optional[Dict[str, int]] = None,
    ) -> pd.DataFrame:
        """Get data from the specified table and columns.

        Args:
            table (str): Table name
            columns (List[str]): Columns to retrieve
            max_rows (int): Maximum number of rows to retrieve
            time_boot_ms_range (Dict[str, int], optional): Time range to filter results
                                                          with format {'start': start_ms, 'end': end_ms}

        Returns:
            pd.DataFrame: DataFrame containing the requested data
        """
        try:
            with duckdb.connect(self.db_path, read_only=True) as conn:
                query = f"SELECT {', '.join(columns)} FROM {table} LIMIT {max_rows}"
                return await asyncio.to_thread(lambda: conn.execute(query).fetchdf())
        except Exception as e:
            self.logger.error(f"Error fetching data from table {table}", error=str(e))
            return pd.DataFrame()

    async def _get_table_data_with_sampling(
        self,
        table: str,
        columns: List[str],
        max_rows: int,
        time_boot_ms_range: Optional[Dict[str, int]] = None,
    ) -> Tuple[pd.DataFrame, bool]:
        """Get data from a table with sampling for large tables.

        Args:
            table (str): Table name
            columns (List[str]): Columns to fetch
            max_rows (int): Maximum number of rows to fetch
            time_boot_ms_range (Dict[str, int], optional): Time range to filter results
                                                          with format {'start': start_ms, 'end': end_ms}

        Returns:
            Tuple[pd.DataFrame, bool]: Table data and whether sampling was applied
        """
        try:
            with duckdb.connect(self.db_path, read_only=True) as conn:
                # Get row count
                try:
                    row_count = await asyncio.to_thread(
                        lambda: conn.execute(
                            f"SELECT COUNT(*) FROM {table}"
                        ).fetchone()[0]
                    )
                    self.logger.info(f"Table {table} has {row_count} rows")
                except:
                    row_count = 0

                # Apply sampling for large tables
                sample_clause = ""
                sampling_applied = False
                limit_clause = ""

                # Only apply sampling if max_rows > 0 (0 means use all data)
                if max_rows > 0 and row_count > max_rows:
                    # Calculate sampling percentage
                    sample_percent = min(100, (max_rows / row_count) * 100)
                    sample_clause = f" USING SAMPLE {sample_percent}%"
                    sampling_applied = True
                    limit_clause = f" LIMIT {max_rows}"
                    self.logger.info(
                        f"Applying {sample_percent:.2f}% sampling to table {table}"
                    )

                # Fetch data
                query = f"SELECT {', '.join(columns)} FROM {table}{sample_clause}{limit_clause}"
                df = await asyncio.to_thread(lambda: conn.execute(query).fetchdf())

                return df, sampling_applied

        except Exception as e:
            self.logger.error(f"Error fetching data from table {table}", error=str(e))
            return pd.DataFrame(), False

    async def detect_anomalies_batch(
        self,
        tables: Optional[List[str]] = None,
        max_rows_per_table: int = 0,  # 0 means use all available data
        contamination: float = DEFAULT_CONTAMINATION,
        time_limit_seconds: float = DEFAULT_TIME_LIMIT_SECONDS,
        time_boot_ms_range: Optional[Dict[str, int]] = None,
    ) -> Dict[str, Any]:
        """
        Detect anomalies across multiple tables with a time limit.

        This method processes as many tables as possible within the time limit,
        prioritizing the most important tables first.

        Args:
            tables (List[str]): List of tables to analyze. If None, uses PRIORITY_TABLES
            max_rows_per_table (int): Maximum rows to analyze per table
            contamination (float): Expected proportion of anomalies
            time_limit_seconds (float): Maximum time to spend on analysis
            time_boot_ms_range (Dict[str, int], optional): Time range to filter results
                                                          with format {'start': start_ms, 'end': end_ms}

        Returns:
            Dict[str, Any]: Combined detection results
        """
        start_time = time.time()
        results = {
            "status": "success",
            "tables_processed": [],
            "tables_skipped": [],
            "anomalies_found": 0,
            "total_rows_analyzed": 0,
            "time_spent": 0,
            "results": {},
        }

        # Use priority tables if none specified
        tables_to_process = tables if tables else self.PRIORITY_TABLES

        # Get all table schemas first if not already loaded
        if not self.table_schemas:
            await self._preload_table_schemas()

        # Filter to tables that actually exist
        existing_tables = [t for t in tables_to_process if t in self.table_schemas]
        if not existing_tables:
            return {"status": "error", "message": "No valid tables found to analyze"}

        # Process tables until time limit
        for table in existing_tables:
            # Check if we've exceeded the time limit
            if time.time() - start_time > time_limit_seconds:
                results["tables_skipped"].extend(
                    [t for t in existing_tables if t not in results["tables_processed"]]
                )
                results["status"] = "partial_success"
                results["message"] = (
                    f"Time limit reached after processing {len(results['tables_processed'])} tables"
                )
                break

            # Get numerical columns for this table
            try:
                all_columns = self.table_schemas.get(table, [])
                numerical_columns = await self.get_numerical_columns(table, all_columns)

                if not numerical_columns:
                    self.logger.warning(f"No numerical columns found in table {table}")
                    results["tables_skipped"].append(table)
                    continue

                # Detect anomalies for this table
                table_result = await self.detect_anomalies(
                    table,
                    numerical_columns,
                    max_rows=max_rows_per_table,
                    contamination=contamination,
                    wait_for_model=True,  # Wait for models to get immediate results
                    time_boot_ms_range=time_boot_ms_range,
                )

                # Add results
                results["results"][table] = table_result
                results["tables_processed"].append(table)
                results["anomalies_found"] += table_result.get("anomaly_count", 0)
                results["total_rows_analyzed"] += table_result.get("row_count", 0)

            except Exception as e:
                self.logger.error(f"Error processing table {table}: {str(e)}")
                results["tables_skipped"].append(table)

        # Calculate time spent
        results["time_spent"] = time.time() - start_time

        return results

    def _format_results(
        self,
        df: pd.DataFrame,
        predictions: np.ndarray,
        anomaly_scores: np.ndarray,
        sampling_applied: bool,
        table: str = "unknown",
    ) -> Dict[str, Any]:
        """Format anomaly detection results.

        Args:
            df (pd.DataFrame): Data frame with the analyzed data
            predictions (np.ndarray): Model predictions (-1 for anomalies, 1 for normal)
            anomaly_scores (np.ndarray): Anomaly scores
            sampling_applied (bool): Whether sampling was applied

        Returns:
            Dict[str, Any]: Formatted results
        """
        # -1 indicates anomaly, 1 indicates normal
        anomalies = predictions == -1

        # Format results
        results = []
        anomaly_count = 0

        # Limit number of results
        process_indices = range(min(len(df), MAX_RESULTS))

        for i in process_indices:
            idx = df.index[i]
            is_anomaly = anomalies[i]
            result = {
                "row_index": int(idx),
                "is_anomaly": bool(is_anomaly),
                "anomaly_score": float(anomaly_scores[i]),
            }

            # Add time_boot_ms to the result if it exists in the dataframe
            if "time_boot_ms" in df.columns:
                result["time_boot_ms"] = (
                    int(df.iloc[i]["time_boot_ms"])
                    if not pd.isna(df.iloc[i]["time_boot_ms"])
                    else None
                )

            results.append(result)
            if is_anomaly:
                anomaly_count += 1

        # Truncate results if too long
        final_row_count = len(df)
        if len(results) > MAX_TOOL_OUTPUT_ROWS:
            self.logger.debug(
                f"Output truncated from {len(results)} to {MAX_TOOL_OUTPUT_ROWS} rows.",
                table=table,
            )
            # Include truncation message
            return {
                "status": "success_truncated",
                "data": results[:MAX_TOOL_OUTPUT_ROWS],
                "message": f"Output truncated to first {MAX_TOOL_OUTPUT_ROWS} anomaly detection results. Original result count: {len(results)}.",
                "anomaly_count": sum(
                    1 for r in results[:MAX_TOOL_OUTPUT_ROWS] if r["is_anomaly"]
                ),
                "row_count": final_row_count,
                "displayed_row_count": MAX_TOOL_OUTPUT_ROWS,
                "columns_used": list(df.columns),
                "sampling_applied": sampling_applied,
            }

        return {
            "status": "success",
            "data": results,
            "anomaly_count": anomaly_count,
            "row_count": final_row_count,
            "columns_used": list(df.columns),
            "sampling_applied": sampling_applied,
        }
