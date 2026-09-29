"""
SkyGuard AI - database.py

SQLite database layer for:

    - Live sensor readings
    - Anomaly detection results
    - Sensor history
    - Prediction / suggested values
    - Seasonal prediction metadata
    - Maintenance analysis

IMPORTANT
---------
This module stores the original sensor reading exactly as received.

Prediction values are stored separately and NEVER replace:

    temperature
    pressure
    humidity
"""


import sqlite3
import os
import json
import re

from datetime import datetime, date

import pandas as pd
import numpy as np

try:
    from .sensor_registry import normalize_sensor_id
except Exception:
    from sensor_registry import normalize_sensor_id


# ============================================================
# DATABASE CONFIGURATION
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

DB_FILE = os.path.join(
    BASE_DIR,
    "skyguard.db"
)


# ============================================================
# DATABASE CONNECTION
# ============================================================

def get_connection():
    """
    Create and return a SQLite connection.
    """

    conn = sqlite3.connect(
        DB_FILE,
        timeout=30,
        check_same_thread=False
    )

    conn.row_factory = sqlite3.Row

    # --------------------------------------------------------
    # SQLite performance / concurrency settings
    # --------------------------------------------------------

    conn.execute(
        "PRAGMA journal_mode=WAL"
    )

    conn.execute(
        "PRAGMA synchronous=NORMAL"
    )

    conn.execute(
        "PRAGMA busy_timeout=30000"
    )

    return conn


# ============================================================
# DATABASE SCHEMA
# ============================================================

def create_database():
    """
    Create the database and perform safe schema migration.

    Existing databases are NOT deleted.

    If prediction columns are missing, they are added
    automatically.
    """

    conn = get_connection()

    try:

        cursor = conn.cursor()

        # ----------------------------------------------------
        # MAIN SENSOR TABLE
        # ----------------------------------------------------

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS sensor_readings (

                id INTEGER PRIMARY KEY AUTOINCREMENT,

                sensor_id TEXT NOT NULL,

                latitude REAL,
                longitude REAL,

                state TEXT,
                district TEXT,
                city TEXT,
                locality TEXT,

                temperature REAL,
                pressure REAL,
                humidity REAL,

                status TEXT,
                anomaly_category TEXT,
                fault_type TEXT,
                severity TEXT,

                confidence REAL,
                health REAL,

                observation_time TEXT,

                timestamp TEXT,
                actual_fetch_time TEXT,
                weather_model_time TEXT,
                api_status TEXT,
                api_error TEXT,
                explanation TEXT,
                evidence_summary TEXT,
                time_gap_seconds REAL,
                humidity_change_points REAL,

                created_at TEXT,

                -- ============================================
                -- PREDICTION INFORMATION
                -- ============================================

                suggested_value REAL,

                prediction_field TEXT,

                prediction_components TEXT,

                prediction_weights TEXT,

                prediction_confidence TEXT,

                prediction_neighbor_count INTEGER,

                seasonal_points_used INTEGER,

                seasonal_uses_day_of_year INTEGER,

                seasonal_uses_day_of_week INTEGER,

                seasonal_uses_month INTEGER,

                seasonal_history_span_days REAL,

                seasonal_strength REAL,

                seasonal_strength_label TEXT,

                trend_points_used INTEGER,

                trend_sequence TEXT,

                prediction_nearest_distance_km REAL,

                prediction_confidence_label TEXT,

                field_predictions TEXT
            )
            """
        )

        # ----------------------------------------------------
        # SAFE MIGRATION FOR EXISTING DATABASE
        #
        # CREATE TABLE IF NOT EXISTS does NOT add new columns
        # to an already existing SQLite table.
        #
        # Therefore we explicitly check and add them.
        # ----------------------------------------------------

        existing_columns = set()

        cursor.execute(
            "PRAGMA table_info(sensor_readings)"
        )

        for column in cursor.fetchall():

            existing_columns.add(
                column["name"]
            )

        # ----------------------------------------------------
        # Columns required by live API / observation metadata.
        # ----------------------------------------------------

        metadata_columns = {
            "timestamp": "TEXT",
            "actual_fetch_time": "TEXT",
            "weather_model_time": "TEXT",
            "api_status": "TEXT",
            "api_error": "TEXT",
            "explanation": "TEXT",
            "evidence_summary": "TEXT",
            "time_gap_seconds": "REAL",
            "humidity_change_points": "REAL",
        }

        for column_name, column_type in metadata_columns.items():
            if column_name not in existing_columns:
                try:
                    cursor.execute(
                        f"ALTER TABLE sensor_readings ADD COLUMN {column_name} {column_type}"
                    )
                    print(f"[DATABASE] Added column: {column_name}")
                except sqlite3.OperationalError as exc:
                    if "duplicate column" not in str(exc).lower():
                        raise

        # Columns required by the improved predictor.py
        # ----------------------------------------------------

        prediction_columns = {

            "suggested_value":
                "REAL",

            "prediction_field":
                "TEXT",

            "prediction_components":
                "TEXT",

            "prediction_weights":
                "TEXT",

            "prediction_confidence":
                "TEXT",

            "prediction_neighbor_count":
                "INTEGER",

            "seasonal_points_used":
                "INTEGER",

            "seasonal_uses_day_of_year":
                "INTEGER",

            "seasonal_uses_day_of_week":
                "INTEGER",

            "seasonal_uses_month":
                "INTEGER",

            "seasonal_history_span_days":
                "REAL",

            "seasonal_strength":
                "REAL",

            "seasonal_strength_label":
                "TEXT",

            "trend_points_used":
                "INTEGER",

            "trend_sequence":
                "TEXT",

            "prediction_nearest_distance_km":
                "REAL",

            "prediction_confidence_label":
                "TEXT",

            "field_predictions":
                "TEXT"
        }

        # ----------------------------------------------------
        # Add missing columns one by one.
        # ----------------------------------------------------

        for column_name, column_type in prediction_columns.items():

            if column_name not in existing_columns:

                try:

                    cursor.execute(
                        f"""
                        ALTER TABLE sensor_readings
                        ADD COLUMN {column_name}
                        {column_type}
                        """
                    )

                    print(
                        f"[DATABASE] Added column: "
                        f"{column_name}"
                    )

                except sqlite3.OperationalError as exc:

                    # If another process added the column
                    # simultaneously, do not crash the backend.

                    if "duplicate column" not in str(
                        exc
                    ).lower():

                        print(
                            f"[DATABASE WARNING] "
                            f"Could not add "
                            f"{column_name}: {exc}"
                        )

        # ----------------------------------------------------
        # INDEX: SENSOR + TIME
        # ----------------------------------------------------

        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS
            idx_sensor_observation

            ON sensor_readings(
                sensor_id,
                observation_time
            )
            """
        )

        # ----------------------------------------------------
        # INDEX: OBSERVATION TIME
        # ----------------------------------------------------

        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS
            idx_observation

            ON sensor_readings(
                observation_time
            )
            """
        )

        # ----------------------------------------------------
        # INDEX: SENSOR ID
        # ----------------------------------------------------

        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS
            idx_sensor

            ON sensor_readings(
                sensor_id
            )
            """
        )

        # ----------------------------------------------------
        # INDEX: STATUS
        # ----------------------------------------------------

        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS
            idx_status

            ON sensor_readings(
                status
            )
            """
        )

        # ----------------------------------------------------
        # INDEX: SENSOR + STATUS
        # ----------------------------------------------------

        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS
            idx_sensor_status

            ON sensor_readings(
                sensor_id,
                status
            )
            """
        )

        conn.commit()

    finally:

        conn.close()


# ============================================================
# SQLITE SAFE VALUE
# ============================================================

def sqlite_safe(value):
    """
    Convert Python / Pandas / NumPy values into SQLite-safe
    values.
    """

    if value is None:
        return None

    # --------------------------------------------------------
    # Pandas / NumPy missing values
    # --------------------------------------------------------

    try:

        missing = pd.isna(value)

        if isinstance(
            missing,
            (bool, np.bool_)
        ):

            if bool(missing):
                return None

    except Exception:

        pass

    # --------------------------------------------------------
    # Pandas Timestamp
    # --------------------------------------------------------

    if isinstance(
        value,
        pd.Timestamp
    ):

        return value.strftime(
            "%Y-%m-%d %H:%M:%S"
        )

    # --------------------------------------------------------
    # Python datetime
    # --------------------------------------------------------

    if isinstance(
        value,
        datetime
    ):

        return value.strftime(
            "%Y-%m-%d %H:%M:%S"
        )

    # --------------------------------------------------------
    # Python date
    # --------------------------------------------------------

    if isinstance(
        value,
        date
    ):

        return value.isoformat()

    # --------------------------------------------------------
    # NumPy integer
    # --------------------------------------------------------

    if isinstance(
        value,
        np.integer
    ):

        return int(value)

    # --------------------------------------------------------
    # NumPy float
    # --------------------------------------------------------

    if isinstance(
        value,
        np.floating
    ):

        return float(value)

    # --------------------------------------------------------
    # NumPy boolean
    # --------------------------------------------------------

    if isinstance(
        value,
        np.bool_
    ):

        return bool(value)

    return value


# ============================================================
# JSON SAFE VALUE
# ============================================================

def json_safe(value):
    """
    Convert dictionaries containing NumPy/Pandas values into
    JSON-safe structures.
    """

    if value is None:
        return None

    if isinstance(
        value,
        dict
    ):

        return {
            str(key): json_safe(val)
            for key, val in value.items()
        }

    if isinstance(
        value,
        (list, tuple)
    ):

        return [
            json_safe(item)
            for item in value
        ]

    if isinstance(
        value,
        np.integer
    ):

        return int(value)

    if isinstance(
        value,
        np.floating
    ):

        return float(value)

    if isinstance(
        value,
        np.bool_
    ):

        return bool(value)

    if isinstance(
        value,
        datetime
    ):

        return value.isoformat()

    if isinstance(
        value,
        date
    ):

        return value.isoformat()

    return value


# ============================================================
# JSON SERIALIZATION
# ============================================================

def json_string(value):
    """
    Safely convert a Python object to JSON text for SQLite.
    """

    if value is None:
        return None

    try:

        return json.dumps(
            json_safe(value),
            separators=(
                ",",
                ":"
            )
        )

    except Exception:

        return None


# ============================================================
# INSERT SQL
# ============================================================

INSERT_SQL = """
    INSERT INTO sensor_readings (

        sensor_id,

        latitude,
        longitude,

        state,
        district,
        city,
        locality,

        temperature,
        pressure,
        humidity,

        status,
        anomaly_category,
        fault_type,
        severity,

        confidence,
        health,

        observation_time,
        timestamp,
        actual_fetch_time,
        weather_model_time,
        api_status,
        api_error,
        explanation,
        evidence_summary,
        time_gap_seconds,
        humidity_change_points,
        created_at,

        -- ================================================
        -- PREDICTION DATA
        -- ================================================

        suggested_value,

        prediction_field,

        prediction_components,

        prediction_weights,

        prediction_confidence,

        prediction_neighbor_count,

        seasonal_points_used,

        seasonal_uses_day_of_year,

        seasonal_uses_day_of_week,

        seasonal_uses_month,

        seasonal_history_span_days,

        seasonal_strength,

        seasonal_strength_label,

        trend_points_used,

        trend_sequence,

        prediction_nearest_distance_km,

        prediction_confidence_label,

        field_predictions
    )

VALUES (
        ?,
        ?,
        ?,
        ?,
        ?,
        ?,
        ?,
        ?,
        ?,
        ?,
        ?,
        ?,
        ?,
        ?,
        ?,
        ?,
        ?,
        ?,
        ?,
        ?,
        ?,
        ?,
        ?,
        ?,
        ?,
        ?,
        ?,
        ?,
        ?,
        ?,
        ?,
        ?,
        ?,
        ?,
        ?,
        ?,
        ?,
        ?,
        ?,
        ?,
        ?,
        ?,
        ?,
        ?,
        ?
    )"""


# ============================================================
# RESULT -> SQL TUPLE
# ============================================================

def result_to_tuple(result):
    """
    Convert one anomaly/prediction result dictionary into the
    exact tuple required by INSERT_SQL.
    """

    if result is None:
        return None

    # --------------------------------------------------------
    # Prediction object
    # --------------------------------------------------------

    prediction = result.get(
        "prediction"
    )

    if not isinstance(
        prediction,
        dict
    ):

        prediction = {}

    # --------------------------------------------------------
    # Suggested value
    #
    # Prefer the top-level value added by main.py.
    # Fall back to prediction object.
    # --------------------------------------------------------

    suggested_value = result.get(
        "suggested_value"
    )

    if suggested_value is None:

        suggested_value = prediction.get(
            "suggested_value"
        )

    # --------------------------------------------------------
    # Prediction field
    # --------------------------------------------------------

    prediction_field = result.get(
        "prediction_field"
    )

    # --------------------------------------------------------
    # Prediction components
    # --------------------------------------------------------

    prediction_components = result.get(
        "prediction_components"
    )

    if prediction_components is None:

        prediction_components = prediction.get(
            "components"
        )

    # --------------------------------------------------------
    # Prediction weights
    # --------------------------------------------------------

    prediction_weights = result.get(
        "prediction_weights"
    )

    if prediction_weights is None:

        prediction_weights = prediction.get(
            "weights_used"
        )

    # --------------------------------------------------------
    # Prediction confidence
    # --------------------------------------------------------

    prediction_confidence = result.get(
        "prediction_confidence"
    )

    if prediction_confidence is None:

        prediction_confidence = prediction.get(
            "confidence"
        )

    # --------------------------------------------------------
    # Spatial neighbor count
    # --------------------------------------------------------

    prediction_neighbor_count = result.get(
        "prediction_neighbor_count"
    )

    if prediction_neighbor_count is None:

        prediction_neighbor_count = prediction.get(
            "spatial_neighbor_count",
            0
        )

    # --------------------------------------------------------
    # Seasonal points
    # --------------------------------------------------------

    seasonal_points_used = result.get(
        "seasonal_points_used"
    )

    if seasonal_points_used is None:

        seasonal_points_used = prediction.get(
            "seasonal_points_used",
            0
        )

    # --------------------------------------------------------
    # Day-of-year
    # --------------------------------------------------------

    seasonal_uses_day_of_year = result.get(
        "seasonal_uses_day_of_year"
    )

    if seasonal_uses_day_of_year is None:

        seasonal_uses_day_of_year = prediction.get(
            "seasonal_uses_day_of_year",
            False
        )

    # --------------------------------------------------------
    # Day-of-week
    # --------------------------------------------------------

    seasonal_uses_day_of_week = result.get(
        "seasonal_uses_day_of_week"
    )

    if seasonal_uses_day_of_week is None:

        seasonal_uses_day_of_week = prediction.get(
            "seasonal_uses_day_of_week",
            False
        )

    # --------------------------------------------------------
    # Month
    # --------------------------------------------------------

    seasonal_uses_month = result.get(
        "seasonal_uses_month"
    )

    if seasonal_uses_month is None:

        seasonal_uses_month = prediction.get(
            "seasonal_uses_month",
            False
        )

    # --------------------------------------------------------
    # Seasonal history span
    # --------------------------------------------------------

    seasonal_history_span_days = result.get(
        "seasonal_history_span_days"
    )

    if seasonal_history_span_days is None:

        seasonal_history_span_days = prediction.get(
            "seasonal_history_span_days",
            0
        )

    # --------------------------------------------------------
    # Seasonal strength
    # --------------------------------------------------------

    seasonal_strength = result.get(
        "seasonal_strength"
    )

    if seasonal_strength is None:

        seasonal_strength = prediction.get(
            "seasonal_strength",
            0.0
        )

    seasonal_strength_label = result.get(
        "seasonal_strength_label"
    )

    if seasonal_strength_label is None:

        seasonal_strength_label = prediction.get(
            "seasonal_strength_label"
        )

    prediction_nearest_distance_km = result.get(
        "prediction_nearest_distance_km"
    )

    if prediction_nearest_distance_km is None:

        prediction_nearest_distance_km = prediction.get(
            "nearest_distance_km"
        )

    prediction_confidence_label = result.get(
        "prediction_confidence_label"
    )

    if prediction_confidence_label is None:

        prediction_confidence_label = prediction.get(
            "confidence_label"
        )

    # --------------------------------------------------------
    # Trend points / sequence
    # --------------------------------------------------------

    trend_points_used = result.get(
        "trend_points_used"
    )

    if trend_points_used is None:

        trend_points_used = prediction.get(
            "trend_points_used",
            0
        )

    trend_sequence = result.get(
        "trend_sequence"
    )

    if trend_sequence is None:

        trend_sequence = prediction.get(
            "trend_sequence",
            []
        )

    # --------------------------------------------------------
    # FINAL SQLITE TUPLE
    # --------------------------------------------------------

    sensor_id = normalize_sensor_id(result.get("sensor_id", ""))

    timestamp = result.get("timestamp")
    observation_time = result.get("observation_time") or timestamp
    actual_fetch_time = result.get("actual_fetch_time") or timestamp
    weather_model_time = result.get("weather_model_time")
    api_status = result.get("api_status")
    api_error = result.get("api_error")
    explanation = result.get("explanation")
    evidence_summary = result.get("evidence_summary")
    time_gap_seconds = result.get("time_gap_seconds")
    humidity_change_points = result.get("humidity_change_points")

    return (

        # ====================================================
        # SENSOR INFORMATION
        # ====================================================

        sqlite_safe(
            sensor_id
        ),

        sqlite_safe(
            result.get(
                "latitude"
            )
        ),

        sqlite_safe(
            result.get(
                "longitude"
            )
        ),

        sqlite_safe(
            result.get(
                "state",
                ""
            )
        ),

        sqlite_safe(
            result.get(
                "district",
                ""
            )
        ),

        sqlite_safe(
            result.get(
                "city",
                ""
            )
        ),

        sqlite_safe(
            result.get(
                "locality",
                ""
            )
        ),

        # ====================================================
        # RAW SENSOR VALUES
        #
        # THESE ARE NEVER REPLACED BY PREDICTIONS.
        # ====================================================

        sqlite_safe(
            result.get(
                "temperature"
            )
        ),

        sqlite_safe(
            result.get(
                "pressure"
            )
        ),

        sqlite_safe(
            result.get(
                "humidity"
            )
        ),

        # ====================================================
        # ANOMALY RESULT
        # ====================================================

        sqlite_safe(
            result.get(
                "status",
                "NORMAL"
            )
        ),

        sqlite_safe(
            result.get(
                "anomaly_category",
                "NORMAL"
            )
        ),

        sqlite_safe(
            result.get(
                "fault_type"
            )
        ),

        sqlite_safe(
            result.get(
                "severity",
                "LOW"
            )
        ),

        # ====================================================
        # SCORE / HEALTH
        # ====================================================

        sqlite_safe(
            result.get(
                "confidence",
                0
            )
        ),

        sqlite_safe(
            result.get(
                "health",
                100
            )
        ),

        # ====================================================
        # TIME
        # ====================================================

        sqlite_safe(
            observation_time
        ),

        sqlite_safe(
            timestamp
        ),

        sqlite_safe(
            actual_fetch_time
        ),

        sqlite_safe(
            weather_model_time
        ),

        sqlite_safe(
            api_status
        ),

        sqlite_safe(
            api_error
        ),

        sqlite_safe(
            explanation
        ),

        sqlite_safe(
            evidence_summary
        ),

        sqlite_safe(
            time_gap_seconds
        ),

        sqlite_safe(
            humidity_change_points
        ),

        datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S"
        ),

        # ====================================================
        # PREDICTION
        # ====================================================

        sqlite_safe(
            suggested_value
        ),

        sqlite_safe(
            prediction_field
        ),

        json_string(
            prediction_components
        ),

        json_string(
            prediction_weights
        ),

        sqlite_safe(
            prediction_confidence
        ),

        sqlite_safe(
            prediction_neighbor_count
        ),

        # ====================================================
        # SEASONAL
        # ====================================================

        sqlite_safe(
            seasonal_points_used
        ),

        sqlite_safe(
            int(
                bool(
                    seasonal_uses_day_of_year
                )
            )
        ),

        sqlite_safe(
            int(
                bool(
                    seasonal_uses_day_of_week
                )
            )
        ),

        sqlite_safe(
            int(
                bool(
                    seasonal_uses_month
                )
            )
        ),

        sqlite_safe(
            seasonal_history_span_days
        ),

        sqlite_safe(
            seasonal_strength
        ),

        sqlite_safe(
            seasonal_strength_label
        ),

        # ====================================================
        # TREND
        # ====================================================

        sqlite_safe(
            trend_points_used
        ),

        json_string(
            trend_sequence
        ),

        sqlite_safe(
            prediction_nearest_distance_km
        ),

        sqlite_safe(
            prediction_confidence_label
        ),

        json_string(
            result.get("field_predictions")
            or prediction.get("field_predictions")
        )
    )


# ============================================================
# SAVE ONE RESULT
# ============================================================

def save_result(result):
    """
    Save one analysis result.
    """

    if result is None:
        return

    conn = get_connection()

    try:

        conn.execute(
            INSERT_SQL,
            result_to_tuple(result)
        )

        conn.commit()

    finally:

        conn.close()


# ============================================================
# SAVE COMPLETE BATCH
# ============================================================

def save_results(results):
    """
    Save a complete batch of analysis results in one transaction.
    """

    if not results:
        return

    conn = get_connection()

    try:

        values = [
            result_to_tuple(result)
            for result in results
            if result is not None
        ]

        if not values:
            return

        conn.executemany(
            INSERT_SQL,
            values
        )

        conn.commit()

    finally:

        conn.close()


# ============================================================
# SENSOR HISTORY
# ============================================================

def get_sensor_history(
    sensor_id,
    limit=100
):
    """
    Return recent history for one sensor.

    Newest observations are returned first.
    """

    conn = get_connection()

    try:

        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT *

            FROM sensor_readings

            WHERE sensor_id = ?

            ORDER BY observation_time DESC

            LIMIT ?
            """,
            (
                normalize_sensor_id(sensor_id),
                int(limit)
            )
        )

        rows = cursor.fetchall()

        return [
            dict(row)
            for row in rows
        ]

    finally:

        conn.close()


# ============================================================
# SENSOR HISTORY BEFORE SPECIFIC TIME
# ============================================================

def get_sensor_history_before(
    sensor_id,
    observation_time,
    limit=100
):
    """
    Return historical observations before a specific time.
    """

    conn = get_connection()

    try:

        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT *

            FROM sensor_readings

            WHERE sensor_id = ?

            AND observation_time < ?

            ORDER BY observation_time DESC

            LIMIT ?
            """,
            (
                str(sensor_id),
                str(observation_time),
                int(limit)
            )
        )

        rows = cursor.fetchall()

        return [
            dict(row)
            for row in rows
        ]

    finally:

        conn.close()


# ============================================================
# LATEST SENSOR READING
# ============================================================

def get_latest_sensor_reading(
    sensor_id
):
    """
    Return the latest stored observation for a sensor.
    """

    conn = get_connection()

    try:

        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT *

            FROM sensor_readings

            WHERE sensor_id = ?

            ORDER BY observation_time DESC

            LIMIT 1
            """,
            (
                str(sensor_id),
            )
        )

        row = cursor.fetchone()

        if row is None:
            return None

        return dict(row)

    finally:

        conn.close()


# ============================================================
# ALL SENSOR IDS
# ============================================================

def get_sensor_ids():
    """
    Return all unique sensor IDs.
    """

    conn = get_connection()

    try:

        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT DISTINCT sensor_id

            FROM sensor_readings

            ORDER BY sensor_id
            """
        )

        rows = cursor.fetchall()

        return [
            row["sensor_id"]
            for row in rows
        ]

    finally:

        conn.close()


# ============================================================
# SENSOR COUNT
# ============================================================

def get_sensor_count():
    """
    Return number of unique sensors.
    """

    conn = get_connection()

    try:

        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT COUNT(
                DISTINCT sensor_id
            ) AS count

            FROM sensor_readings
            """
        )

        row = cursor.fetchone()

        return int(
            row["count"]
        )

    finally:

        conn.close()


# ============================================================
# GENERAL HISTORY
# ============================================================

def get_history(
    limit=100
):
    """
    Return recent analysis history.
    """

    conn = get_connection()

    try:

        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT *

            FROM sensor_readings

            ORDER BY observation_time DESC

            LIMIT ?
            """,
            (
                int(limit),
            )
        )

        rows = cursor.fetchall()

        return [
            dict(row)
            for row in rows
        ]

    finally:

        conn.close()


# ============================================================
# PREDICTION HISTORY
# ============================================================

def get_prediction_history(
    sensor_id=None,
    limit=100
):
    """
    Return stored prediction information.

    This is useful for checking whether seasonal prediction
    is actually being generated and persisted.
    """

    conn = get_connection()

    try:

        cursor = conn.cursor()

        if sensor_id is None:

            cursor.execute(
                """
                SELECT
                    id,
                    sensor_id,
                    observation_time,

                    temperature,
                    pressure,
                    humidity,

                    status,

                    suggested_value,
                    prediction_field,
                    prediction_components,
                    prediction_weights,
                    prediction_confidence,
                    prediction_confidence_label,

                    prediction_neighbor_count,

                    seasonal_points_used,
                    seasonal_uses_day_of_year,
                    seasonal_uses_day_of_week,
                    seasonal_uses_month,
                    seasonal_history_span_days,
                    seasonal_strength,
                    seasonal_strength_label,

                    trend_points_used,
                    trend_sequence,
                    prediction_nearest_distance_km

                FROM sensor_readings

                WHERE suggested_value IS NOT NULL

                ORDER BY observation_time DESC, id DESC

                LIMIT ?
                """,
                (
                    int(limit),
                )
            )

        else:

            cursor.execute(
                """
                SELECT
                    id,
                    sensor_id,
                    observation_time,

                    temperature,
                    pressure,
                    humidity,

                    status,

                    suggested_value,
                    prediction_field,
                    prediction_components,
                    prediction_weights,
                    prediction_confidence,
                    prediction_confidence_label,

                    prediction_neighbor_count,

                    seasonal_points_used,
                    seasonal_uses_day_of_year,
                    seasonal_uses_day_of_week,
                    seasonal_uses_month,
                    seasonal_history_span_days,
                    seasonal_strength,
                    seasonal_strength_label,

                    trend_points_used,
                    trend_sequence,
                    prediction_nearest_distance_km

                FROM sensor_readings

                WHERE sensor_id = ?

                AND suggested_value IS NOT NULL

                ORDER BY observation_time DESC, id DESC

                LIMIT ?
                """,
                (
                    normalize_sensor_id(sensor_id),
                    int(limit)
                )
            )

        rows = cursor.fetchall()

        return [
            dict(row)
            for row in rows
        ]

    finally:

        conn.close()


# ============================================================
# CLEAR DATABASE
# ============================================================

def clear_database():
    """
    Delete all sensor history.

    WARNING:
        This permanently deletes stored readings.
    """

    conn = get_connection()

    try:

        conn.execute(
            "DELETE FROM sensor_readings"
        )

        # ----------------------------------------------------
        # Reset AUTOINCREMENT counter.
        # ----------------------------------------------------

        conn.execute(
            """
            DELETE FROM sqlite_sequence
            WHERE name = 'sensor_readings'
            """
        )

        conn.commit()

    finally:

        conn.close()

    print(
        "DATABASE CLEARED"
    )


# ============================================================
# DATABASE INFORMATION
# ============================================================

def get_database_info():
    """
    Return basic database statistics.
    """

    conn = get_connection()

    try:

        cursor = conn.cursor()

        # ----------------------------------------------------
        # Total records
        # ----------------------------------------------------

        cursor.execute(
            """
            SELECT COUNT(*) AS count
            FROM sensor_readings
            """
        )

        total_records = int(
            cursor.fetchone()["count"]
        )

        # ----------------------------------------------------
        # Unique sensors
        # ----------------------------------------------------

        cursor.execute(
            """
            SELECT COUNT(
                DISTINCT sensor_id
            ) AS count

            FROM sensor_readings
            """
        )

        total_sensors = int(
            cursor.fetchone()["count"]
        )

        # ----------------------------------------------------
        # Stored predictions
        # ----------------------------------------------------

        cursor.execute(
            """
            SELECT COUNT(*) AS count

            FROM sensor_readings

            WHERE suggested_value IS NOT NULL
            """
        )

        prediction_count = int(
            cursor.fetchone()["count"]
        )

        # ----------------------------------------------------
        # Anomalies
        # ----------------------------------------------------

        cursor.execute(
            """
            SELECT COUNT(*) AS count

            FROM sensor_readings

            WHERE status = 'SENSOR_ANOMALY'
            """
        )

        anomaly_count = int(
            cursor.fetchone()["count"]
        )

        return {
            "database_file": DB_FILE,
            "total_records": total_records,
            "total_sensors": total_sensors,
            "prediction_records": prediction_count,
            "sensor_anomaly_records": anomaly_count
        }

    finally:

        conn.close()


# ============================================================
# INITIALIZE DATABASE
# ============================================================

create_database()


# ============================================================
# STARTUP MESSAGE
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("          SKYGUARD AI DATABASE")
    print("=" * 60)

    print(
        f"Database: {DB_FILE}"
    )

    info = get_database_info()

    print(
        f"Records : {info['total_records']}"
    )

    print(
        f"Sensors : {info['total_sensors']}"
    )

    print(
        f"Predictions: {info['prediction_records']}"
    )

    print(
        f"Anomalies  : {info['sensor_anomaly_records']}"
    )

    print("=" * 60)
 