import pandas as pd


# ============================================================
# SKYGUARD AI
# DATA PREPROCESSING
# ============================================================

def clean_sensor_data(df):

    if df is None:
        raise ValueError("No data received.")

    if not isinstance(df, pd.DataFrame):

        df = pd.DataFrame(df)

    df = df.copy()

    # ========================================================
    # REQUIRED WEATHER/SENSOR COLUMNS
    # ========================================================

    required_columns = [
        "sensor_id",
        "latitude",
        "longitude",
        "temperature",
        "pressure",
        "humidity"
    ]

    for column in required_columns:

        if column not in df.columns:

            raise ValueError(
                f"Missing column: {column}"
            )

    # ========================================================
    # LOCATION COLUMNS
    # ========================================================

    location_columns = [
        "state",
        "district",
        "city",
        "locality"
    ]

    for column in location_columns:

        if column not in df.columns:

            df[column] = ""

    # ========================================================
    # TIMESTAMP
    # ========================================================

    if "observation_time" in df.columns:

        timestamp_column = "observation_time"

    elif "timestamp" in df.columns:

        timestamp_column = "timestamp"

    else:

        raise ValueError(
            "Missing timestamp column. "
            "CSV must contain timestamp "
            "or observation_time."
        )

    # ========================================================
    # SENSOR ID
    # ========================================================

    df["sensor_id"] = (
        df["sensor_id"]
        .astype(str)
        .str.strip()
    )

    # ========================================================
    # GPS
    # ========================================================

    df["latitude"] = pd.to_numeric(
        df["latitude"],
        errors="coerce"
    )

    df["longitude"] = pd.to_numeric(
        df["longitude"],
        errors="coerce"
    )

    # ========================================================
    # WEATHER
    # ========================================================

    df["temperature"] = pd.to_numeric(
        df["temperature"],
        errors="coerce"
    )

    df["pressure"] = pd.to_numeric(
        df["pressure"],
        errors="coerce"
    )

    df["humidity"] = pd.to_numeric(
        df["humidity"],
        errors="coerce"
    )

    # ========================================================
    # LOCATION TEXT
    # ========================================================

    for column in location_columns:

        df[column] = (
            df[column]
            .fillna("")
            .astype(str)
            .str.strip()
        )

    # ========================================================
    # LIVE API STATUS / ERROR
    # ========================================================

    if "api_status" not in df.columns:
        df["api_status"] = "API_OK"

    if "api_error" not in df.columns:
        df["api_error"] = ""

    df["api_status"] = (
        df["api_status"]
        .fillna("UNKNOWN")
        .astype(str)
        .str.strip()
        .str.upper()
    )

    df["api_error"] = (
        df["api_error"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    # ========================================================
    # OBSERVATION TIME
    # ========================================================

    df["observation_time"] = pd.to_datetime(
        df[timestamp_column],
        errors="coerce",
        utc=True
    )

    # ========================================================
    # TIMESTAMP STRING
    # ========================================================

    df["timestamp"] = (
        df["observation_time"]
        .dt.strftime(
            "%Y-%m-%d %H:%M:%S UTC"
        )
    )

    # ========================================================
    # REMOVE INVALID DATA
    # ========================================================

    # IMPORTANT:
    # Do NOT drop rows because temperature/pressure/humidity are missing.
    # Missing weather fields are meaningful telemetry states and must reach
    # anomaly.py so it can classify DATA_UNAVAILABLE / missing data.
    df = df.dropna(
        subset=[
            "sensor_id",
            "latitude",
            "longitude",
            "observation_time"
        ]
    )

    # ========================================================
    # SORT
    # ========================================================

    df = df.sort_values(
        [
            "sensor_id",
            "observation_time"
        ]
    ).reset_index(
        drop=True
    )

    # ========================================================
    # FINAL COLUMN ORDER
    # ========================================================

    final_columns = [

        "sensor_id",

        "latitude",
        "longitude",

        "state",
        "district",
        "city",
        "locality",

        "temperature",
        "pressure",
        "humidity",

        "timestamp",
        "observation_time",
        "api_status",
        "api_error"
    ]

    df = df[
        final_columns
    ]

    # ========================================================
    # DEBUG
    # ========================================================

    print("\n========================================")
    print("       CLEANED SENSOR DATA")
    print("========================================")

    print(
        df.to_string(
            index=False
        )
    )

    print("\nColumns passed to anomaly detector:")

    print(
        list(df.columns)
    )

    return df