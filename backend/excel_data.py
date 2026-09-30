import os
from datetime import datetime, timezone

import pandas as pd


# ============================================================
# SKYGUARD AI
# EXCEL DATA SOURCE
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Put your Excel file inside:
# backend/data/skyguard_data.xlsx
DATA_DIR = os.path.join(BASE_DIR, "data")

EXCEL_FILE = os.path.join(
    DATA_DIR,
    "SkyGuard_1000_Rows_5_Sensors_Test_Data.xlsx"
)


# ============================================================
# INTERNAL STATE
# ============================================================

# Current batch position.
# Each call to get_sensor_data() returns the next batch.
_current_batch = 0


# ============================================================
# COLUMN HELPERS
# ============================================================

def normalize_column_name(column):
    """
    Convert Excel column names into a common format.
    """

    return (
        str(column)
        .strip()
        .lower()
        .replace(" ", "_")
        .replace("-", "_")
        .replace("/", "_")
        .replace("(", "")
        .replace(")", "")
        .replace("%", "percent")
    )


def find_column(df, possible_names):
    """
    Find a column using multiple possible Excel column names.
    """

    normalized = {
        normalize_column_name(column): column
        for column in df.columns
    }

    for name in possible_names:

        key = normalize_column_name(name)

        if key in normalized:
            return normalized[key]

    return None


# ============================================================
# VALUE CONVERSION
# ============================================================

def safe_float(value):
    """
    Safely convert a value to float.
    """

    if pd.isna(value):
        return None

    try:
        return float(value)
    except (ValueError, TypeError):
        return None


def clean_text(value):
    """
    Safely convert text fields.
    """

    if pd.isna(value):
        return ""

    return str(value).strip()


# ============================================================
# TIMESTAMP HANDLING
# ============================================================

def convert_timestamp(value):
    """
    Convert Excel timestamp into the format used by SkyGuard.
    """

    if pd.isna(value):
        return None

    try:

        # Already a datetime
        if isinstance(value, datetime):

            dt = value

        else:

            dt = pd.to_datetime(
                value,
                errors="coerce"
            )

            if pd.isna(dt):
                return None

            # pandas Timestamp -> Python datetime
            if hasattr(dt, "to_pydatetime"):
                dt = dt.to_pydatetime()

        # Make timezone aware
        if dt.tzinfo is None:

            dt = dt.replace(
                tzinfo=timezone.utc
            )

        else:

            dt = dt.astimezone(
                timezone.utc
            )

        return dt.strftime(
            "%Y-%m-%d %H:%M:%S UTC"
        )

    except Exception:

        return None


# ============================================================
# LOAD EXCEL
# ============================================================

def load_excel():

    if not os.path.isfile(EXCEL_FILE):

        raise FileNotFoundError(
            f"Excel file not found:\n{EXCEL_FILE}\n\n"
            "Place your Excel file inside backend/data/"
        )

    print("\n========================================")
    print("       SKYGUARD AI EXCEL DATA")
    print("========================================")

    print(
        f"Reading Excel:\n{EXCEL_FILE}"
    )

    df = pd.read_excel(
        EXCEL_FILE
    )

    if df.empty:

        raise ValueError(
            "The Excel file contains no data."
        )

    print(
        f"Excel rows loaded: {len(df)}"
    )

    print(
        "\nExcel columns:"
    )

    for column in df.columns:
        print(
            f" - {column}"
        )

    return df


# ============================================================
# CONVERT EXCEL ROW
# ============================================================

def convert_excel_row(
    row,
    columns
):

    # --------------------------------------------------------
    # SENSOR ID
    # --------------------------------------------------------

    sensor_id_column = columns["sensor_id"]

    sensor_id = clean_text(
        row[sensor_id_column]
    )

    if not sensor_id:

        sensor_id = "UNKNOWN"


    # --------------------------------------------------------
    # LOCATION
    # --------------------------------------------------------

    latitude = safe_float(
        row[columns["latitude"]]
    )

    longitude = safe_float(
        row[columns["longitude"]]
    )


    state = clean_text(
        row[columns["state"]]
    )

    district = clean_text(
        row[columns["district"]]
    )

    city = clean_text(
        row[columns["city"]]
    )

    locality = clean_text(
        row[columns["locality"]]
    )


    # --------------------------------------------------------
    # WEATHER VALUES
    # --------------------------------------------------------

    temperature = safe_float(
        row[columns["temperature"]]
    )

    pressure = safe_float(
        row[columns["pressure"]]
    )

    humidity = safe_float(
        row[columns["humidity"]]
    )


    # --------------------------------------------------------
    # TIME
    # --------------------------------------------------------

    timestamp = None

    if columns["timestamp"] is not None:

        timestamp = convert_timestamp(
            row[columns["timestamp"]]
        )


    # --------------------------------------------------------
    # TIME DIFFERENCE
    # --------------------------------------------------------

    time_difference = None

    if columns["time_difference"] is not None:

        time_difference = row[
            columns["time_difference"]
        ]

        if pd.isna(time_difference):

            time_difference = None

        else:

            try:
                time_difference = float(
                    time_difference
                )
            except (ValueError, TypeError):
                pass


    # --------------------------------------------------------
    # FALLBACK TIMESTAMP
    # --------------------------------------------------------

    if timestamp is None:

        timestamp = datetime.now(
            timezone.utc
        ).strftime(
            "%Y-%m-%d %H:%M:%S UTC"
        )


    # --------------------------------------------------------
    # CREATE SKYGUARD RESULT
    # --------------------------------------------------------

    result = {

        # Sensor
        "sensor_id": sensor_id,

        # GPS
        "latitude": latitude,
        "longitude": longitude,

        # Location
        "state": state,
        "district": district,
        "city": city,
        "locality": locality,

        # Weather
        "temperature": temperature,
        "pressure": pressure,
        "humidity": humidity,

        # Time
        "timestamp": timestamp,
        "observation_time": timestamp,
        "weather_model_time": timestamp,
        "actual_fetch_time": timestamp,

        # Excel source
        "api_status": "EXCEL_OK",
        "api_error": "",

        # Preserve time difference if present
        "time_difference": time_difference
    }

    return result


# ============================================================
# FIND EXCEL COLUMNS
# ============================================================

def detect_columns(df):

    columns = {

        "sensor_id": find_column(
            df,
            [
                "sensor_id",
                "sensor",
                "sensorid",
                "id"
            ]
        ),

        "latitude": find_column(
            df,
            [
                "latitude",
                "lat"
            ]
        ),

        "longitude": find_column(
            df,
            [
                "longitude",
                "long",
                "lon",
                "lng"
            ]
        ),

        "state": find_column(
            df,
            [
                "state"
            ]
        ),

        "district": find_column(
            df,
            [
                "district"
            ]
        ),

        "city": find_column(
            df,
            [
                "city"
            ]
        ),

        "locality": find_column(
            df,
            [
                "locality",
                "location",
                "area"
            ]
        ),

        "temperature": find_column(
            df,
            [
                "temperature",
                "temperature_c",
                "temp",
                "temp_c"
            ]
        ),

        "pressure": find_column(
            df,
            [
                "pressure",
                "pressure_hpa",
                "press",
                "hpa"
            ]
        ),

        "humidity": find_column(
            df,
            [
                "humidity",
                "relative_humidity",
                "relative_humidity_percent",
                "rh"
            ]
        ),

        "timestamp": find_column(
            df,
            [
                "timestamp",
                "time",
                "datetime",
                "date_time",
                "observation_time",
                "observation_datetime"
            ]
        ),

        "time_difference": find_column(
            df,
            [
                "time_difference",
                "time_difference_seconds",
                "time_diff",
                "time_gap",
                "time_gap_seconds",
                "difference"
            ]
        )
    }


    # --------------------------------------------------------
    # REQUIRED COLUMNS
    # --------------------------------------------------------

    required = [
        "sensor_id",
        "latitude",
        "longitude",
        "temperature",
        "pressure",
        "humidity"
    ]

    missing = [
        name
        for name in required
        if columns[name] is None
    ]

    if missing:

        raise ValueError(
            "Required Excel columns are missing:\n"
            + "\n".join(
                f" - {item}"
                for item in missing
            )
        )

    return columns


# ============================================================
# CREATE BATCHES
# ============================================================

def create_batches(df):

    # --------------------------------------------------------
    # If timestamp exists, use timestamp as the batch.
    # This preserves the historical time sequence.
    # --------------------------------------------------------

    timestamp_column = find_column(
        df,
        [
            "timestamp",
            "time",
            "datetime",
            "date_time",
            "observation_time",
            "observation_datetime"
        ]
    )

    if timestamp_column is not None:

        temp = df.copy()

        temp["_skyguard_timestamp"] = pd.to_datetime(
            temp[timestamp_column],
            errors="coerce"
        )

        valid_timestamp = (
            temp["_skyguard_timestamp"]
            .notna()
        )

        if valid_timestamp.any():

            temp = temp.sort_values(
                "_skyguard_timestamp"
            )

            groups = []

            for _, group in temp.groupby(
                "_skyguard_timestamp",
                sort=True
            ):

                groups.append(
                    group.drop(
                        columns=[
                            "_skyguard_timestamp"
                        ]
                    )
                )

            if groups:

                return groups


    # --------------------------------------------------------
    # Fallback:
    # If no usable timestamp exists, use sensor count.
    #
    # Example:
    # 5 sensors
    # rows 1-5   -> batch 1
    # rows 6-10  -> batch 2
    # rows 11-15 -> batch 3
    # --------------------------------------------------------

    sensor_column = find_column(
        df,
        [
            "sensor_id",
            "sensor",
            "sensorid",
            "id"
        ]
    )

    if sensor_column is None:

        return [df]

    sensor_count = (
        df[sensor_column]
        .nunique()
    )

    if sensor_count <= 0:

        return [df]

    batches = []

    for start in range(
        0,
        len(df),
        sensor_count
    ):

        batch = df.iloc[
            start:start + sensor_count
        ]

        if not batch.empty:

            batches.append(
                batch
            )

    return batches


# ============================================================
# GET SENSOR DATA
# ============================================================

def get_sensor_data():

    global _current_batch

    try:

        # ----------------------------------------------------
        # LOAD EXCEL
        # ----------------------------------------------------

        df = load_excel()


        # ----------------------------------------------------
        # DETECT COLUMNS
        # ----------------------------------------------------

        columns = detect_columns(
            df
        )


        # ----------------------------------------------------
        # CREATE HISTORICAL BATCHES
        # ----------------------------------------------------

        batches = create_batches(
            df
        )

        if not batches:

            print(
                "No usable batches found."
            )

            return pd.DataFrame()


        # ----------------------------------------------------
        # SELECT CURRENT BATCH
        # ----------------------------------------------------

        if _current_batch >= len(batches):

            # Restart from beginning after reaching
            # the end of the Excel dataset.
            _current_batch = 0


        current_df = batches[
            _current_batch
        ].copy()


        print("\n========================================")
        print(
            f"Excel batch "
            f"{_current_batch + 1}/{len(batches)}"
        )
        print(
            f"Rows: {len(current_df)}"
        )
        print("========================================")


        # ----------------------------------------------------
        # MOVE TO NEXT BATCH
        # ----------------------------------------------------

        _current_batch += 1


        # ----------------------------------------------------
        # CONVERT ROWS
        # ----------------------------------------------------

        results = []

        for _, row in current_df.iterrows():

            try:

                result = convert_excel_row(
                    row,
                    columns
                )

                results.append(
                    result
                )

            except Exception as error:

                print(
                    "\nExcel row processing error:"
                )

                print(error)


        if not results:

            return pd.DataFrame()


        # ----------------------------------------------------
        # CREATE DATAFRAME
        # ----------------------------------------------------

        result_df = pd.DataFrame(
            results
        )


        # ----------------------------------------------------
        # KEEP THE SAME MAIN COLUMNS AS WEATHER API
        # ----------------------------------------------------

        preferred_columns = [

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
            "weather_model_time",
            "actual_fetch_time",

            "api_status",
            "api_error",

            "time_difference"
        ]


        existing_columns = [
            column
            for column in preferred_columns
            if column in result_df.columns
        ]


        result_df = result_df[
            existing_columns
        ]


        # ----------------------------------------------------
        # DISPLAY
        # ----------------------------------------------------

        print("\n========================================")
        print("       EXCEL SENSOR DATA")
        print("========================================")

        print(
            result_df.to_string(
                index=False
            )
        )


        return result_df


    except FileNotFoundError as error:

        print(
            f"\nERROR: {error}"
        )

        return pd.DataFrame()


    except Exception as error:

        print(
            "\nERROR reading Excel data:"
        )

        print(error)

        return pd.DataFrame()


# ============================================================
# RESET EXCEL REPLAY
# ============================================================

def reset_excel_data():

    global _current_batch

    _current_batch = 0

    print(
        "SkyGuard Excel replay reset."
    )


# ============================================================
# DIRECT TEST
# ============================================================

if __name__ == "__main__":

    df = get_sensor_data()

    if not df.empty:

        print("\n========================================")
        print("       EXCEL TEST COMPLETE")
        print("========================================")

        print(
            f"\nSensors in current batch: {len(df)}"
        )

        print("\nFinal columns:")

        for column in df.columns:

            print(
                f" - {column}"
            )