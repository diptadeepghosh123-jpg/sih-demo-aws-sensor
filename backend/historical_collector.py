import os
import time
import pandas as pd
from datetime import datetime, timezone

from weather_api import get_sensor_data


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

CSV_FILE = os.path.join(
    BASE_DIR,
    "historical_weather_data.csv"
)

# Exact collection interval
FETCH_INTERVAL_SECONDS = 600


# ============================================================
# CSV COLUMNS
# ============================================================

CSV_COLUMNS = [
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
    "observation_time"
]


# ============================================================
# CHECK CSV
# ============================================================

def create_csv_if_needed():

    # --------------------------------------------------------
    # IMPORTANT:
    # If the CSV already exists, DO NOTHING.
    # Existing historical data will remain untouched.
    # --------------------------------------------------------

    if os.path.exists(CSV_FILE):

        print("Existing historical CSV found.")
        print("Using existing file:")
        print(CSV_FILE)

        return

    # --------------------------------------------------------
    # Only create the CSV if it does NOT exist.
    # --------------------------------------------------------

    df = pd.DataFrame(columns=CSV_COLUMNS)

    df.to_csv(
        CSV_FILE,
        index=False
    )

    print("New historical CSV created:")
    print(CSV_FILE)


# ============================================================
# FETCH LIVE DATA
# ============================================================

def fetch_live_data():

    try:

        print("Fetching live data...")

        data = get_sensor_data()

        if data is None:

            print("No data received.")

            return []

        if isinstance(data, pd.DataFrame):

            return data.to_dict("records")

        if isinstance(data, list):

            return data

        if isinstance(data, dict):

            return [data]

        print("Unknown data format.")

        return []

    except Exception as e:

        print()
        print("Error while fetching live data:")
        print(e)

        return []


# ============================================================
# PREPARE DATA
# ============================================================

def prepare_data(data):

    if not data:

        return pd.DataFrame(
            columns=CSV_COLUMNS
        )

    rows = []

    # --------------------------------------------------------
    # One collection timestamp for this complete batch
    # --------------------------------------------------------

    collection_time = datetime.now(
        timezone.utc
    )

    timestamp = collection_time.strftime(
        "%Y-%m-%d %H:%M:%S UTC"
    )

    # --------------------------------------------------------
    # Create one row for every sensor
    # --------------------------------------------------------

    for item in data:

        row = {}

        for column in CSV_COLUMNS:

            row[column] = item.get(
                column,
                None
            )

        # ----------------------------------------------------
        # IMPORTANT:
        # Use collector time as observation time.
        # ----------------------------------------------------

        row["timestamp"] = timestamp

        row["observation_time"] = timestamp

        rows.append(row)

    df = pd.DataFrame(
        rows,
        columns=CSV_COLUMNS
    )

    return df


# ============================================================
# APPEND DATA TO EXISTING CSV
# ============================================================

def append_to_csv(df):

    if df.empty:

        print("Nothing to save.")

        return

    # --------------------------------------------------------
    # IMPORTANT:
    # mode="a" means APPEND.
    #
    # Existing rows are NOT deleted.
    # --------------------------------------------------------

    df.to_csv(
        CSV_FILE,
        mode="a",
        header=False,
        index=False
    )

    print()
    print("Data appended successfully.")
    print("Rows added:", len(df))
    print(
        "Timestamp:",
        df.iloc[0]["timestamp"]
    )
    print(
        "CSV:",
        CSV_FILE
    )


# ============================================================
# ONE COLLECTION CYCLE
# ============================================================

def collect_once():

    print()
    print("=" * 60)

    current_time = datetime.now(
        timezone.utc
    )

    print(
        "Collection started:",
        current_time.strftime(
            "%Y-%m-%d %H:%M:%S UTC"
        )
    )

    print("=" * 60)

    # --------------------------------------------------------
    # Fetch live data
    # --------------------------------------------------------

    data = fetch_live_data()

    if not data:

        print(
            "Collection failed or returned no data."
        )

        return False

    # --------------------------------------------------------
    # Prepare rows
    # --------------------------------------------------------

    df = prepare_data(data)

    # --------------------------------------------------------
    # Append rows
    # --------------------------------------------------------

    append_to_csv(df)

    print()
    print(
        "Sensors collected:",
        len(df)
    )

    return True


# ============================================================
# CONTINUOUS COLLECTOR
# ============================================================

def run_collector():

    # --------------------------------------------------------
    # Check/create CSV ONCE.
    #
    # If it already exists, it is NOT recreated.
    # --------------------------------------------------------

    create_csv_if_needed()

    print()
    print("=" * 60)
    print(
        "       SKYGUARD HISTORICAL DATA COLLECTOR"
    )
    print("=" * 60)

    print()
    print("CSV:")
    print(CSV_FILE)

    print()
    print(
        "Collection interval: EXACTLY 10 MINUTES (600 seconds)"
    )

    print()
    print(
        "Press CTRL + C to stop."
    )

    print()

    # --------------------------------------------------------
    # First collection immediately
    # --------------------------------------------------------

    next_run = time.time()

    while True:

        try:

            # ------------------------------------------------
            # Wait until scheduled time
            # ------------------------------------------------

            current_time = time.time()

            if current_time < next_run:

                wait_time = (
                    next_run - current_time
                )

                print(
                    f"Waiting {wait_time:.1f} seconds..."
                )

                time.sleep(wait_time)

            # ------------------------------------------------
            # Collect
            # ------------------------------------------------

            collect_once()

            # ------------------------------------------------
            # Schedule next collection exactly 10 minutes
            # after the scheduled collection time.
            # ------------------------------------------------

            next_run += FETCH_INTERVAL_SECONDS

            # ------------------------------------------------
            # If an API request took too long and the scheduled
            # time has already passed, schedule one new cycle
            # instead of running multiple collections together.
            # ------------------------------------------------

            if next_run <= time.time():

                next_run = (
                    time.time()
                    + FETCH_INTERVAL_SECONDS
                )

            print()
            print(
                "Next collection scheduled for:",
                datetime.fromtimestamp(
                    next_run,
                    timezone.utc
                ).strftime(
                    "%Y-%m-%d %H:%M:%S UTC"
                )
            )

            print()

        except KeyboardInterrupt:

            print()
            print("=" * 60)
            print(
                "Collector stopped by user."
            )
            print("=" * 60)

            break

        except Exception as e:

            print()
            print(
                "Unexpected error:"
            )
            print(e)

            print()
            print(
                "Retrying after 10 minutes..."
            )

            next_run = (
                time.time()
                + FETCH_INTERVAL_SECONDS
            )


# ============================================================
# PROGRAM START
# ============================================================

if __name__ == "__main__":

    run_collector()