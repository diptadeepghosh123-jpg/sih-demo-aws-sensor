import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone

import pandas as pd
import requests
from dotenv import load_dotenv

from sensor_registry import get_all_sensors


# ============================================================
# SKYGUARD AI
# WEATHER API
# ============================================================

# Load configuration from the places users commonly keep the project .env.
# A non-blank process environment variable always wins.
# Otherwise, try backend/.env, project/.env, then the current working directory.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)
ENV_FILES = [
    os.path.join(BASE_DIR, ".env"),
    os.path.join(PROJECT_DIR, ".env"),
    os.path.join(os.getcwd(), ".env"),
]

_process_key = (os.getenv("WINDY_API_KEY") or "").strip()
if not _process_key:
    for _env_file in ENV_FILES:
        if not os.path.isfile(_env_file):
            continue
        load_dotenv(_env_file, override=True)
        if (os.getenv("WINDY_API_KEY") or "").strip():
            break

WINDY_API_KEY = (os.getenv("WINDY_API_KEY") or "").strip().strip('"').strip("'")

if WINDY_API_KEY:
    print("WINDY_API_KEY: loaded")
else:
    print("WINDY_API_KEY: missing (set it in environment or backend/.env)")

WINDY_URL = "https://api.windy.com/api/point-forecast/v2"

WINDY_MODEL = "gfs"


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def get_first_value(data, key):
    """
    Get the first value from a Windy API array.
    """

    if key not in data:
        return None

    value = data[key]

    if isinstance(value, list):

        if len(value) == 0:
            return None

        return value[0]

    return value


def convert_temperature_to_celsius(value, unit):
    """
    Convert Windy temperature to Celsius.
    """

    if value is None:
        return None

    value = float(value)

    unit = str(unit).lower()

    if unit in ["k", "kelvin"]:
        return value - 273.15

    if unit in ["c", "°c", "degc", "celsius"]:
        return value

    # Safety fallback:
    # The values returned by your current API are Kelvin.
    return value - 273.15


def convert_pressure_to_hpa(value, unit):
    """
    Convert Windy pressure to hPa.
    """

    if value is None:
        return None

    value = float(value)

    unit = str(unit).lower()

    if unit in ["pa", "pascal", "pascals"]:
        return value / 100.0

    if unit in ["hpa", "mb", "mbar"]:
        return value

    # Safety fallback:
    # Your current API is returning pressure in Pa.
    return value / 100.0


def convert_humidity(value, unit):
    """
    Convert relative humidity to percentage.
    """

    if value is None:
        return None

    value = float(value)

    unit = str(unit).lower()

    if unit in ["%", "percent", "percentage"]:
        return value

    # Windy normally returns RH as percentage.
    return value


def convert_timestamp(timestamp):
    """
    Convert Windy timestamp in milliseconds
    to readable UTC time.
    """

    if timestamp is None:
        return None

    try:

        timestamp = float(timestamp)

        # Windy uses milliseconds since Unix epoch.
        if timestamp > 100000000000:
            timestamp = timestamp / 1000.0

        dt = datetime.fromtimestamp(
            timestamp,
            tz=timezone.utc
        )

        return dt.strftime(
            "%Y-%m-%d %H:%M:%S UTC"
        )

    except Exception:

        return None


# ============================================================
# GET WEATHER FOR ONE SENSOR
# ============================================================

def get_weather_for_sensor(sensor):

    sensor_id = str(sensor["sensor_id"])

    latitude = float(sensor["latitude"])
    longitude = float(sensor["longitude"])

    print("\n========================================")
    print(f"Requesting weather for {sensor_id}")
    print(f"Coordinates: {latitude}, {longitude}")
    print("========================================")

    # --------------------------------------------------------
    # API REQUEST
    # --------------------------------------------------------

    payload = {

        "lat": latitude,

        "lon": longitude,

        "model": WINDY_MODEL,

        "parameters": [
            "temp",
            "rh",
            "pressure"
        ],

        "levels": [
            "surface"
        ],

        "key": WINDY_API_KEY
    }

    # --------------------------------------------------------
    # API REQUEST WITH SMALL RETRY FOR TRANSIENT TLS ERRORS
    # --------------------------------------------------------

    max_attempts = 3
    response = None

    for attempt in range(1, max_attempts + 1):

        try:

            response = requests.post(
                WINDY_URL,
                json=payload,
                headers={
                    "Connection": "close"
                },
                timeout=30
            )

            response.raise_for_status()
            break

        except (
            requests.exceptions.SSLError,
            requests.exceptions.ConnectionError
        ) as error:

            if attempt == max_attempts:
                raise

            wait_seconds = 1.5 * attempt

            print(
                f"⚠ {sensor_id}: transient network/TLS error "
                f"(attempt {attempt}/{max_attempts})"
            )

            print(
                f"  Retrying in {wait_seconds:.1f}s..."
            )

            time.sleep(wait_seconds)

    # requests succeeded; decode JSON exactly once.
    data = response.json()

    # --------------------------------------------------------
    # CHECK RESPONSE
    # --------------------------------------------------------

    if not isinstance(data, dict):

        raise ValueError(
            "Invalid response received from Windy API."
        )

    # --------------------------------------------------------
    # SHOW RESPONSE FIELDS
    # --------------------------------------------------------

    print("\nWindy response fields:")

    for key in data.keys():

        print(" -", key)

    # --------------------------------------------------------
    # GET UNITS
    # --------------------------------------------------------

    units = data.get("units", {})

    temp_unit = units.get(
        "temp-surface",
        "K"
    )

    pressure_unit = units.get(
        "pressure-surface",
        "Pa"
    )

    humidity_unit = units.get(
        "rh-surface",
        "%"
    )

    print("\nAPI units:")

    print(
        "Temperature:",
        temp_unit
    )

    print(
        "Pressure   :",
        pressure_unit
    )

    print(
        "Humidity   :",
        humidity_unit
    )

    # --------------------------------------------------------
    # GET RAW VALUES
    # --------------------------------------------------------

    raw_temperature = get_first_value(
        data,
        "temp-surface"
    )

    raw_pressure = get_first_value(
        data,
        "pressure-surface"
    )

    raw_humidity = get_first_value(
        data,
        "rh-surface"
    )

    # --------------------------------------------------------
    # GET TIMESTAMP
    # --------------------------------------------------------

    raw_timestamp = get_first_value(
        data,
        "ts"
    )

    # --------------------------------------------------------
    # CONVERT VALUES
    # --------------------------------------------------------

    temperature = convert_temperature_to_celsius(
        raw_temperature,
        temp_unit
    )

    pressure = convert_pressure_to_hpa(
        raw_pressure,
        pressure_unit
    )

    humidity = convert_humidity(
        raw_humidity,
        humidity_unit
    )

    observation_time = convert_timestamp(
        raw_timestamp
    )

    # --------------------------------------------------------
    # FALLBACK TIMESTAMP
    # --------------------------------------------------------

    if observation_time is None:

        observation_time = datetime.now(timezone.utc).strftime(
            "%Y-%m-%d %H:%M:%S UTC"
        )

    # --------------------------------------------------------
    # ACTUAL FETCH TIME vs WEATHER MODEL TIME
    # --------------------------------------------------------
    # Windy's "ts" is the model/forecast valid time, not the moment we
    # polled the API. Using it as the sensor observation time produces
    # large, misleading time gaps (e.g. ~6h) in the anomaly engine.
    # We keep both: "weather_model_time" preserves Windy's own ts for
    # reference/debugging, while "observation_time"/"timestamp" (used by
    # anomaly.py's row_time()/time_gap()) reflect the actual moment this
    # reading was fetched.

    actual_fetch_time = datetime.now(timezone.utc).strftime(
        "%Y-%m-%d %H:%M:%S UTC"
    )

    weather_model_time = observation_time

    # --------------------------------------------------------
    # CREATE RESULT
    # --------------------------------------------------------

    result = {

        # Sensor
        "sensor_id": sensor_id,

        # GPS
        "latitude": latitude,
        "longitude": longitude,

        # Location
        "state": sensor.get("state", ""),
        "district": sensor.get("district", ""),
        "city": sensor.get("city", ""),
        "locality": sensor.get("locality", ""),

        # Weather
        "temperature": temperature,
        "pressure": pressure,
        "humidity": humidity,

        # Time
        # Real polling time - this is what anomaly/time-gap logic reads.
        "timestamp": actual_fetch_time,
        "observation_time": actual_fetch_time,
        # Windy model/forecast valid time - kept separately for reference.
        "weather_model_time": weather_model_time,
        "actual_fetch_time": actual_fetch_time
    }

    # --------------------------------------------------------
    # DISPLAY RESULT
    # --------------------------------------------------------

    print("\nWeather data received:")

    print(
        f"Temperature : {temperature:.2f} °C"
        if temperature is not None
        else "Temperature : None"
    )

    print(
        f"Pressure    : {pressure:.2f} hPa"
        if pressure is not None
        else "Pressure    : None"
    )

    print(
        f"Humidity    : {humidity:.2f} %"
        if humidity is not None
        else "Humidity    : None"
    )

    print(
        "Observation :",
        observation_time
    )

    return result


# ============================================================
# BUILD A SENSOR-SPECIFIC API FAILURE ROW
# ============================================================

def build_api_failure_result(sensor, api_status, api_error=""):
    """
    Preserve the registered sensor even when the live weather API fails.

    The anomaly engine uses api_status/api_error plus the missing weather
    values to classify the reading as DATA_UNAVAILABLE instead of silently
    dropping the sensor from the batch.
    """

    now = datetime.now(timezone.utc).strftime(
        "%Y-%m-%d %H:%M:%S UTC"
    )

    return {
        "sensor_id": str(sensor.get("sensor_id", "UNKNOWN")),
        "latitude": sensor.get("latitude"),
        "longitude": sensor.get("longitude"),
        "state": sensor.get("state", ""),
        "district": sensor.get("district", ""),
        "city": sensor.get("city", ""),
        "locality": sensor.get("locality", ""),
        "temperature": None,
        "pressure": None,
        "humidity": None,
        "timestamp": now,
        "observation_time": now,
        "weather_model_time": None,
        "actual_fetch_time": now,
        "api_status": str(api_status),
        "api_error": str(api_error or "")
    }


# ============================================================
# GET WEATHER FOR ALL SENSORS
# ============================================================

def get_sensor_data():

    print("========================================")
    print("       SKYGUARD AI WEATHER API")
    print("========================================")

    # --------------------------------------------------------
    # CHECK API KEY
    # --------------------------------------------------------

    # --------------------------------------------------------
    # LOAD SENSOR REGISTRY
    # --------------------------------------------------------

    try:

        sensors = get_all_sensors()

    except Exception as error:

        print("\nERROR loading sensor registry:")
        print(error)

        return pd.DataFrame()

    # --------------------------------------------------------
    # CHECK API KEY
    # --------------------------------------------------------

    if not WINDY_API_KEY:

        print("\nERROR:")
        print("WINDY_API_KEY is missing.")
        print("\nPreserving all registered sensors as CONFIG_ERROR rows.")

        failure_rows = [
            build_api_failure_result(
                sensor,
                "CONFIG_ERROR",
                "WINDY_API_KEY is missing"
            )
            for sensor in sensors
        ]

        return pd.DataFrame(failure_rows)

    # --------------------------------------------------------
    # CHECK SENSOR REGISTRY
    # --------------------------------------------------------

    if not sensors:

        print("\nERROR:")
        print("No sensors found in sensor_registry.csv.")

        return pd.DataFrame()

    print(
        f"\nSensors loaded: {len(sensors)}"
    )

    # --------------------------------------------------------
    # REQUEST DATA (bounded parallel fetch)
    # --------------------------------------------------------
    # Sensors are fetched concurrently so total request time does not
    # grow linearly with sensor count. The worker pool size is capped
    # (not tied to a fixed sensor count) so any number of registered
    # sensors is supported.

    def fetch_one(sensor):

        sensor_id = sensor.get(
            "sensor_id",
            "UNKNOWN"
        )

        try:

            result = get_weather_for_sensor(
                sensor
            )

            # A successful request can still contain a missing weather
            # field. Keep it and let anomaly.py classify the data state.
            result["api_status"] = "API_OK"
            result["api_error"] = ""

            print(
                f"\n✓ {sensor_id} weather received"
            )

            return result

        except requests.exceptions.Timeout as error:

            print(
                f"\n✗ {sensor_id}: Windy API timeout"
            )

            return build_api_failure_result(
                sensor,
                "API_TIMEOUT",
                str(error) or "Windy API request timed out"
            )

        except requests.exceptions.RequestException as error:

            print(
                f"\n✗ {sensor_id}: API request failed"
            )
            print(
                f"  {error}"
            )

            return build_api_failure_result(
                sensor,
                "API_ERROR",
                str(error) or "Windy API request failed"
            )

        except (ValueError, KeyError, TypeError) as error:

            print(
                f"\n✗ {sensor_id}: invalid API response"
            )
            print(
                f"  {error}"
            )

            return build_api_failure_result(
                sensor,
                "INVALID_RESPONSE",
                str(error) or "Invalid Windy API response"
            )

        except Exception as error:

            print(
                f"\n✗ {sensor_id}: unexpected API error"
            )
            print(
                f"  {error}"
            )

            return build_api_failure_result(
                sensor,
                "NETWORK_ERROR",
                str(error) or "Unexpected weather API failure"
            )

    total_sensors = len(sensors)

    # 6 is a concurrency cap, not a sensor-count limit — any number of
    # registered sensors is supported, just fetched in bounded batches.
    worker_count = min(2, max(1, total_sensors))

    results_by_index = [None] * total_sensors

    with ThreadPoolExecutor(max_workers=worker_count) as executor:

        future_to_index = {
            executor.submit(fetch_one, sensor): index
            for index, sensor in enumerate(sensors)
        }

        for future in as_completed(future_to_index):

            index = future_to_index[future]

            try:
                results_by_index[index] = future.result()
            except Exception as error:
                results_by_index[index] = build_api_failure_result(
                    sensors[index],
                    "NETWORK_ERROR",
                    str(error) or "Unexpected weather API failure"
                )

    # Preserve original sensor-registry order for a stable dashboard.
    results = [row for row in results_by_index if row is not None]

    # --------------------------------------------------------
    # CHECK RESULTS
    # --------------------------------------------------------

    if not results:

        print("\nERROR:")
        print("No weather data received.")

        return pd.DataFrame()

    # --------------------------------------------------------
    # CREATE DATAFRAME
    # --------------------------------------------------------

    df = pd.DataFrame(results)

    # --------------------------------------------------------
    # COLUMN ORDER
    # --------------------------------------------------------

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
        "api_error"
    ]

    existing_columns = [

        column

        for column in preferred_columns

        if column in df.columns
    ]

    df = df[existing_columns]

    # --------------------------------------------------------
    # DISPLAY FINAL DATA
    # --------------------------------------------------------

    print("\n========================================")
    print("       WEATHER DATA RECEIVED")
    print("========================================")

    print(
        df.to_string(index=False)
    )

    return df


# ============================================================
# DIRECT TEST
# ============================================================

if __name__ == "__main__":

    df = get_sensor_data()

    if not df.empty:

        print("\n========================================")
        print("       WEATHER API TEST COMPLETE")
        print("========================================")

        print(
            f"\nTotal sensors: {len(df)}"
        )

        print("\nFinal columns:")

        for column in df.columns:

            print(
                " -",
                column
            )