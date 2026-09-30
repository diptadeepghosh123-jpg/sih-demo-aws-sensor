import os
import time
from datetime import datetime, timezone

import pandas as pd
import requests
from dotenv import load_dotenv

from sensor_registry import get_all_sensors


# ============================================================
# SKYGUARD AI
# WEATHER API
# ============================================================

# Keep the old .env loading structure so the rest of the project
# does not need any configuration changes.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)
ENV_FILES = [
    os.path.join(BASE_DIR, ".env"),
    os.path.join(PROJECT_DIR, ".env"),
    os.path.join(os.getcwd(), ".env"),
]

for _env_file in ENV_FILES:
    if not os.path.isfile(_env_file):
        continue

    load_dotenv(_env_file, override=False)


# Open-Meteo does not require an API key for the free public endpoint.
# The old WINDY_API_KEY can remain in .env; it is simply not used.
OPENMETEO_URL = "https://api.open-meteo.com/v1/forecast"


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def convert_temperature_to_celsius(value, unit="°C"):
    """
    Convert temperature to Celsius.
    Open-Meteo current temperature is returned in °C by default.
    """

    if value is None:
        return None

    value = float(value)
    unit = str(unit).lower()

    if unit in ["k", "kelvin"]:
        return value - 273.15

    return value


def convert_pressure_to_hpa(value, unit="hPa"):
    """
    Convert pressure to hPa.
    Open-Meteo surface_pressure is returned in hPa.
    """

    if value is None:
        return None

    value = float(value)
    unit = str(unit).lower()

    if unit in ["pa", "pascal", "pascals"]:
        return value / 100.0

    return value


def convert_humidity(value, unit="%"):
    """
    Convert relative humidity to percentage.
    Open-Meteo relative_humidity_2m is returned in %.
    """

    if value is None:
        return None

    return float(value)


def convert_timestamp(timestamp):
    """
    Convert an Open-Meteo ISO timestamp to readable UTC time.
    """

    if timestamp is None:
        return None

    try:

        # Open-Meteo normally returns ISO8601, e.g.
        # 2026-09-29T10:00
        text = str(timestamp).strip()

        if text.endswith("Z"):
            text = text[:-1] + "+00:00"

        dt = datetime.fromisoformat(text)

        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        else:
            dt = dt.astimezone(timezone.utc)

        return dt.strftime(
            "%Y-%m-%d %H:%M:%S UTC"
        )

    except Exception:

        return None


def request_openmeteo_weather(sensors):
    """
    Fetch all registered sensor locations in ONE Open-Meteo request.

    Open-Meteo supports comma-separated latitude/longitude values for
    multiple locations. The returned JSON is a list with one structure
    per requested location.
    """

    if not sensors:
        return []

    latitudes = ",".join(
        str(float(sensor["latitude"]))
        for sensor in sensors
    )

    longitudes = ",".join(
        str(float(sensor["longitude"]))
        for sensor in sensors
    )

    params = {
        "latitude": latitudes,
        "longitude": longitudes,
        "current": (
            "temperature_2m,"
            "relative_humidity_2m,"
            "surface_pressure"
        ),
        "timezone": "UTC"
    }

    print("\n========================================")
    print("Requesting Open-Meteo weather")
    print(f"Locations: {len(sensors)}")
    print("========================================")

    max_attempts = 3
    response = None

    for attempt in range(1, max_attempts + 1):

        try:

            response = requests.get(
                OPENMETEO_URL,
                params=params,
                headers={
                    "Connection": "close",
                    "User-Agent": "SkyGuard-AI/1.0"
                },
                timeout=30
            )

            response.raise_for_status()
            break

        except (
            requests.exceptions.SSLError,
            requests.exceptions.ConnectionError,
            requests.exceptions.Timeout
        ) as error:

            if attempt == max_attempts:
                raise

            wait_seconds = 1.5 * attempt

            print(
                "⚠ Open-Meteo transient network/TLS error "
                f"(attempt {attempt}/{max_attempts})"
            )

            print(
                f"  Retrying in {wait_seconds:.1f}s..."
            )

            time.sleep(wait_seconds)

    data = response.json()

    if not isinstance(data, (dict, list)):

        raise ValueError(
            "Invalid response received from Open-Meteo API."
        )

    return data


# ============================================================
# GET WEATHER FOR ONE SENSOR
# ============================================================

def get_weather_for_sensor(sensor, weather_data=None):

    sensor_id = str(sensor["sensor_id"])

    latitude = float(sensor["latitude"])
    longitude = float(sensor["longitude"])

    print("\n========================================")
    print(f"Processing weather for {sensor_id}")
    print(f"Coordinates: {latitude}, {longitude}")
    print("========================================")

    # If this function is called directly, request only this sensor.
    if weather_data is None:

        weather_data = request_openmeteo_weather(
            [sensor]
        )

    # Open-Meteo returns a list for multiple locations and a dict
    # for a single location.
    if isinstance(weather_data, list):

        if not weather_data:
            raise ValueError(
                f"No Open-Meteo data returned for {sensor_id}."
            )

        location_data = weather_data[0]

    elif isinstance(weather_data, dict):

        location_data = weather_data

    else:

        raise ValueError(
            f"Invalid Open-Meteo location data for {sensor_id}."
        )

    if not isinstance(location_data, dict):

        raise ValueError(
            f"Invalid Open-Meteo location response for {sensor_id}."
        )

    # --------------------------------------------------------
    # GET CURRENT WEATHER
    # --------------------------------------------------------

    current = location_data.get(
        "current",
        {}
    )

    current_units = location_data.get(
        "current_units",
        {}
    )

    if not isinstance(current, dict):

        raise ValueError(
            f"Missing current weather data for {sensor_id}."
        )

    # --------------------------------------------------------
    # SHOW RESPONSE FIELDS
    # --------------------------------------------------------

    print("\nOpen-Meteo response fields:")

    for key in location_data.keys():
        print(" -", key)

    # --------------------------------------------------------
    # GET UNITS
    # --------------------------------------------------------

    temp_unit = current_units.get(
        "temperature_2m",
        "°C"
    )

    pressure_unit = current_units.get(
        "surface_pressure",
        "hPa"
    )

    humidity_unit = current_units.get(
        "relative_humidity_2m",
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
    # GET CURRENT VALUES
    # --------------------------------------------------------

    raw_temperature = current.get(
        "temperature_2m"
    )

    raw_pressure = current.get(
        "surface_pressure"
    )

    raw_humidity = current.get(
        "relative_humidity_2m"
    )

    raw_timestamp = current.get(
        "time"
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
    # Open-Meteo's current time represents the weather-model data time.
    # The actual fetch time remains separate so anomaly.py does not see
    # artificial multi-hour gaps caused by model timestamps.

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

        # Weather model valid time - kept separately for reference.
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
    Preserve the registered sensor even when the weather API fails.

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

    print("Data source: Open-Meteo")
    print("API key: Not required")

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
    # REQUEST DATA
    # --------------------------------------------------------
    # All registered sensors are sent in ONE Open-Meteo request.
    # This keeps the API-call count low while preserving one result
    # per sensor for the rest of SkyGuard.

    try:

        weather_data = request_openmeteo_weather(
            sensors
        )

        if isinstance(weather_data, list):

            if len(weather_data) != len(sensors):

                raise ValueError(
                    "Open-Meteo returned a different number of "
                    "locations than requested."
                )

            location_data_by_index = weather_data

        elif isinstance(weather_data, dict):

            if len(sensors) != 1:

                raise ValueError(
                    "Open-Meteo returned a single location response "
                    "for multiple requested sensors."
                )

            location_data_by_index = [weather_data]

        else:

            raise ValueError(
                "Invalid Open-Meteo response."
            )

    except requests.exceptions.Timeout as error:

        print(
            "\n✗ Open-Meteo API timeout"
        )

        print(
            f"  {error}"
        )

        failure_rows = [
            build_api_failure_result(
                sensor,
                "API_TIMEOUT",
                str(error) or "Open-Meteo API request timed out"
            )
            for sensor in sensors
        ]

        return pd.DataFrame(failure_rows)

    except requests.exceptions.RequestException as error:

        print(
            "\n✗ Open-Meteo API request failed"
        )

        print(
            f"  {error}"
        )

        failure_rows = [
            build_api_failure_result(
                sensor,
                "API_ERROR",
                str(error) or "Open-Meteo API request failed"
            )
            for sensor in sensors
        ]

        return pd.DataFrame(failure_rows)

    except Exception as error:

        print(
            "\n✗ Open-Meteo API processing error"
        )

        print(
            f"  {error}"
        )

        failure_rows = [
            build_api_failure_result(
                sensor,
                "NETWORK_ERROR",
                str(error) or "Unexpected Open-Meteo API failure"
            )
            for sensor in sensors
        ]

        return pd.DataFrame(failure_rows)

    # --------------------------------------------------------
    # BUILD SENSOR RESULTS
    # --------------------------------------------------------

    results = []

    for index, sensor in enumerate(sensors):

        sensor_id = sensor.get(
            "sensor_id",
            "UNKNOWN"
        )

        try:

            result = get_weather_for_sensor(
                sensor,
                location_data_by_index[index]
            )

            # A successful request can still contain a missing weather
            # field. Keep it and let anomaly.py classify the data state.
            result["api_status"] = "API_OK"
            result["api_error"] = ""

            print(
                f"\n✓ {sensor_id} weather received"
            )

            results.append(result)

        except (ValueError, KeyError, TypeError) as error:

            print(
                f"\n✗ {sensor_id}: invalid Open-Meteo response"
            )

            print(
                f"  {error}"
            )

            results.append(
                build_api_failure_result(
                    sensor,
                    "INVALID_RESPONSE",
                    str(error) or "Invalid Open-Meteo API response"
                )
            )

        except Exception as error:

            print(
                f"\n✗ {sensor_id}: unexpected API error"
            )

            print(
                f"  {error}"
            )

            results.append(
                build_api_failure_result(
                    sensor,
                    "NETWORK_ERROR",
                    str(error) or "Unexpected weather API failure"
                )
            )

    # Preserve original sensor-registry order for a stable dashboard.
    results = [
        row
        for row in results
        if row is not None
    ]

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
