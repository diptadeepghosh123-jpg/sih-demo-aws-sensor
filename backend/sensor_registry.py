# ============================================================
# SkyGuard AI - Dynamic Sensor Registry
# ============================================================
#
# Responsibilities:
#
#   1. Load sensors from sensor_registry.csv
#   2. Save sensors to sensor_registry.csv
#   3. Add sensors dynamically
#   4. Delete sensors dynamically
#   5. Update sensors dynamically
#   6. Get one sensor
#   7. Get all sensors
#   8. Get sensor count
#   9. Get states dynamically
#  10. Get districts dynamically
#  11. Get sensors by state
#  12. Find nearby sensors using latitude/longitude
#  13. Provide registry information
#
# IMPORTANT:
#   - There is NO fixed limit of 5 sensors.
#   - sensor_registry.csv is the persistent registry.
#   - Do not hardcode sensor IDs here.
#   - Spatial detection can use get_nearby_sensors().
#
# ============================================================

import os
import re
import math
import threading

import pandas as pd


# ============================================================
# PATH CONFIGURATION
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

REGISTRY_FILE = os.path.join(
    BASE_DIR,
    "sensor_registry.csv"
)


# ============================================================
# REQUIRED CSV COLUMNS
# ============================================================

REQUIRED_COLUMNS = [
    "sensor_id",
    "latitude",
    "longitude",
    "state",
    "district",
    "city",
    "locality"
]


# ============================================================
# FILE LOCK
# ============================================================

_REGISTRY_LOCK = threading.RLock()


# ============================================================
# INTERNAL TEXT CLEANING
# ============================================================

def _clean_text(value):
    """
    Convert a value to a clean string.

    None/NaN becomes an empty string.
    """

    if value is None:
        return ""

    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass

    return str(value).strip()


# ============================================================
# CANONICAL SENSOR ID FORMAT
# ============================================================
#
# Sensor IDs must be identical everywhere (CSV, database.py, anomaly.py,
# maintenance.py, predictions, frontend, chatbot) because history/
# maintenance/prediction lookups use exact `WHERE sensor_id = ?` matching.
# "AWS-006" and "AWS_006" would otherwise be treated as two different
# sensors. normalize_sensor_id() is the single canonical conversion used
# any time a sensor ID enters the system (registration, live weather
# fetch, API requests, database writes/reads).
# ============================================================

def normalize_sensor_id(value):
    """
    Convert any sensor ID spelling to one canonical format:
    uppercase, words separated by a single underscore.

    Examples:
        "aws-006"   -> "AWS_006"
        " AWS 006 " -> "AWS_006"
        "AWS__006"  -> "AWS_006"
        "AWS_006"   -> "AWS_006"
    """

    text = _clean_text(value)

    if not text:
        return ""

    text = text.upper()

    # Any run of whitespace, hyphens, or underscores becomes one underscore.
    text = re.sub(r"[\s_\-]+", "_", text)

    return text.strip("_")


# ============================================================
# INTERNAL REGISTRY CREATION
# ============================================================

def _ensure_registry_file():
    """
    Create sensor_registry.csv if it does not exist.
    """

    if os.path.exists(REGISTRY_FILE):
        return

    df = pd.DataFrame(
        columns=REQUIRED_COLUMNS
    )

    df.to_csv(
        REGISTRY_FILE,
        index=False
    )


# ============================================================
# INTERNAL COORDINATE VALIDATION
# ============================================================

def _validate_coordinates(
    latitude,
    longitude
):
    """
    Validate latitude and longitude.

    Returns:
        (True, None)
    OR
        (False, error_message)
    """

    try:
        latitude = float(latitude)
        longitude = float(longitude)

    except (
        TypeError,
        ValueError
    ):

        return (
            False,
            "Latitude and longitude must be numeric."
        )

    if latitude < -90 or latitude > 90:

        return (
            False,
            "Latitude must be between -90 and 90."
        )

    if longitude < -180 or longitude > 180:

        return (
            False,
            "Longitude must be between -180 and 180."
        )

    return True, None


# ============================================================
# LOAD REGISTRY
# ============================================================

def load_sensor_registry():
    """
    Load the complete sensor registry.

    The number of sensors is dynamic.

    Returns:
        pandas.DataFrame
    """

    with _REGISTRY_LOCK:

        _ensure_registry_file()

        df = pd.read_csv(
            REGISTRY_FILE,
            dtype={
                "sensor_id": str
            }
        )

    # --------------------------------------------------------
    # Make sure all required columns exist
    # --------------------------------------------------------

    for column in REQUIRED_COLUMNS:

        if column not in df.columns:

            raise ValueError(
                f"Sensor registry is missing column: {column}"
            )

    # --------------------------------------------------------
    # Sensor ID
    # --------------------------------------------------------
    # Canonicalize on every read so "AWS-006", "aws_006", and "AWS_006"
    # are always treated as the exact same sensor everywhere downstream
    # (weather fetch, database writes/queries, anomaly engine, frontend,
    # chatbot). This is the single choke point all registry reads pass
    # through, so normalizing here fixes ID drift globally.

    df["sensor_id"] = (
        df["sensor_id"]
        .fillna("")
        .astype(str)
        .map(normalize_sensor_id)
    )

    # --------------------------------------------------------
    # Coordinates
    # --------------------------------------------------------

    df["latitude"] = pd.to_numeric(
        df["latitude"],
        errors="coerce"
    )

    df["longitude"] = pd.to_numeric(
        df["longitude"],
        errors="coerce"
    )

    # --------------------------------------------------------
    # Location fields
    # --------------------------------------------------------

    for column in [
        "state",
        "district",
        "city",
        "locality"
    ]:

        df[column] = (
            df[column]
            .fillna("")
            .astype(str)
            .str.strip()
        )

    # --------------------------------------------------------
    # Remove empty sensor IDs
    # --------------------------------------------------------

    df = df[
        df["sensor_id"] != ""
    ]

    # --------------------------------------------------------
    # Remove invalid coordinates
    # --------------------------------------------------------

    df = df.dropna(
        subset=[
            "latitude",
            "longitude"
        ]
    )

    # --------------------------------------------------------
    # Coordinate range validation
    # --------------------------------------------------------

    df = df[
        (df["latitude"] >= -90)
        &
        (df["latitude"] <= 90)
        &
        (df["longitude"] >= -180)
        &
        (df["longitude"] <= 180)
    ]

    # --------------------------------------------------------
    # Remove duplicate sensor IDs
    # --------------------------------------------------------

    df = df.drop_duplicates(
        subset=["sensor_id"],
        keep="first"
    )

    return df[
        REQUIRED_COLUMNS
    ].reset_index(
        drop=True
    )


# ============================================================
# INTERNAL SAVE FUNCTION
# ============================================================

def _save_sensor_registry(df):
    """
    Save registry safely to CSV.
    """

    df = df.copy()

    # --------------------------------------------------------
    # Ensure columns
    # --------------------------------------------------------

    for column in REQUIRED_COLUMNS:

        if column not in df.columns:

            df[column] = ""

    # --------------------------------------------------------
    # Correct column order
    # --------------------------------------------------------

    df = df[
        REQUIRED_COLUMNS
    ]

    # --------------------------------------------------------
    # Clean sensor IDs
    # --------------------------------------------------------

    df["sensor_id"] = (
        df["sensor_id"]
        .fillna("")
        .astype(str)
        .map(normalize_sensor_id)
    )

    # --------------------------------------------------------
    # Coordinates
    # --------------------------------------------------------

    df["latitude"] = pd.to_numeric(
        df["latitude"],
        errors="coerce"
    )

    df["longitude"] = pd.to_numeric(
        df["longitude"],
        errors="coerce"
    )

    # --------------------------------------------------------
    # Location fields
    # --------------------------------------------------------

    for column in [
        "state",
        "district",
        "city",
        "locality"
    ]:

        df[column] = (
            df[column]
            .fillna("")
            .astype(str)
            .str.strip()
        )

    # --------------------------------------------------------
    # Remove invalid records
    # --------------------------------------------------------

    df = df[
        (df["sensor_id"] != "")
        &
        df["latitude"].notna()
        &
        df["longitude"].notna()
    ]

    # --------------------------------------------------------
    # Remove duplicates
    # --------------------------------------------------------

    df = df.drop_duplicates(
        subset=["sensor_id"],
        keep="first"
    )

    # --------------------------------------------------------
    # Temporary file
    # --------------------------------------------------------

    temp_file = (
        REGISTRY_FILE
        + ".tmp"
    )

    # --------------------------------------------------------
    # Write
    # --------------------------------------------------------

    df.to_csv(
        temp_file,
        index=False
    )

    # --------------------------------------------------------
    # Replace original
    # --------------------------------------------------------

    os.replace(
        temp_file,
        REGISTRY_FILE
    )


# ============================================================
# SENSOR EXISTS
# ============================================================

def sensor_exists(sensor_id):
    """
    Check whether a sensor exists.
    """

    sensor_id = normalize_sensor_id(
        sensor_id
    )

    if not sensor_id:
        return False

    registry = load_sensor_registry()

    return bool(
        (
            registry["sensor_id"]
            .str.lower()
            == sensor_id.lower()
        ).any()
    )


# ============================================================
# GET ONE SENSOR
# ============================================================

def get_sensor(sensor_id):
    """
    Get one sensor by sensor ID.

    Returns:
        dict
        or None
    """

    sensor_id = normalize_sensor_id(
        sensor_id
    )

    if not sensor_id:
        return None

    registry = load_sensor_registry()

    result = registry[
        registry["sensor_id"]
        .str.lower()
        == sensor_id.lower()
    ]

    if result.empty:
        return None

    return result.iloc[0].to_dict()


# ============================================================
# GET ALL SENSORS
# ============================================================

def get_all_sensors():
    """
    Return every registered sensor.

    There is no fixed sensor limit.
    """

    registry = load_sensor_registry()

    return registry.to_dict(
        orient="records"
    )


# ============================================================
# GET SENSOR COUNT
# ============================================================

def get_sensor_count():
    """
    Return current number of registered sensors.
    """

    registry = load_sensor_registry()

    return int(
        len(registry)
    )


# ============================================================
# GET STATES
# ============================================================

def get_states():
    """
    Return all states currently represented
    by registered sensors.
    """

    registry = load_sensor_registry()

    states = []

    for state in registry["state"]:

        state = _clean_text(
            state
        )

        if (
            state
            and state not in states
        ):

            states.append(
                state
            )

    return sorted(
        states,
        key=str.lower
    )


# ============================================================
# GET DISTRICTS
# ============================================================

def get_districts(state=None):
    """
    Return districts dynamically.

    If state is supplied, only districts belonging
    to that state are returned.
    """

    registry = load_sensor_registry()

    if state:

        state = _clean_text(
            state
        )

        registry = registry[
            registry["state"]
            .str.lower()
            == state.lower()
        ]

    districts = []

    for district in registry["district"]:

        district = _clean_text(
            district
        )

        if (
            district
            and district not in districts
        ):

            districts.append(
                district
            )

    return sorted(
        districts,
        key=str.lower
    )


# ============================================================
# GET CITIES
# ============================================================

def get_cities(
    state=None,
    district=None
):
    """
    Return cities dynamically.

    Optional filtering:
        state
        district
    """

    registry = load_sensor_registry()

    if state:

        state = _clean_text(
            state
        )

        registry = registry[
            registry["state"]
            .str.lower()
            == state.lower()
        ]

    if district:

        district = _clean_text(
            district
        )

        registry = registry[
            registry["district"]
            .str.lower()
            == district.lower()
        ]

    cities = []

    for city in registry["city"]:

        city = _clean_text(
            city
        )

        if (
            city
            and city not in cities
        ):

            cities.append(
                city
            )

    return sorted(
        cities,
        key=str.lower
    )


# ============================================================
# GET SENSORS BY STATE
# ============================================================

def get_sensors_by_state(state):
    """
    Return sensors belonging to a state.
    """

    state = _clean_text(
        state
    )

    if not state:
        return []

    registry = load_sensor_registry()

    result = registry[
        registry["state"]
        .str.lower()
        == state.lower()
    ]

    return result.to_dict(
        orient="records"
    )


# ============================================================
# GET SENSORS BY DISTRICT
# ============================================================

def get_sensors_by_district(
    district,
    state=None
):
    """
    Return sensors belonging to a district.

    State can optionally be supplied to narrow the search.
    """

    district = _clean_text(
        district
    )

    if not district:
        return []

    registry = load_sensor_registry()

    result = registry[
        registry["district"]
        .str.lower()
        == district.lower()
    ]

    if state:

        state = _clean_text(
            state
        )

        result = result[
            result["state"]
            .str.lower()
            == state.lower()
        ]

    return result.to_dict(
        orient="records"
    )


# ============================================================
# GET SENSORS BY CITY
# ============================================================

def get_sensors_by_city(
    city,
    district=None,
    state=None
):
    """
    Return sensors belonging to a city.

    Optional district/state filters.
    """

    city = _clean_text(
        city
    )

    if not city:
        return []

    registry = load_sensor_registry()

    result = registry[
        registry["city"]
        .str.lower()
        == city.lower()
    ]

    if district:

        district = _clean_text(
            district
        )

        result = result[
            result["district"]
            .str.lower()
            == district.lower()
        ]

    if state:

        state = _clean_text(
            state
        )

        result = result[
            result["state"]
            .str.lower()
            == state.lower()
        ]

    return result.to_dict(
        orient="records"
    )


# ============================================================
# HAVERSINE DISTANCE
# ============================================================

def _haversine_distance(
    latitude_1,
    longitude_1,
    latitude_2,
    longitude_2
):
    """
    Calculate distance between two coordinates.

    Returns:
        distance in kilometres
    """

    earth_radius_km = 6371.0

    lat1 = math.radians(
        float(latitude_1)
    )

    lat2 = math.radians(
        float(latitude_2)
    )

    delta_lat = math.radians(
        float(latitude_2)
        -
        float(latitude_1)
    )

    delta_lon = math.radians(
        float(longitude_2)
        -
        float(longitude_1)
    )

    a = (
        math.sin(delta_lat / 2) ** 2
        +
        math.cos(lat1)
        *
        math.cos(lat2)
        *
        math.sin(delta_lon / 2) ** 2
    )

    a = max(
        0.0,
        min(1.0, a)
    )

    c = 2 * math.atan2(
        math.sqrt(a),
        math.sqrt(1 - a)
    )

    return (
        earth_radius_km
        * c
    )


# ============================================================
# GET NEARBY SENSORS
# ============================================================

def get_nearby_sensors(
    latitude,
    longitude,
    radius_km=50.0,
    exclude_sensor_id=None
):
    """
    Dynamically find sensors within a radius.

    This does NOT assume five sensors.

    Example:

        get_nearby_sensors(
            22.5726,
            88.3639,
            radius_km=50
        )

    Returns sensors sorted from nearest to farthest.
    """

    # --------------------------------------------------------
    # Validate input
    # --------------------------------------------------------

    valid, error = _validate_coordinates(
        latitude,
        longitude
    )

    if not valid:

        raise ValueError(
            error
        )

    try:

        radius_km = float(
            radius_km
        )

    except (
        TypeError,
        ValueError
    ):

        raise ValueError(
            "Radius must be numeric."
        )

    if radius_km <= 0:

        raise ValueError(
            "Radius must be greater than 0."
        )

    # --------------------------------------------------------
    # Load current registry
    # --------------------------------------------------------

    registry = load_sensor_registry()

    nearby = []

    # --------------------------------------------------------
    # Compare every registered sensor
    # --------------------------------------------------------

    for _, row in registry.iterrows():

        sensor_id = _clean_text(
            row["sensor_id"]
        )

        # ----------------------------------------------------
        # Exclude source sensor if requested
        # ----------------------------------------------------

        if (
            exclude_sensor_id
            and sensor_id.lower()
            ==
            _clean_text(
                exclude_sensor_id
            ).lower()
        ):
            continue

        sensor_latitude = float(
            row["latitude"]
        )

        sensor_longitude = float(
            row["longitude"]
        )

        distance = _haversine_distance(
            latitude,
            longitude,
            sensor_latitude,
            sensor_longitude
        )

        # ----------------------------------------------------
        # Within requested radius
        # ----------------------------------------------------

        if distance <= radius_km:

            sensor = row.to_dict()

            sensor["distance_km"] = round(
                distance,
                3
            )

            nearby.append(
                sensor
            )

    # --------------------------------------------------------
    # Nearest sensor first
    # --------------------------------------------------------

    nearby.sort(
        key=lambda sensor:
        sensor["distance_km"]
    )

    return nearby


# ============================================================
# ADD SENSOR
# ============================================================

def add_sensor(
    sensor_id,
    latitude,
    longitude,
    state="",
    district="",
    city="",
    locality=""
):
    """
    Add a new sensor dynamically.

    Sensor ID must be unique.
    """

    sensor_id = normalize_sensor_id(
        sensor_id
    )

    state = _clean_text(
        state
    )

    district = _clean_text(
        district
    )

    city = _clean_text(
        city
    )

    locality = _clean_text(
        locality
    )

    # --------------------------------------------------------
    # Sensor ID validation
    # --------------------------------------------------------

    if not sensor_id:

        raise ValueError(
            "Sensor ID is required."
        )

    # --------------------------------------------------------
    # Coordinate validation
    # --------------------------------------------------------

    valid, error = _validate_coordinates(
        latitude,
        longitude
    )

    if not valid:

        raise ValueError(
            error
        )

    latitude = float(
        latitude
    )

    longitude = float(
        longitude
    )

    # --------------------------------------------------------
    # Locked operation
    # --------------------------------------------------------

    with _REGISTRY_LOCK:

        _ensure_registry_file()

        df = pd.read_csv(
            REGISTRY_FILE,
            dtype={
                "sensor_id": str
            }
        )

        # ----------------------------------------------------
        # Ensure required columns
        # ----------------------------------------------------

        for column in REQUIRED_COLUMNS:

            if column not in df.columns:

                raise ValueError(
                    f"Sensor registry is missing column: {column}"
                )

        # ----------------------------------------------------
        # Duplicate ID check
        # ----------------------------------------------------

        duplicate = df[
            df["sensor_id"]
            .fillna("")
            .astype(str)
            .map(normalize_sensor_id)
            .str.lower()
            ==
            sensor_id.lower()
        ]

        if not duplicate.empty:

            raise ValueError(
                f"Sensor '{sensor_id}' already exists."
            )

        # ----------------------------------------------------
        # New sensor
        # ----------------------------------------------------

        new_sensor = {
            "sensor_id": sensor_id,
            "latitude": latitude,
            "longitude": longitude,
            "state": state,
            "district": district,
            "city": city,
            "locality": locality
        }

        new_row = pd.DataFrame(
            [new_sensor]
        )

        # ----------------------------------------------------
        # Add
        # ----------------------------------------------------

        df = pd.concat(
            [
                df,
                new_row
            ],
            ignore_index=True
        )

        # ----------------------------------------------------
        # Save
        # ----------------------------------------------------

        _save_sensor_registry(
            df
        )

    return {
        "success": True,
        "message": (
            f"Sensor '{sensor_id}' "
            "added successfully."
        ),
        "sensor": new_sensor,
        "sensor_count": get_sensor_count()
    }


# ============================================================
# DELETE SENSOR
# ============================================================

def delete_sensor(sensor_id):
    """
    Delete a sensor dynamically.
    """

    sensor_id = normalize_sensor_id(
        sensor_id
    )

    if not sensor_id:

        raise ValueError(
            "Sensor ID is required."
        )

    with _REGISTRY_LOCK:

        _ensure_registry_file()

        df = pd.read_csv(
            REGISTRY_FILE,
            dtype={
                "sensor_id": str
            }
        )

        # ----------------------------------------------------
        # Locate sensor
        # ----------------------------------------------------

        mask = (
            df["sensor_id"]
            .fillna("")
            .astype(str)
            .map(normalize_sensor_id)
            .str.lower()
            ==
            sensor_id.lower()
        )

        if not mask.any():

            raise ValueError(
                f"Sensor '{sensor_id}' does not exist."
            )

        # ----------------------------------------------------
        # Save deleted sensor
        # ----------------------------------------------------

        deleted_sensor = (
            df.loc[mask]
            .iloc[0]
            .to_dict()
        )

        # ----------------------------------------------------
        # Remove
        # ----------------------------------------------------

        df = df[
            ~mask
        ].reset_index(
            drop=True
        )

        # ----------------------------------------------------
        # Save
        # ----------------------------------------------------

        _save_sensor_registry(
            df
        )

    return {
        "success": True,
        "message": (
            f"Sensor '{sensor_id}' "
            "deleted successfully."
        ),
        "sensor": deleted_sensor,
        "sensor_count": get_sensor_count()
    }


# ============================================================
# UPDATE SENSOR
# ============================================================

def update_sensor(
    sensor_id,
    latitude=None,
    longitude=None,
    state=None,
    district=None,
    city=None,
    locality=None
):
    """
    Update an existing sensor.

    Only supplied fields are changed.

    Sensor ID itself remains unchanged.
    """

    sensor_id = normalize_sensor_id(
        sensor_id
    )

    if not sensor_id:

        raise ValueError(
            "Sensor ID is required."
        )

    with _REGISTRY_LOCK:

        _ensure_registry_file()

        df = pd.read_csv(
            REGISTRY_FILE,
            dtype={
                "sensor_id": str
            }
        )

        # ----------------------------------------------------
        # Find sensor
        # ----------------------------------------------------

        mask = (
            df["sensor_id"]
            .fillna("")
            .astype(str)
            .map(normalize_sensor_id)
            .str.lower()
            ==
            sensor_id.lower()
        )

        if not mask.any():

            raise ValueError(
                f"Sensor '{sensor_id}' does not exist."
            )

        index = df[
            mask
        ].index[0]

        # ----------------------------------------------------
        # Update latitude/longitude
        # ----------------------------------------------------

        if (
            latitude is not None
            or longitude is not None
        ):

            current_latitude = (
                df.at[
                    index,
                    "latitude"
                ]
            )

            current_longitude = (
                df.at[
                    index,
                    "longitude"
                ]
            )

            if latitude is None:
                latitude = current_latitude

            if longitude is None:
                longitude = current_longitude

            valid, error = _validate_coordinates(
                latitude,
                longitude
            )

            if not valid:

                raise ValueError(
                    error
                )

            df.at[
                index,
                "latitude"
            ] = float(latitude)

            df.at[
                index,
                "longitude"
            ] = float(longitude)

        # ----------------------------------------------------
        # Update state
        # ----------------------------------------------------

        if state is not None:

            df.at[
                index,
                "state"
            ] = _clean_text(
                state
            )

        # ----------------------------------------------------
        # Update district
        # ----------------------------------------------------

        if district is not None:

            df.at[
                index,
                "district"
            ] = _clean_text(
                district
            )

        # ----------------------------------------------------
        # Update city
        # ----------------------------------------------------

        if city is not None:

            df.at[
                index,
                "city"
            ] = _clean_text(
                city
            )

        # ----------------------------------------------------
        # Update locality
        # ----------------------------------------------------

        if locality is not None:

            df.at[
                index,
                "locality"
            ] = _clean_text(
                locality
            )

        # ----------------------------------------------------
        # Save
        # ----------------------------------------------------

        _save_sensor_registry(
            df
        )

        updated_sensor = (
            df.loc[index]
            .to_dict()
        )

    return {
        "success": True,
        "message": (
            f"Sensor '{sensor_id}' "
            "updated successfully."
        ),
        "sensor": updated_sensor,
        "sensor_count": get_sensor_count()
    }


# ============================================================
# REGISTRY INFORMATION
# ============================================================

def get_registry_info():
    """
    Return complete dynamic registry information.
    """

    sensors = get_all_sensors()

    states = get_states()

    return {
        "sensor_count": len(sensors),
        "state_count": len(states),
        "states": states,
        "sensors": sensors
    }


# ============================================================
# REGISTRY SUMMARY
# ============================================================

def get_registry_summary():
    """
    Return lightweight registry information.

    Useful for dashboard cards.
    """

    sensors = get_all_sensors()

    states = get_states()

    return {
        "sensor_count": len(sensors),
        "state_count": len(states),
        "states": states
    }


# ============================================================
# TEST / COMMAND-LINE CHECK
# ============================================================

if __name__ == "__main__":

    print("=" * 70)
    print("SkyGuard AI - Dynamic Sensor Registry")
    print("=" * 70)

    try:

        sensors = get_all_sensors()

        print(
            f"\nTotal registered sensors: "
            f"{len(sensors)}"
        )

        print(
            f"Registered states: "
            f"{get_states()}"
        )

        print(
            "\nRegistered sensors:"
        )

        for sensor in sensors:

            print(
                f"  {sensor['sensor_id']} | "
                f"Lat: {sensor['latitude']} | "
                f"Lon: {sensor['longitude']} | "
                f"State: {sensor['state']} | "
                f"District: {sensor['district']} | "
                f"City: {sensor['city']} | "
                f"Locality: {sensor['locality']}"
            )

        print(
            "\nRegistry summary:"
        )

        print(
            get_registry_summary()
        )

    except Exception as exc:

        print(
            "\nRegistry error:"
        )

        print(
            str(exc)
        )