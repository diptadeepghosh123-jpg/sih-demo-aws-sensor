import math
import statistics


# ============================================================
# SPATIAL ANOMALY DETECTION
# ============================================================

# Maximum distance to consider another AWS as a neighbor.
# This is a network limit, not a rule saying sensors must
# have similar weather.
MAX_NEIGHBOR_DISTANCE_KM = 44.0


# ============================================================
# DISTANCE CALCULATION
# ============================================================

def calculate_distance(lat1, lon1, lat2, lon2):

    R = 6371.0

    lat1 = math.radians(float(lat1))
    lat2 = math.radians(float(lat2))

    dlat = lat2 - lat1
    dlon = math.radians(float(lon2) - float(lon1))

    a = (
        math.sin(dlat / 2) ** 2
        +
        math.cos(lat1)
        * math.cos(lat2)
        * math.sin(dlon / 2) ** 2
    )

    a = max(0.0, min(1.0, a))

    c = 2 * math.atan2(
        math.sqrt(a),
        math.sqrt(1 - a)
    )

    return R * c


# ============================================================
# SAFE NUMBER
# ============================================================

def safe_float(value):

    try:
        return float(value)

    except (TypeError, ValueError):

        return None


# ============================================================
# DISTANCE WEIGHT
# ============================================================

def calculate_distance_weight(distance):

    if distance <= 0:
        return 1.0

    # Closer sensors get more influence.
    #
    # Example:
    # 5 km  -> strong
    # 20 km -> moderate
    # 40 km -> weak

    weight = 1.0 / (1.0 + distance)

    return weight


# ============================================================
# HEALTHY SENSOR CHECK
# ============================================================

def is_valid_sensor(sensor):

    temperature = safe_float(
        sensor.get("temperature")
    )

    pressure = safe_float(
        sensor.get("pressure")
    )

    humidity = safe_float(
        sensor.get("humidity")
    )

    latitude = safe_float(
        sensor.get("latitude")
    )

    longitude = safe_float(
        sensor.get("longitude")
    )

    if (
        temperature is None
        or pressure is None
        or humidity is None
        or latitude is None
        or longitude is None
    ):

        return False

    # Physical validity checks.

    if temperature < -50 or temperature > 60:
        return False

    if pressure < 850 or pressure > 1100:
        return False

    if humidity < 3 or humidity > 100:
        return False

    return True


# ============================================================
# FIND NEIGHBORS
# ============================================================

def find_neighbors(current, all_sensors):

    current_lat = safe_float(
        current.get("latitude")
    )

    current_lon = safe_float(
        current.get("longitude")
    )

    current_id = str(
        current.get("sensor_id")
    )

    if current_lat is None or current_lon is None:

        return []

    neighbors = []

    for sensor in all_sensors:

        sensor_id = str(
            sensor.get("sensor_id")
        )

        # Don't compare sensor with itself.

        if sensor_id == current_id:
            continue

        # Ignore invalid sensors.

        if not is_valid_sensor(sensor):
            continue

        sensor_lat = safe_float(
            sensor.get("latitude")
        )

        sensor_lon = safe_float(
            sensor.get("longitude")
        )

        distance = calculate_distance(
            current_lat,
            current_lon,
            sensor_lat,
            sensor_lon
        )

        if distance <= MAX_NEIGHBOR_DISTANCE_KM:

            weight = calculate_distance_weight(
                distance
            )

            neighbors.append({
                "sensor": sensor,
                "distance": distance,
                "weight": weight
            })

    # Closest sensors first.

    neighbors.sort(
        key=lambda item: item["distance"]
    )

    return neighbors


# ============================================================
# WEIGHTED AVERAGE
# ============================================================

def weighted_average(values):

    if not values:
        return None

    total_value = 0.0
    total_weight = 0.0

    for value, weight in values:

        total_value += value * weight
        total_weight += weight

    if total_weight == 0:
        return None

    return total_value / total_weight


# ============================================================
# HUMIDITY SPATIAL ANALYSIS
# ============================================================

def analyze_humidity(current, neighbors):

    current_humidity = safe_float(
        current.get("humidity")
    )

    if current_humidity is None:

        return {
            "anomaly": False,
            "reason": "Invalid humidity value",
            "score": 0,
            "expected_humidity": None,
            "humidity_difference": None,
            "neighbor_count": 0,
            "regional_change": False
        }

    if not neighbors:

        return {
            "anomaly": False,
            "reason": "No valid nearby sensors available",
            "score": 0,
            "expected_humidity": None,
            "humidity_difference": None,
            "neighbor_count": 0,
            "regional_change": False
        }

    humidity_values = []

    weighted_values = []

    for item in neighbors:

        humidity = safe_float(
            item["sensor"].get("humidity")
        )

        if humidity is None:
            continue

        humidity_values.append(humidity)

        weighted_values.append(
            (
                humidity,
                item["weight"]
            )
        )

    if not humidity_values:

        return {
            "anomaly": False,
            "reason": "No valid neighboring humidity values",
            "score": 0,
            "expected_humidity": None,
            "humidity_difference": None,
            "neighbor_count": 0,
            "regional_change": False
        }

    # --------------------------------------------------------
    # LOCAL PEER VALUES
    # --------------------------------------------------------

    median_humidity = statistics.median(
        humidity_values
    )

    expected_humidity = weighted_average(
        weighted_values
    )

    if expected_humidity is None:

        expected_humidity = median_humidity

    difference = abs(
        current_humidity - expected_humidity
    )

    # --------------------------------------------------------
    # PEER SPREAD
    # --------------------------------------------------------

    if len(humidity_values) >= 2:

        peer_spread = (
            max(humidity_values)
            - min(humidity_values)
        )

    else:

        peer_spread = 0.0

    # --------------------------------------------------------
    # REGIONAL CONSISTENCY
    # --------------------------------------------------------
    #
    # If neighboring sensors are all close together,
    # their consensus is stronger.
    #
    # If neighboring sensors themselves vary greatly,
    # we reduce the strength of the spatial conclusion.
    # --------------------------------------------------------

    regional_change = False

    if len(humidity_values) >= 3:

        if peer_spread <= 15:

            regional_change = False

        else:

            regional_change = True

    # ========================================================
    # EVIDENCE SCORE
    # ========================================================

    score = 0

    reasons = []

    # --------------------------------------------------------
    # EXTREME BUT PHYSICALLY VALID HUMIDITY
    # --------------------------------------------------------

    if current_humidity <= 10:

        score += 1

        reasons.append(
            f"Very low humidity reading "
            f"({round(current_humidity, 2)}%)"
        )

    elif current_humidity >= 95:

        score += 1

        reasons.append(
            f"Very high humidity reading "
            f"({round(current_humidity, 2)}%)"
        )

    # --------------------------------------------------------
    # STRONG LOCAL DIFFERENCE
    # --------------------------------------------------------

    if difference >= 30:

        # If nearby sensors strongly agree with each other,
        # the isolated sensor becomes more suspicious.

        if peer_spread <= 15:

            score += 3

            reasons.append(
                "Humidity differs strongly from "
                "the local sensor consensus"
            )

        else:

            score += 1

            reasons.append(
                "Humidity differs from nearby sensors, "
                "but nearby conditions are also variable"
            )

    elif difference >= 20:

        if peer_spread <= 15:

            score += 2

            reasons.append(
                "Humidity differs significantly "
                "from nearby sensors"
            )

        else:

            score += 1

            reasons.append(
                "Moderate humidity difference from "
                "nearby sensors"
            )

    elif difference >= 10:

        score += 1

        reasons.append(
            "Humidity differs moderately from "
            "nearby sensors"
        )

    # --------------------------------------------------------
    # REGIONAL CHANGE
    # --------------------------------------------------------

    if regional_change:

        reasons.append(
            "Nearby sensors show considerable "
            "humidity variation"
        )

    # --------------------------------------------------------
    # FINAL SPATIAL DECISION
    # --------------------------------------------------------

    anomaly = score >= 2

    if not reasons:

        reason = (
            "Humidity agrees with nearby "
            "sensor conditions"
        )

    else:

        reason = "; ".join(reasons)

    return {
        "anomaly": anomaly,
        "reason": reason,
        "score": score,
        "expected_humidity": round(
            expected_humidity,
            2
        ),
        "humidity_difference": round(
            difference,
            2
        ),
        "neighbor_count": len(
            humidity_values
        ),
        "regional_change": regional_change
    }


# ============================================================
# MAIN SPATIAL ANOMALY DETECTION
# ============================================================

def detect_spatial_anomaly(current, all_sensors):

    current_temperature = safe_float(
        current.get("temperature")
    )

    current_pressure = safe_float(
        current.get("pressure")
    )

    current_humidity = safe_float(
        current.get("humidity")
    )

    if (
        current_temperature is None
        or current_pressure is None
        or current_humidity is None
    ):

        return {
            "anomaly": False,
            "reason": "Invalid current sensor values",
            "score": 0,
            "neighbor_count": 0
        }

    # ========================================================
    # FIND DYNAMIC NEIGHBORS
    # ========================================================

    neighbors = find_neighbors(
        current,
        all_sensors
    )

    if not neighbors:

        return {
            "anomaly": False,
            "reason": "No valid nearby sensors available",
            "score": 0,
            "neighbor_count": 0
        }

    # ========================================================
    # HUMIDITY ANALYSIS
    # ========================================================

    humidity_result = analyze_humidity(
        current,
        neighbors
    )

    # ========================================================
    # TEMPERATURE ANALYSIS
    # ========================================================

    temperature_values = []

    temperature_weighted_values = []

    for item in neighbors:

        temperature = safe_float(
            item["sensor"].get("temperature")
        )

        if temperature is None:
            continue

        temperature_values.append(
            temperature
        )

        temperature_weighted_values.append(
            (
                temperature,
                item["weight"]
            )
        )

    expected_temperature = weighted_average(
        temperature_weighted_values
    )

    if expected_temperature is None:
        # BUGFIX: this used statistics.median(temperature_values) here,
        # which crashes with StatisticsError if temperature_values is
        # also empty. Today that can't happen -- is_valid_sensor()
        # requires every neighbor to already have a valid temperature --
        # but that's an implicit dependency on a function defined far
        # above, not something this code enforces itself. If validation
        # is ever loosened, this silently becomes a hard crash instead
        # of a graceful "not enough evidence" result.
        if temperature_values:
            expected_temperature = statistics.median(
                temperature_values
            )
        else:
            expected_temperature = current_temperature

    temperature_difference = abs(
        current_temperature
        - expected_temperature
    )

    # ========================================================
    # PRESSURE ANALYSIS
    # ========================================================

    pressure_values = []

    pressure_weighted_values = []

    for item in neighbors:

        pressure = safe_float(
            item["sensor"].get("pressure")
        )

        if pressure is None:
            continue

        pressure_values.append(
            pressure
        )

        pressure_weighted_values.append(
            (
                pressure,
                item["weight"]
            )
        )

    expected_pressure = weighted_average(
        pressure_weighted_values
    )

    if expected_pressure is None:
        # Same defensive fallback as expected_temperature above.
        if pressure_values:
            expected_pressure = statistics.median(
                pressure_values
            )
        else:
            expected_pressure = current_pressure

    pressure_difference = abs(
        current_pressure
        - expected_pressure
    )

    # ========================================================
    # GENERAL SPATIAL EVIDENCE
    # ========================================================

    score = humidity_result["score"]

    reasons = []

    if humidity_result["reason"]:
        reasons.append(
            humidity_result["reason"]
        )

    # Temperature

    if temperature_difference >= 8:

        score += 2

        reasons.append(
            "Temperature differs significantly "
            "from nearby sensors"
        )

    elif temperature_difference >= 5:

        score += 1

        reasons.append(
            "Temperature differs moderately "
            "from nearby sensors"
        )

    # Pressure

    if pressure_difference >= 25:

        score += 2

        reasons.append(
            "Pressure differs significantly "
            "from nearby sensors"
        )

    elif pressure_difference >= 15:

        score += 1

        reasons.append(
            "Pressure differs moderately "
            "from nearby sensors"
        )

    # ========================================================
    # FINAL RESULT
    # ========================================================

    anomaly = score >= 2

    if not reasons:

        reason = (
            "Sensor agrees with nearby "
            "healthy sensors"
        )

    else:

        reason = "; ".join(reasons)

    return {
        "anomaly": anomaly,
        "reason": reason,
        "score": score,

        "neighbor_count": len(neighbors),

        "expected_temperature": round(
            expected_temperature,
            2
        ),

        "temperature_difference": round(
            temperature_difference,
            2
        ),

        "expected_pressure": round(
            expected_pressure,
            2
        ),

        "pressure_difference": round(
            pressure_difference,
            2
        ),

        "expected_humidity": humidity_result[
            "expected_humidity"
        ],

        "humidity_difference": humidity_result[
            "humidity_difference"
        ],

        "humidity_score": humidity_result[
            "score"
        ],

        "regional_humidity_change": humidity_result[
            "regional_change"
        ]
    }