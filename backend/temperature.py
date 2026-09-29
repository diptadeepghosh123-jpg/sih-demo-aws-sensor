import math
import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

MIN_HISTORY_FOR_BASELINE = 3

# A very large temperature movement in a short time is
# suspicious, but temperature itself has NO fixed fault limit.
MAX_REASONABLE_TEMP = 60.0
MIN_REASONABLE_TEMP = -50.0

# Temporal thresholds
SUDDEN_CHANGE_C = 5.0
EXTREME_CHANGE_C = 8.0

# Temperature-rate thresholds
SUSPICIOUS_RATE_C_PER_MIN = 15.0
EXTREME_RATE_C_PER_MIN = 30.0

# Persistent abnormal temperature
PERSISTENCE_READINGS = 3

# Peer comparison
MIN_PEERS = 2


# ============================================================
# SAFE NUMBER
# ============================================================

def safe_float(value, default=0.0):

    try:
        value = float(value)

        if math.isnan(value) or math.isinf(value):
            return default

        return value

    except Exception:
        return default


# ============================================================
# ROBUST Z-SCORE
# ============================================================

def robust_z_score(value, values):

    values = np.asarray(values, dtype=float)

    values = values[
        np.isfinite(values)
    ]

    if len(values) < 3:
        return 0.0

    median = np.median(values)

    mad = np.median(
        np.abs(values - median)
    )

    if mad < 0.01:
        std = np.std(values)

        if std < 0.01:
            return 0.0

        return abs(value - median) / std

    return abs(
        value - median
    ) / (1.4826 * mad)


# ============================================================
# GET SENSOR HISTORY
# ============================================================

def prepare_history(sensor_id, history):

    if history is None:
        return pd.DataFrame()

    if not isinstance(history, pd.DataFrame):
        history = pd.DataFrame(history)

    if history.empty:
        return pd.DataFrame()

    history = history.copy()

    if "sensor_id" in history.columns:

        history = history[
            history["sensor_id"].astype(str)
            == str(sensor_id)
        ]

    if "temperature" not in history.columns:
        return pd.DataFrame()

    history["temperature"] = pd.to_numeric(
        history["temperature"],
        errors="coerce"
    )

    history = history.dropna(
        subset=["temperature"]
    )

    if "observation_time" in history.columns:

        history["observation_time"] = pd.to_datetime(
            history["observation_time"],
            errors="coerce",
            utc=True
        )

    return history.sort_values(
        "observation_time"
        if "observation_time" in history.columns
        else "temperature"
    )


# ============================================================
# TEMPORAL ANALYSIS
# ============================================================

def calculate_temporal_score(
    current_temperature,
    history
):

    if history.empty:

        return {
            "score": 0.0,
            "change": 0.0,
            "rate": 0.0,
            "baseline": current_temperature,
            "history_z": 0.0,
            "sudden_change": False,
            "extreme_change": False
        }

    values = history[
        "temperature"
    ].astype(float).tolist()

    baseline = float(
        np.median(values[-10:])
    )

    previous = float(
        values[-1]
    )

    change = abs(
        current_temperature - previous
    )

    # --------------------------------------------------------
    # Timestamp-aware rate
    # --------------------------------------------------------

    rate = 0.0

    if (
        "observation_time" in history.columns
        and len(history) >= 1
    ):

        previous_time = history.iloc[-1][
            "observation_time"
        ]

        current_time = history.iloc[-1].get(
            "_current_time",
            pd.NaT
        )

        if pd.notna(previous_time) and pd.notna(current_time):

            seconds = (
                current_time - previous_time
            ).total_seconds()

            if seconds > 0:

                rate = (
                    change / seconds
                ) * 60.0

    # --------------------------------------------------------
    # Historical deviation
    # --------------------------------------------------------

    history_z = robust_z_score(
        current_temperature,
        values[-10:]
    )

    sudden_change = (
        change >= SUDDEN_CHANGE_C
        or rate >= SUSPICIOUS_RATE_C_PER_MIN
    )

    extreme_change = (
        change >= EXTREME_CHANGE_C
        or rate >= EXTREME_RATE_C_PER_MIN
    )

    # --------------------------------------------------------
    # Convert evidence into score
    # --------------------------------------------------------

    score = 0.0

    # Historical deviation
    if history_z >= 4:
        score += 0.35

    elif history_z >= 3:
        score += 0.25

    elif history_z >= 2:
        score += 0.15

    # Sudden change
    if extreme_change:
        score += 0.40

    elif sudden_change:
        score += 0.25

    # Strong baseline deviation
    baseline_difference = abs(
        current_temperature - baseline
    )

    if baseline_difference >= 10:
        score += 0.25

    elif baseline_difference >= 7:
        score += 0.18

    elif baseline_difference >= 5:
        score += 0.10

    return {
        "score": min(score, 1.0),
        "change": change,
        "rate": rate,
        "baseline": baseline,
        "history_z": history_z,
        "sudden_change": sudden_change,
        "extreme_change": extreme_change
    }


# ============================================================
# PEER / SPATIAL ANALYSIS
# ============================================================

def calculate_peer_score(
    current_sensor,
    all_sensors
):

    if all_sensors is None:
        return {
            "score": 0.0,
            "peer_median": None,
            "peer_difference": 0.0,
            "peer_count": 0
        }

    if not isinstance(all_sensors, pd.DataFrame):

        all_sensors = pd.DataFrame(
            all_sensors
        )

    if all_sensors.empty:
        return {
            "score": 0.0,
            "peer_median": None,
            "peer_difference": 0.0,
            "peer_count": 0
        }

    sensor_id = str(
        current_sensor.get(
            "sensor_id",
            ""
        )
    )

    peers = all_sensors[
        all_sensors["sensor_id"].astype(str)
        != sensor_id
    ].copy()

    if "temperature" not in peers.columns:
        return {
            "score": 0.0,
            "peer_median": None,
            "peer_difference": 0.0,
            "peer_count": 0
        }

    peers["temperature"] = pd.to_numeric(
        peers["temperature"],
        errors="coerce"
    )

    peers = peers.dropna(
        subset=["temperature"]
    )

    if len(peers) < MIN_PEERS:

        return {
            "score": 0.0,
            "peer_median": None,
            "peer_difference": 0.0,
            "peer_count": len(peers)
        }

    peer_values = peers[
        "temperature"
    ].values

    peer_median = float(
        np.median(peer_values)
    )

    current_temperature = safe_float(
        current_sensor.get(
            "temperature"
        )
    )

    difference = abs(
        current_temperature
        - peer_median
    )

    peer_z = robust_z_score(
        current_temperature,
        peer_values
    )

    score = 0.0

    if difference >= 10:
        score = 0.40

    elif difference >= 7:
        score = 0.30

    elif difference >= 5:
        score = 0.20

    elif difference >= 3:
        score = 0.10

    if peer_z >= 4:
        score = max(score, 0.40)

    elif peer_z >= 3:
        score = max(score, 0.30)

    return {
        "score": min(score, 1.0),
        "peer_median": peer_median,
        "peer_difference": difference,
        "peer_count": len(peers)
    }


# ============================================================
# PERSISTENCE ANALYSIS
# ============================================================

def calculate_persistence_score(
    current_temperature,
    history
):

    if history.empty:

        return {
            "score": 0.0,
            "persistent": False,
            "count": 0
        }

    values = history[
        "temperature"
    ].astype(float).tolist()

    recent = values[
        -PERSISTENCE_READINGS:
    ]

    if len(recent) < PERSISTENCE_READINGS:

        return {
            "score": 0.0,
            "persistent": False,
            "count": len(recent)
        }

    baseline = float(
        np.median(values[:-len(recent)])
    ) if len(values) > len(recent) else float(
        np.median(values)
    )

    deviations = [
        abs(value - baseline)
        for value in recent
    ]

    abnormal_count = sum(
        deviation >= 5
        for deviation in deviations
    )

    persistent = (
        abnormal_count
        >= PERSISTENCE_READINGS
    )

    score = 0.30 if persistent else 0.0

    return {
        "score": score,
        "persistent": persistent,
        "count": abnormal_count
    }


# ============================================================
# MAIN TEMPERATURE ANALYZER
# ============================================================

def analyze_temperature(
    sensor,
    history,
    all_sensors
):

    current_temperature = safe_float(
        sensor.get(
            "temperature"
        )
    )

    # --------------------------------------------------------
    # Physical sanity only
    # --------------------------------------------------------

    physically_invalid = (
        current_temperature < MIN_REASONABLE_TEMP
        or current_temperature > MAX_REASONABLE_TEMP
    )

    # --------------------------------------------------------
    # Prepare history
    # --------------------------------------------------------

    history_df = prepare_history(
        sensor.get("sensor_id"),
        history
    )

    # Add current timestamp so rate can be calculated
    if not history_df.empty:

        current_time = pd.to_datetime(
            sensor.get(
                "observation_time",
                sensor.get("timestamp")
            ),
            errors="coerce",
            utc=True
        )

        history_df = history_df.copy()

        history_df["_current_time"] = current_time

    # --------------------------------------------------------
    # Evidence
    # --------------------------------------------------------

    temporal = calculate_temporal_score(
        current_temperature,
        history_df
    )

    peer = calculate_peer_score(
        sensor,
        all_sensors
    )

    persistence = calculate_persistence_score(
        current_temperature,
        history_df
    )

    # --------------------------------------------------------
    # Combined evidence
    # --------------------------------------------------------

    score = (
        temporal["score"] * 0.45
        + peer["score"] * 0.30
        + persistence["score"] * 0.25
    )

    # Physical impossibility is deterministic.
    if physically_invalid:
        score = 1.0

    # --------------------------------------------------------
    # Strong combined anomaly condition
    #
    # We explicitly require evidence rather than using
    # temperature > 45 as the decision.
    # --------------------------------------------------------

    strong_temporal = (
        temporal["extreme_change"]
        and temporal["history_z"] >= 2
    )

    strong_isolated_change = (
        temporal["change"] >= 8
        and peer["peer_difference"] >= 5
    )

    persistent_isolated = (
        persistence["persistent"]
        and peer["peer_difference"] >= 5
    )

    anomaly = (
        physically_invalid
        or score >= 0.65
        or strong_temporal
        or strong_isolated_change
        or persistent_isolated
    )

    # --------------------------------------------------------
    # Explanation
    # --------------------------------------------------------

    evidence = []

    if temporal["sudden_change"]:

        evidence.append(
            f"sudden temperature change "
            f"{temporal['change']:.2f}°C"
        )

    if temporal["rate"] > 0:

        evidence.append(
            f"temperature rate "
            f"{temporal['rate']:.2f}°C/min"
        )

    if temporal["history_z"] >= 2:

        evidence.append(
            f"historical deviation "
            f"(robust z={temporal['history_z']:.2f})"
        )

    if peer["peer_median"] is not None:

        evidence.append(
            f"peer difference "
            f"{peer['peer_difference']:.2f}°C"
        )

    if persistence["persistent"]:

        evidence.append(
            "persistent abnormal temperature"
        )

    if physically_invalid:

        evidence.append(
            "temperature outside physical range"
        )

    return {
        "temperature_anomaly": bool(anomaly),
        "temperature_score": round(
            float(score),
            4
        ),
        "baseline_temperature": round(
            float(temporal["baseline"]),
            3
        ),
        "temperature_change": round(
            float(temporal["change"]),
            3
        ),
        "temperature_rate_c_per_min": round(
            float(temporal["rate"]),
            3
        ),
        "historical_z_score": round(
            float(temporal["history_z"]),
            3
        ),
        "peer_median_temperature": (
            round(
                float(peer["peer_median"]),
                3
            )
            if peer["peer_median"] is not None
            else None
        ),
        "peer_temperature_difference": round(
            float(peer["peer_difference"]),
            3
        ),
        "persistent_temperature": bool(
            persistence["persistent"]
        ),
        "temperature_evidence": evidence,
        "physical_temperature_fault": bool(
            physically_invalid
        )
    }