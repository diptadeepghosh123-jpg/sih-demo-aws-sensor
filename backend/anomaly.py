"""
SkyGuard AI - anomaly.py
Robust, simple hybrid anomaly engine.

Rules:
- Temperature: time-band differential logic, sequential-pattern checks, and local-neighbor double-check.
- Pressure: percentage-change logic + physical operating guardrail.
- Humidity: independent 5-minute movement, rate, red-line, and distance-weighted neighbor logic.
  The input may arrive every 5 seconds, 10 seconds, 1 minute, 5 minutes,
  or at irregular intervals. We calculate the real timestamp difference.
- ML is supporting evidence only. ML alone can never create a sensor fault.
"""

from datetime import datetime, timezone
import math
import statistics

try:
    import pandas as pd
except Exception:
    pd = None

try:
    from .database import get_sensor_history
except Exception:
    from database import get_sensor_history

try:
    from .model import predict_probability
except Exception:
    try:
        from model import predict_probability
    except Exception:
        predict_probability = None

try:
    from .model import predict_fault_type
except Exception:
    try:
        from model import predict_fault_type
    except Exception:
        predict_fault_type = None

try:
    from .sensor_registry import get_all_sensors, normalize_sensor_id
except Exception:
    try:
        from sensor_registry import get_all_sensors, normalize_sensor_id
    except Exception:
        get_all_sensors = None
        def normalize_sensor_id(value):
            return str(value or "").strip().upper().replace("-", "_")

try:
    from .features import build_ml_features
except Exception:
    try:
        from features import build_ml_features
    except Exception:
        # Fallback keeps the engine running even if features.py is unavailable
        def build_ml_features(change_percent, change_rate, historical_score,
                              spatial_score, regional_agreement,
                              persistence_score, multivariate_score,
                              time_gap_seconds):
            return {
                "change_percent": change_percent,
                "change_rate": change_rate,
                "historical_score": historical_score,
                "spatial_score": spatial_score,
                "regional_agreement": regional_agreement,
                "persistence_score": persistence_score,
                "multivariate_score": multivariate_score,
                "time_gap_seconds": time_gap_seconds,
            }

# ============================================================
# OPTIONAL EXPECTED-VALUE PREDICTOR
# ============================================================
# The predictor imports shared helpers from anomaly.py, so importing it
# here at module load time creates a circular-import failure when main.py
# imports anomaly first. Keep the reference empty and load predictor only
# when a prediction is actually requested.
predict_expected_value = None


# ============================================================
# CONFIGURATION
# ============================================================

NEIGHBOR_RADIUS_KM = 44.0

# History must be large enough for the humidity 5-minute window.
HISTORY_LIMIT = 120

HUMIDITY_WINDOW_SECONDS = 300.0
HUMIDITY_STUCK_SECONDS = 300.0
HUMIDITY_STUCK_TOLERANCE = 0.05
HUMIDITY_MIN = 3.0
HUMIDITY_MAX = 100.0
HUMIDITY_REDLINE = 7.0

# Extreme temperature is evidence, not an automatic sensor-fault rule.
TEMPERATURE_EXTREME_HIGH = 60.0
TEMPERATURE_EXTREME_LOW = -45.0

# Temperature timing bands from the project specification.
TEMP_FAST_SECONDS = 60.0
TEMP_MEDIUM_SECONDS = 600.0
TEMP_LONG_SECONDS = 1800.0
TEMP_FAST_NORMAL_PCT = 3.0
TEMP_MEDIUM_NORMAL_PCT = 5.0
TEMP_WATCH_PCT = 7.0
TEMP_SEQUENCE_WINDOW_SECONDS = 120.0
TEMP_PERSISTENT_LEVEL_MIN_CONTINUATIONS = 2
TEMP_PERSISTENT_LEVEL_SPREAD_C = 2.0

# Humidity is intentionally independent from temperature.
HUMIDITY_EVENT_MIN_POINTS = 7.0
HUMIDITY_EVENT_MAX_POINTS = 10.0
HUMIDITY_RATE_STRONG_POINTS_PER_MIN = 8.0
HUMIDITY_MIN_OBSERVATION_SECONDS = 300.0

# Frozen/fixed-value sensor detection
FROZEN_CHECK_SECONDS = 1800.0          # 30 min -> WATCH
FROZEN_STAGE_SECONDS = 86400.0         # 24 h TOTAL -> GLITCH SUSPECT
FROZEN_CONFIRM_SECONDS = 172800.0      # 48 h TOTAL -> confirmed SENSOR_ANOMALY
FROZEN_ABSOLUTE_TOLERANCE = 0.0005
FROZEN_REQUIRED_POINTS = 3
FROZEN_HISTORY_LIMIT = 50000

HUMIDITY_GRADUAL_WINDOW_SECONDS = 900.0
HUMIDITY_PERSISTENT_WINDOW_SECONDS = 1800.0
HUMIDITY_SMALL_STEP_POINTS = 5.0
HUMIDITY_GRADUAL_MAX_POINTS = 10.0
HUMIDITY_DISTANCE_BASE_TOLERANCE = 8.0
HUMIDITY_DISTANCE_TOLERANCE_PER_KM = 0.55
HUMIDITY_DISTANCE_MAX_TOLERANCE = 35.0

LOCAL_CLUSTER_MIN_SUPPORT = 2
LOCAL_CLUSTER_TOLERANCE_C = 5.0

HYBRID_WEIGHT = 0.85
ML_WEIGHT = 0.15

# Silent Sensor Thresholds
SILENT_SENSOR_WATCH_SECONDS = 1800.0     # 30 min of silence  -> WATCH
SILENT_SENSOR_ANOMALY_SECONDS = 7200.0   # 2 hours of silence -> SENSOR_ANOMALY


# ============================================================
# BASIC HELPERS
# ============================================================

def clamp(value, low=0.0, high=1.0):
    try:
        return max(low, min(high, float(value)))
    except Exception:
        return low

def safe_float(value, default=None):
    try:
        if value is None:
            return default
        value = float(value)
        return default if not math.isfinite(value) else value
    except Exception:
        return default

def row_time(row):
    value = row.get("observation_time", row.get("timestamp"))
    if value is None or value == "":
        return None
    try:
        if pd is not None:
            t = pd.to_datetime(value, utc=True)
            if pd.isna(t):
                return None
            return t.to_pydatetime()
    except Exception:
        pass
    try:
        t = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        return t if t.tzinfo else t.replace(tzinfo=timezone.utc)
    except Exception:
        return None

def percentage_change(current, previous):
    current = safe_float(current)
    previous = safe_float(previous)
    if current is None or previous is None:
        return 0.0
    if abs(previous) < 1e-9:
        return 100.0 if abs(current) > 1e-9 else 0.0
    return abs(current - previous) / abs(previous) * 100.0

def time_gap(current, previous):
    if not previous:
        return None
    t1 = row_time(current)
    t0 = row_time(previous)
    if t1 is None or t0 is None:
        return None
    gap = (t1 - t0).total_seconds()
    return gap if gap >= 0 else None

def get_previous(history):
    return history[-1] if history else None

def normalize_input(data):
    if data is None:
        return []
    if hasattr(data, "to_dict"):
        try:
            rows = data.to_dict(orient="records")
        except Exception:
            rows = []
    elif isinstance(data, dict):
        rows = [data]
    else:
        rows = list(data)

    clean = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        item = dict(row)
        if "sensor_id" in item:
            item["sensor_id"] = normalize_sensor_id(item.get("sensor_id"))
        for field in ("temperature", "pressure", "humidity", "latitude", "longitude"):
            if field in item:
                item[field] = safe_float(item[field])
        clean.append(item)

    clean.sort(key=lambda x: row_time(x) or datetime.min.replace(tzinfo=timezone.utc))
    return clean

# ============================================================
# PREDICTION HELPERS
# ============================================================

def _prediction_trend_points(history, field):
    """Return the number of valid recent points used by predictor.py's trend estimator."""
    points = []
    for row in history:
        try:
            t = row_time(row)
        except Exception:
            t = None
        value = safe_float(row.get(field))
        if t is not None and value is not None:
            points.append((t, value))
    return min(8, len(points))


def generate_prediction(current, history, batch, field):
    """
    Additive expected-value prediction layer mapping to frontend contract.
    """
    empty = {
        "prediction_available": False,
        "expected_value": None,
        "trend_points": 0,
        "spatial_information": {
            "neighbor_count": 0,
            "value": None,
        },
        "prediction_components": {},
        "prediction_weights": {},
        "prediction_confidence": None,
        "prediction_actual_value": safe_float(current.get(field)),
        "prediction_deviation": None,
        "prediction_absolute_deviation": None,
        "prediction_field": field,
        "suggested_value": None,
        "prediction_neighbor_count": 0,
        "seasonal_points_used": 0,
        "seasonal_uses_day_of_year": False,
        "seasonal_uses_day_of_week": False,
        "seasonal_uses_month": False,
        "seasonal_history_span_days": 0.0,
        "seasonal_strength": 0.0,
        "trend_points_used": 0,
        "trend_sequence": [],
        "prediction_nearest_distance_km": None,
        "prediction_confidence_label": None,
        "seasonal_strength_label": None,
    }

    global predict_expected_value

    if predict_expected_value is None:
        try:
            try:
                from .predictor import predict_expected_value as _predict_expected_value
            except Exception:
                from predictor import predict_expected_value as _predict_expected_value
            predict_expected_value = _predict_expected_value
        except Exception:
            return empty

    try:
        prediction = predict_expected_value(
            current=current,
            history=history,
            batch=batch,
            field=field,
            healthy_sensor_ids=None,
        )
    except Exception:
        return empty

    if not prediction:
        return empty

    expected = safe_float(prediction.get("suggested_value"))
    actual = safe_float(current.get(field))
    deviation = None
    if expected is not None and actual is not None:
        deviation = actual - expected

    components = prediction.get("components") or {}
    spatial_value = safe_float(components.get("spatial"))
    trend_points = _prediction_trend_points(history, field)

    return {
        "prediction_available": expected is not None,
        "expected_value": expected,
        "trend_points": trend_points,
        "spatial_information": {
            "neighbor_count": int(prediction.get("spatial_neighbor_count") or 0),
            "value": spatial_value,
            "nearest_distance_km": prediction.get("nearest_distance_km"),
        },
        "prediction_components": components,
        "prediction_weights": prediction.get("weights_used") or {},
        "prediction_confidence": prediction.get("confidence"),
        "prediction_confidence_label": prediction.get("confidence_label"),
        "prediction_actual_value": actual,
        "prediction_deviation": deviation,
        "prediction_absolute_deviation": abs(deviation) if deviation is not None else None,
        "prediction_field": field,
        "suggested_value": expected,
        "prediction_neighbor_count": int(prediction.get("spatial_neighbor_count") or 0),
        "prediction_nearest_distance_km": prediction.get("nearest_distance_km"),
        "seasonal_points_used": int(prediction.get("seasonal_points_used") or 0),
        "seasonal_uses_day_of_year": bool(prediction.get("seasonal_uses_day_of_year", False)),
        "seasonal_uses_day_of_week": bool(prediction.get("seasonal_uses_day_of_week", False)),
        "seasonal_uses_month": bool(prediction.get("seasonal_uses_month", False)),
        "seasonal_history_span_days": float(prediction.get("seasonal_history_span_days") or 0.0),
        "seasonal_strength": float(prediction.get("seasonal_strength") or 0.0),
        "seasonal_strength_label": prediction.get("seasonal_strength_label"),
        "trend_points_used": trend_points,
        "trend_sequence": prediction.get("trend_sequence") or [],
    }


def generate_all_predictions(current, history, batch):
    """
    Generate additive expected-value predictions independently for all
    three weather parameters. This does not change anomaly decisions.
    """
    return {
        prediction_field: generate_prediction(
            current=current,
            history=history,
            batch=batch,
            field=prediction_field,
        )
        for prediction_field in ("temperature", "pressure", "humidity")
    }

# ============================================================
# TEMPERATURE / PRESSURE TEMPORAL LOGIC
# ============================================================

def sudden_change_score(change_percent, gap):
    change_percent = abs(safe_float(change_percent, 0.0))
    if change_percent <= 5:
        score = 0.0
    elif change_percent <= 7:
        score = 0.20
    elif change_percent <= 10:
        score = 0.45
    elif change_percent <= 15:
        score = 0.70
    else:
        score = 1.0
    if gap is None:
        return score
        
    if gap <= 10:
        factor = 1.0
    elif gap <= 30:
        factor = 0.95
    elif gap <= 60:
        factor = 0.90
    elif gap <= 120:
        factor = 0.75
    elif gap <= 300:
        factor = 0.60
    elif gap <= 600:
        factor = 0.40
    else:
        factor = 0.20
    return clamp(score * factor)

def temperature_time_rule(change_percent, gap):
    change = abs(safe_float(change_percent, 0.0))
    gap = safe_float(gap)

    if gap is None:
        gap = TEMP_MEDIUM_SECONDS

    if gap <= TEMP_FAST_SECONDS:
        if change <= TEMP_FAST_NORMAL_PCT:
            return "NORMAL", 0.0
        if change <= TEMP_WATCH_PCT:
            return "WATCH", 0.35
        if change <= 10:
            return "ANOMALY_EVIDENCE", 0.65
        if change <= 15:
            return "ANOMALY_EVIDENCE", 0.85
        return "ANOMALY_EVIDENCE", 1.0

    if gap <= TEMP_MEDIUM_SECONDS:
        if change <= TEMP_MEDIUM_NORMAL_PCT:
            return "NORMAL", 0.0
        if change <= TEMP_WATCH_PCT:
            return "WATCH", 0.30
        if change <= 10:
            return "ANOMALY_EVIDENCE", 0.60
        if change <= 15:
            return "ANOMALY_EVIDENCE", 0.82
        return "ANOMALY_EVIDENCE", 1.0

    if change <= TEMP_WATCH_PCT:
        return "WATCH", 0.35 if change > TEMP_MEDIUM_NORMAL_PCT else 0.0
    if change <= 10:
        return "ANOMALY_EVIDENCE", 0.55
    if change <= 15:
        return "ANOMALY_EVIDENCE", 0.75
    return "ANOMALY_EVIDENCE", 1.0

def temperature_sequence_analysis(history, current):
    rows = list(history[-8:]) + [current]
    rows = [r for r in rows if row_time(r) is not None and safe_float(r.get("temperature")) is not None]
    rows.sort(key=row_time)

    result = {
        "recovery": False,
        "erratic": False,
        "persistent_level": False,
        "pre_jump_value": None,
        "new_level": None,
        "continuation_count": 0,
        "level_direction": None,
        "direction_consistent": True,
        "large_move_count": 0,
        "sequence_score": 0.0,
        "description": "insufficient sequence history",
    }
    if len(rows) < 3:
        return result

    transitions = []
    for a, b in zip(rows, rows[1:]):
        gap = (row_time(b) - row_time(a)).total_seconds()
        if gap < 0:
            continue
        old = safe_float(a.get("temperature"))
        new = safe_float(b.get("temperature"))
        pct = percentage_change(new, old)
        direction = 1 if new > old else (-1 if new < old else 0)
        transitions.append({"gap": gap, "pct": pct, "direction": direction, "old": old, "new": new})

    recent = [x for x in transitions if x["gap"] <= TEMP_SEQUENCE_WINDOW_SECONDS]
    large = [x for x in recent if x["pct"] > TEMP_WATCH_PCT]
    result["large_move_count"] = len(large)

    if len(recent) >= 2:
        dirs = [x["direction"] for x in recent if x["direction"]]
        flips = sum(1 for a, b in zip(dirs, dirs[1:]) if a != b)
        result["direction_consistent"] = flips <= 1
        if flips >= 2 and any(x["pct"] > 3 for x in recent):
            result["erratic"] = True

    for i in range(max(0, len(transitions) - 4), len(transitions) - 1):
        first = transitions[i]
        second = transitions[i + 1]
        if first["pct"] <= TEMP_WATCH_PCT:
            continue
        if first["direction"] == 0 or second["direction"] == 0:
            continue
        if first["direction"] == second["direction"]:
            continue
        if second["gap"] > TEMP_SEQUENCE_WINDOW_SECONDS:
            continue
        jump = abs(first["new"] - first["old"])
        recovery = abs(second["new"] - first["new"])
        if jump > 0 and recovery >= 0.50 * jump:
            result["recovery"] = True
            result["description"] = "large movement followed by coherent recovery"
            result["sequence_score"] = max(result["sequence_score"], 0.25)

    (
        result["persistent_level"],
        result["pre_jump_value"],
        result["new_level"],
        result["continuation_count"],
        result["level_direction"],
    ) = _temperature_persistent_level(transitions)

    if result["persistent_level"]:
        result["description"] = "large jump followed by persistence at a new level"
        result["sequence_score"] = max(result["sequence_score"], 0.65)

    if result["erratic"]:
        result["description"] = "large direction reversals / non-sequential movement"
        result["sequence_score"] = max(result["sequence_score"], 0.85)
    elif result["large_move_count"] >= 2 and not result["recovery"] and not result["persistent_level"]:
        result["description"] = "repeated large movement without recovery"
        result["sequence_score"] = max(result["sequence_score"], 0.70)

    return result

def _temperature_persistent_level(transitions):
    if len(transitions) < 1:
        return False, None, None, 0, None

    for i, first in enumerate(transitions):
        if first["pct"] <= TEMP_WATCH_PCT or first["direction"] == 0:
            continue
        if first["gap"] > TEMP_SEQUENCE_WINDOW_SECONDS:
            continue

        follow = transitions[i + 1:]
        if len(follow) < TEMP_PERSISTENT_LEVEL_MIN_CONTINUATIONS:
            continue

        follow = follow[:TEMP_PERSISTENT_LEVEL_MIN_CONTINUATIONS]
        if any(x["direction"] == -first["direction"] and x["pct"] > 3.0 for x in follow):
            continue

        level_values = [first["new"]] + [x["new"] for x in follow]
        old_value = first["old"]
        jump = abs(first["new"] - old_value)
        spread = max(level_values) - min(level_values)

        if jump <= 0 or spread > TEMP_PERSISTENT_LEVEL_SPREAD_C:
            continue

        if any(abs(x["new"] - old_value) < 0.50 * jump for x in follow):
            continue

        return True, old_value, first["new"], len(follow), first["direction"]

    return False, None, None, 0, None


# ============================================================
# HISTORY & PERSISTENCE
# ============================================================

def historical_score(current_value, history, field):
    current_value = safe_float(current_value)
    values = [safe_float(row.get(field)) for row in history]
    values = [v for v in values if v is not None]

    if current_value is None or len(values) < 3:
        return 0.0, None, None

    median = statistics.median(values)
    deviations = [abs(v - median) for v in values]
    mad = statistics.median(deviations)

    if field == "humidity":
        scale = max(2.0, mad * 3.0)
    elif field == "temperature":
        scale = max(1.0, mad * 3.0)
    else:
        scale = max(2.0, mad * 3.0)

    distance = abs(current_value - median)

    if distance <= scale:
        score = 0.0
    elif distance <= 2 * scale:
        score = 0.30
    elif distance <= 3 * scale:
        score = 0.60
    else:
        score = 0.90

    return clamp(score), median, mad

def persistence_score(history, field):
    if len(history) < 3:
        return 0.0

    rows = history[-8:]
    changes = []

    for a, b in zip(rows, rows[1:]):
        old = safe_float(a.get(field))
        new = safe_float(b.get(field))
        if old is None or new is None:
            continue
        if field == "humidity":
            changes.append(abs(new - old))
        else:
            changes.append(percentage_change(new, old))

    if not changes:
        return 0.0

    if field == "humidity":
        unusual = sum(x >= 5 for x in changes)
    else:
        unusual = sum(x > 5 for x in changes)

    return clamp(unusual / max(1, len(changes)))


# ============================================================
# HUMIDITY-ONLY 5-MINUTE LOGIC
# ============================================================

def humidity_change_score(delta_points, actual_gap):
    delta_points = abs(safe_float(delta_points, 0.0))

    if delta_points < 5:
        score = 0.0
    elif delta_points < 10:
        score = 0.20
    elif delta_points < 20:
        score = 0.45
    elif delta_points < 30:
        score = 0.70
    else:
        score = 1.0

    if actual_gap is None or actual_gap <= 0:
        return score

    if actual_gap <= 30:
        factor = 1.00
    elif actual_gap <= 60:
        factor = 0.95
    elif actual_gap <= 120:
        factor = 0.85
    elif actual_gap <= 300:
        factor = 0.70
    else:
        factor = 0.40

    return clamp(score * factor)

def humidity_5min_reference(history, current):
    now = row_time(current)
    current_h = safe_float(current.get("humidity"))

    result = {
        "available": False,
        "baseline": None,
        "current": current_h,
        "delta": 0.0,
        "duration": 0.0,
        "reference_time": None,
        "stable_new_level": False,
        "reverted": False,
    }

    if now is None or current_h is None:
        return result

    target = now.timestamp() - HUMIDITY_WINDOW_SECONDS
    candidates = []

    for row in history:
        t = row_time(row)
        h = safe_float(row.get("humidity"))
        if t is None or h is None:
            continue
        seconds_from_target = abs(t.timestamp() - target)
        if t < now:
            candidates.append((seconds_from_target, t, h))

    if not candidates:
        return result

    candidates.sort(key=lambda x: x[0])
    _, reference_time, baseline = candidates[0]
    duration = (now - reference_time).total_seconds()

    if duration < HUMIDITY_MIN_OBSERVATION_SECONDS:
        return result

    result["available"] = True
    result["baseline"] = baseline
    result["delta"] = current_h - baseline
    result["duration"] = duration
    result["reference_time"] = reference_time.isoformat()

    window = []
    for row in history:
        t = row_time(row)
        h = safe_float(row.get("humidity"))
        if t is None or h is None:
            continue
        age = (now - t).total_seconds()
        if 0 <= age <= HUMIDITY_WINDOW_SECONDS:
            window.append((t, h))

    window.append((now, current_h))
    window.sort(key=lambda x: x[0])

    if len(window) >= 3:
        tail = [x[1] for x in window[-3:]]
        result["stable_new_level"] = max(tail) - min(tail) <= 2.0
        peak = max(x[1] for x in window)
        trough = min(x[1] for x in window)
        excursion = max(abs(peak - baseline), abs(trough - baseline))
        result["reverted"] = (excursion >= 10.0 and abs(current_h - baseline) <= 5.0)

    return result

def frozen_parameter_evidence(current, history, field):
    current_value = safe_float(current.get(field))
    now = row_time(current)
    result = {
        "candidate": False,
        "stage_24h": False,
        "confirmed": False,
        "duration_seconds": 0.0,
        "value": current_value,
        "spread": None,
        "points": 0,
    }

    if current_value is None or now is None:
        return result

    observations = []
    for row in history:
        t = row_time(row)
        if t is None or t > now:
            continue
        raw = row.get(field)
        value = safe_float(raw)
        observations.append((t, value))

    observations.append((now, current_value))
    observations.sort(key=lambda x: x[0])

    if len(observations) < FROZEN_REQUIRED_POINTS:
        return result

    run_start = now
    run_values = [current_value]
    previous_time = now

    for t, value in reversed(observations[:-1]):
        if value is None:
            break
        gap = (previous_time - t).total_seconds()
        if gap <= 0:
            break
        if gap > FROZEN_CHECK_SECONDS:
            break
        if abs(value - current_value) > FROZEN_ABSOLUTE_TOLERANCE:
            break
        run_start = t
        run_values.append(value)
        previous_time = t

    duration = max(0.0, (now - run_start).total_seconds())
    result["duration_seconds"] = duration
    result["spread"] = max(run_values) - min(run_values)
    result["points"] = len(run_values)
    result["candidate"] = duration >= FROZEN_CHECK_SECONDS
    result["stage_24h"] = duration >= FROZEN_STAGE_SECONDS
    result["confirmed"] = duration >= FROZEN_CONFIRM_SECONDS
    return result

def frozen_sensor_evidence(current, history):
    return {field: frozen_parameter_evidence(current, history, field)
            for field in ("temperature", "pressure", "humidity")}

def humidity_stuck_score(current, history):
    current_h = safe_float(current.get("humidity"))
    now = row_time(current)

    if current_h is None or now is None:
        return 0.0

    points = []
    for row in history:
        t = row_time(row)
        h = safe_float(row.get("humidity"))
        if t is None or h is None:
            continue
        age = (now - t).total_seconds()
        if 0 <= age <= HUMIDITY_STUCK_SECONDS:
            points.append((t, h))

    if len(points) < 3:
        return 0.0

    values = [h for _, h in points] + [current_h]
    if max(values) - min(values) > HUMIDITY_STUCK_TOLERANCE:
        return 0.0

    duration = (now - points[0][0]).total_seconds()
    if duration < HUMIDITY_STUCK_SECONDS:
        return 0.0

    previous = get_previous(history)
    if not previous:
        return 0.0

    old_t = safe_float(previous.get("temperature"))
    new_t = safe_float(current.get("temperature"))
    old_p = safe_float(previous.get("pressure"))
    new_p = safe_float(current.get("pressure"))

    temp_moved = (old_t is not None and new_t is not None and percentage_change(new_t, old_t) > 5)
    pressure_moved = (old_p is not None and new_p is not None and abs(new_p - old_p) >= 2)

    return 1.0 if (temp_moved or pressure_moved) else 0.0


# ============================================================
# DISTANCE / NEIGHBORS
# ============================================================

def distance_km(lat1, lon1, lat2, lon2):
    lat1 = safe_float(lat1)
    lon1 = safe_float(lon1)
    lat2 = safe_float(lat2)
    lon2 = safe_float(lon2)

    if None in (lat1, lon1, lat2, lon2):
        return None

    r = 6371.0
    p1 = math.radians(lat1)
    p2 = math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)

    a = (math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2)
    return r * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

def nearby_rows(current, batch):
    result = []
    for row in batch:
        if str(row.get("sensor_id")) == str(current.get("sensor_id")):
            continue
        distance = distance_km(
            current.get("latitude"), current.get("longitude"),
            row.get("latitude"), row.get("longitude"),
        )
        if distance is not None and distance <= NEIGHBOR_RADIUS_KM:
            result.append((distance, row))
    return result


# ============================================================
# SPATIAL / REGIONAL LOGIC
# ============================================================

def local_temperature_cluster(current, peers):
    value = safe_float(current.get("temperature"))
    if value is None:
        return {"supported": False, "support_count": 0, "median": None, "spread": None}

    close = []
    for _, peer in peers:
        v = safe_float(peer.get("temperature"))
        if v is not None and abs(v - value) <= LOCAL_CLUSTER_TOLERANCE_C:
            close.append(v)

    supported = len(close) >= LOCAL_CLUSTER_MIN_SUPPORT
    return {
        "supported": supported,
        "support_count": len(close),
        "median": statistics.median(close) if close else None,
        "spread": (max(close) - min(close)) if len(close) > 1 else None,
    }

def temperature_spatial(current, batch):
    temperature = safe_float(current.get("temperature"))
    peers = nearby_rows(current, batch) if temperature is not None else []
    if not peers:
        return {"spatial": 0.0, "regional": 0.5, "count": 0, "median": None, "spread": None,
                "local_cluster_supported": False, "local_cluster_support_count": 0,
                "local_cluster_median": None, "local_cluster_spread": None,
                "nearest_distance_km": None, "peer_movement_median": None, "support_ratio": 0.0}

    weighted, peer_movements, raw_values = [], [], []
    close_support_weight = 0.0
    total_weight = 0.0

    for distance, peer in sorted(peers, key=lambda x: x[0]):
        value = safe_float(peer.get("temperature"))
        if value is None:
            continue
        weight = 1.0 / (1.0 + distance)
        diff = abs(temperature - value)
        weighted.append((weight, value, diff, distance))
        raw_values.append(value)
        total_weight += weight

        if diff <= 5.0:
            close_support_weight += weight

        try:
            ph = get_sensor_history(str(peer.get("sensor_id")), limit=3) or []
            pp = get_previous(ph)
            old = safe_float(pp.get("temperature")) if pp else None
            if old is not None:
                peer_movements.append(percentage_change(value, old))
        except Exception:
            pass

    if not weighted:
        return {"spatial": 0.0, "regional": 0.5, "count": 0, "median": None, "spread": None,
                "local_cluster_supported": False, "local_cluster_support_count": 0,
                "local_cluster_median": None, "local_cluster_spread": None,
                "nearest_distance_km": None, "peer_movement_median": None, "support_ratio": 0.0}

    weighted.sort(key=lambda x: x[2])
    cumulative = 0.0
    weighted_median = weighted[-1][1]
    half = total_weight / 2.0
    for w, value, _, _ in weighted:
        cumulative += w
        if cumulative >= half:
            weighted_median = value
            break

    weighted_diff = abs(temperature - weighted_median)
    support_ratio = close_support_weight / max(total_weight, 1e-9)
    close_count = sum(1 for _, _, diff, d in weighted if diff <= 5 and d <= 15)
    local_supported = close_count >= LOCAL_CLUSTER_MIN_SUPPORT or support_ratio >= 0.55

    nearest_distance = weighted[0][3]
    expected_gradient = 2.0 + 0.12 * nearest_distance
    ratio = weighted_diff / max(expected_gradient, 1.0)
    
    if local_supported:
        spatial = 0.0 if ratio <= 1.5 else clamp((ratio - 1.5) / 3.0)
    else:
        spatial = clamp((ratio - 1.0) / 2.5)

    target_history = get_sensor_history(str(current.get("sensor_id")), limit=3) or []
    target_previous = get_previous(target_history)
    target_change = 0.0
    if target_previous:
        old = safe_float(target_previous.get("temperature"))
        if old is not None:
            target_change = percentage_change(temperature, old)

    if peer_movements:
        peer_movement_median = statistics.median(peer_movements)
        if target_change > 7 and peer_movement_median > 5:
            regional = 0.90
        elif target_change > 5 and peer_movement_median >= 3:
            regional = 0.75
        elif target_change > 7 and peer_movement_median < 3:
            regional = 0.10
        elif target_change > 5 and peer_movement_median < 2:
            regional = 0.20
        else:
            regional = 0.50
    else:
        peer_movement_median = None
        regional = 0.50

    return {
        "spatial": clamp(spatial),
        "regional": clamp(regional),
        "count": len(raw_values),
        "median": weighted_median,
        "spread": max(raw_values) - min(raw_values) if len(raw_values) > 1 else 0.0,
        "local_cluster_supported": bool(local_supported),
        "local_cluster_support_count": int(close_count),
        "local_cluster_median": weighted_median if local_supported else None,
        "local_cluster_spread": max(raw_values) - min(raw_values) if len(raw_values) > 1 else 0.0,
        "nearest_distance_km": nearest_distance,
        "peer_movement_median": peer_movement_median,
        "support_ratio": support_ratio,
    }


def pressure_spatial(current, batch):
    pressure = safe_float(current.get("pressure"))
    empty = {
        "spatial": 0.0, "regional": 0.5, "count": 0, "median": None, "spread": None,
        "nearest_distance_km": None, "nearest_difference": None, "peer_delta_median": None,
        "support_ratio": 0.0, "target_change": 0.0,
    }

    if pressure is None:
        return empty

    peers = nearby_rows(current, batch)
    weighted, peer_changes = [], []
    total_weight = 0.0
    close_weight = 0.0

    for distance, peer in sorted(peers, key=lambda x: x[0]):
        peer_pressure = safe_float(peer.get("pressure"))
        if peer_pressure is None:
            continue
        weight = 1.0 / (1.0 + max(distance, 0.0))
        difference = abs(pressure - peer_pressure)
        weighted.append((weight, peer_pressure, difference, distance))
        total_weight += weight

        tolerance = max(2.0, 1.5 + 0.08 * distance)
        if difference <= tolerance:
            close_weight += weight

        try:
            peer_history = get_sensor_history(str(peer.get("sensor_id")), limit=3) or []
            peer_previous = get_previous(peer_history)
            old_peer_pressure = safe_float(peer_previous.get("pressure")) if peer_previous else None
            if old_peer_pressure is not None:
                peer_changes.append(percentage_change(peer_pressure, old_peer_pressure))
        except Exception:
            pass

    if not weighted:
        return empty

    weighted.sort(key=lambda x: x[1])
    half = total_weight / 2.0
    cumulative = 0.0
    weighted_median = weighted[-1][1]
    for weight, value, _, _ in weighted:
        cumulative += weight
        if cumulative >= half:
            weighted_median = value
            break

    nearest_distance = min(x[3] for x in weighted)
    nearest_difference = min(x[2] for x in weighted)
    spread = max(x[1] for x in weighted) - min(x[1] for x in weighted)
    support_ratio = close_weight / max(total_weight, 1e-12)

    expected_gradient = max(2.0, 1.5 + 0.08 * nearest_distance)
    ratio = abs(pressure - weighted_median) / expected_gradient
    spatial = clamp((ratio - 1.0) / 2.0)

    target_history = get_sensor_history(str(current.get("sensor_id")), limit=3) or []
    target_previous = get_previous(target_history)
    previous_pressure = safe_float(target_previous.get("pressure")) if target_previous else None
    target_change = percentage_change(pressure, previous_pressure)

    peer_delta_median = statistics.median(peer_changes) if peer_changes else None

    if peer_changes and target_change > 5:
        if peer_delta_median is not None and peer_delta_median >= 3:
            regional = 0.85
        elif peer_delta_median is not None and peer_delta_median < 1.5:
            regional = 0.15
        else:
            regional = 0.50
    else:
        regional = 0.50

    return {
        "spatial": spatial, "regional": regional, "count": len(weighted),
        "median": weighted_median, "spread": spread, "nearest_distance_km": nearest_distance,
        "nearest_difference": nearest_difference, "peer_delta_median": peer_delta_median,
        "support_ratio": support_ratio, "target_change": target_change,
    }

def humidity_spatial(current, batch):
    humidity = safe_float(current.get("humidity"))
    peers = nearby_rows(current, batch) if humidity is not None else []
    if not peers:
        return {"spatial": 0.0, "regional": 0.5, "count": 0, "median": None, "spread": None,
                "nearest_distance_km": None, "peer_delta_median": None,
                "nearest_difference": None, "support_ratio": 0.0}

    weighted, peer_deltas = [], []
    total_weight = 0.0
    matching_weight = 0.0

    for distance, peer in sorted(peers, key=lambda x: x[0]):
        value = safe_float(peer.get("humidity"))
        if value is None:
            continue
        weight = 1.0 / (1.0 + distance)
        diff = abs(humidity - value)
        tolerance = min(HUMIDITY_DISTANCE_MAX_TOLERANCE, HUMIDITY_DISTANCE_BASE_TOLERANCE + HUMIDITY_DISTANCE_TOLERANCE_PER_KM * distance)
        weighted.append((weight, value, diff, distance, tolerance))
        total_weight += weight
        
        if diff <= tolerance:
            matching_weight += weight

        try:
            ph = get_sensor_history(str(peer.get("sensor_id")), limit=30) or []
            ref = humidity_5min_reference(ph, peer)
            if ref["available"]:
                peer_deltas.append(abs(ref["delta"]))
            else:
                pp = get_previous(ph)
                old = safe_float(pp.get("humidity")) if pp else None
                if old is not None:
                    peer_deltas.append(abs(value - old))
        except Exception:
            pass

    if not weighted:
        return {"spatial": 0.0, "regional": 0.5, "count": 0, "median": None, "spread": None,
                "nearest_distance_km": None, "peer_delta_median": None,
                "nearest_difference": None, "support_ratio": 0.0}

    weighted.sort(key=lambda x: x[2])
    cumulative = 0.0
    half = total_weight / 2.0
    median = weighted[-1][1]
    for w, value, _, _, _ in weighted:
        cumulative += w
        if cumulative >= half:
            median = value
            break

    nearest = weighted[0]
    nearest_difference = nearest[2]
    nearest_distance = nearest[3]
    nearest_tolerance = nearest[4]
    support_ratio = matching_weight / max(total_weight, 1e-9)

    nearest_ratio = nearest_difference / max(nearest_tolerance, 1.0)
    if support_ratio >= 0.65:
        spatial = 0.0 if nearest_ratio <= 1.5 else clamp((nearest_ratio - 1.5) / 2.5)
    else:
        spatial = clamp((nearest_ratio - 0.8) / 2.0)

    peer_delta_median = statistics.median(peer_deltas) if peer_deltas else None
    target_history = get_sensor_history(str(current.get("sensor_id")), limit=30) or []
    target_ref = humidity_5min_reference(target_history, current)
    target_delta = abs(target_ref["delta"]) if target_ref["available"] else 0.0

    if peer_delta_median is None:
        regional = 0.50
    else:
        movement_difference = abs(target_delta - peer_delta_median)
        if target_delta >= HUMIDITY_EVENT_MIN_POINTS:
            if movement_difference <= max(10.0, 0.35 * target_delta) and support_ratio >= 0.55:
                regional = 0.90
            elif peer_delta_median >= 10 and support_ratio >= 0.45:
                regional = 0.75
            elif peer_delta_median <= 5:
                regional = 0.10
            else:
                regional = 0.35
        elif target_delta >= 10 and peer_delta_median >= 10 and support_ratio >= 0.55:
            regional = 0.75
        else:
            regional = 0.50

    return {
        "spatial": clamp(spatial), "regional": clamp(regional), "count": len(weighted),
        "median": median, "spread": max(x[1] for x in weighted) - min(x[1] for x in weighted) if len(weighted) > 1 else 0.0,
        "nearest_distance_km": nearest_distance, "nearest_difference": nearest_difference,
        "nearest_tolerance": nearest_tolerance, "peer_delta_median": peer_delta_median,
        "support_ratio": support_ratio, "target_5min_delta": target_delta,
    }

# ============================================================
# PHYSICAL GUARDRAILS
# ============================================================

def physical_faults(row):
    faults = []
    humidity = safe_float(row.get("humidity"))
    pressure = safe_float(row.get("pressure"))
    temperature = safe_float(row.get("temperature"))

    if humidity is not None and (humidity < HUMIDITY_MIN or humidity > HUMIDITY_MAX):
        faults.append("Humidity Sensor")
    if pressure is not None and (pressure < 950 or pressure > 1050):
        faults.append("Pressure Sensor")
    if temperature is not None and (temperature >= TEMPERATURE_EXTREME_HIGH or temperature <= TEMPERATURE_EXTREME_LOW):
        faults.append("Extreme Temperature")

    return faults

# ============================================================
# MULTIVARIATE
# ============================================================

def multivariate_score(data):
    temp = data["temperature"]["change"]
    pressure = data["pressure"]["change"]
    humidity = data["humidity"]["delta_points"]
    humidity_complete = bool(data["humidity"].get("window_complete", False))

    abnormal = 0
    if temp > 7: abnormal += 1
    if pressure > 7: abnormal += 1
    if humidity_complete and humidity is not None and humidity >= 10: abnormal += 1

    if abnormal >= 3: return 0.60
    if abnormal >= 2: return 0.35
    return 0.0

# ============================================================
# FAULT TAXONOMY
# ============================================================

def detect_missing_data(current, previous):
    missing = []
    if previous is None:
        return missing
    for field in ("temperature", "pressure", "humidity"):
        current_value = safe_float(current.get(field))
        previous_value = safe_float(previous.get(field))
        if current_value is None and previous_value is not None:
            missing.append(field)
    return missing

def detect_silent_sensors(batch, reference_time=None):
    if get_all_sensors is None:
        return []
    try:
        known_sensors = get_all_sensors() or []
    except Exception:
        return []

    present_ids = {str(row.get("sensor_id")) for row in batch if row.get("sensor_id") is not None}

    if reference_time is None:
        batch_times = [row_time(r) for r in batch if row_time(r) is not None]
        reference_time = max(batch_times) if batch_times else datetime.now(timezone.utc)

    silent_results = []
    for sensor in known_sensors:
        sensor_id = str(sensor.get("sensor_id"))
        if sensor_id in present_ids:
            continue

        try:
            history = get_sensor_history(sensor_id, limit=1) or []
        except Exception:
            history = []

        last_reading = get_previous(history)
        if last_reading is None:
            continue

        last_time = row_time(last_reading)
        if last_time is None:
            continue

        silence_seconds = (reference_time - last_time).total_seconds()
        if silence_seconds < SILENT_SENSOR_WATCH_SECONDS:
            continue

        if silence_seconds >= SILENT_SENSOR_ANOMALY_SECONDS:
            status = "SENSOR_ANOMALY"
            severity = "CRITICAL" if silence_seconds >= SILENT_SENSOR_ANOMALY_SECONDS * 4 else "HIGH"
            score = clamp(0.75 + min(0.20, silence_seconds / (SILENT_SENSOR_ANOMALY_SECONDS * 20)))
            fault_type = f"Communication Timeout (no reading for {silence_seconds / 3600.0:.1f}h)"
        else:
            status = "WATCH"
            severity = "LOW"
            score = 0.40
            fault_type = "None (monitoring)"

        explanation = (f"This station has sent no reading at all for {silence_seconds / 3600.0:.1f} hours "
                       f"(last seen {last_time.isoformat()}), even though it is a registered station. "
                       f"This looks like the sensor itself or its communication link is offline.")

        silent_results.append({
            "sensor_id": sensor_id,
            "latitude": sensor.get("latitude"),
            "longitude": sensor.get("longitude"),
            "state": sensor.get("state", ""),
            "district": sensor.get("district", ""),
            "city": sensor.get("city", ""),
            "locality": sensor.get("locality", ""),
            "temperature": None,
            "pressure": None,
            "humidity": None,
            "timestamp": reference_time.isoformat(),
            "observation_time": reference_time.isoformat(),
            "status": status,
            "temperature_status": "NO_DATA",
            "humidity_status": "NO_DATA",
            "pressure_status": "NO_DATA",
            "humidity_observation_state": "NO_DATA",
            "anomaly_category": status,
            "fault_type": fault_type,
            "missing_data_fields": ["temperature", "pressure", "humidity"],
            "severity": severity,
            "score": round(score, 3),
            "confidence": round(score * 100, 2),
            "health": round(max(0.0, 100.0 - score * 100.0), 2),
            "strongest_parameter": None,
            "strongest_change": 0.0,
            "time_gap_seconds": round(silence_seconds, 2),
            "last_seen": last_time.isoformat(),
            "silence_seconds": round(silence_seconds, 2),
            "physical_faults": [],
            "frozen_status": "NORMAL",
            "frozen_field": None,
            "explanation": explanation,
            "evidence_summary": f"sensor_id={sensor_id}: no reading received for {silence_seconds:.0f}s",
        })

    return silent_results

def classify_fault_type(status, strongest_field, physical, frozen_status, frozen_field, missing_fields, selected, strongest_change, strongest_gap):
    if missing_fields:
        label = " / ".join(field.title() for field in missing_fields)
        return f"Communication Dropout / Missing Data ({label})"

    if status == "NORMAL": return "None"
    if status == "WATCH": return "None (monitoring)"
    if status == "WEATHER_EVENT": return "None (genuine regional weather event)"
    if status == "GLITCH_SUSPECT": return f"{strongest_field.title()} Frozen/Fixed Value (24h, unconfirmed)"
    if frozen_status == "SENSOR_ANOMALY" and frozen_field == strongest_field: return f"{strongest_field.title()} Frozen/Stuck Sensor (48h confirmed)"
    if physical: return physical[0]

    change_score = safe_float(selected.get("change_score"), 0.0)
    persistence = safe_float(selected.get("persistence"), 0.0)
    history_dev = safe_float(selected.get("history"), 0.0)
    gap = strongest_gap or 0.0

    is_sudden = change_score >= 0.70 and gap <= 300
    is_drift = not is_sudden and persistence >= 0.70 and history_dev >= 0.50

    if is_sudden: return f"{strongest_field.title()} Sudden Spike / Step Fault"
    if is_drift: return f"{strongest_field.title()} Calibration Drift"
    return f"{strongest_field.title()} Sensor Anomaly"

def generate_explanation(status, fault_type, strongest_field, current, previous, strongest_change, strongest_gap, selected_spatial, missing_fields, frozen_evidence):
    # Humidity's decision is driven by the 5-minute-window RH-point delta
    # (the same value reported as "humidity_change_points" in the output),
    # not a simple percentage change between two adjacent readings. Using
    # the right unit here keeps the explanation text consistent with the
    # evidence fields instead of quoting an unrelated (often ~0%) figure.
    if strongest_field == "humidity":
        change_txt = f"{strongest_change:.1f} percentage points"
    else:
        change_txt = f"{strongest_change:.1f}%"

    if missing_fields:
        label = ", ".join(missing_fields)
        return f"No {label} reading arrived in the latest packet even though the station reported a valid value previously. This looks like a communication dropout or data-logger fault, not a real environmental change."

    if status == "NORMAL":
        return f"{strongest_field.title()} is within its normal, expected range."

    if status == "WATCH":
        return f"{strongest_field.title()} changed by {change_txt} since the last reading -- larger than usual, but not yet confirmed as a fault. Being monitored."

    if status == "WEATHER_EVENT":
        return f"{strongest_field.title()} changed by {change_txt}, and nearby stations show a similar shift (regional agreement={selected_spatial.get('regional', 0.0):.0%}), so this looks like a genuine local weather event rather than a sensor fault."

    if status == "GLITCH_SUSPECT":
        duration_h = frozen_evidence[strongest_field]["duration_seconds"] / 3600.0
        return f"{strongest_field.title()} has reported the exact same value for {duration_h:.1f} hours -- past the 24-hour suspect threshold but not yet the 48-hour confirmation threshold for a frozen sensor."

    if "Frozen" in fault_type or "Stuck" in fault_type:
        duration_h = frozen_evidence[strongest_field]["duration_seconds"] / 3600.0
        stuck_value = frozen_evidence[strongest_field].get("value")
        return f"{strongest_field.title()} has been stuck at {stuck_value} for {duration_h:.1f} hours, well past the 48-hour confirmation threshold. Most likely the sensor is physically stuck or disconnected."

    if "Drift" in fault_type:
        return f"{strongest_field.title()} has been steadily deviating from its expected value over a sustained period with no sudden jump -- a pattern typical of slow calibration drift rather than a one-off spike."

    if "Spike" in fault_type or "Step" in fault_type:
        gap_txt = f"{strongest_gap:.0f}s" if strongest_gap else "an unknown interval"
        current_value = current.get(strongest_field)
        previous_value = previous.get(strongest_field) if previous else None
        return f"{strongest_field.title()} jumped from {previous_value} to {current_value} in {gap_txt} ({change_txt} change) -- too fast to be physically plausible, and nearby stations don't show the same change (regional agreement={selected_spatial.get('regional', 0.0):.0%}). This looks like a sensor spike rather than real weather."

    if fault_type in ("Humidity Sensor", "Pressure Sensor", "Extreme Temperature"):
        current_value = current.get(strongest_field)
        return f"{strongest_field.title()} reading of {current_value} is outside the physically plausible operating range for this instrument."

    return f"{strongest_field.title()} changed by {change_txt} and disagrees with nearby stations (regional agreement={selected_spatial.get('regional', 0.0):.0%}), which points to a sensor-level fault rather than real weather."

# ============================================================
# ML - SUPPORT ONLY
# ============================================================

def ml_score(features):
    if predict_probability is None:
        return 0.0
    try:
        return clamp(predict_probability(features))
    except Exception:
        return 0.0

def temperature_rule_decision(temperature, change, gap, history_score, spatial_score, regional, persistence, local_cluster_supported, sequence=None):
    temperature = safe_float(temperature)
    change = abs(safe_float(change, 0.0))
    history_score = clamp(history_score)
    spatial_score = clamp(spatial_score)
    regional = clamp(regional)
    persistence = clamp(persistence)
    sequence = sequence or {}

    strong = []
    supporting = []

    hard_extreme = (temperature is not None and (temperature >= TEMPERATURE_EXTREME_HIGH or temperature <= TEMPERATURE_EXTREME_LOW))
    if hard_extreme:
        if local_cluster_supported and regional >= 0.70:
            return "WEATHER_EVENT", 0.80, strong, ["regional_extreme_supported"]
        return "SENSOR_ANOMALY", 0.90, ["temperature_absolute_boundary"], supporting

    if sequence.get("persistent_level"):
        if local_cluster_supported and spatial_score < 0.35:
            return "WEATHER_EVENT", 0.75, strong, ["persistent_temperature_level_supported_by_nearby_sensors"]
        if spatial_score >= 0.70 or regional < 0.30:
            return "SENSOR_ANOMALY", 0.85, ["persistent_temperature_level_isolated_from_nearby_sensors"], supporting
        return "WATCH", 0.60, strong, ["persistent_temperature_level_requires_neighbor_confirmation"]

    if change <= 5.0:
        return "NORMAL", 0.10, strong, supporting
    if change <= 7.0:
        if regional >= 0.80:
            return "WEATHER_EVENT", 0.50, strong, ["possible_weather_movement"]
        return "WATCH", 0.35, strong, ["temperature_5_to_7_percent_flag"]

    if regional >= 0.80:
        return "WEATHER_EVENT", max(0.70, regional), strong, ["regional_weather_support"]

    # A very large temperature jump/drop is treated as an anomaly on its own
    # when it occurs within the operational observation window.  Regional
    # agreement is checked first so genuine widespread weather movement can
    # still be classified as WEATHER_EVENT.
    absolute_jump = None
    if temperature is not None and gap is not None and gap <= 600.0:
        # `change` is percentage-based; use the previous absolute movement
        # supplied through the sequence when available.
        if sequence.get("pre_jump_value") is not None:
            absolute_jump = abs(temperature - safe_float(sequence.get("pre_jump_value"), temperature))

    if change > 15.0 and gap is not None and gap <= 600.0 and regional < 0.80:
        return "SENSOR_ANOMALY", 0.90, strong or ["very_large_temperature_step"], supporting

    if absolute_jump is not None and absolute_jump >= 5.0 and regional < 0.80:
        return "SENSOR_ANOMALY", 0.88, strong or ["large_absolute_temperature_jump"], supporting

    if sequence.get("erratic"): strong.append("non_sequential_temperature_movement")
    if spatial_score >= 0.80 and not local_cluster_supported: strong.append("nearby_sensor_disagreement")

    if history_score >= 0.85: supporting.append("strong_sensor_history_deviation")
    elif history_score >= 0.50: supporting.append("sensor_history_deviation")
    if persistence >= 0.80 and not local_cluster_supported: supporting.append("persistent_sensor_isolation")

    if change > 15 and gap is not None and gap <= 60 and regional < 0.50:
        return "SENSOR_ANOMALY", 0.90, strong or ["very_large_sudden_change"], supporting

    if strong:
        return "SENSOR_ANOMALY", 0.82, strong, supporting

    support_count = sum([history_score >= 0.50, persistence >= 0.70, regional < 0.30])
    if change > 10 and gap is not None and gap <= 120 and support_count >= 2:
        return "SENSOR_ANOMALY", 0.75, strong, supporting

    return "WATCH", max(0.35, 0.45 if change > 10 else 0.35), strong, supporting

def final_decision(hybrid, ml, physical, change, gap, regional, history_score, persistence):
    if physical: return "SENSOR_ANOMALY", max(0.85, hybrid)
    if change > 7 and regional >= 0.80: return "WEATHER_EVENT", (HYBRID_WEIGHT * hybrid + ML_WEIGHT * ml)
    if change > 15 and gap is not None and gap <= 60 and regional < 0.50: return "SENSOR_ANOMALY", max(0.85, HYBRID_WEIGHT * hybrid + ML_WEIGHT * ml)
    
    support = sum([history_score >= 0.50, persistence >= 0.70, regional < 0.30, ml >= 0.75])
    if change > 10 and gap is not None and gap <= 120 and support >= 2: return "SENSOR_ANOMALY", (HYBRID_WEIGHT * hybrid + ML_WEIGHT * ml)
    if hybrid >= 0.55: return "WATCH", (HYBRID_WEIGHT * hybrid + ML_WEIGHT * ml)
    return "NORMAL", (HYBRID_WEIGHT * hybrid + ML_WEIGHT * ml)


# ============================================================
# MAIN DETECTOR
# ============================================================

def detect_anomaly(data):
    batch = normalize_input(data)
    if not batch:
        return []

    results = []

    for current in batch:
        sensor_id = str(current.get("sensor_id", ""))

        # --------------------------------------------------------
        # Base Information Dictionary (DRY Optimization)
        # --------------------------------------------------------
        base_metadata = {
            "sensor_id": sensor_id,
            "latitude": current.get("latitude"),
            "longitude": current.get("longitude"),
            "state": current.get("state", ""),
            "district": current.get("district", ""),
            "city": current.get("city", ""),
            "locality": current.get("locality", ""),
            "temperature": current.get("temperature"),
            "pressure": current.get("pressure"),
            "humidity": current.get("humidity"),
            "timestamp": current.get("timestamp", current.get("observation_time", "")),
            "observation_time": current.get("observation_time", current.get("timestamp", "")),
            "actual_fetch_time": current.get("actual_fetch_time"),
            "weather_model_time": current.get("weather_model_time"),
            "api_status": current.get("api_status", "UNKNOWN"),
            "api_error": current.get("api_error", ""),
        }

        # --------------------------------------------------------
        # LIVE API / DATA AVAILABILITY GUARD
        # --------------------------------------------------------
        api_status = str(current.get("api_status", "UNKNOWN")).strip().upper()
        api_error = str(current.get("api_error", "")).strip()

        weather_values = [
            safe_float(current.get("temperature")),
            safe_float(current.get("pressure")),
            safe_float(current.get("humidity"))
        ]

        api_failure_statuses = {"API_TIMEOUT", "API_ERROR", "NETWORK_ERROR", "CONFIG_ERROR", "INVALID_RESPONSE"}
        weather_data_missing = any(value is None for value in weather_values)

        if api_status in api_failure_statuses or (api_status == "API_OK" and weather_data_missing):
            explanation = f"Weather data could not be analyzed because the live data source returned {api_status}."
            if api_error: explanation += f" Error: {api_error}"
            
            evidence_summary = f"sensor_id={sensor_id}: api_status={api_status}"
            if api_error: evidence_summary += f", api_error={api_error}"
            evidence_summary += "; no reading was analyzed."

            results.append({
                **base_metadata,
                "status": "DATA_UNAVAILABLE",
                "temperature_status": "DATA_UNAVAILABLE",
                "pressure_status": "DATA_UNAVAILABLE",
                "humidity_status": "DATA_UNAVAILABLE",
                "anomaly_category": "API_FAILURE",
                "fault_type": "DATA_SOURCE_FAILURE",
                "severity": "INFO",
                "score": 0.0,
                "confidence": 0.0,
                "health": None,
                "missing_data_fields": [],
                "spatial_score": None,
                "regional_agreement": None,
                "api_status": api_status,
                "api_error": api_error,
                "analysis_available": False,
                "explanation": explanation,
                "evidence_summary": evidence_summary,
            })
            continue

        # --------------------------------------------------------
        # PARAMETER EVALUATION
        # --------------------------------------------------------
        try:
            history = get_sensor_history(sensor_id, limit=max(HISTORY_LIMIT, FROZEN_HISTORY_LIMIT)) or []
        except Exception:
            history = []
        history.sort(key=lambda x: row_time(x) or datetime.min.replace(tzinfo=timezone.utc))

        previous = get_previous(history)
        gap = time_gap(current, previous)
        missing_fields = detect_missing_data(current, previous)
        parameters = {}

        for field in ("temperature", "pressure", "humidity"):
            current_value = safe_float(current.get(field))
            previous_value = safe_float(previous.get(field)) if previous else None

            if field == "humidity":
                delta_points = abs(current_value - previous_value) if current_value is not None and previous_value is not None else 0.0
                humidity_window = humidity_5min_reference(history, current)
                humidity_gap = humidity_window["duration"] if humidity_window["available"] else gap
                window_delta = abs(humidity_window["delta"]) if humidity_window["available"] else delta_points
                change_score = humidity_change_score(window_delta, humidity_gap)
                history_value, median, mad = historical_score(current_value, history, "humidity")
                persistence = persistence_score(history, "humidity")

                parameters[field] = {
                    "change": percentage_change(current_value, previous_value),
                    "delta_points": delta_points,
                    "window_delta": window_delta,
                    "window_gap": humidity_gap,
                    "change_score": change_score,
                    "history": history_value,
                    "persistence": persistence,
                    "median": median,
                    "mad": mad,
                    "window_complete": bool(humidity_window.get("available")),
                }
            else:
                change = percentage_change(current_value, previous_value)
                change_score = sudden_change_score(change, gap)
                history_value, median, mad = historical_score(current_value, history, field)
                persistence = persistence_score(history, field)

                parameters[field] = {
                    "change": change,
                    "delta_points": None,
                    "window_delta": None,
                    "window_gap": None,
                    "change_score": change_score,
                    "history": history_value,
                    "persistence": persistence,
                    "median": median,
                    "mad": mad,
                }

        # ----------------------------------------------------
        # HUMIDITY SPECIAL EVIDENCE
        # ----------------------------------------------------
        humidity_window = humidity_5min_reference(history, current)
        humidity_stuck = humidity_stuck_score(current, history)
        frozen_evidence = frozen_sensor_evidence(current, history)
        humidity_spatial_result = humidity_spatial(current, batch)
        pressure_spatial_result = pressure_spatial(current, batch)

        # ----------------------------------------------------
        # INDEPENDENT TEMPERATURE + HUMIDITY DECISIONS
        # ----------------------------------------------------
        physical = physical_faults(current)
        temperature_spatial_result = temperature_spatial(current, batch)
        temperature_sequence = temperature_sequence_analysis(history, current)

        temp_decision, temp_score, temp_strong_rules, temp_supporting = temperature_rule_decision(
            temperature=current.get("temperature"),
            change=parameters["temperature"]["change"],
            gap=gap,
            history_score=parameters["temperature"]["history"],
            spatial_score=temperature_spatial_result["spatial"],
            regional=temperature_spatial_result["regional"],
            persistence=parameters["temperature"]["persistence"],
            local_cluster_supported=temperature_spatial_result.get("local_cluster_supported", False),
            sequence=temperature_sequence,
        )

        humidity_delta = parameters["humidity"]["window_delta"]
        humidity_gap = parameters["humidity"]["window_gap"]
        regional_h = humidity_spatial_result["regional"]
        humidity_spatial_score = humidity_spatial_result["spatial"]
        peer_delta = humidity_spatial_result.get("peer_delta_median")
        nearest_diff = humidity_spatial_result.get("nearest_difference")
        support_ratio = humidity_spatial_result.get("support_ratio", 0.0)
        humidity_rate = humidity_delta / (humidity_gap / 60.0) if humidity_gap and humidity_gap > 0 else 0.0

        humidity_rules, humidity_strong = [], []
        humidity_evidence = 0.0
        humidity_window_available = bool(humidity_window.get("available"))
        humidity_observation_seconds = safe_float(humidity_window.get("duration"), 0.0)

        if not humidity_window_available or humidity_observation_seconds < HUMIDITY_MIN_OBSERVATION_SECONDS:
            humidity_status = "INSUFFICIENT_OBSERVATION"
            humidity_score = 0.0
            humidity_rules.append("humidity_requires_5min_observation")
        else:
            if humidity_delta >= HUMIDITY_EVENT_MIN_POINTS:
                humidity_rules.append("significant_5min_humidity_change")
                humidity_evidence = max(humidity_evidence, 0.45)
            if humidity_delta >= HUMIDITY_EVENT_MAX_POINTS:
                humidity_rules.append("large_5min_humidity_change")
                humidity_evidence = max(humidity_evidence, 0.60)
            if humidity_rate >= HUMIDITY_RATE_STRONG_POINTS_PER_MIN:
                humidity_rules.append("rapid_humidity_change_rate")
                humidity_evidence = max(humidity_evidence, 0.55)

            recent_rows = []
            now_h_time = row_time(current)
            if now_h_time is not None:
                for row in history:
                    rt = row_time(row)
                    rh = safe_float(row.get("humidity"))
                    if rt is None or rh is None: continue
                    age = (now_h_time - rt).total_seconds()
                    if 0 <= age <= HUMIDITY_PERSISTENT_WINDOW_SECONDS:
                        recent_rows.append((rt, rh))
                current_h = safe_float(current.get("humidity"))
                if current_h is not None:
                    recent_rows.append((now_h_time, current_h))

            recent_rows.sort(key=lambda x: x[0])
            steps = []
            for a, b in zip(recent_rows, recent_rows[1:]):
                step = b[1] - a[1]
                step_gap = (b[0] - a[0]).total_seconds()
                if step_gap > 0:
                    steps.append({
                        "delta": step, "abs_delta": abs(step), "gap": step_gap,
                        "direction": 1 if step > 0 else (-1 if step < 0 else 0),
                    })

            meaningful_steps = [x for x in steps if x["abs_delta"] >= 0.5]
            small_steps = [x for x in meaningful_steps if x["abs_delta"] <= HUMIDITY_SMALL_STEP_POINTS]
            latest_humidity_step = meaningful_steps[-1] if meaningful_steps else None
            direction_flips = 0
            for a, b in zip(meaningful_steps, meaningful_steps[1:]):
                if a["direction"] and b["direction"] and a["direction"] != b["direction"]:
                    direction_flips += 1

            decrease_steps = [x for x in meaningful_steps if x["delta"] < 0]
            orderly_decrease = (len(decrease_steps) >= 2 and len(small_steps) >= max(2, len(decrease_steps) // 2) and direction_flips <= 2 and humidity_observation_seconds >= HUMIDITY_GRADUAL_WINDOW_SECONDS)
            orderly_pattern = (len(small_steps) >= 2 and direction_flips <= 2)

            if orderly_decrease or orderly_pattern:
                humidity_rules.append("orderly_humidity_sequence")
                humidity_evidence = min(humidity_evidence, 0.40)
            if humidity_window.get("reverted"):
                humidity_rules.append("humidity_excursion_reverted")
                humidity_evidence = min(humidity_evidence, 0.35)

            local_weather_support = (humidity_delta >= HUMIDITY_EVENT_MIN_POINTS and peer_delta is not None and peer_delta >= max(5.0, 0.65 * humidity_delta) and regional_h >= 0.70 and support_ratio >= 0.50)
            nearest_disagreement = (humidity_delta >= HUMIDITY_EVENT_MIN_POINTS and humidity_spatial_score >= 0.65 and support_ratio < 0.45)
            persistent_fault_support = (humidity_observation_seconds >= HUMIDITY_PERSISTENT_WINDOW_SECONDS and humidity_stuck >= 1.0 and not local_weather_support)

            if persistent_fault_support:
                humidity_strong.append("persistent_humidity_stuck")
                humidity_evidence = max(humidity_evidence, 0.85)
            if nearest_disagreement:
                humidity_strong.append("nearest_sensor_humidity_disagreement")
                humidity_evidence = max(humidity_evidence, 0.75)

            sudden_humidity_step = bool(
                latest_humidity_step
                and latest_humidity_step.get("abs_delta", 0.0) >= 10.0
                and latest_humidity_step.get("gap", 0.0) <= 600.0
            )

            if local_weather_support:
                humidity_status, humidity_score = "WEATHER_EVENT", max(0.70, regional_h)
            elif sudden_humidity_step and regional_h < 0.70:
                humidity_status, humidity_score = "SENSOR_ANOMALY", max(0.85, humidity_evidence, 0.85)
                humidity_strong.append("sudden_humidity_step")
            elif humidity_delta >= 15.0 and regional_h < 0.60 and not orderly_pattern:
                humidity_status, humidity_score = "SENSOR_ANOMALY", max(0.85, humidity_evidence)
            elif persistent_fault_support:
                humidity_status, humidity_score = "SENSOR_ANOMALY", max(0.85, humidity_evidence)
            elif nearest_disagreement and not orderly_pattern:
                humidity_status, humidity_score = "SENSOR_ANOMALY", max(0.75, humidity_evidence)
            elif humidity_delta >= HUMIDITY_EVENT_MAX_POINTS and regional_h < 0.40 and not orderly_pattern:
                humidity_status, humidity_score = "WATCH", max(0.55, humidity_evidence)
            elif humidity_delta >= HUMIDITY_EVENT_MIN_POINTS:
                humidity_status, humidity_score = "WATCH", max(0.40, humidity_evidence)
            elif orderly_pattern or humidity_delta < HUMIDITY_EVENT_MIN_POINTS:
                humidity_status, humidity_score = "NORMAL", max(0.10, min(0.35, humidity_evidence))
            else:
                humidity_status, humidity_score = "WATCH", max(0.30, humidity_evidence)

        # ----------------------------------------------------
        # PRESSURE / PHYSICAL DECISION
        # ----------------------------------------------------
        pressure_physical = [f for f in physical if f == "Pressure Sensor"]
        pressure_change = parameters["pressure"]["change"]
        pressure_persistence = parameters["pressure"]["persistence"]
        pressure_change_score = parameters["pressure"]["change_score"]
        pressure_spatial_score = pressure_spatial_result.get("spatial", 0.0)
        pressure_regional = pressure_spatial_result.get("regional", 0.5)

        if pressure_physical:
            pressure_status, pressure_score = "SENSOR_ANOMALY", 0.95
        elif current.get("pressure") is None or previous is None or previous.get("pressure") is None:
            pressure_status, pressure_score = "NORMAL", 0.0
        elif pressure_change <= 5.0:
            pressure_status, pressure_score = "NORMAL", pressure_change_score
        elif pressure_change <= 7.0:
            pressure_status, pressure_score = "WATCH", max(0.20, pressure_change_score)
        elif pressure_spatial_score >= 0.65 and pressure_regional < 0.40:
            pressure_status, pressure_score = "SENSOR_ANOMALY", max(0.75, pressure_change_score)
        elif pressure_change > 10.0 and pressure_persistence >= 0.70 and pressure_regional < 0.50:
            pressure_status, pressure_score = "SENSOR_ANOMALY", max(0.70, pressure_change_score)
        else:
            pressure_status, pressure_score = "WATCH", max(0.35, pressure_change_score)

        # ----------------------------------------------------
        # FROZEN / FIXED VALUE DECISION
        # ----------------------------------------------------
        frozen_confirmed = [f for f, e in frozen_evidence.items() if e["confirmed"]]
        frozen_stage_24h = [f for f, e in frozen_evidence.items() if e.get("stage_24h", False)]
        frozen_candidates = [f for f, e in frozen_evidence.items() if e["candidate"]]

        if frozen_confirmed:
            frozen_field = max(frozen_confirmed, key=lambda f: frozen_evidence[f]["duration_seconds"])
            frozen_status, frozen_score = "SENSOR_ANOMALY", 0.95
        elif frozen_stage_24h:
            frozen_field = max(frozen_stage_24h, key=lambda f: frozen_evidence[f]["duration_seconds"])
            frozen_status, frozen_score = "GLITCH_SUSPECT", 0.75
        elif frozen_candidates:
            frozen_field = max(frozen_candidates, key=lambda f: frozen_evidence[f]["duration_seconds"])
            frozen_status, frozen_score = "WATCH", 0.60
        else:
            frozen_field, frozen_status, frozen_score = None, "NORMAL", 0.0

        # ----------------------------------------------------
        # OVERALL SENSOR STATE
        # ----------------------------------------------------
        decisions = [
            ("temperature", temp_decision, temp_score),
            ("humidity", humidity_status, humidity_score),
            ("pressure", pressure_status, pressure_score),
        ]
        if frozen_status != "NORMAL":
            decisions.append((frozen_field, frozen_status, frozen_score))
        for missing_field in missing_fields:
            decisions.append((missing_field, "SENSOR_ANOMALY", 0.90))

        anomalies = [d for d in decisions if d[1] == "SENSOR_ANOMALY"]
        weather_events = [d for d in decisions if d[1] == "WEATHER_EVENT"]
        glitch_suspects = [d for d in decisions if d[1] == "GLITCH_SUSPECT"]
        watches = [d for d in decisions if d[1] == "WATCH"]

        if anomalies:
            strongest_field, status, score = max(anomalies, key=lambda x: x[2])
        elif weather_events:
            strongest_field, status, score = max(weather_events, key=lambda x: x[2])
        elif glitch_suspects:
            strongest_field, status, score = max(glitch_suspects, key=lambda x: x[2])
        elif watches:
            strongest_field, status, score = max(watches, key=lambda x: x[2])
        else:
            strongest_field = max(decisions, key=lambda x: x[2])[0]
            status, score = "NORMAL", 0.10

        if strongest_field == "temperature":
            selected = parameters["temperature"]
            selected_spatial = temperature_spatial_result
            strongest_change = selected["change"]
            strongest_gap = gap
            history_score = selected["history"]
            persistence = selected["persistence"]
        elif strongest_field == "humidity":
            selected = parameters["humidity"]
            selected_spatial = humidity_spatial_result
            # Humidity is decided from the 5-minute-window RH-point delta
            # ("window_delta", same value exposed as humidity_change_points
            # in the output), not the adjacent-reading percentage change in
            # "change" -- using "change" here made the explanation text
            # ("Humidity changed by 0.0%") disagree with the evidence that
            # actually triggered the decision.
            strongest_change = selected["window_delta"] if selected.get("window_delta") is not None else selected["change"]
            strongest_gap = selected["window_gap"]
            history_score = selected["history"]
            persistence = selected["persistence"]
        else:
            selected = parameters["pressure"]
            selected_spatial = pressure_spatial_result
            strongest_change = selected["change"]
            strongest_gap = gap
            history_score = selected["history"]
            persistence = selected["persistence"]

        # ----------------------------------------------------
        # MULTIVARIATE + SUPPORTING ML
        # ----------------------------------------------------
        multivariate = multivariate_score(parameters)
        change_score = selected["change_score"]
        hybrid = clamp(
            0.40 * change_score + 0.20 * history_score + 0.15 * selected_spatial["spatial"] +
            0.10 * persistence + 0.10 * multivariate + 0.05 * (1.0 - selected_spatial["regional"])
        )

        if strongest_field == "humidity" and not humidity_window["available"]:
            hybrid *= 0.70

        ml_change = selected["window_delta"] if strongest_field == "humidity" else strongest_change
        ml_gap = selected["window_gap"] or 0.0 if strongest_field == "humidity" else strongest_gap or 0.0
        rate = ml_change / ml_gap * 60.0 if ml_gap > 0 else 0.0

        ml_features = build_ml_features(
            change_percent=ml_change, change_rate=rate, historical_score=history_score,
            spatial_score=selected_spatial["spatial"], regional_agreement=selected_spatial["regional"],
            persistence_score=persistence, multivariate_score=multivariate, time_gap_seconds=ml_gap,
        )

        ml = ml_score(ml_features)
        ml_fault_prediction = predict_fault_type(ml_features) if predict_fault_type is not None else None

        ml_agrees = None
        if ml >= 0.75: ml_agrees = status in ("SENSOR_ANOMALY", "WEATHER_EVENT")
        elif ml <= 0.25: ml_agrees = status == "NORMAL"

        # ----------------------------------------------------
        # OUTPUT DECISION AND EXPLANATION
        # ----------------------------------------------------
        score = clamp(score)

        if status == "NORMAL":
            severity, health, category, fault_type, confidence = "NONE", 100.0, "NORMAL", "None", (1.0 - score) * 100
        elif status == "WATCH":
            severity, health, category, fault_type, confidence = "LOW", max(0.0, 100.0 - score * 25.0), "WATCH", "None (monitoring)", score * 100
        elif status == "GLITCH_SUSPECT":
            severity, health, category, fault_type, confidence = "MEDIUM", max(0.0, 100.0 - score * 35.0), "GLITCH_SUSPECT", f"{strongest_field.title()} Frozen/Fixed Value (24h, unconfirmed)", score * 100
        elif status == "WEATHER_EVENT":
            severity, health, category, fault_type, confidence = "MEDIUM", 100.0, "REGIONAL_WEATHER_EVENT", "None (genuine regional weather event)", score * 100
        else:
            if score >= 0.90: severity = "CRITICAL"
            elif score >= 0.75: severity = "HIGH"
            elif score >= 0.55: severity = "MEDIUM"
            else: severity = "LOW"
            health = max(0.0, 100.0 - score * 55.0)
            category = "SENSOR_ANOMALY"
            fault_type = classify_fault_type(status, strongest_field, physical, frozen_status, frozen_field, missing_fields, selected, strongest_change, strongest_gap)
            confidence = score * 100

        explanation = generate_explanation(status, fault_type, strongest_field, current, previous, strongest_change, strongest_gap, selected_spatial, missing_fields, frozen_evidence)

        if strongest_field == "humidity":
            evidence_summary = (f"humidity: {parameters['humidity']['window_delta']:.2f} RH points over {parameters['humidity']['window_gap']:.1f}s (5-minute humidity window), "
                                f"history={parameters['humidity']['history']:.2f}, spatial={humidity_spatial_result['spatial']:.2f}, "
                                f"regional={humidity_spatial_result['regional']:.2f}, persistence={parameters['humidity']['persistence']:.2f}.")
        else:
            evidence_summary = (f"{strongest_field}: {strongest_change:.2f}% change, gap={strongest_gap}, history={history_score:.2f}, "
                                f"spatial={selected_spatial['spatial']:.2f}, regional={selected_spatial['regional']:.2f}, persistence={persistence:.2f}.")

        if frozen_confirmed: evidence_summary += " fixed-value sensor anomaly: " + ", ".join(frozen_confirmed) + " unchanged for at least 24 hours."
        elif frozen_candidates: evidence_summary += " frozen-value candidate: " + ", ".join(frozen_candidates) + " unchanged for at least 30 minutes."
        if missing_fields: evidence_summary += " missing/blank fields: " + ", ".join(missing_fields) + "."

        ml_fault_type_agrees = None
        if ml_fault_prediction is not None:
            predicted_class = ml_fault_prediction["class"]
            expected_class = None
            if category == "NORMAL": expected_class = "NORMAL"
            elif category == "REGIONAL_WEATHER_EVENT": expected_class = "WEATHER_EVENT"
            elif category == "SENSOR_ANOMALY":
                if "Spike" in fault_type or "Step" in fault_type: expected_class = "SUDDEN_SPIKE"
                elif "Drift" in fault_type: expected_class = "CALIBRATION_DRIFT"
                else: expected_class = "ISOLATED_SENSOR_FAULT"
            if expected_class is not None:
                ml_fault_type_agrees = (predicted_class == expected_class)

        # ----------------------------------------------------
        # EXPECTED-VALUE PREDICTION INTEGRATION
        # ----------------------------------------------------
        prediction_enabled = status in ("WATCH", "GLITCH_SUSPECT", "SENSOR_ANOMALY")

        prediction_data = generate_prediction(
            current, history, batch, strongest_field
        ) if prediction_enabled else {
            "prediction_available": False, "expected_value": None, "trend_points": 0,
            "spatial_information": {"neighbor_count": 0, "value": None},
            "prediction_components": {}, "prediction_weights": {}, "prediction_confidence": None,
            "prediction_confidence_label": None,
            "prediction_actual_value": safe_float(current.get(strongest_field)), "prediction_deviation": None,
            "prediction_absolute_deviation": None, "prediction_field": strongest_field, "suggested_value": None,
            "prediction_neighbor_count": 0, "seasonal_points_used": 0, "seasonal_uses_day_of_year": False,
            "seasonal_uses_day_of_week": False, "seasonal_uses_month": False, "seasonal_history_span_days": 0.0,
            "seasonal_strength": 0.0, "seasonal_strength_label": None, "trend_points_used": 0,
            "trend_sequence": [], "prediction_nearest_distance_km": None,
        }

        # Additive all-field prediction data for the seasonal dashboard.
        # These values never replace raw sensor readings and never affect
        # the anomaly decision above.
        field_predictions = (
            generate_all_predictions(current, history, batch)
            if prediction_enabled else {}
        )

        # Assemble output payload using dictionary unpacking for clean code layout
        results.append({
            **base_metadata,
            **prediction_data,
            "field_predictions": field_predictions,

            "status": status,
            "temperature_status": temp_decision,
            "humidity_status": humidity_status,
            "pressure_status": pressure_status,
            "humidity_observation_state": (
                "INSUFFICIENT_OBSERVATION" if not humidity_window.get("available")
                else "MINIMUM_5_MIN" if humidity_observation_seconds < 600
                else "EXTENDED_5_TO_30_MIN" if humidity_observation_seconds < 1800
                else "PERSISTENT_OVER_30_MIN"
            ),
            "anomaly_category": category,
            "fault_type": fault_type,
            "missing_data_fields": missing_fields,
            "severity": severity,
            "score": round(score, 3),
            "confidence": round(confidence, 2),
            "health": round(health, 2),

            "temperature_evidence_score": round(temp_score, 3),
            "humidity_evidence_score": round(humidity_score, 3),
            "pressure_evidence_score": round(pressure_score, 3),
            "strongest_parameter": strongest_field,
            "strongest_change": round(strongest_change, 3),
            "time_gap_seconds": round(strongest_gap, 2) if strongest_gap is not None else None,

            "temperature_change": round(parameters["temperature"]["change"], 3),
            "pressure_change": round(parameters["pressure"]["change"], 3),
            "pressure_spatial_score": round(pressure_spatial_result.get("spatial", 0.0), 3),
            "pressure_regional_support": round(pressure_spatial_result.get("regional", 0.5), 3),
            "pressure_nearest_difference": pressure_spatial_result.get("nearest_difference"),
            "humidity_change_points": round(parameters["humidity"]["window_delta"], 3),
            "humidity_observation_complete": bool(humidity_window["available"]),
            "humidity_observation_seconds": round(humidity_window["duration"], 2),
            "humidity_observation_baseline": humidity_window["baseline"],
            "humidity_observation_delta": round(humidity_window["delta"], 3),
            "humidity_observation_stable": bool(humidity_window["stable_new_level"]),
            "humidity_observation_reverted": bool(humidity_window["reverted"]),
            "humidity_stuck_score": round(humidity_stuck, 3),

            "historical_median": parameters["temperature"]["median"],
            "historical_mad": parameters["temperature"]["mad"],
            "temperature_score": round(parameters["temperature"]["change_score"], 3),
            "pressure_score": round(parameters["pressure"]["change_score"], 3),
            "humidity_score": round(parameters["humidity"]["change_score"], 3),

            "historical_score": round(history_score, 3),
            "spatial_score": round(selected_spatial["spatial"], 3),
            "regional_agreement": round(selected_spatial["regional"], 3),
            "neighbor_count": selected_spatial["count"],
            "neighbor_median": selected_spatial["median"],
            "neighbor_spread": selected_spatial["spread"],
            "persistence_score": round(persistence, 3),
            "multivariate_score": round(multivariate, 3),
            "hybrid_score": round(hybrid, 3),

            "ml_anomaly": bool(ml >= 0.50),
            "ml_score": round(ml, 3),
            "ml_confidence": round(ml * 100, 2),
            "ml_agrees_with_hybrid": ml_agrees,
            "ml_fault_type": ml_fault_prediction["class"] if ml_fault_prediction is not None else None,
            "ml_fault_type_confidence": round(ml_fault_prediction["confidence"] * 100, 2) if ml_fault_prediction is not None else None,
            "ml_fault_type_probabilities": {k: round(v, 3) for k, v in ml_fault_prediction["probabilities"].items()} if ml_fault_prediction is not None else None,
            "ml_fault_type_agrees": ml_fault_type_agrees,

            "evidence_percentage": round(hybrid * 100, 2),
            "temperature_sequence": temperature_sequence,
            "temperature_time_rule": temperature_time_rule(parameters["temperature"]["change"], gap),
            "temperature_extreme_boundary": bool(safe_float(current.get("temperature")) is not None and (safe_float(current.get("temperature")) >= TEMPERATURE_EXTREME_HIGH or safe_float(current.get("temperature")) <= TEMPERATURE_EXTREME_LOW)),
            "humidity_rate_points_per_min": round(humidity_rate, 3),
            "humidity_nearest_difference": nearest_diff,
            "humidity_peer_delta_median": peer_delta,
            "humidity_neighbor_support_ratio": round(support_ratio, 3),

            "physical_faults": physical,
            "frozen_status": frozen_status,
            "frozen_field": frozen_field,
            "frozen_candidate_parameters": frozen_candidates,
            "frozen_24h_suspect_parameters": frozen_stage_24h,
            "frozen_confirmed_parameters": frozen_confirmed,
            "frozen_duration_seconds": round(frozen_evidence[frozen_field]["duration_seconds"], 2) if frozen_field is not None else 0.0,
            "frozen_value": frozen_evidence[frozen_field]["value"] if frozen_field is not None else None,
            "frozen_spread": frozen_evidence[frozen_field]["spread"] if frozen_field is not None else None,
            "frozen_evidence": frozen_evidence,

            "explanation": explanation,
            "evidence_summary": evidence_summary,
        })

    results.extend(detect_silent_sensors(batch))
    return results


def inject_anomaly(sensor_id="AWS_001", anomaly_type="temperature_spike"):
    base_data = {
        "sensor_id": sensor_id,
        "temperature": 28.0,
        "pressure": 1013.0,
        "humidity": 65.0,
        "latitude": 22.57,
        "longitude": 88.36,
        "observation_time": datetime.now(timezone.utc).isoformat(),
    }
    if anomaly_type == "temperature_spike": base_data["temperature"] = 45.0
    elif anomaly_type == "temperature_drop": base_data["temperature"] = 5.0
    elif anomaly_type == "pressure_spike": base_data["pressure"] = 1090.0
    elif anomaly_type == "pressure_drop": base_data["pressure"] = 900.0
    elif anomaly_type == "humidity_spike": base_data["humidity"] = 98.0
    elif anomaly_type == "humidity_drop": base_data["humidity"] = 10.0
    elif anomaly_type == "sensor_fault":
        base_data["temperature"] = 80.0
        base_data["pressure"] = 1200.0
        base_data["humidity"] = 110.0
    return base_data

# ============================================================
# COMPATIBILITY
# ============================================================

detect = detect_anomaly
analyze = detect_anomaly
analyze_batch = detect_anomaly

if __name__ == "__main__":
    print("SKYGUARD AI - HYBRID ANOMALY ENGINE")
    print("Temperature + Pressure + Humidity")