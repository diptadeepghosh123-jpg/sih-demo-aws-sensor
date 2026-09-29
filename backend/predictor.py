"""
SkyGuard AI - predictor.py
Suggested/corrected-value estimator.

Purpose
-------
When anomaly.py flags a reading as SENSOR_ANOMALY, this module computes
a suggested "what the value probably should be" WITHOUT ever touching
the raw stored reading. The two are shown side by side (see
predict_expected_value()'s return shape) -- the flagged value stays
exactly what the sensor reported; the suggestion is separate, derived
evidence for an operator to act on.

Three independent signals, blended by how much data backs each one:

  1. SPATIAL ESTIMATE
     Distance-weighted average of the same parameter from nearby,
     currently-healthy sensors right now. Reuses anomaly.py's own
     distance_km()/nearby_rows() so both modules agree on what
     "nearby" means and neither duplicates (or re-introduces a bug
     in) the other's distance logic.

  2. TREND ESTIMATE
     A short linear extrapolation from this sensor's own last few
     valid readings -- "what would this sensor's own trajectory
     predict for right now". Least trusted of the three when others
     are available, because it is computed FROM the same sensor that
     may already be at fault.

  3. SEASONAL BASELINE (the ML component -- "seasonal learning")
     A station's temperature/pressure/humidity are not flat; they
     follow an hour-of-day pattern, and given enough history, a
     day-of-year pattern too. Rather than one fixed model pretrained
     on synthetic data (which would be meaningless here -- there is
     no real climate signal to fake), this trains a small regressor
     PER SENSOR PER FIELD directly on that sensor's own accumulated
     history, refit fresh each time it's asked. That means:
       - With too little history, it honestly returns None instead
         of fabricating a confident number.
       - It genuinely gets better the longer the system runs, because
         it is always learning from that station's own real data.
       - Day-of-year seasonality only activates once the history
         actually spans enough calendar days to support it; until
         then it falls back to hour-of-day only.
"""

import math
from datetime import datetime, timezone

import numpy as np

try:
    from sklearn.ensemble import RandomForestRegressor
except Exception:
    RandomForestRegressor = None

try:
    from .anomaly import safe_float, row_time, distance_km, nearby_rows
except Exception:
    from anomaly import safe_float, row_time, distance_km, nearby_rows


# ============================================================
# CONFIGURATION
# ============================================================

# Minimum valid (time, value) points required before each estimator
# will produce a number at all, rather than silently guessing.
TREND_MIN_POINTS = 3
SEASONAL_MIN_POINTS = 20            # roughly a day of hourly-ish data
SEASONAL_DAY_OF_YEAR_MIN_SPAN_DAYS = 20  # history must span this long
                                          # before day-of-year features
                                          # are trusted enough to use

# Don't extrapolate a linear trend further into the future than this;
# beyond it, a straight-line guess is more likely to be nonsense than
# a genuine "expected value", so we fall back to the last known value.
TREND_MAX_EXTRAPOLATION_SECONDS = 3600.0

# Blend weights when more than one estimate is available. Spatial is
# trusted most because it comes from OTHER sensors, unaffected by this
# sensor's own possible fault. Trend is trusted least for the opposite
# reason. Renormalized over whichever sources actually produced a
# number for this reading.
BASE_WEIGHTS = {
    "spatial": 0.50,
    "seasonal": 0.30,
    "trend": 0.20,
}


# ============================================================
# 1. SPATIAL ESTIMATE
# ============================================================

def _spatial_estimate(current, batch, field, healthy_sensor_ids=None):
    """
    Distance-weighted average of `field` from nearby sensors in the
    current batch. If `healthy_sensor_ids` is given, only sensors in
    that set are used -- letting a caller exclude neighbors that are
    themselves currently flagged as anomalous, so one bad station
    doesn't pollute the estimate for another.
    """
    peers = nearby_rows(current, batch)
    if not peers:
        return None

    weighted_sum = 0.0
    total_weight = 0.0
    used = 0
    nearest_distance_km = None

    for distance, peer in peers:
        peer_id = str(peer.get("sensor_id"))
        if healthy_sensor_ids is not None and peer_id not in healthy_sensor_ids:
            continue

        value = safe_float(peer.get(field))
        if value is None:
            continue

        weight = 1.0 / (1.0 + distance)
        weighted_sum += value * weight
        total_weight += weight
        used += 1
        if nearest_distance_km is None or distance < nearest_distance_km:
            nearest_distance_km = float(distance)

    if used == 0 or total_weight <= 0:
        return None

    return {
        "value": weighted_sum / total_weight,
        "neighbor_count": used,
        "nearest_distance_km": nearest_distance_km,
    }


# ============================================================
# 2. TREND ESTIMATE
# ============================================================

def _trend_estimate(history, field, current_time):
    """
    Linear extrapolation from this sensor's own last few valid
    readings of `field`, projected to `current_time`. Capped so it
    never extrapolates further than TREND_MAX_EXTRAPOLATION_SECONDS
    past the last known point -- beyond that it just returns the last
    known value rather than a runaway straight-line guess.
    """
    points = []
    for row in history:
        t = row_time(row)
        v = safe_float(row.get(field))
        if t is not None and v is not None:
            points.append((t, v))

    points.sort(key=lambda p: p[0])
    points = points[-8:]  # a short recent window is enough for a trend

    if len(points) < TREND_MIN_POINTS or current_time is None:
        return None

    t0 = points[0][0]
    xs = np.array([(t - t0).total_seconds() for t, _ in points])
    ys = np.array([v for _, v in points])

    trend_sequence = [
        {"timestamp": t.isoformat(), "value": float(v)}
        for t, v in points
    ]

    if np.ptp(xs) <= 0:
        # All points at (effectively) the same timestamp -- no trend
        # to extrapolate; just use the most recent value.
        return {
            "value": float(ys[-1]),
            "points_used": len(points),
            "trend_sequence": trend_sequence,
        }

    slope, intercept = np.polyfit(xs, ys, 1)

    target_x = (current_time - t0).total_seconds()
    last_x = xs[-1]

    if target_x - last_x > TREND_MAX_EXTRAPOLATION_SECONDS:
        # Too far past the last real reading to trust a straight line.
        predicted = ys[-1]
    else:
        predicted = slope * target_x + intercept

    return {
        "value": float(predicted),
        "points_used": len(points),
        "trend_sequence": trend_sequence,
    }


# ============================================================
# 3. SEASONAL BASELINE (ML)
# ============================================================

def _cyclical_hour_features(dt):
    hour_fraction = dt.hour + dt.minute / 60.0
    angle = 2 * math.pi * hour_fraction / 24.0
    return math.sin(angle), math.cos(angle)


def _cyclical_day_of_year_features(dt):
    day_fraction = dt.timetuple().tm_yday / 365.25
    angle = 2 * math.pi * day_fraction
    return math.sin(angle), math.cos(angle)


def _seasonal_estimate(history, field, current_time):
    """
    Trains a small regressor ON THIS SENSOR'S OWN HISTORY (not a
    pretrained model) mapping hour-of-day (and, given enough calendar
    span, day-of-year) to `field`'s value, then predicts for
    `current_time`. Returns None -- honestly, not a guess -- if there
    isn't enough history to support this yet.
    """
    if RandomForestRegressor is None or current_time is None:
        return None

    points = []
    for row in history:
        t = row_time(row)
        v = safe_float(row.get(field))
        if t is not None and v is not None:
            points.append((t, v))

    if len(points) < SEASONAL_MIN_POINTS:
        return None

    span_days = (max(t for t, _ in points) - min(t for t, _ in points)).total_seconds() / 86400.0
    use_day_of_year = span_days >= SEASONAL_DAY_OF_YEAR_MIN_SPAN_DAYS

    X = []
    y = []
    for t, v in points:
        hour_sin, hour_cos = _cyclical_hour_features(t)
        if use_day_of_year:
            doy_sin, doy_cos = _cyclical_day_of_year_features(t)
            X.append([hour_sin, hour_cos, doy_sin, doy_cos])
        else:
            X.append([hour_sin, hour_cos])
        y.append(v)

    try:
        model = RandomForestRegressor(
            n_estimators=100,
            max_depth=6,
            min_samples_leaf=3,
            random_state=42,
        )
        model.fit(X, y)

        hour_sin, hour_cos = _cyclical_hour_features(current_time)
        if use_day_of_year:
            doy_sin, doy_cos = _cyclical_day_of_year_features(current_time)
            features = [[hour_sin, hour_cos, doy_sin, doy_cos]]
        else:
            features = [[hour_sin, hour_cos]]

        predicted = model.predict(features)[0]
        # This is a maturity/availability indicator for the seasonal
        # signal, not a claim of predictive accuracy.  It is deliberately
        # kept separate from the fitted model and does not alter prediction.
        seasonal_strength = 1.0 if use_day_of_year else 0.5
        seasonal_strength_label = "strong" if use_day_of_year else "developing"
        return {
            "value": float(predicted),
            "points_used": len(points),
            "day_of_year_used": use_day_of_year,
            "history_span_days": round(span_days, 1),
            "seasonal_strength": seasonal_strength,
            "seasonal_strength_label": seasonal_strength_label,
        }
    except Exception:
        # A seasonal estimate is a bonus signal, never a hard
        # dependency -- if fitting fails for any reason, fall back
        # silently to the other two sources.
        return None


# ============================================================
# COMBINED SUGGESTION
# ============================================================

def predict_expected_value(current, history, batch, field,
                            healthy_sensor_ids=None):
    """
    Returns a suggested value for `field` on the sensor described by
    `current`, blending spatial + trend + seasonal estimates by
    confidence. Never modifies `current` -- this is meant to be shown
    ALONGSIDE the raw flagged reading, not in place of it.

    Returns None if none of the three sources could produce anything
    (e.g. a brand-new, isolated sensor with no neighbors and no
    history yet) -- callers should keep showing the raw value alone
    in that case rather than a fabricated suggestion.
    """
    current_time = row_time(current)

    spatial = _spatial_estimate(current, batch, field, healthy_sensor_ids)
    trend = _trend_estimate(history, field, current_time)
    seasonal = _seasonal_estimate(history, field, current_time)

    components = {}
    if spatial is not None:
        components["spatial"] = spatial["value"]
    if seasonal is not None:
        components["seasonal"] = seasonal["value"]
    if trend is not None:
        components["trend"] = trend["value"]

    if not components:
        return None

    weight_sum = sum(BASE_WEIGHTS[name] for name in components)
    suggested_value = sum(
        components[name] * BASE_WEIGHTS[name] for name in components
    ) / weight_sum

    # Confidence is a simple evidence-coverage score, not a calibrated
    # probability.  It increases only when independent estimator sources
    # are actually available.
    confidence_by_source_count = {1: 0.50, 2: 0.75, 3: 0.90}
    confidence = confidence_by_source_count.get(len(components), 0.50)
    confidence_label = (
        "high" if confidence >= 0.90
        else "medium" if confidence >= 0.75
        else "low"
    )

    trend_sequence = (
        trend.get("trend_sequence", [])
        if trend is not None else []
    )

    seasonal_strength = (
        float(seasonal.get("seasonal_strength", 0.0))
        if seasonal is not None else 0.0
    )
    seasonal_strength_label = (
        seasonal.get("seasonal_strength_label")
        if seasonal is not None else "unavailable"
    )

    return {
        "suggested_value": round(suggested_value, 3),
        "components": {k: round(v, 3) for k, v in components.items()},
        "weights_used": {
            name: round(BASE_WEIGHTS[name] / weight_sum, 3)
            for name in components
        },
        "spatial_neighbor_count": (
            spatial["neighbor_count"] if spatial is not None else 0
        ),
        "nearest_distance_km": (
            round(float(spatial["nearest_distance_km"]), 3)
            if spatial is not None and spatial.get("nearest_distance_km") is not None
            else None
        ),
        "seasonal_points_used": (
            seasonal["points_used"] if seasonal is not None else 0
        ),
        "seasonal_uses_day_of_year": (
            seasonal["day_of_year_used"] if seasonal is not None else False
        ),
        # These keys are kept for the frontend/database contract.
        "seasonal_uses_day_of_week": False,
        "seasonal_uses_month": False,
        "seasonal_history_span_days": (
            float(seasonal["history_span_days"]) if seasonal is not None else 0.0
        ),
        "seasonal_strength": seasonal_strength,
        "seasonal_strength_label": seasonal_strength_label,
        "trend_points_used": (
            trend["points_used"] if trend is not None else 0
        ),
        "trend_sequence": trend_sequence,
        "confidence": confidence,
        "confidence_label": confidence_label,
    }


if __name__ == "__main__":
    print("SkyGuard AI - suggested-value predictor (spatial + trend + seasonal ML)")