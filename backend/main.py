from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import re
from fastapi.middleware.cors import CORSMiddleware

from weather_api import get_sensor_data
from processing import clean_sensor_data
from anomaly import inject_anomaly, detect_anomaly
from database import (
    create_database,
    save_result,
    get_history,
    get_sensor_history
)
from maintenance import predict_maintenance
from sensor_registry import (
    add_sensor as registry_add_sensor,
    delete_sensor as registry_delete_sensor,
    get_all_sensors,
    get_sensor,
    get_sensor_count,
    get_states,
    get_districts,
    get_cities,
    get_registry_summary,
    update_sensor as registry_update_sensor,
    normalize_sensor_id,
)


# ============================================================
# FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="SkyGuard AI",
    description="Real-Time Weather Sensor Anomaly Detection System",
    version="1.0.0"
)


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"]
)


# ============================================================
# DATABASE INITIALIZATION
# ============================================================

create_database()


# ============================================================
# VALID ANOMALY TYPES
# ============================================================

VALID_ANOMALY_TYPES = {
    "temperature_spike",
    "temperature_drop",
    "pressure_spike",
    "pressure_drop",
    "humidity_spike",
    "humidity_drop",
    "sensor_fault"
}


class ChatRequest(BaseModel):
    message: str
    sensor_id: str | None = None


# ============================================================
# HELPER: CONVERT DATA TO LIST OF DICTS
# ============================================================

def _rows_from(data):
    """
    Convert DataFrame/list-like sensor data into
    a plain list of dictionaries.
    """

    if data is None:
        return []

    if hasattr(data, "to_dict"):
        return data.to_dict(orient="records")

    return list(data)


# ============================================================
# HELPER: FETCH AND CLEAN LIVE SENSOR DATA
# ============================================================

def _fetch_live_batch():
    """
    Fetch live sensor data and clean it before anomaly detection.

    A 502 error is returned when the live weather feed
    provides no usable data.
    """

    try:
        data = get_sensor_data()
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Unable to fetch live sensor data: {str(exc)}"
        )

    if data is None:
        raise HTTPException(
            status_code=502,
            detail=(
                "No live sensor data was returned. "
                "Check the weather API, API key, network, "
                "and sensor registry."
            )
        )

    try:
        data = clean_sensor_data(data)
    except ValueError as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Sensor data validation failed: {str(exc)}"
        )
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Sensor data processing failed: {str(exc)}"
        )

    rows = _rows_from(data)

    if not rows:
        raise HTTPException(
            status_code=502,
            detail=(
                "No usable live sensor data was returned. "
                "Check the weather API, API key, network, "
                "and sensor registry."
            )
        )

    return rows


# ============================================================
# HELPER: SAVE RESULTS SAFELY
# ============================================================

def _save_results(results):
    """
    Save valid anomaly-analysis results to the database.
    """

    if not results:
        return

    for result in results:

        if result is None:
            continue

        try:
            save_result(result)
        except Exception as exc:
            # Do not destroy a successful analysis just because
            # database storage failed for one result.
            print(
                f"[DATABASE WARNING] "
                f"Could not save result: {exc}"
            )


# ============================================================
# HEALTH / HOME
# ============================================================

@app.get("/")
def home():
    return {
        "message": "SkyGuard AI is running",
        "status": "online",
        "service": "weather-anomaly-detection"
    }


# ============================================================
# API HEALTH CHECK
# ============================================================

@app.get("/api/health")
def health():
    """
    Simple backend health endpoint.

    This does not call the external weather API.
    It only confirms that FastAPI is running.
    """

    return {
        "status": "healthy",
        "service": "SkyGuard AI",
        "backend": "online"
    }


# ============================================================
# DYNAMIC SENSOR REGISTRY
# ============================================================

@app.get("/api/sensors")
def sensors():
    return {
        "success": True,
        "sensors": get_all_sensors(),
        "count": get_sensor_count(),
    }


@app.get("/api/sensors/summary")
def sensor_summary():
    return get_registry_summary()


@app.get("/api/sensors/states")
def sensor_states():
    return {"success": True, "states": get_states()}


@app.get("/api/sensors/districts")
def sensor_districts(state: str | None = None):
    return {"success": True, "districts": get_districts(state)}


@app.get("/api/sensors/cities")
def sensor_cities(state: str | None = None, district: str | None = None):
    return {"success": True, "cities": get_cities(state=state, district=district)}


@app.get("/api/sensors/{sensor_id}")
def sensor(sensor_id: str):
    result = get_sensor(sensor_id)
    if result is None:
        raise HTTPException(status_code=404, detail=f"Sensor {sensor_id} not found")
    return {"success": True, "sensor": result}


@app.post("/api/sensors")
def create_sensor(payload: dict):
    try:
        result = registry_add_sensor(
            sensor_id=payload.get("sensor_id"),
            latitude=payload.get("latitude"),
            longitude=payload.get("longitude"),
            state=payload.get("state", ""),
            district=payload.get("district", ""),
            city=payload.get("city", ""),
            locality=payload.get("locality", ""),
        )
        return result
    except (ValueError, TypeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.put("/api/sensors/{sensor_id}")
def update_sensor(sensor_id: str, payload: dict):
    try:
        result = registry_update_sensor(
            sensor_id=sensor_id,
            latitude=payload.get("latitude"),
            longitude=payload.get("longitude"),
            state=payload.get("state"),
            district=payload.get("district"),
            city=payload.get("city"),
            locality=payload.get("locality"),
        )
    except ValueError as exc:
        if "does not exist" in str(exc).lower():
            raise HTTPException(status_code=404, detail=str(exc))
        raise HTTPException(status_code=400, detail=str(exc))
    return result


@app.delete("/api/sensors/{sensor_id}")
def delete_sensor(sensor_id: str):
    try:
        result = registry_delete_sensor(sensor_id)
    except ValueError as exc:
        if "does not exist" in str(exc).lower():
            raise HTTPException(status_code=404, detail=str(exc))
        raise HTTPException(status_code=400, detail=str(exc))
    return result


# ============================================================
# ANALYZE LIVE SENSOR DATA
# ============================================================

@app.get("/api/analyze")
def analyze():
    """
    Fetch live weather sensor data,
    clean it,
    run anomaly detection,
    save results,
    and return the analysis.
    """

    rows = _fetch_live_batch()

    try:
        results = detect_anomaly(rows)
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Anomaly detection failed: {str(exc)}"
        )

    if results is None:
        results = []

    _save_results(results)

    anomaly_count = sum(
        1
        for result in results
        # Matches maintenance.py's definition of a concerning reading.
        # Previously this only counted SENSOR_ANOMALY, so /api/analyze's
        # anomaly_count and the maintenance predictor's degradation
        # score could silently disagree on the same sensor's history.
        if result.get("status") in ("SENSOR_ANOMALY", "GLITCH_SUSPECT")
    )

    return {
        "success": True,
        "sensor_count": len(results),
        "anomaly_count": anomaly_count,
        "sensors": results
    }


# ============================================================
# SKYGUARD ASSISTANT
# ============================================================

_SENSOR_PATTERN = re.compile(r"\bAWS[_-]\d+\b", re.IGNORECASE)

_STATUS_INFO = {
    "NORMAL": {
        "label": "Normal",
        "simple": "The latest reading looks normal. The system did not find a significant sensor problem.",
        "action": "No immediate action is required. Keep monitoring normally.",
    },
    "WATCH": {
        "label": "Watch",
        "simple": "The system noticed an unusual reading or change, but there is not enough evidence yet to confirm a sensor fault.",
        "action": "Keep watching the next few readings. A repeated problem may be escalated automatically.",
    },
    "GLITCH_SUSPECT": {
        "label": "Glitch Suspect",
        "simple": "The sensor may be giving a stuck, repeated, or otherwise suspicious value. SkyGuard is waiting for more evidence before confirming it as a fault.",
        "action": "Check the sensor and compare the next readings. Do not treat one suspicious value as a confirmed failure yet.",
    },
    "SENSOR_ANOMALY": {
        "label": "Sensor Anomaly",
        "simple": "The reading is inconsistent with the expected sensor behavior and SkyGuard has classified it as a sensor-side anomaly.",
        "action": "Inspect the sensor, connection, calibration, or data path and compare with nearby sensors.",
    },
    "WEATHER_EVENT": {
        "label": "Weather Event",
        "simple": "The change looks more like a genuine regional weather change because nearby sensors show similar behavior.",
        "action": "Treat this primarily as a weather condition rather than an isolated sensor fault, while continuing to monitor it.",
    },
    "DATA_UNAVAILABLE": {
        "label": "Data Unavailable",
        "simple": "SkyGuard could not get a usable weather reading from the live data source, so it did not make an anomaly decision from missing values.",
        "action": "Check the live data source, API key, network connection, and sensor/data feed.",
    },
}

_FAULT_INFO = {
    "DATA_SOURCE_FAILURE": "The live data source failed or was not configured, so there is no trustworthy reading to analyze.",
    "SUDDEN_SPIKE": "The sensor changed much faster than expected over the available time gap.",
    "CALIBRATION_DRIFT": "The sensor appears to be drifting away from the expected pattern instead of making one isolated jump.",
    "ISOLATED_SENSOR_FAULT": "The sensor behaves differently from nearby sensors, suggesting a local sensor/data problem.",
    "STUCK_SENSOR": "The sensor appears to be repeating or holding a value for too long.",
    "PHYSICAL_ISSUE": "The pattern suggests a possible physical or hardware-related problem.",
    "WEATHER_EVENT": "Multiple nearby sensors show compatible changes, which supports a genuine weather event.",
    "None": "No specific fault type was assigned.",
}


def _extract_sensor_id(message, provided=None):
    if provided:
        return normalize_sensor_id(provided)
    match = _SENSOR_PATTERN.search(message or "")
    return match.group(0).upper().replace("-", "_") if match else None


def _latest_by_sensor(records):
    latest = {}
    for row in records or []:
        sid = normalize_sensor_id(row.get("sensor_id"))
        if sid and sid not in latest:
            latest[sid] = row
    return latest


def _chat_status_summary(records):
    latest = _latest_by_sensor(records)
    counts = {
        "NORMAL": 0,
        "WATCH": 0,
        "GLITCH_SUSPECT": 0,
        "SENSOR_ANOMALY": 0,
        "WEATHER_EVENT": 0,
        "DATA_UNAVAILABLE": 0,
    }
    for row in latest.values():
        status = str(row.get("status") or "NORMAL").upper()
        if status in counts:
            counts[status] += 1
    return latest, counts


def _clean_value(value):
    if value in (None, "", "None", "null", "NULL"):
        return None
    return value


def _format_reading(row):
    parts = []
    for field, label, unit in (
        ("temperature", "Temperature", "°C"),
        ("pressure", "Pressure", "hPa"),
        ("humidity", "Humidity", "%"),
    ):
        value = _clean_value(row.get(field))
        if value is not None:
            parts.append(f"{label} {value}{unit}")
    return ", ".join(parts) if parts else "No usable weather reading is currently stored."


def _format_prediction(row):
    suggested = _clean_value(row.get("suggested_value"))
    field = row.get("prediction_field") or "value"
    confidence = _clean_value(row.get("prediction_confidence"))
    confidence_label = _clean_value(row.get("prediction_confidence_label"))
    strength = _clean_value(row.get("seasonal_strength"))
    strength_label = _clean_value(row.get("seasonal_strength_label"))
    trend_points = _clean_value(row.get("trend_points_used"))
    nearest = _clean_value(row.get("prediction_nearest_distance_km"))
    if suggested is None:
        return "No stored prediction is available for this sensor yet. SkyGuard generates prediction information only when the detection state requires it and enough history is available."
    parts = [f"Expected {field}: {suggested}"]
    if confidence_label is not None:
        parts.append(f"confidence {confidence_label}")
    elif confidence is not None:
        parts.append(f"confidence {confidence}")
    if strength_label is not None:
        parts.append(f"seasonality {strength_label}")
    elif strength is not None:
        parts.append(f"seasonal strength {strength}")
    if trend_points is not None:
        parts.append(f"trend points {trend_points}")
    if nearest is not None:
        parts.append(f"nearest sensor distance {nearest} km")
    return "; ".join(str(p) for p in parts) + "."


def _status_details(row):
    status = str(row.get("status") or "NORMAL").upper()
    info = _STATUS_INFO.get(status, {
        "label": status.replace("_", " ").title(),
        "simple": "SkyGuard returned a status that is not covered by the standard assistant explanations.",
        "action": "Review the detailed detection result in the dashboard.",
    })
    fault = str(row.get("fault_type") or "None")
    category = str(row.get("anomaly_category") or status)
    explanation = str(row.get("explanation") or "").strip()
    evidence = str(row.get("evidence_summary") or "").strip()
    reason = _FAULT_INFO.get(fault)
    if not reason and category == "API_FAILURE":
        reason = "The live data source did not return a usable reading."
    details = {
        "status": status,
        "label": info["label"],
        "simple": info["simple"],
        "action": info["action"],
        "fault_type": fault,
        "category": category,
        "reason": reason or explanation or "No extra reason was stored.",
        "stored_explanation": explanation,
        "evidence": evidence,
        "severity": row.get("severity"),
        "confidence": row.get("confidence"),
        "health": row.get("health"),
    }
    return details


def _human_anomaly_line(sensor_id, row):
    d = _status_details(row)
    return (
        f"{sensor_id}: {d['label']}. {d['simple']} "
        f"Reason: {d['reason']} Action: {d['action']}"
    )


def _matches_any(text, terms):
    return any(term in text for term in terms)


@app.post("/api/chat")
def chat(request: ChatRequest):
    """
    SkyGuard Assistant.

    This is intentionally data-grounded: the assistant reads the same
    sensor registry/history/prediction results produced by the SkyGuard
    detection pipeline and explains them in plain language. It does not
    invent weather readings or replace the anomaly engine's decision.
    """
    message = str(request.message or "").strip()
    if not message:
        raise HTTPException(status_code=400, detail="Message cannot be empty")

    text = message.lower()
    sensor_id = _extract_sensor_id(message, request.sensor_id)

    try:
        registry = get_all_sensors()
        history = get_history(300)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Chat context unavailable: {exc}")

    latest, counts = _chat_status_summary(history)

    # --------------------------------------------------------
    # General conversation / capability
    # --------------------------------------------------------
    if any(word in text for word in ("hello", "hi", "hey")) and len(text.split()) <= 4:
        return {
            "success": True,
            "reply": "Hello. I am SkyGuard Assistant. I read SkyGuard's detected results and can explain anomalies, Watch states, glitches, weather events, sensor health, predictions, and network status in simple language.",
            "source": "skyguard_context",
        }

    if "help" in text or "what can you do" in text:
        return {
            "success": True,
            "reply": "Ask me things like: 'What anomalies are present?', 'Why is AWS_001 on Watch?', 'Explain the glitch on AWS_002 simply', 'What should I do about this anomaly?', 'What is the prediction for AWS_003?', or 'How many sensors are registered?'.",
            "source": "skyguard_context",
        }

    if _matches_any(text, ("how many sensor", "how many station", "sensor count", "station count")):
        count = len(registry or [])
        return {
            "success": True,
            "reply": f"There are {count} registered sensor stations in the SkyGuard network.",
            "source": "sensor_registry",
            "sensor_count": count,
        }

    # --------------------------------------------------------
    # Network-wide anomaly / event questions
    # --------------------------------------------------------
    anomaly_terms = ("anomal", "alert", "critical", "fault", "problem", "issue")
    watch_terms = ("watch", "monitor", "warning")
    glitch_terms = ("glitch", "stuck", "frozen", "fixed value")
    weather_terms = ("weather event", "regional event", "weather change")
    unavailable_terms = ("unavailable", "no data", "data source", "offline")
    simplify_terms = ("simplify", "simple", "explain", "what happened", "why", "reason", "meaning", "means", "details", "what should")

    if not sensor_id:
        if _matches_any(text, glitch_terms):
            rows = [r for r in latest.values() if str(r.get("status") or "").upper() == "GLITCH_SUSPECT"]
            if rows:
                lines = [_human_anomaly_line(str(r.get("sensor_id")), r) for r in rows[:8]]
                return {"success": True, "reply": "Current suspected glitches:\n" + "\n".join(f"• {line}" for line in lines), "source": "sensor_history", "status": "GLITCH_SUSPECT"}
            return {"success": True, "reply": "I do not see a current GLITCH_SUSPECT state in the latest stored result for any sensor.", "source": "sensor_history"}

        if _matches_any(text, watch_terms) and not _matches_any(text, anomaly_terms):
            rows = [r for r in latest.values() if str(r.get("status") or "").upper() == "WATCH"]
            if rows:
                lines = [_human_anomaly_line(str(r.get("sensor_id")), r) for r in rows[:8]]
                return {"success": True, "reply": "Current Watch states:\n" + "\n".join(f"• {line}" for line in lines), "source": "sensor_history", "status": "WATCH"}
            return {"success": True, "reply": "There are no current WATCH states in the latest stored results.", "source": "sensor_history"}

        if _matches_any(text, weather_terms):
            rows = [r for r in latest.values() if str(r.get("status") or "").upper() == "WEATHER_EVENT"]
            if rows:
                lines = [_human_anomaly_line(str(r.get("sensor_id")), r) for r in rows[:8]]
                return {"success": True, "reply": "Detected weather events:\n" + "\n".join(f"• {line}" for line in lines), "source": "sensor_history", "status": "WEATHER_EVENT"}
            return {"success": True, "reply": "I do not see a current WEATHER_EVENT state in the latest stored results.", "source": "sensor_history"}

        if _matches_any(text, unavailable_terms):
            rows = [r for r in latest.values() if str(r.get("status") or "").upper() == "DATA_UNAVAILABLE"]
            if rows:
                names = ", ".join(str(r.get("sensor_id")) for r in rows[:12])
                return {"success": True, "reply": f"{len(rows)} sensor(s) currently have unavailable data: {names}. This means SkyGuard did not receive a usable live reading, so no normal/anomaly decision was made from missing data.", "source": "sensor_history", "status": "DATA_UNAVAILABLE"}

        if _matches_any(text, anomaly_terms):
            problem_rows = [
                r for r in latest.values()
                if str(r.get("status") or "").upper() in {"SENSOR_ANOMALY", "GLITCH_SUSPECT", "WATCH", "WEATHER_EVENT"}
            ]
            if problem_rows:
                lines = [_human_anomaly_line(str(r.get("sensor_id")), r) for r in problem_rows[:10]]
                return {
                    "success": True,
                    "reply": "Here are the latest detected non-normal states:\n" + "\n".join(f"• {line}" for line in lines),
                    "source": "sensor_history",
                    "status_counts": counts,
                }
            return {
                "success": True,
                "reply": "I do not see a current sensor anomaly, glitch, Watch state, or weather-event state in the latest stored results.",
                "source": "sensor_history",
                "status_counts": counts,
            }

    # --------------------------------------------------------
    # Sensor-specific questions
    # --------------------------------------------------------
    if sensor_id:
        row = latest.get(sensor_id)
        if row is None:
            return {
                "success": True,
                "reply": f"I could not find a stored detection result for {sensor_id}. Check the sensor ID or run a live analysis first.",
                "source": "sensor_history",
                "sensor_id": sensor_id,
            }

        status = str(row.get("status") or "NORMAL").upper()
        details = _status_details(row)

        if _matches_any(text, ("prediction", "predict", "expected", "seasonal", "trend")):
            return {
                "success": True,
                "reply": f"{sensor_id}: {_format_prediction(row)}",
                "source": "prediction_history",
                "sensor_id": sensor_id,
            }

        if _matches_any(text, ("health", "maintenance", "degradation")):
            sensor_history = get_sensor_history(sensor_id, limit=100)
            maintenance = predict_maintenance(sensor_id, sensor_history)
            return {
                "success": True,
                "reply": f"{sensor_id} health: {maintenance.get('degradation', 'UNKNOWN')}. {maintenance.get('maintenance_reason', '')}",
                "source": "maintenance",
                "sensor_id": sensor_id,
                "maintenance": maintenance,
            }

        if _matches_any(text, ("temperature", "pressure", "humidity", "reading", "value")):
            return {
                "success": True,
                "reply": f"{sensor_id}: {_format_reading(row)}. Current detected state: {details['label']}.",
                "source": "sensor_history",
                "sensor_id": sensor_id,
            }

        if _matches_any(text, simplify_terms) or _matches_any(text, anomaly_terms + watch_terms + glitch_terms + weather_terms + unavailable_terms):
            reply = f"{sensor_id} is currently {details['label']}. {details['simple']}"
            if details["reason"]:
                reply += f" Reason: {details['reason']}"
            if details["evidence"] and any(k in text for k in ("detail", "evidence", "why", "explain")):
                reply += f" Evidence: {details['evidence']}"
            reply += f" What to do: {details['action']}"
            return {
                "success": True,
                "reply": reply,
                "source": "sensor_history",
                "sensor_id": sensor_id,
                "detection": details,
            }

        return {
            "success": True,
            "reply": f"{sensor_id} latest detected state: {details['label']}. {details['simple']} Ask 'why?', 'what happened?', 'show the reading', 'show the prediction', or 'how is the sensor health?'.",
            "source": "sensor_history",
            "sensor_id": sensor_id,
            "detection": details,
        }

    # --------------------------------------------------------
    # General network overview
    # --------------------------------------------------------
    if _matches_any(text, ("status", "network", "overview", "overall")):
        count = len(registry or [])
        return {
            "success": True,
            "reply": (
                f"Network overview: {count} registered sensors. Latest states: "
                f"{counts['NORMAL']} normal, {counts['WATCH']} Watch, "
                f"{counts['GLITCH_SUSPECT']} glitch suspects, {counts['SENSOR_ANOMALY']} sensor anomalies, "
                f"{counts['WEATHER_EVENT']} weather events, and {counts['DATA_UNAVAILABLE']} unavailable."
            ),
            "source": "sensor_history",
            "status_counts": counts,
        }

    return {
        "success": True,
        "reply": "I answer from SkyGuard's detected sensor results. Try: 'What anomalies are present?', 'Why is AWS_001 on Watch?', 'Explain the glitch on AWS_002 simply', 'What is the reading of AWS_003?', or 'What is the prediction for AWS_003?'",
        "source": "skyguard_context",
    }


# ============================================================
# TEST / INJECT ANOMALY
# ============================================================

@app.get("/api/test/{anomaly_type}")
def test_anomaly(
    anomaly_type: str,
    sensor_id: str = "AWS_001"
):
    """
    Inject a synthetic anomaly into one live sensor.

    The remaining live sensors are kept so that the anomaly
    detector can still perform spatial / regional comparisons.
    """

    # --------------------------------------------------------
    # VALIDATE ANOMALY TYPE
    # --------------------------------------------------------

    sensor_id = normalize_sensor_id(sensor_id)

    if anomaly_type not in VALID_ANOMALY_TYPES:

        raise HTTPException(
            status_code=400,
            detail={
                "message": "Unknown anomaly type",
                "received": anomaly_type,
                "valid_types": sorted(
                    VALID_ANOMALY_TYPES
                )
            }
        )

    # --------------------------------------------------------
    # FETCH CURRENT LIVE BATCH
    # --------------------------------------------------------

    rows = _fetch_live_batch()

    # --------------------------------------------------------
    # CREATE SYNTHETIC ANOMALOUS READING
    # --------------------------------------------------------

    try:

        injected_row = inject_anomaly(
            sensor_id=sensor_id,
            anomaly_type=anomaly_type
        )

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=(
                f"Could not create test anomaly: {str(exc)}"
            )
        )

    # --------------------------------------------------------
    # REPLACE SENSOR READING
    # --------------------------------------------------------

    rows = [
        row
        for row in rows
        if str(row.get("sensor_id")) != str(sensor_id)
    ]

    rows.append(injected_row)

    # --------------------------------------------------------
    # RUN REAL DETECTOR
    # --------------------------------------------------------

    try:

        results = detect_anomaly(rows)

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=(
                f"Test anomaly detection failed: {str(exc)}"
            )
        )

    # --------------------------------------------------------
    # SAVE RESULTS
    # --------------------------------------------------------

    _save_results(results)

    # --------------------------------------------------------
    # FIND INJECTED SENSOR RESULT
    # --------------------------------------------------------

    injected_result = next(
        (
            result
            for result in results
            if str(result.get("sensor_id"))
            == str(sensor_id)
        ),
        None
    )

    return {
        "success": True,
        "sensors": results,
        "injected_sensor_id": sensor_id,
        "injected_anomaly_type": anomaly_type,
        "injected_result": injected_result
    }


# ============================================================
# SENSOR HISTORY
# ============================================================

@app.get("/api/history")
def history():
    """
    Return recent sensor analysis history.
    """

    try:
        rows = get_history()

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=f"Could not load sensor history: {str(exc)}"
        )

    return {
        "success": True,
        "history": rows
    }


# ============================================================
# SENSOR-SPECIFIC MAINTENANCE
# ============================================================

@app.get("/api/maintenance/{sensor_id}")
def maintenance(
    sensor_id: str
):
    """
    Calculate maintenance status for one sensor
    using its recent historical readings.
    """

    if not sensor_id.strip():

        raise HTTPException(
            status_code=400,
            detail="sensor_id cannot be empty"
        )

    try:

        sensor_history = get_sensor_history(
            sensor_id,
            20
        )

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=(
                f"Could not load history for "
                f"{sensor_id}: {str(exc)}"
            )
        )

    try:

        result = predict_maintenance(
            sensor_id,
            sensor_history
        )

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=(
                f"Maintenance prediction failed: "
                f"{str(exc)}"
            )
        )

    return {
        "success": True,
        "sensor_id": sensor_id,
        "history_count": len(sensor_history),
        "maintenance": result
    }
 