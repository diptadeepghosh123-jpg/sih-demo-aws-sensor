def predict_maintenance(
    sensor_id,
    history
):

    # No history
    if len(history) == 0:

        return {
            "degradation": "UNKNOWN",
            "maintenance_required": False,
            "maintenance_reason":
                "Not enough sensor history"
        }

    # ========================================================
    # Count anomalies
    # ========================================================

    anomaly_count = 0

    for record in history:

        # BUGFIX: anomaly.py never outputs the literal status "ANOMALY".
        # It outputs SENSOR_ANOMALY / GLITCH_SUSPECT / WATCH / WEATHER_EVENT
        # / NORMAL. The old check here matched nothing, so this function
        # always reported the sensor as HEALTHY regardless of real history.
        if record.get("status") in ("SENSOR_ANOMALY", "GLITCH_SUSPECT"):

            anomaly_count += 1

    total_readings = len(history)

    # ========================================================
    # Calculate anomaly percentage
    # ========================================================

    anomaly_percentage = (
        anomaly_count
        / total_readings
    ) * 100

    # ========================================================
    # Determine degradation
    # ========================================================

    if anomaly_percentage >= 60:

        degradation = "CRITICAL"

        maintenance_required = True

        reason = (
            "Sensor shows repeated anomalous readings"
        )

    elif anomaly_percentage >= 30:

        degradation = "WARNING"

        maintenance_required = True

        reason = (
            "Sensor is showing increasing abnormal behavior"
        )

    else:

        degradation = "HEALTHY"

        maintenance_required = False

        reason = (
            "Sensor is operating normally"
        )

    return {

        "degradation":
            degradation,

        "maintenance_required":
            maintenance_required,

        "maintenance_reason":
            reason,

        "anomaly_percentage":
            round(
                anomaly_percentage,
                2
            )
    }






