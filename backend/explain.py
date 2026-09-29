def explain_anomaly(sensor, ml_prediction):

    reasons = []

    temperature = float(
        sensor["temperature"]
    )

    pressure = float(
        sensor["pressure"]
    )

    humidity = float(
        sensor["humidity"]
    )

    # Temperature
    if temperature > 45:
        reasons.append(
            "Temperature is unusually high"
        )

    elif temperature < -20:
        reasons.append(
            "Temperature is unusually low"
        )

    # Pressure
    if pressure < 950:
        reasons.append(
            "Atmospheric pressure is unusually low"
        )

    elif pressure > 1050:
        reasons.append(
            "Atmospheric pressure is unusually high"
        )

    # Humidity
    if humidity > 100:
        reasons.append(
            "Humidity exceeds the physical range"
        )

    elif humidity < 0:
        reasons.append(
            "Humidity is below the physical range"
        )

    # ML result
    if ml_prediction == 1:
        reasons.append(
            "Machine learning model detected an unusual pattern"
        )

    if len(reasons) == 0:

        return {
            "explanation":
                "Sensor values are consistent with the learned normal pattern."
        }

    return {
        "explanation":
            "; ".join(reasons)
    }