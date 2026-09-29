import math


def detect_multivariate_anomaly(sensor):

    temperature = float(
        sensor["temperature"]
    )

    pressure = float(
        sensor["pressure"]
    )

    humidity = float(
        sensor["humidity"]
    )

    reasons = []

    # --------------------------------
    # Temperature + Humidity relationship
    # --------------------------------

    if temperature > 40 and humidity > 90:

        reasons.append(
            "Unusual temperature-humidity combination"
        )

    # --------------------------------
    # Temperature + Pressure relationship
    # --------------------------------

    if temperature > 45 and pressure < 950:

        reasons.append(
            "Unusual temperature-pressure combination"
        )

    # --------------------------------
    # Pressure + Humidity relationship
    # --------------------------------

    if pressure < 950 and humidity > 90:

        reasons.append(
            "Unusual pressure-humidity combination"
        )

    # --------------------------------
    # Extreme combined condition
    # --------------------------------

    if (
        temperature > 45
        and pressure < 950
        and humidity > 90
    ):

        reasons.append(
            "Multiple weather parameters are inconsistent"
        )

    # --------------------------------
    # Result
    # --------------------------------

    if len(reasons) > 0:

        return {
            "anomaly": True,
            "reason": ", ".join(reasons)
        }

    return {
        "anomaly": False,
        "reason": "Normal multivariate relationship"
    }