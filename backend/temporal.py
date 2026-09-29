import statistics


# ============================================================
# TEMPORAL ANOMALY DETECTION
# ============================================================

def safe_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def calculate_z_score(value, values):
    if len(values) < 3:
        return None

    try:
        mean_value = statistics.mean(values)
        std_value = statistics.stdev(values)

        if std_value == 0:
            return None

        return abs(value - mean_value) / std_value

    except statistics.StatisticsError:
        return None


def detect_temporal_anomaly(current, history):

    # ========================================================
    # 1. NO HISTORY
    # ========================================================

    if not history:

        return {
            "anomaly": False,
            "reason": "No historical data available",
            "frozen": False
        }

    current_temperature = safe_float(current.get("temperature"))
    current_pressure = safe_float(current.get("pressure"))
    current_humidity = safe_float(current.get("humidity"))

    if (
        current_temperature is None
        or current_pressure is None
        or current_humidity is None
    ):

        return {
            "anomaly": False,
            "reason": "Current sensor data is invalid",
            "frozen": False
        }

    # ========================================================
    # 2. CLEAN HISTORY
    # ========================================================

    temperatures = []
    pressures = []
    humidities = []

    for row in history:

        temperature = safe_float(row.get("temperature"))
        pressure = safe_float(row.get("pressure"))
        humidity = safe_float(row.get("humidity"))

        if (
            temperature is None
            or pressure is None
            or humidity is None
        ):
            continue

        temperatures.append(temperature)
        pressures.append(pressure)
        humidities.append(humidity)

    if not temperatures:

        return {
            "anomaly": False,
            "reason": "No valid historical data available",
            "frozen": False
        }

    # ========================================================
    # 3. LATEST HISTORICAL READING
    # ========================================================

    previous_temperature = temperatures[-1]
    previous_pressure = pressures[-1]
    previous_humidity = humidities[-1]

    reasons = []
    frozen_reasons = []

    # ========================================================
    # 4. SUDDEN TEMPERATURE CHANGE
    # ========================================================

    temperature_change = (
        current_temperature - previous_temperature
    )

    if abs(temperature_change) >= 5:

        if temperature_change > 0:

            reasons.append(
                f"Sudden temperature increase of "
                f"{round(temperature_change, 2)}°C"
            )

        else:

            reasons.append(
                f"Sudden temperature decrease of "
                f"{round(abs(temperature_change), 2)}°C"
            )

    # ========================================================
    # 5. SUDDEN PRESSURE CHANGE
    # ========================================================

    pressure_change = (
        current_pressure - previous_pressure
    )

    if abs(pressure_change) >= 10:

        if pressure_change > 0:

            reasons.append(
                f"Sudden pressure increase of "
                f"{round(pressure_change, 2)} hPa"
            )

        else:

            reasons.append(
                f"Sudden pressure decrease of "
                f"{round(abs(pressure_change), 2)} hPa"
            )

    # ========================================================
    # 6. IMPROVED HUMIDITY CHANGE
    # ========================================================

    humidity_change = (
        current_humidity - previous_humidity
    )

    absolute_humidity_change = abs(humidity_change)

    # Strong sudden change
    if absolute_humidity_change >= 20:

        if humidity_change > 0:

            reasons.append(
                f"Large sudden humidity increase of "
                f"{round(humidity_change, 2)}%"
            )

        else:

            reasons.append(
                f"Large sudden humidity decrease of "
                f"{round(abs(humidity_change), 2)}%"
            )

    # Moderate sudden change
    elif absolute_humidity_change >= 15:

        if humidity_change > 0:

            reasons.append(
                f"Sudden humidity increase of "
                f"{round(humidity_change, 2)}%"
            )

        else:

            reasons.append(
                f"Sudden humidity decrease of "
                f"{round(abs(humidity_change), 2)}%"
            )

    # ========================================================
    # 7. HISTORICAL MEDIAN
    # ========================================================
    #
    # Median is more resistant to previous abnormal readings
    # than the simple mean.
    # ========================================================

    median_temperature = statistics.median(temperatures)
    median_pressure = statistics.median(pressures)
    median_humidity = statistics.median(humidities)

    temperature_difference = abs(
        current_temperature - median_temperature
    )

    pressure_difference = abs(
        current_pressure - median_pressure
    )

    humidity_difference = abs(
        current_humidity - median_humidity
    )

    # ========================================================
    # TEMPERATURE HISTORICAL DEVIATION
    # ========================================================

    if temperature_difference > 10:

        reasons.append(
            f"Temperature differs from historical pattern by "
            f"{round(temperature_difference, 2)}°C"
        )

    # ========================================================
    # PRESSURE HISTORICAL DEVIATION
    # ========================================================

    if pressure_difference > 20:

        reasons.append(
            f"Pressure differs from historical pattern by "
            f"{round(pressure_difference, 2)} hPa"
        )

    # ========================================================
    # HUMIDITY HISTORICAL DEVIATION
    # ========================================================

    if humidity_difference > 25:

        reasons.append(
            f"Humidity differs from historical pattern by "
            f"{round(humidity_difference, 2)}%"
        )

    # ========================================================
    # 8. VERY LOW / VERY HIGH HUMIDITY CONTEXT
    # ========================================================
    #
    # This is NOT an automatic sensor fault.
    #
    # It only adds temporal evidence.
    # Spatial analysis will later determine whether nearby
    # sensors support the reading.
    # ========================================================

    if current_humidity <= 10:

        reasons.append(
            f"Extremely low humidity reading of "
            f"{round(current_humidity, 2)}%"
        )

    elif current_humidity >= 95:

        reasons.append(
            f"Extremely high humidity reading of "
            f"{round(current_humidity, 2)}%"
        )

    # ========================================================
    # 9. FROZEN / STUCK SENSOR
    # ========================================================

    frozen = False

    # --------------------------------------------------------
    # Temperature
    # --------------------------------------------------------

    if len(temperatures) >= 4:

        recent_temperatures = temperatures[-4:]

        if (
            max(recent_temperatures)
            - min(recent_temperatures)
            <= 0.05
        ):

            if (
                abs(
                    current_temperature
                    - recent_temperatures[-1]
                )
                <= 0.05
            ):

                frozen = True

                frozen_reasons.append(
                    "Temperature sensor appears frozen or stuck"
                )

    # --------------------------------------------------------
    # Pressure
    # --------------------------------------------------------

    if len(pressures) >= 4:

        recent_pressures = pressures[-4:]

        if (
            max(recent_pressures)
            - min(recent_pressures)
            <= 0.05
        ):

            if (
                abs(
                    current_pressure
                    - recent_pressures[-1]
                )
                <= 0.05
            ):

                frozen = True

                frozen_reasons.append(
                    "Pressure sensor appears frozen or stuck"
                )

    # --------------------------------------------------------
    # Humidity
    # --------------------------------------------------------

    if len(humidities) >= 4:

        recent_humidities = humidities[-4:]

        if (
            max(recent_humidities)
            - min(recent_humidities)
            <= 0.05
        ):

            if (
                abs(
                    current_humidity
                    - recent_humidities[-1]
                )
                <= 0.05
            ):

                frozen = True

                frozen_reasons.append(
                    "Humidity sensor appears frozen or stuck"
                )

    if frozen:

        reasons.extend(frozen_reasons)

    # ========================================================
    # 10. HUMIDITY DRIFT
    # ========================================================

    if len(humidities) >= 5:

        recent_humidities = humidities[-5:]

        first_value = recent_humidities[0]
        last_value = recent_humidities[-1]

        drift = last_value - first_value

        # Large consistent movement
        if abs(drift) >= 30:

            if drift > 0:

                reasons.append(
                    f"Large humidity drift of "
                    f"{round(drift, 2)}% across recent readings"
                )

            else:

                reasons.append(
                    f"Large humidity drift of "
                    f"{round(abs(drift), 2)}% across recent readings"
                )

    # ========================================================
    # 11. HUMIDITY OSCILLATION / INSTABILITY
    # ========================================================

    if len(humidities) >= 5:

        recent_humidities = humidities[-5:]

        changes = []

        for i in range(1, len(recent_humidities)):

            changes.append(
                recent_humidities[i]
                - recent_humidities[i - 1]
            )

        large_changes = [
            change
            for change in changes
            if abs(change) >= 15
        ]

        if len(large_changes) >= 3:

            reasons.append(
                "Humidity readings show unusually high "
                "short-term variability"
            )

    # ========================================================
    # 12. Z-SCORE
    # ========================================================

    temperature_z = calculate_z_score(
        current_temperature,
        temperatures
    )

    pressure_z = calculate_z_score(
        current_pressure,
        pressures
    )

    humidity_z = calculate_z_score(
        current_humidity,
        humidities
    )

    if temperature_z is not None and temperature_z >= 3:

        reasons.append(
            f"Temperature is statistically unusual "
            f"(Z-score {round(temperature_z, 2)})"
        )

    if pressure_z is not None and pressure_z >= 3:

        reasons.append(
            f"Pressure is statistically unusual "
            f"(Z-score {round(pressure_z, 2)})"
        )

    if humidity_z is not None and humidity_z >= 3:

        reasons.append(
            f"Humidity is statistically unusual "
            f"(Z-score {round(humidity_z, 2)})"
        )

    # ========================================================
    # 13. FINAL RESULT
    # ========================================================

    if reasons:

        return {
            "anomaly": True,
            "reason": "; ".join(reasons),
            "frozen": frozen
        }

    return {
        "anomaly": False,
        "reason": "Normal temporal behavior",
        "frozen": False
    }