def estimate_corrected_values(
    sensor,
    all_sensors
):

    temperature_values = []
    pressure_values = []
    humidity_values = []

    # --------------------------------
    # Collect healthy nearby sensors
    # --------------------------------

    for other in all_sensors:

        if (
            other["sensor_id"]
            == sensor["sensor_id"]
        ):
            continue

        temperature = float(
            other["temperature"]
        )

        pressure = float(
            other["pressure"]
        )

        humidity = float(
            other["humidity"]
        )

        # Ignore abnormal values
        if (
            temperature < -20
            or temperature > 45
            or pressure < 950
            or pressure > 1050
            or humidity < 0
            or humidity > 100
        ):
            continue

        temperature_values.append(
            temperature
        )

        pressure_values.append(
            pressure
        )

        humidity_values.append(
            humidity
        )

    # --------------------------------
    # No healthy stations
    # --------------------------------

    if len(temperature_values) == 0:

        return {
            "temperature": None,
            "pressure": None,
            "humidity": None,
            "available": False
        }

    # --------------------------------
    # Calculate estimated values
    # --------------------------------

    corrected_temperature = round(
        sum(temperature_values)
        / len(temperature_values),
        2
    )

    corrected_pressure = round(
        sum(pressure_values)
        / len(pressure_values),
        2
    )

    corrected_humidity = round(
        sum(humidity_values)
        / len(humidity_values),
        2
    )

    return {
        "temperature":
            corrected_temperature,

        "pressure":
            corrected_pressure,

        "humidity":
            corrected_humidity,

        "available": True
    }