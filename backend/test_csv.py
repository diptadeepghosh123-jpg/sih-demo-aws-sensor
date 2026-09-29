import sys
import os
import pandas as pd


# ============================================================
# SKYGUARD AI IMPORTS
# ============================================================

from weather_api import get_sensor_data
from processing import clean_sensor_data
from anomaly import detect_anomaly
from database import create_database, save_result


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

CSV_FILE = os.path.join(
    BASE_DIR,
    "test_weather_data.csv"
)


# ============================================================
# DISPLAY SENSOR MAPPING
# ============================================================

def show_sensor_mapping(df):

    print("\n========================================")
    print("          SENSOR MAPPING")
    print("========================================")

    required_mapping_columns = [
        "sensor_id",
        "latitude",
        "longitude",
        "state",
        "district",
        "city",
        "locality"
    ]

    available_columns = [
        column
        for column in required_mapping_columns
        if column in df.columns
    ]

    if "sensor_id" not in df.columns:

        print(
            "\nERROR: sensor_id column is missing."
        )

        return


    if not available_columns:

        print(
            "\nNo sensor mapping information found."
        )

        return


    # --------------------------------------------------------
    # Remove duplicate sensor entries
    # --------------------------------------------------------

    mapping = (
        df[available_columns]
        .drop_duplicates(
            subset=["sensor_id"]
        )
    )


    print(
        "\nRegistered sensors:"
    )

    print(
        "----------------------------------------"
    )

    print(
        mapping.to_string(
            index=False
        )
    )


    print(
        "\nTotal unique sensors:",
        df["sensor_id"].nunique()
    )


# ============================================================
# DISPLAY CSV STRUCTURE
# ============================================================

def show_csv_structure(df):

    print("\n========================================")
    print("          CSV STRUCTURE")
    print("========================================")

    print(
        "\nColumns:"
    )

    for column in df.columns:

        print(
            " -",
            column
        )


# ============================================================
# FETCH FRESH WEATHER DATA
# ============================================================

def create_csv():

    print("========================================")
    print("       SKYGUARD AI CSV CREATOR")
    print("========================================")


    # ========================================================
    # INITIALIZE DATABASE
    # ========================================================

    create_database()


    # ========================================================
    # FETCH WINDY DATA
    # ========================================================

    print(
        "\nFetching weather data..."
    )

    df = get_sensor_data()


    # ========================================================
    # CHECK DATA
    # ========================================================

    if df.empty:

        print(
            "\nERROR:"
        )

        print(
            "No weather data received."
        )

        return


    # ========================================================
    # DISPLAY RAW DATA
    # ========================================================

    print(
        "\nWeather data received:"
    )

    print(
        "----------------------------------------"
    )

    print(
        df.to_string(
            index=False
        )
    )


    # ========================================================
    # SAVE CSV
    # ========================================================

    try:

        df.to_csv(
            CSV_FILE,
            index=False
        )

    except Exception as error:

        print(
            "\nERROR saving CSV:"
        )

        print(error)

        return


    print(
        "\nCSV saved successfully:"
    )

    print(
        CSV_FILE
    )


    # ========================================================
    # SHOW STRUCTURE
    # ========================================================

    show_csv_structure(
        df
    )


    # ========================================================
    # SHOW SENSOR MAPPING
    # ========================================================

    show_sensor_mapping(
        df
    )


    # ========================================================
    # IMPORTANT MESSAGE
    # ========================================================

    print(
        "\n========================================"
    )

    print(
        "          CSV CREATED"
    )

    print(
        "========================================"
    )

    print(
        "\nYou can now open:"
    )

    print(
        CSV_FILE
    )

    print(
        "\nYou may manually modify the CSV."
    )

    print(
        "\nIMPORTANT:"
    )

    print(
        "Do NOT run 'python test_csv.py'"
    )

    print(
        "after manually modifying the CSV."
    )

    print(
        "\nThat command fetches fresh Windy data"
    )

    print(
        "and overwrites the CSV."
    )

    print(
        "\nTo test your modified CSV, run:"
    )

    print(
        "python test_csv.py --test"
    )


# ============================================================
# READ MODIFIED CSV
# ============================================================

def read_csv():

    if not os.path.exists(
        CSV_FILE
    ):

        print(
            "\nERROR:"
        )

        print(
            "test_weather_data.csv was not found."
        )

        print(
            "\nRun first:"
        )

        print(
            "python test_csv.py"
        )

        return None


    try:

        df = pd.read_csv(
            CSV_FILE
        )

    except pd.errors.EmptyDataError:

        print(
            "\nERROR:"
        )

        print(
            "test_weather_data.csv is empty."
        )

        return None

    except Exception as error:

        print(
            "\nERROR reading CSV:"
        )

        print(error)

        return None


    return df


# ============================================================
# TEST MODIFIED CSV
# ============================================================

def test_modified_csv():

    print("========================================")
    print("   SKYGUARD AI - CSV HYBRID TEST")
    print("========================================")


    # ========================================================
    # INITIALIZE DATABASE
    # ========================================================

    create_database()


    # ========================================================
    # READ CSV
    # ========================================================

    df = read_csv()


    if df is None:

        return


    # ========================================================
    # SHOW CSV DATA
    # ========================================================

    print(
        "\nData loaded from CSV:"
    )

    print(
        "----------------------------------------"
    )

    print(
        df.to_string(
            index=False
        )
    )


    # ========================================================
    # SHOW CSV STRUCTURE
    # ========================================================

    show_csv_structure(
        df
    )


    # ========================================================
    # SHOW SENSOR MAPPING
    # ========================================================

    show_sensor_mapping(
        df
    )


    # ========================================================
    # CLEAN DATA
    # ========================================================

    print(
        "\nCleaning and validating data..."
    )

    try:

        df = clean_sensor_data(
            df
        )

    except Exception as error:

        print(
            "\nCSV validation error:"
        )

        print(error)

        return


    # ========================================================
    # DISPLAY CLEAN DATA
    # ========================================================

    print(
        "\nCleaned data:"
    )

    print(
        "----------------------------------------"
    )

    print(
        df.to_string(
            index=False
        )
    )


    # ========================================================
    # RUN HYBRID MODEL
    # ========================================================

    print(
        "\n========================================"
    )

    print(
        "          RUNNING HYBRID MODEL"
    )

    print(
        "========================================"
    )


    try:

        results = detect_anomaly(
            df
        )

    except Exception as error:

        print(
            "\nERROR during anomaly detection:"
        )

        print(
            error
        )

        return


    # ========================================================
    # SAVE RESULTS
    # ========================================================

    print(
        "\nSaving results to database..."
    )


    for result in results:

        try:

            save_result(
                result
            )

        except Exception as error:

            print(
                "\nDatabase error for sensor:",
                result.get(
                    "sensor_id",
                    "UNKNOWN"
                )
            )

            print(
                error
            )


    # ========================================================
    # DISPLAY RESULTS
    # ========================================================

    print(
        "\n========================================"
    )

    print(
        "       HYBRID MODEL RESULTS"
    )

    print(
        "========================================"


    )


    for result in results:

        print(
            "\n----------------------------------------"
        )


        # ----------------------------------------------------
        # SENSOR
        # ----------------------------------------------------

        print(
            "Sensor ID   :",
            result.get(
                "sensor_id",
                "N/A"
            )
        )


        # ----------------------------------------------------
        # LOCATION
        # ----------------------------------------------------

        print(
            "Latitude    :",
            result.get(
                "latitude",
                "N/A"
            )
        )

        print(
            "Longitude   :",
            result.get(
                "longitude",
                "N/A"
            )
        )

        print(
            "State       :",
            result.get(
                "state",
                "N/A"
            )
        )

        print(
            "District    :",
            result.get(
                "district",
                "N/A"
            )
        )

        print(
            "City        :",
            result.get(
                "city",
                "N/A"
            )
        )

        print(
            "Locality    :",
            result.get(
                "locality",
                "N/A"
            )
        )


        # ----------------------------------------------------
        # WEATHER
        # ----------------------------------------------------

        print(
            "\nWeather:"
        )

        print(
            "Temperature :",
            result.get(
                "temperature",
                "N/A"
            ),
            "°C"
        )

        print(
            "Pressure    :",
            result.get(
                "pressure",
                "N/A"
            ),
            "hPa"
        )

        print(
            "Humidity    :",
            result.get(
                "humidity",
                "N/A"
            ),
            "%"
        )


        # ----------------------------------------------------
        # TIME
        # ----------------------------------------------------

        print(
            "Observation :",
            result.get(
                "observation_time",
                result.get(
                    "timestamp",
                    "N/A"
                )
            )
        )


        # ----------------------------------------------------
        # DETECTION
        # ----------------------------------------------------

        print(
            "\nDetection:"
        )

        print(
            "Status      :",
            result.get(
                "status",
                "N/A"
            )
        )

        print(
            "Category    :",
            result.get(
                "anomaly_category",
                "N/A"
            )
        )

        print(
            "Fault Type  :",
            result.get(
                "fault_type",
                "N/A"
            )
        )

        print(
            "Severity    :",
            result.get(
                "severity",
                "N/A"
            )
        )

        print(
            "Confidence  :",
            result.get(
                "confidence",
                "N/A"
            ),
            "%"
        )

        print(
            "Health      :",
            result.get(
                "health",
                "N/A"
            ),
            "%"
        )


        # ----------------------------------------------------
        # EXPLANATION
        # ----------------------------------------------------

        print(
            "\nExplanation:"
        )

        print(
            result.get(
                "explanation",
                "No explanation available."
            )
        )


        # ----------------------------------------------------
        # TEMPERATURE ANALYSIS
        # ----------------------------------------------------

        if (
            "temperature_score"
            in result
        ):

            print(
                "\nTemperature Analysis:"
            )

            print(
                "Score:",
                result.get(
                    "temperature_score"
                )
            )

            print(
                "Reason:",
                result.get(
                    "temperature_reason",
                    "N/A"
                )
            )


        # ----------------------------------------------------
        # SPATIAL ANALYSIS
        # ----------------------------------------------------

        if (
            "neighbor_count"
            in result
        ):

            print(
                "\nSpatial Analysis:"
            )

            print(
                "Neighbor Count:",
                result.get(
                    "neighbor_count"
                )
            )

            print(
                "Neighbor Median:",
                result.get(
                    "neighbor_median"
                )
            )

            print(
                "Neighbor Spread:",
                result.get(
                    "neighbor_spread"
                )
            )


    # ========================================================
    # COMPLETE
    # ========================================================

    print(
        "\n========================================"
    )

    print(
        "          TEST COMPLETE"
    )

    print(
        "========================================"
    )


# ============================================================
# PROGRAM START
# ============================================================

if __name__ == "__main__":

    # --------------------------------------------------------
    # --test
    # --------------------------------------------------------
    #
    # Reads the existing CSV.
    #
    # Does NOT call Windy.
    #
    # --------------------------------------------------------

    if (
        len(sys.argv) > 1
        and sys.argv[1].lower() == "--test"
    ):

        test_modified_csv()

    else:

        # ----------------------------------------------------
        # Normal mode
        #
        # Fetch fresh Windy data
        # and create/update CSV.
        # ----------------------------------------------------

        create_csv()