# ============================================================
# SKYGUARD AI
# HISTORICAL WEATHER STREAM REPLAY
# ============================================================

import os
import sys

import pandas as pd


BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

# ============================================================
# EXCEL FILE
# ============================================================

EXCEL_FILE = os.path.join(
    BASE_DIR,
    "SkyGuard_Anomaly_Test_100.xlsx"
)


# ============================================================
# IMPORT DATABASE
# ============================================================

from database import (
    create_database,
    clear_database,
    save_result
)

from processing import (
    clean_sensor_data
)

from anomaly import (
    detect_anomaly
)


# ============================================================
# LOAD EXCEL
# ============================================================

def load_excel():

    if not os.path.exists(EXCEL_FILE):
        raise FileNotFoundError(
            "Historical fixture not found. Place "
            f"SkyGuard_Anomaly_Test_100.xlsx in {BASE_DIR} "
            "or update EXCEL_FILE to your own test dataset."
        )

    try:

        df = pd.read_excel(
            EXCEL_FILE
        )

    except Exception as e:

        raise RuntimeError(
            f"Could not read Excel file:\n{e}"
        )

    if df.empty:

        raise ValueError(
            "Historical Excel file is empty."
        )

    print(
        f"Excel loaded successfully: "
        f"{len(df)} rows"
    )

    return df


# ============================================================
# PREPARE EXCEL
# ============================================================

def prepare_excel(
    df
):

    # --------------------------------------------------------
    # REMOVE EMPTY EXCEL COLUMNS
    # --------------------------------------------------------

    df = df.loc[
        :,
        ~df.columns.astype(str).str.startswith(
            "Unnamed:"
        )
    ]

    # --------------------------------------------------------
    # CLEAN SENSOR DATA
    # --------------------------------------------------------

    df = clean_sensor_data(
        df
    )

    # --------------------------------------------------------
    # FIND TIME COLUMN
    # --------------------------------------------------------

    time_column = None

    if "observation_time" in df.columns:

        time_column = "observation_time"

    elif "timestamp" in df.columns:

        time_column = "timestamp"

    if time_column is None:

        raise ValueError(
            "Excel has no timestamp column."
        )

    # --------------------------------------------------------
    # CONVERT TIMESTAMP
    # --------------------------------------------------------

    df[time_column] = pd.to_datetime(
        df[time_column],
        errors="coerce",
        utc=True
    )

    # --------------------------------------------------------
    # REMOVE INVALID TIMESTAMPS
    # --------------------------------------------------------

    df = df.dropna(
        subset=[
            time_column
        ]
    )

    # --------------------------------------------------------
    # CHECK SENSOR ID
    # --------------------------------------------------------

    if "sensor_id" not in df.columns:

        raise ValueError(
            "Excel has no sensor_id column."
        )

    # --------------------------------------------------------
    # SORT CHRONOLOGICALLY
    # --------------------------------------------------------

    df = df.sort_values(
        [
            time_column,
            "sensor_id"
        ]
    )

    df = df.reset_index(
        drop=True
    )

    return df


# ============================================================
# REPLAY
# ============================================================

def replay(
    df
):

    time_column = (
        "observation_time"
        if "observation_time" in df.columns
        else "timestamp"
    )

    grouped = list(
        df.groupby(
            time_column,
            sort=True
        )
    )

    total_batches = len(grouped)

    print()
    print("=" * 70)
    print("SKYGUARD AI - HISTORICAL WEATHER REPLAY")
    print("=" * 70)

    print(f"Excel rows     : {len(df)}")
    print(f"Time batches   : {total_batches}")
    print(f"Sensors        : {df['sensor_id'].nunique()}")

    print(
        "Sensor IDs     : "
        + ", ".join(
            sorted(
                df["sensor_id"]
                .astype(str)
                .unique()
            )
        )
    )

    print()
    print("Processing chronologically...")
    print("-" * 70)

    anomaly_count = 0
    normal_count = 0
    weather_event_count = 0
    watch_count = 0

    sensor_anomalies = {}

    for index, (timestamp, batch) in enumerate(
        grouped,
        start=1
    ):

        # Current batch is NOT saved before detection.
        # Detector can therefore use only previous history
        # plus the current batch.
        results = detect_anomaly(
            batch.copy()
        )

        batch_has_anomaly = False
        batch_has_weather = False
        batch_has_watch = False

        for result in results:

            status = str(
                result.get(
                    "status",
                    "NORMAL"
                )
            )

            sensor_id = str(
                result.get(
                    "sensor_id",
                    ""
                )
            )

            # ====================================================
            # DEBUG ANOMALY EVIDENCE
            # ====================================================

            if status in (
                "WATCH",
                "GLITCH_SUSPECT",
                "SENSOR_ANOMALY",
                "SUSPICIOUS"
            ):

                print()
                print("----- DETECTION DEBUG -----")

                print(
                    f"Sensor          : {sensor_id}"
                )

                print(
                    f"Time            : {timestamp}"
                )

                print(
                    f"Status          : {status}"
                )

                debug_score = result.get("score")
                if debug_score is None:
                    debug_score = result.get("decision_score")
                if debug_score is None:
                    debug_score = result.get("hybrid_score")

                print(
                    f"Score           : "
                    f"{debug_score}"
                )

                print(
                    f"Severity        : "
                    f"{result.get('severity')}"
                )

                print(
                    f"Temperature     : "
                    f"{result.get('temperature')}"
                )

                print(
                    f"Temperature Δ   : "
                    f"{result.get('temperature_change')}"
                )

                temperature_rate = result.get("temperature_rate_c_per_min")
                if temperature_rate is None:
                    temperature_rate = result.get("temperature_rate_points_per_min")

                print(
                    f"Temperature rate: "
                    f"{temperature_rate} °C/min"
                )

                print(
                    f"Time gap        : "
                    f"{result.get('time_gap_seconds')}"
                )

                history_score = result.get("historical_score")
                if history_score is None:
                    history_score = result.get("decision_evidence", {}).get("historical_score")

                print(
                    f"History score   : "
                    f"{history_score}"
                )

                print(
                    f"Spatial score   : "
                    f"{result.get('spatial_score')}"
                )

                regional_score = result.get("regional_agreement")
                if regional_score is None:
                    regional_score = result.get("decision_evidence", {}).get("regional_agreement")

                print(
                    f"Regional score  : "
                    f"{regional_score}"
                )

                print(
                    f"Persistence     : "
                    f"{result.get('persistence_score')}"
                )

                ml_score_value = result.get("ml_score")
                if ml_score_value is None:
                    ml_score_value = result.get("decision_evidence", {}).get("ml_score")

                print(
                    f"ML score        : "
                    f"{ml_score_value}"
                )

                print(
                    f"Explanation     : "
                    f"{result.get('explanation')}"
                )

                print(
                    "---------------------------"
                )

            # ====================================================
            # SENSOR ANOMALY
            # ====================================================

            if status in (
                "SENSOR_ANOMALY",
                "SUSPICIOUS"
            ):

                batch_has_anomaly = True
                anomaly_count += 1

                sensor_anomalies[sensor_id] = (
                    sensor_anomalies.get(
                        sensor_id,
                        0
                    ) + 1
                )

                print()
                print(
                    "!!! SENSOR ANOMALY DETECTED !!!"
                )

                print(
                    f"Sensor      : {sensor_id}"
                )

                print(
                    f"Time        : {timestamp}"
                )

                print(
                    f"Status      : {status}"
                )

                print(
                    f"Severity    : "
                    f"{result.get('severity')}"
                )

                print(
                    f"Score       : "
                    f"{result.get('score')}"
                )

                print(
                    f"Temperature : "
                    f"{result.get('temperature')}"
                )

                print(
                    f"ΔT          : "
                    f"{result.get('temperature_change')}"
                )

                print(
                    f"Rate        : "
                    f"{result.get('temperature_rate_points_per_min')}"
                )

                print(
                    f"Time gap    : "
                    f"{result.get('time_gap_seconds')}"
                )

                print(
                    f"Neighbors   : "
                    f"{result.get('neighbor_count')}"
                )

                print(
                    f"Neighbor T  : "
                    f"{result.get('neighbor_median')}"
                )

                print(
                    f"Explanation : "
                    f"{result.get('explanation')}"
                )

                print()

            elif status == "WEATHER_EVENT":

                batch_has_weather = True
                weather_event_count += 1

            elif status in (
                "WATCH",
                "UNCERTAIN"
            ):

                batch_has_watch = True
                watch_count += 1

            else:

                normal_count += 1

        # ========================================================
        # SAVE ONLY AFTER DETECTION
        # ========================================================

        for result in results:
            save_result(result)

        # ========================================================
        # BATCH STATUS
        # ========================================================

        if batch_has_anomaly:
            batch_status = "SENSOR ANOMALY"

        elif batch_has_weather:
            batch_status = "WEATHER EVENT"

        elif batch_has_watch:
            batch_status = "WATCH"

        else:
            batch_status = "NORMAL"

        print(
            f"[{index:03d}/{total_batches}] "
            f"{timestamp}  "
            f"Rows: {len(batch):<3}  "
            f"{batch_status}"
        )

    # ========================================================
    # FINAL REPORT
    # ========================================================

    print()
    print("=" * 70)
    print("HISTORICAL REPLAY COMPLETED")
    print("=" * 70)

    print(
        f"Total Excel rows    : {len(df)}"
    )

    print(
        f"Time batches        : {total_batches}"
    )

    print(
        f"Normal results      : {normal_count}"
    )

    print(
        f"Watch results       : {watch_count}"
    )

    print(
        f"Weather events      : {weather_event_count}"
    )

    print(
        f"Sensor anomalies    : {anomaly_count}"
    )

    print()
    print("Sensor anomaly counts:")
    print("-" * 40)

    if sensor_anomalies:

        for sensor_id, count in sorted(
            sensor_anomalies.items()
        ):

            print(
                f"{sensor_id:<12}: {count}"
            )

    else:

        print(
            "No sensor anomalies detected."
        )

    print()
    print("=" * 70)


# ============================================================
# MAIN
# ============================================================

def main():

    print()

    print(
        "Clearing previous detection history..."
    )

    # --------------------------------------------------------
    # CLEAN DATABASE
    # --------------------------------------------------------

    clear_database()

    create_database()

    print(
        "Database ready."
    )

    # --------------------------------------------------------
    # LOAD EXCEL
    # --------------------------------------------------------

    df = load_excel()

    # --------------------------------------------------------
    # PREPARE DATA
    # --------------------------------------------------------

    df = prepare_excel(
        df
    )

    # --------------------------------------------------------
    # REPLAY
    # --------------------------------------------------------

    replay(
        df
    )

    print()

    print(
        "Database history has been updated."
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()