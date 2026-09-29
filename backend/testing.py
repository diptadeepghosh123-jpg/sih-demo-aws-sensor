import os
import pandas as pd

# ============================================================
# SENSOR REGISTRY CREATOR
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REGISTRY_FILE = os.path.join(BASE_DIR, "sensor_registry.csv")


# ------------------------------------------------------------
# SENSOR DATA
# ------------------------------------------------------------
# IMPORTANT:
# These are PROTOTYPE coordinates for testing.
# Replace them later with the actual official AWS locations.

sensor_data = [
    {
        "sensor_id": "AWS_001",
        "latitude": 22.5726,
        "longitude": 88.3639,
        "state": "West Bengal",
        "district": "Kolkata",
        "city": "Kolkata",
        "locality": "Central Kolkata"
    },

    {
        "sensor_id": "AWS_002",
        "latitude": 22.5958,
        "longitude": 88.4100,
        "state": "West Bengal",
        "district": "Kolkata",
        "city": "Kolkata",
        "locality": "Salt Lake"
    },

    {
        "sensor_id": "AWS_003",
        "latitude": 22.5200,
        "longitude": 88.3500,
        "state": "West Bengal",
        "district": "Kolkata",
        "city": "Kolkata",
        "locality": "South Kolkata"
    },

    {
        "sensor_id": "AWS_004",
        "latitude": 22.6100,
        "longitude": 88.3900,
        "state": "West Bengal",
        "district": "Kolkata",
        "city": "Kolkata",
        "locality": "Lake Town"
    },

    {
        "sensor_id": "AWS_005",
        "latitude": 22.5500,
        "longitude": 88.4300,
        "state": "West Bengal",
        "district": "Kolkata",
        "city": "Kolkata",
        "locality": "Eastern Kolkata"
    }
]


# ------------------------------------------------------------
# CREATE DATAFRAME
# ------------------------------------------------------------

df = pd.DataFrame(sensor_data)


# ------------------------------------------------------------
# CHECK REQUIRED COLUMNS
# ------------------------------------------------------------

required_columns = [
    "sensor_id",
    "latitude",
    "longitude",
    "state",
    "district",
    "city",
    "locality"
]

for column in required_columns:

    if column not in df.columns:
        raise ValueError(
            f"Missing required column: {column}"
        )


# ------------------------------------------------------------
# REMOVE DUPLICATE SENSOR IDs
# ------------------------------------------------------------

df = df.drop_duplicates(
    subset=["sensor_id"],
    keep="first"
)


# ------------------------------------------------------------
# SAVE CSV
# ------------------------------------------------------------

df.to_csv(
    REGISTRY_FILE,
    index=False
)


# ------------------------------------------------------------
# DISPLAY RESULT
# ------------------------------------------------------------

print("========================================")
print("       SENSOR REGISTRY CREATED")
print("========================================")

print("\nFile:")
print(REGISTRY_FILE)

print("\nTotal sensors:")
print(len(df))

print("\nSensor registry:")
print("----------------------------------------")

print(df.to_string(index=False))

print("\n========================================")
print("       REGISTRY CREATION COMPLETE")
print("========================================")