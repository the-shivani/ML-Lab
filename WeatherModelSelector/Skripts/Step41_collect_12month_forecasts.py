import os
import time
import requests
import pandas as pd
from datetime import datetime

# ============================================================
# STEP 1: COLLECT 12-MONTH HISTORICAL FORECAST DATA
# ============================================================

BASE_PATH = r"C:\Projects\WeatherModelSelector"
DATA_PATH = os.path.join(BASE_PATH, "data")

LOCATIONS_FILE = os.path.join(DATA_PATH, "Locations.csv")
OUTPUT_FILE = os.path.join(
    DATA_PATH,
    "historical_multivariable_forecast_12months.csv"
)

API_URL = "https://historical-forecast-api.open-meteo.com/v1/forecast"

# ------------------------------------------------------------
# 12-MONTH PERIOD
# ------------------------------------------------------------

START_DATE = "2025-10-01"
END_DATE = "2026-09-30"

# ------------------------------------------------------------
# MODELS
# ------------------------------------------------------------

MODELS = [
    "ecmwf_ifs",
    "gfs_seamless",
    "icon_seamless",
    "gem_seamless"
]

# ------------------------------------------------------------
# WEATHER VARIABLES
# ------------------------------------------------------------

HOURLY_VARIABLES = [
    "temperature_2m",
    "relative_humidity_2m",
    "precipitation",
    "wind_speed_10m",
    "pressure_msl"
]

# ------------------------------------------------------------
# LOAD LOCATIONS
# ------------------------------------------------------------

locations = pd.read_csv(LOCATIONS_FILE)

print("=" * 75)
print("       12-MONTH HISTORICAL WEATHER FORECAST COLLECTOR")
print("=" * 75)

print("\nLocations file:")
print(LOCATIONS_FILE)

print("\nColumns found:")
print(list(locations.columns))

print("\nTotal locations:", len(locations))

print("\nHistorical period:")
print(START_DATE, "to", END_DATE)

print("\nModels:")
print(", ".join(MODELS))

print("\nParameters:")
print(", ".join(HOURLY_VARIABLES))

print("\nOutput:")
print(OUTPUT_FILE)

# ------------------------------------------------------------
# CHECK REQUIRED LOCATION COLUMNS
# ------------------------------------------------------------

required_location_columns = [
    "City",
    "Latitude",
    "Longitude"
]

for column in required_location_columns:
    if column not in locations.columns:
        raise ValueError(
            f"Required column '{column}' not found in Locations.csv"
        )

# ------------------------------------------------------------
# STORAGE
# ------------------------------------------------------------

all_rows = []

session = requests.Session()

# ------------------------------------------------------------
# PROCESS EACH LOCATION
# ------------------------------------------------------------

for location_number, location_row in locations.iterrows():

    city = str(location_row["City"])
    latitude = float(location_row["Latitude"])
    longitude = float(location_row["Longitude"])

    print("\n" + "=" * 75)
    print(
        f"LOCATION {location_number + 1}/{len(locations)}: {city}"
    )
    print(f"Latitude : {latitude}")
    print(f"Longitude: {longitude}")
    print("=" * 75)

    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": START_DATE,
        "end_date": END_DATE,
        "hourly": ",".join(HOURLY_VARIABLES),
        "models": ",".join(MODELS),
        "timezone": "UTC"
    }

    success = False

    for attempt in range(1, 4):

        try:

            print(
                f"  Request attempt {attempt}/3..."
            )

            response = session.get(
                API_URL,
                params=params,
                timeout=120
            )

            print(
                "  HTTP status:",
                response.status_code
            )

            if response.status_code != 200:

                print(
                    "  API error:",
                    response.text[:500]
                )

                time.sleep(5)
                continue

            data = response.json()

            hourly = data.get("hourly")

            if hourly is None:
                print("  ERROR: No hourly data returned.")
                time.sleep(5)
                continue

            times = hourly.get("time", [])

            if not times:
                print("  ERROR: No timestamps returned.")
                time.sleep(5)
                continue

            print(
                f"  Received {len(times)} hourly timestamps."
            )

            # ------------------------------------------------
            # CREATE ROWS
            # ------------------------------------------------

            for i, timestamp in enumerate(times):

                for model in MODELS:

                    # Open-Meteo multi-model responses normally
                    # use names such as:
                    # temperature_2m_ecmwf_ifs
                    # temperature_2m_gfs_seamless
                    # etc.

                    def get_value(variable):

                        key = f"{variable}_{model}"

                        if key in hourly:
                            values = hourly[key]

                            if i < len(values):
                                return values[i]

                        # Fallback for possible single-model style
                        if variable in hourly:
                            values = hourly[variable]

                            if i < len(values):
                                return values[i]

                        return None

                    row = {
                        "location": city,
                        "latitude": latitude,
                        "longitude": longitude,
                        "datetime": timestamp,
                        "model": model,
                        "temperature": get_value(
                            "temperature_2m"
                        ),
                        "humidity": get_value(
                            "relative_humidity_2m"
                        ),
                        "precipitation": get_value(
                            "precipitation"
                        ),
                        "wind_speed": get_value(
                            "wind_speed_10m"
                        ),
                        "pressure": get_value(
                            "pressure_msl"
                        )
                    }

                    all_rows.append(row)

            print(
                f"  Successfully processed {city}."
            )

            success = True
            break

        except Exception as e:

            print(
                f"  ERROR on attempt {attempt}: {e}"
            )

            time.sleep(5)

    if not success:

        print(
            f"  FAILED: {city}"
        )

    # Small delay to avoid sending requests too quickly
    time.sleep(1)

# ------------------------------------------------------------
# CREATE DATAFRAME
# ------------------------------------------------------------

print("\n" + "=" * 75)
print("CREATING FINAL DATASET")
print("=" * 75)

df = pd.DataFrame(all_rows)

if df.empty:
    raise ValueError(
        "No data was collected. Dataset is empty."
    )

# ------------------------------------------------------------
# CLEAN DATETIME
# ------------------------------------------------------------

df["datetime"] = pd.to_datetime(
    df["datetime"],
    errors="coerce"
)

# ------------------------------------------------------------
# REMOVE INVALID ROWS
# ------------------------------------------------------------

before = len(df)

df = df.dropna(
    subset=[
        "location",
        "datetime",
        "model"
    ]
)

after = len(df)

print(
    f"\nRows removed because of invalid basic data: "
    f"{before - after}"
)

# ------------------------------------------------------------
# REMOVE DUPLICATES
# ------------------------------------------------------------

before = len(df)

df = df.drop_duplicates(
    subset=[
        "location",
        "datetime",
        "model"
    ]
)

after = len(df)

print(
    f"Duplicate rows removed: {before - after}"
)

# ------------------------------------------------------------
# SORT
# ------------------------------------------------------------

df = df.sort_values(
    [
        "location",
        "datetime",
        "model"
    ]
).reset_index(drop=True)

# ------------------------------------------------------------
# SAVE
# ------------------------------------------------------------

df.to_csv(
    OUTPUT_FILE,
    index=False
)

# ------------------------------------------------------------
# FINAL REPORT
# ------------------------------------------------------------

print("\n" + "=" * 75)
print("12-MONTH DATA COLLECTION COMPLETE")
print("=" * 75)

print("\nRows:", len(df))

print(
    "Locations:",
    df["location"].nunique()
)

print(
    "Models:",
    df["model"].nunique()
)

print(
    "Start:",
    df["datetime"].min()
)

print(
    "End:",
    df["datetime"].max()
)

print(
    "Unique timestamps:",
    df["datetime"].nunique()
)

print("\nModel counts:")
print(
    df["model"].value_counts()
)

print("\nMissing values:")
print(
    df[
        [
            "temperature",
            "humidity",
            "precipitation",
            "wind_speed",
            "pressure"
        ]
    ].isna().sum()
)

print("\nSaved to:")
print(OUTPUT_FILE)

print("\nFirst 10 rows:")
print(df.head(10).to_string(index=False))

print("\n" + "=" * 75)
print("DONE")
print("=" * 75)