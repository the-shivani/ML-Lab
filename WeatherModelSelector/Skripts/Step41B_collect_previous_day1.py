import os
import time
import requests
import pandas as pd

# ============================================================
# STEP 41B - COLLECT 12-MONTH 24-HOUR-AHEAD FORECASTS
# ============================================================

BASE_PATH = r"C:\Projects\WeatherModelSelector"
DATA_PATH = os.path.join(BASE_PATH, "data")

LOCATIONS_FILE = os.path.join(
    DATA_PATH,
    "Locations.csv"
)

OUTPUT_FILE = os.path.join(
    DATA_PATH,
    "historical_multivariable_forecast_12months_previous_day1.csv"
)

API_URL = "https://previous-runs-api.open-meteo.com/v1/forecast"

START_DATE = "2025-10-01"
END_DATE = "2026-09-30"

MODELS = [
    "ecmwf_ifs",
    "gfs_seamless",
    "icon_seamless",
    "gem_seamless"
]

# These are the 24-hour-ahead variables
VARIABLES = [
    "temperature_2m_previous_day1",
    "relative_humidity_2m_previous_day1",
    "precipitation_previous_day1",
    "wind_speed_10m_previous_day1",
    "pressure_msl_previous_day1"
]

print("=" * 75)
print("STEP 41B - 12-MONTH PREVIOUS-DAY1 FORECAST COLLECTOR")
print("=" * 75)

print("\nForecast period:")
print(START_DATE, "to", END_DATE)

print("\nLead time:")
print("24 hours (previous_day1)")

print("\nModels:")
print(", ".join(MODELS))

# ============================================================
# LOAD LOCATIONS
# ============================================================

locations = pd.read_csv(LOCATIONS_FILE)

required_columns = [
    "City",
    "Latitude",
    "Longitude"
]

for column in required_columns:
    if column not in locations.columns:
        raise ValueError(
            f"Missing required column: {column}"
        )

print("\nTotal locations:", len(locations))

# ============================================================
# STORAGE
# ============================================================

all_rows = []

session = requests.Session()

# ============================================================
# PROCESS LOCATIONS
# ============================================================

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
        "hourly": ",".join(VARIABLES),
        "models": ",".join(MODELS),
        "timezone": "UTC"
    }

    success = False

    for attempt in range(1, 4):

        try:

            print(
                f"Request attempt {attempt}/3..."
            )

            response = session.get(
                API_URL,
                params=params,
                timeout=180
            )

            print(
                "HTTP status:",
                response.status_code
            )

            if response.status_code != 200:

                print(
                    "API error:",
                    response.text[:500]
                )

                time.sleep(5)
                continue

            data = response.json()

            hourly = data.get("hourly")

            if hourly is None:
                print("ERROR: No hourly data returned.")
                time.sleep(5)
                continue

            times = hourly.get("time", [])

            if not times:
                print("ERROR: No timestamps returned.")
                time.sleep(5)
                continue

            print(
                "Received timestamps:",
                len(times)
            )

            # ====================================================
            # CREATE ROWS
            # ====================================================

            for i, timestamp in enumerate(times):

                for model in MODELS:

                    def get_model_value(variable):

                        key = f"{variable}_{model}"

                        values = hourly.get(key)

                        if values is not None and i < len(values):
                            return values[i]

                        return None

                    row = {
                        "location": city,
                        "latitude": latitude,
                        "longitude": longitude,
                        "datetime": timestamp,
                        "model": model,

                        "temperature": get_model_value(
                            "temperature_2m_previous_day1"
                        ),

                        "humidity": get_model_value(
                            "relative_humidity_2m_previous_day1"
                        ),

                        "precipitation": get_model_value(
                            "precipitation_previous_day1"
                        ),

                        "wind_speed": get_model_value(
                            "wind_speed_10m_previous_day1"
                        ),

                        "pressure": get_model_value(
                            "pressure_msl_previous_day1"
                        )
                    }

                    all_rows.append(row)

            print(
                f"Successfully processed {city}"
            )

            success = True
            break

        except Exception as e:

            print(
                f"ERROR on attempt {attempt}: {e}"
            )

            time.sleep(5)

    if not success:

        print(
            f"FAILED: {city}"
        )

    time.sleep(1)

# ============================================================
# CREATE DATAFRAME
# ============================================================

print("\n" + "=" * 75)
print("CREATING FINAL DATASET")
print("=" * 75)

df = pd.DataFrame(all_rows)

if df.empty:
    raise ValueError(
        "No data was collected."
    )

# ============================================================
# CLEAN
# ============================================================

df["datetime"] = pd.to_datetime(
    df["datetime"],
    errors="coerce"
)

df = df.dropna(
    subset=[
        "location",
        "datetime",
        "model"
    ]
)

# Remove duplicate keys
df = df.drop_duplicates(
    subset=[
        "location",
        "datetime",
        "model"
    ]
)

df = df.sort_values(
    [
        "location",
        "datetime",
        "model"
    ]
).reset_index(drop=True)

# ============================================================
# VALIDATION
# ============================================================

print("\n" + "=" * 75)
print("VALIDATION")
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
    "Unique timestamps:",
    df["datetime"].nunique()
)

print(
    "Start:",
    df["datetime"].min()
)

print(
    "End:",
    df["datetime"].max()
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

# ============================================================
# EXPECTED ROW COUNT
# ============================================================

expected_rows = (
    len(locations)
    * 8760
    * len(MODELS)
)

print("\nExpected rows:", expected_rows)
print("Actual rows:", len(df))

if len(df) == expected_rows:
    print("PASS: Expected row count matched.")
else:
    print("WARNING: Expected row count did not match.")

# ============================================================
# SAVE
# ============================================================

df.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\nSaved to:")
print(OUTPUT_FILE)

# ============================================================
# SAMPLE CHECK
# ============================================================

print("\nSample - Pune first timestamp:")

sample = df[
    (df["location"] == "Pune") &
    (
        df["datetime"]
        == pd.Timestamp("2025-10-01 00:00:00")
    )
]

print(
    sample[
        [
            "model",
            "temperature",
            "humidity",
            "precipitation",
            "wind_speed",
            "pressure"
        ]
    ].to_string(index=False)
)

print("\n" + "=" * 75)
print("STEP 41B COMPLETE")
print("=" * 75)
