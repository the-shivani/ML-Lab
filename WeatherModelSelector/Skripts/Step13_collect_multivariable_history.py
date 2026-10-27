import requests
import pandas as pd
import os
import time

# ============================================================
# CONFIGURATION
# ============================================================

BASE_PATH = r"C:\Projects\WeatherModelSelector"

DATA_PATH = os.path.join(BASE_PATH, "data")

LOCATIONS_FILE = os.path.join(
    DATA_PATH,
    "Locations.csv"
)

OUTPUT_FILE = os.path.join(
    DATA_PATH,
    "historical_multivariable_forecast.csv"
)

START_DATE = "2026-09-01"
END_DATE = "2026-09-30"

MODELS = [
    "ecmwf_ifs",
    "gfs_seamless",
    "icon_seamless",
    "gem_seamless"
]

# ============================================================
# VARIABLES
# ============================================================

HOURLY_VARIABLES = (
    "temperature_2m,"
    "relative_humidity_2m,"
    "precipitation,"
    "wind_speed_10m,"
    "surface_pressure"
)

# ============================================================
# LOAD LOCATIONS
# ============================================================

locations = pd.read_csv(LOCATIONS_FILE)

print("=" * 70)
print("MULTI-PARAMETER HISTORICAL WEATHER COLLECTOR")
print("=" * 70)

print()
print("Locations found:", len(locations))
print("Date range:", START_DATE, "to", END_DATE)
print()

all_data = []

# ============================================================
# COLLECT DATA
# ============================================================

for index, row in locations.iterrows():

    location = row["City"]
    latitude = row["Latitude"]
    longitude = row["Longitude"]

    print("-" * 70)
    print(
        f"{index + 1}/{len(locations)} "
        f"{location}"
    )

    for model in MODELS:

        print(f"  Collecting {model}...")

        url = "https://previous-runs-api.open-meteo.com/v1/forecast"

        params = {
            "latitude": latitude,
            "longitude": longitude,
            "start_date": START_DATE,
            "end_date": END_DATE,
            "hourly": HOURLY_VARIABLES,
            "models": model,
            "timezone": "auto"
        }

        success = False

        for attempt in range(1, 4):

            try:

                response = requests.get(
                    url,
                    params=params,
                    timeout=60
                )

                response.raise_for_status()

                data = response.json()

                hourly = data["hourly"]

                df = pd.DataFrame({
                    "location": location,
                    "latitude": latitude,
                    "longitude": longitude,
                    "datetime": hourly["time"],

                    "model": model,

                    "temperature": hourly[
                        "temperature_2m"
                    ],

                    "humidity": hourly[
                        "relative_humidity_2m"
                    ],

                    "precipitation": hourly[
                        "precipitation"
                    ],

                    "wind_speed": hourly[
                        "wind_speed_10m"
                    ],

                    "pressure": hourly[
                        "surface_pressure"
                    ]
                })

                all_data.append(df)

                print(
                    f"    ✓ {len(df)} rows"
                )

                success = True
                break

            except Exception as error:

                print(
                    f"    API attempt "
                    f"{attempt}/3 failed: {error}"
                )

                if attempt < 3:

                    print(
                        "    Waiting 5 seconds..."
                    )

                    time.sleep(5)

        if not success:

            print(
                f"    ✗ FAILED: {model}"
            )

# ============================================================
# COMBINE
# ============================================================

if len(all_data) == 0:

    print()
    print("ERROR: No data collected.")
    raise SystemExit

final_df = pd.concat(
    all_data,
    ignore_index=True
)

# ============================================================
# CLEAN
# ============================================================

final_df["datetime"] = pd.to_datetime(
    final_df["datetime"]
)

final_df = final_df.drop_duplicates(
    subset=[
        "location",
        "datetime",
        "model"
    ]
)

final_df = final_df.sort_values(
    [
        "location",
        "datetime",
        "model"
    ]
)

# ============================================================
# SAVE
# ============================================================

final_df.to_csv(
    OUTPUT_FILE,
    index=False
)

# ============================================================
# SUMMARY
# ============================================================

print()
print("=" * 70)
print("COLLECTION COMPLETE")
print("=" * 70)

print()
print("Total rows:", len(final_df))

print()
print("Models:")
print(
    final_df["model"].value_counts()
)

print()
print("Locations:")
print(
    final_df["location"].nunique()
)

print()
print("Columns:")
print(
    list(final_df.columns)
)

print()
print("Saved to:")
print(OUTPUT_FILE)

print()
print("=" * 70)