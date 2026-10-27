# ================================================================
# STEP 30: LIVE FORECAST WITH PRESSURE MSL
# ================================================================

import os
import requests
import pandas as pd

BASE_PATH = r"C:\Projects\WeatherModelSelector"

DATA_PATH = os.path.join(BASE_PATH, "data")
RESULTS_PATH = os.path.join(BASE_PATH, "results")

LOCATIONS_FILE = os.path.join(DATA_PATH, "Locations.csv")

OUTPUT_FILE = os.path.join(
    RESULTS_PATH,
    "live_pressure_msl_forecasts.csv"
)

MODELS = [
    "ecmwf_ifs",
    "gfs_seamless",
    "icon_seamless",
    "gem_seamless"
]

print("=" * 70)
print("STEP 30: LIVE PRESSURE MSL FORECAST")
print("=" * 70)

# ------------------------------------------------
# LOAD LOCATIONS
# ------------------------------------------------

locations = pd.read_csv(LOCATIONS_FILE)

print(f"\nTotal locations: {len(locations)}")

all_rows = []

# ------------------------------------------------
# FETCH FORECAST
# ------------------------------------------------

for _, row in locations.iterrows():

    city = row["City"]
    latitude = row["Latitude"]
    longitude = row["Longitude"]

    print(f"\nFetching: {city}")

    url = "https://api.open-meteo.com/v1/forecast"

    params = {
        "latitude": latitude,
        "longitude": longitude,
        "hourly": (
            "temperature_2m,"
            "relative_humidity_2m,"
            "precipitation,"
            "wind_speed_10m,"
            "pressure_msl"
        ),
        "models": ",".join(MODELS),
        "forecast_days": 1,
        "timezone": "UTC"
    }

    try:

        response = requests.get(
            url,
            params=params,
            timeout=30
        )

        response.raise_for_status()

        data = response.json()

        hourly = data["hourly"]

        # First forecast hour
        i = 0

        forecast_datetime = hourly["time"][i]

        for model in MODELS:

            # Open-Meteo returns model-specific variables
            # in the hourly response using model suffixes.

            suffix = "_" + model

            temp_key = "temperature_2m" + suffix
            humidity_key = "relative_humidity_2m" + suffix
            precipitation_key = "precipitation" + suffix
            wind_key = "wind_speed_10m" + suffix
            pressure_key = "pressure_msl" + suffix

            # Some API responses may use unsuffixed keys
            # when only one model is returned.
            # This script checks both forms.

            def get_value(key, fallback):

                if key in hourly:
                    return hourly[key][i]

                if fallback in hourly:
                    return hourly[fallback][i]

                return None

            temperature = get_value(
                temp_key,
                "temperature_2m"
            )

            humidity = get_value(
                humidity_key,
                "relative_humidity_2m"
            )

            precipitation = get_value(
                precipitation_key,
                "precipitation"
            )

            wind_speed = get_value(
                wind_key,
                "wind_speed_10m"
            )

            pressure_msl = get_value(
                pressure_key,
                "pressure_msl"
            )

            all_rows.append({
                "location": city,
                "latitude": latitude,
                "longitude": longitude,
                "forecast_datetime": forecast_datetime,
                "model": model,
                "temperature": temperature,
                "humidity": humidity,
                "precipitation": precipitation,
                "wind_speed": wind_speed,
                "pressure_msl": pressure_msl
            })

        print("  OK")

    except Exception as e:

        print(f"  ERROR: {e}")

# ------------------------------------------------
# SAVE
# ------------------------------------------------

result = pd.DataFrame(all_rows)

result.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\n")
print("=" * 70)
print("STEP 30 COMPLETED")
print("=" * 70)

print(f"\nRows saved: {len(result)}")
print(f"Locations: {result['location'].nunique()}")
print(f"Models: {result['model'].nunique()}")

print("\nPressure MSL sample:")

print(
    result[
        [
            "location",
            "model",
            "pressure_msl"
        ]
    ].head(20).to_string(index=False)
)

print("\nSaved file:")
print(OUTPUT_FILE)

print("=" * 70)