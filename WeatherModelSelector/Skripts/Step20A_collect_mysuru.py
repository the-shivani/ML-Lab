import requests
import pandas as pd
import os
import time

# ============================================================
# STEP 20A - COLLECT MISSING MYSURU ERA5 DATA
# ============================================================

BASE_PATH = r"C:\Projects\WeatherModelSelector"

DATA_PATH = os.path.join(
    BASE_PATH,
    "data"
)

OUTPUT_FILE = os.path.join(
    DATA_PATH,
    "era5_mysuru.csv"
)

URL = "https://archive-api.open-meteo.com/v1/archive"


latitude = 12.2958
longitude = 76.6394

START_DATE = "2026-09-01"
END_DATE = "2026-09-30"

HOURLY_VARIABLES = (
    "temperature_2m,"
    "relative_humidity_2m,"
    "precipitation,"
    "wind_speed_10m,"
    "surface_pressure"
)


print("=" * 70)
print("STEP 20A - MYSURU ERA5 DATA")
print("=" * 70)


params = {

    "latitude":
        latitude,

    "longitude":
        longitude,

    "start_date":
        START_DATE,

    "end_date":
        END_DATE,

    "hourly":
        HOURLY_VARIABLES,

    "timezone":
        "UTC"
}


data = None


for attempt in range(1, 6):

    try:

        print()
        print(
            f"Attempt {attempt}/5..."
        )

        response = requests.get(
            URL,
            params=params,
            timeout=180
        )

        response.raise_for_status()

        data = response.json()

        print(
            "Success!"
        )

        break

    except Exception as error:

        print(
            "Failed:",
            error
        )

        if attempt < 5:

            print(
                "Waiting 10 seconds..."
            )

            time.sleep(10)


if data is None:

    print()
    print(
        "ERROR: Could not download Mysuru data."
    )

    raise SystemExit


hourly = data["hourly"]


df = pd.DataFrame({

    "location":
        "Mysuru",

    "latitude":
        latitude,

    "longitude":
        longitude,

    "datetime":
        hourly["time"],

    "reference_temperature":
        hourly["temperature_2m"],

    "reference_humidity":
        hourly["relative_humidity_2m"],

    "reference_precipitation":
        hourly["precipitation"],

    "reference_wind_speed":
        hourly["wind_speed_10m"],

    "reference_pressure":
        hourly["surface_pressure"]
})


print()
print(
    "Rows received:",
    len(df)
)


df.to_csv(
    OUTPUT_FILE,
    index=False
)


print()
print(
    "Saved to:"
)

print(
    OUTPUT_FILE
)

print()
print("=" * 70)
print("STEP 20A FINISHED")
print("=" * 70)