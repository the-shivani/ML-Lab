import os
import time
import requests
import pandas as pd

BASE_PATH = r"C:\Projects\WeatherModelSelector"
DATA_PATH = os.path.join(BASE_PATH, "data")

LOCATIONS_FILE = os.path.join(DATA_PATH, "Locations.csv")
OUTPUT_FILE = os.path.join(
    DATA_PATH,
    "era5_multivariable_reference_12months.csv"
)

API_URL = "https://archive-api.open-meteo.com/v1/archive"

START_DATE = "2025-10-01"
END_DATE = "2026-09-30"

HOURLY_VARIABLES = [
    "temperature_2m",
    "relative_humidity_2m",
    "precipitation",
    "wind_speed_10m",
    "pressure_msl"
]

locations = pd.read_csv(LOCATIONS_FILE)

print("=" * 70)
print("12-MONTH ERA5 REFERENCE DATA COLLECTION")
print("=" * 70)

print(f"Locations: {len(locations)}")
print(f"Period: {START_DATE} to {END_DATE}")
print()

all_rows = []

session = requests.Session()

for location_number, location_row in locations.iterrows():

    city = str(location_row["City"])
    latitude = float(location_row["Latitude"])
    longitude = float(location_row["Longitude"])

    print("=" * 70)
    print(f"LOCATION {location_number + 1}/{len(locations)}: {city}")
    print(f"Latitude : {latitude}")
    print(f"Longitude: {longitude}")
    print("=" * 70)

    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": START_DATE,
        "end_date": END_DATE,
        "hourly": ",".join(HOURLY_VARIABLES),
        "timezone": "UTC"
    }

    success = False

    for attempt in range(1, 4):

        try:
            print(f"Attempt {attempt}/3...")

            response = session.get(
                API_URL,
                params=params,
                timeout=180
            )

            print("HTTP status:", response.status_code)

            if response.status_code != 200:
                print(response.text[:500])
                time.sleep(5)
                continue

            data = response.json()
            hourly = data.get("hourly", {})
            times = hourly.get("time", [])

            if not times:
                print("No timestamps returned.")
                time.sleep(5)
                continue

            print("Hourly timestamps:", len(times))

            for i, timestamp in enumerate(times):

                all_rows.append({
                    "location": city,
                    "latitude": latitude,
                    "longitude": longitude,
                    "datetime": timestamp,
                    "reference_temperature":
                        hourly["temperature_2m"][i],
                    "reference_humidity":
                        hourly["relative_humidity_2m"][i],
                    "reference_precipitation":
                        hourly["precipitation"][i],
                    "reference_wind_speed":
                        hourly["wind_speed_10m"][i],
                    "reference_pressure":
                        hourly["pressure_msl"][i]
                })

            print("SUCCESS:", city)
            success = True
            break

        except Exception as e:
            print("ERROR:", e)
            time.sleep(5)

    if not success:
        print("FAILED:", city)

    time.sleep(1)

print()
print("=" * 70)
print("CREATING ERA5 DATASET")
print("=" * 70)

df = pd.DataFrame(all_rows)

if df.empty:
    raise ValueError("No ERA5 data was collected.")

df["datetime"] = pd.to_datetime(
    df["datetime"],
    errors="coerce"
)

df = df.dropna(
    subset=["location", "datetime"]
)

df = df.drop_duplicates(
    subset=["location", "datetime"]
)

df = df.sort_values(
    ["location", "datetime"]
).reset_index(drop=True)

df.to_csv(
    OUTPUT_FILE,
    index=False
)

print()
print("=" * 70)
print("COLLECTION COMPLETE")
print("=" * 70)

print("Rows:", len(df))
print("Locations:", df["location"].nunique())
print("Start:", df["datetime"].min())
print("End:", df["datetime"].max())
print("Unique timestamps:", df["datetime"].nunique())

print()
print("Missing values:")
print(df.isna().sum())

print()
print("Saved to:")
print(OUTPUT_FILE)

print()
print("First 5 rows:")
print(df.head().to_string(index=False))

print()
print("DONE")
