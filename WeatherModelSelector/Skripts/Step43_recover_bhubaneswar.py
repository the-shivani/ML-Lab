import os
import time
import requests
import pandas as pd

BASE_PATH = r"C:\Projects\WeatherModelSelector"
DATA_PATH = os.path.join(BASE_PATH, "data")

OUTPUT_FILE = os.path.join(
    DATA_PATH,
    "era5_multivariable_reference_12months.csv"
)

API_URL = "https://archive-api.open-meteo.com/v1/archive"

START_DATE = "2025-10-01"
END_DATE = "2026-09-30"

CITY = "Bhubaneswar"
LATITUDE = 20.2961
LONGITUDE = 85.8245

HOURLY_VARIABLES = [
    "temperature_2m",
    "relative_humidity_2m",
    "precipitation",
    "wind_speed_10m",
    "pressure_msl"
]

print("=" * 70)
print("RECOVERING BHUBANESWAR ERA5 DATA")
print("=" * 70)

params = {
    "latitude": LATITUDE,
    "longitude": LONGITUDE,
    "start_date": START_DATE,
    "end_date": END_DATE,
    "hourly": ",".join(HOURLY_VARIABLES),
    "timezone": "UTC"
}

session = requests.Session()

data = None

for attempt in range(1, 6):

    try:
        print(f"Attempt {attempt}/5...")

        response = session.get(
            API_URL,
            params=params,
            timeout=180
        )

        print("HTTP status:", response.status_code)

        if response.status_code == 200:
            data = response.json()
            break

        print(response.text[:500])

    except Exception as e:
        print("ERROR:", e)

    time.sleep(10)

if data is None:
    raise RuntimeError("Could not download Bhubaneswar data.")

hourly = data.get("hourly", {})
times = hourly.get("time", [])

print("Hourly timestamps:", len(times))

if len(times) != 8760:
    raise ValueError(
        f"Expected 8760 timestamps, but received {len(times)}."
    )

rows = []

for i, timestamp in enumerate(times):

    rows.append({
        "location": CITY,
        "latitude": LATITUDE,
        "longitude": LONGITUDE,
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

bhubaneswar_df = pd.DataFrame(rows)

bhubaneswar_df["datetime"] = pd.to_datetime(
    bhubaneswar_df["datetime"]
)

print()
print("Bhubaneswar rows:", len(bhubaneswar_df))
print("Missing values:")
print(bhubaneswar_df.isna().sum())

print()
print("Reading existing ERA5 dataset...")

existing_df = pd.read_csv(
    OUTPUT_FILE,
    parse_dates=["datetime"]
)

print("Existing rows:", len(existing_df))
print("Existing locations:", existing_df["location"].nunique())

existing_df = existing_df[
    existing_df["location"] != CITY
].copy()

final_df = pd.concat(
    [existing_df, bhubaneswar_df],
    ignore_index=True
)

final_df = final_df.drop_duplicates(
    subset=["location", "datetime"]
)

final_df = final_df.sort_values(
    ["location", "datetime"]
).reset_index(drop=True)

final_df.to_csv(
    OUTPUT_FILE,
    index=False
)

print()
print("=" * 70)
print("RECOVERY COMPLETE")
print("=" * 70)

print("Final rows:", len(final_df))
print("Final locations:", final_df["location"].nunique())
print("Start:", final_df["datetime"].min())
print("End:", final_df["datetime"].max())

print()
print("Rows per location:")
print(
    final_df.groupby("location").size().value_counts().sort_index()
)

print()
print("Bhubaneswar rows:",
      len(final_df[final_df["location"] == CITY]))

print()
print("Saved to:")
print(OUTPUT_FILE)

print()
print("DONE")
