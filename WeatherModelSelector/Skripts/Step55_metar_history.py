import os
import requests
import pandas as pd

BASE = r"C:\Projects\WeatherModelSelector"

OUTPUT_DIR = os.path.join(BASE, "data", "observations")
os.makedirs(OUTPUT_DIR, exist_ok=True)

OUTPUT_FILE = os.path.join(
    OUTPUT_DIR, "metar_observation_history.csv"
)

STATIONS = {
    "VABB": "Mumbai",
    "VIDP": "Delhi",
    "VOBL": "Bengaluru"
}

API_URL = "https://aviationweather.gov/api/data/metar"

print("=" * 65)
print("STEP 55 - METAR OBSERVATION HISTORY")
print("=" * 65)

response = requests.get(
    API_URL,
    params={
        "ids": ",".join(STATIONS),
        "format": "json",
        "hours": 48
    },
    headers={
        "User-Agent": "WeatherModelSelector/1.0",
        "Accept": "application/json"
    },
    timeout=60
)

print("HTTP status:", response.status_code)
response.raise_for_status()

if not response.content:
    raise SystemExit("No observations returned.")

data = response.json()

if not isinstance(data, list):
    raise SystemExit("Unexpected API response format.")

records = []

for item in data:

    station = item.get("icaoId")

    if station not in STATIONS:
        continue

    raw_time = item.get("obsTime")

    if isinstance(raw_time, (int, float)):
        obs_time = pd.to_datetime(
            raw_time, unit="s", utc=True, errors="coerce"
        )
    else:
        obs_time = pd.to_datetime(
            raw_time, utc=True, errors="coerce"
        )

    if pd.isna(obs_time):
        continue

    wind_knots = pd.to_numeric(
        item.get("wspd"), errors="coerce"
    )

    records.append({
        "station": station,
        "location": STATIONS[station],
        "observation_time_utc": obs_time,
        "latitude": item.get("lat"),
        "longitude": item.get("lon"),
        "temperature": item.get("temp"),
        "dew_point": item.get("dewp"),
        "wind_speed_kmh": (
            wind_knots * 1.852
            if pd.notna(wind_knots) else None
        ),
        "wind_direction": item.get("wdir"),
        "visibility": item.get("visib"),
        "raw_metar": item.get("rawOb")
    })

new = pd.DataFrame(records)

if new.empty:
    raise SystemExit("No usable station observations.")

if os.path.exists(OUTPUT_FILE):
    old = pd.read_csv(OUTPUT_FILE)
    combined = pd.concat(
        [old, new], ignore_index=True
    )
else:
    combined = new.copy()

combined["observation_time_utc"] = pd.to_datetime(
    combined["observation_time_utc"],
    utc=True
)

combined = (
    combined
    .drop_duplicates(
        subset=["station", "observation_time_utc"],
        keep="last"
    )
    .sort_values(["station", "observation_time_utc"])
)

combined.to_csv(OUTPUT_FILE, index=False)

print("\nOBSERVATION HISTORY SUMMARY")
print("New API records:", len(new))
print("Total unique records:", len(combined))

print("\nObservations per city:")
print(combined.groupby("location").size())

print("\nLatest observation per city (IST):")

latest = combined.groupby("location").tail(1).copy()

latest["time_ist"] = (
    latest["observation_time_utc"]
    .dt.tz_convert("Asia/Kolkata")
)

print(
    latest[
        ["location", "time_ist", "temperature", "wind_speed_kmh"]
    ].to_string(index=False)
)

print("\nSaved:", OUTPUT_FILE)
print("STEP 55 COMPLETE")
