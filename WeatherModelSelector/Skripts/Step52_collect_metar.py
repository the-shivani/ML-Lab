import os
import requests
import pandas as pd

BASE = r"C:\Projects\WeatherModelSelector"
OUTPUT = os.path.join(BASE, "data", "observations")

os.makedirs(OUTPUT, exist_ok=True)

STATIONS = {
    "VAPO": "Pune",
    "VABB": "Mumbai",
    "VIDP": "Delhi",
    "VOBL": "Bengaluru"
}

URL = "https://aviationweather.gov/api/data/metar"

print("=" * 65)
print("STEP 52 - REAL AIRPORT WEATHER OBSERVATIONS")
print("=" * 65)

params = {
    "ids": ",".join(STATIONS.keys()),
    "format": "json",
    "hours": 24
}

try:
    response = requests.get(
        URL,
        params=params,
        headers={
            "User-Agent": "WeatherModelSelector/1.0",
            "Accept": "application/json"
        },
        timeout=60
    )

    print("HTTP status:", response.status_code)
    print("Request URL:", response.url)

    response.raise_for_status()

    if response.status_code == 204 or not response.content:
        print("No observations returned.")
        raise SystemExit(0)

    data = response.json()

    if not isinstance(data, list) or len(data) == 0:
        print("Unexpected or empty response:", str(data)[:1000])
        raise SystemExit(0)

    print("Total observations:", len(data))
    print("Available fields:", list(data[0].keys()))

    records = []

    for observation in data:

        station = observation.get("icaoId")

        if station not in STATIONS:
            continue

        obs_time = pd.to_datetime(
            observation.get("obsTime"),
            unit="s",
            utc=True,
            errors="coerce"
        )

        if pd.isna(obs_time):
            continue

        wind_knots = pd.to_numeric(
            observation.get("wspd"),
            errors="coerce"
        )

        records.append({
            "station": station,
            "city": STATIONS[station],
            "observation_time_utc": obs_time,
            "latitude": observation.get("lat"),
            "longitude": observation.get("lon"),
            "temperature": observation.get("temp"),
            "dew_point": observation.get("dewp"),
            "wind_speed_kmh": (
                wind_knots * 1.852
                if pd.notna(wind_knots)
                else None
            ),
            "wind_direction": observation.get("wdir"),
            "pressure_msl": observation.get("slp"),
            "visibility": observation.get("visib"),
            "raw_metar": observation.get("rawOb")
        })

    df = pd.DataFrame(records)

    if df.empty:
        print("No usable observations for selected stations.")
        raise SystemExit(0)

    df["observation_time_ist"] = (
        df["observation_time_utc"]
        .dt.tz_convert("Asia/Kolkata")
    )

    df = df.sort_values([
        "city",
        "observation_time_utc"
    ])

    path = os.path.join(
        OUTPUT,
        "metar_observations.csv"
    )

    df.to_csv(path, index=False)

    print("\nOBSERVATIONS BY CITY")
    print(df.groupby("city").size())

    print("\nLATEST OBSERVATIONS")
    print(
        df.groupby("city")
        .tail(1)[
            [
                "city",
                "observation_time_ist",
                "temperature",
                "dew_point",
                "wind_speed_kmh",
                "pressure_msl"
            ]
        ].to_string(index=False)
    )

    print("\nSaved:", path)

except requests.RequestException as error:
    print("API request failed:", error)

except ValueError as error:
    print("Response parsing failed:", error)

print("\nSTEP 52 COMPLETED")
