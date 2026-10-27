import requests
import pandas as pd
import os
import urllib3

# ============================================================
# STEP 16C - TEST IMD STATION FILTER
# ============================================================

urllib3.disable_warnings(
    urllib3.exceptions.InsecureRequestWarning
)

BASE_PATH = r"C:\Projects\WeatherModelSelector"

MAPPING_FILE = os.path.join(
    BASE_PATH,
    "data",
    "imd_station_mapping.csv"
)

OBS_URL = (
    "https://wis2box.imd.gov.in/"
    "oapi/collections/"
    "urn:wmo:md:in-imd:"
    "surface-based-observations.synop/items"
)

# ============================================================
# LOAD PUNE STATION
# ============================================================

mapping = pd.read_csv(
    MAPPING_FILE
)

pune = mapping[
    mapping["location"] == "Pune"
].iloc[0]

station_id = pune["station_id"]

print("=" * 70)
print("STEP 16C - TEST IMD STATION FILTER")
print("=" * 70)

print()
print("Location:", pune["location"])
print("Station:", pune["station_name"])
print("WIGOS ID:", station_id)

# ============================================================
# QUERY EXACT STATION
# ============================================================

params = {
    "f": "json",
    "limit": 20,
    "wigos_station_identifier": station_id
}

print()
print("Requesting observations for exact station...")
print()

response = requests.get(
    OBS_URL,
    params=params,
    timeout=(15, 45),
    verify=False
)

print(
    "HTTP Status:",
    response.status_code
)

print(
    "Request URL:",
    response.url
)

response.raise_for_status()

data = response.json()

features = data.get(
    "features",
    []
)

print()
print(
    "Observations returned:",
    len(features)
)

print(
    "Total matched:",
    data.get("numberMatched")
)

# ============================================================
# DISPLAY RESULTS
# ============================================================

for i, feature in enumerate(
    features,
    start=1
):

    properties = feature.get(
        "properties",
        {}
    )

    print()
    print("-" * 70)
    print(
        "OBSERVATION",
        i
    )
    print("-" * 70)

    print(
        "Name:",
        properties.get("name")
    )

    print(
        "Station ID:",
        properties.get(
            "wigos_station_identifier"
        )
    )

    print(
        "Time:",
        properties.get(
            "phenomenonTime"
        )
    )

    print(
        "Units:",
        properties.get("units")
    )

    print(
        "Value:",
        properties.get("value")
    )

print()
print("=" * 70)
print("STEP 16C COMPLETE")
print("=" * 70)