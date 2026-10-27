import requests
import pandas as pd
import os
import urllib3

# ============================================================
# STEP 16B - TEST IMD REAL-TIME OBSERVATION
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
# LOAD STATION MAPPING
# ============================================================

mapping = pd.read_csv(
    MAPPING_FILE
)

print("=" * 70)
print("STEP 16B - TEST IMD REAL-TIME OBSERVATION")
print("=" * 70)

print()

# ============================================================
# TEST PUNE
# ============================================================

station = mapping[
    mapping["location"] == "Pune"
].iloc[0]

station_id = station["station_id"]
station_name = station["station_name"]

print(
    "Location:",
    station["location"]
)

print(
    "Station:",
    station_name
)

print(
    "Station ID:",
    station_id
)

print()

# ============================================================
# REQUEST LATEST OBSERVATIONS
# ============================================================

params = {
    "f": "json",
    "limit": 5
}

print(
    "Requesting IMD observations..."
)

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

response.raise_for_status()

data = response.json()

# ============================================================
# BASIC INFORMATION
# ============================================================

print()
print(
    "Number of observations returned:",
    len(data.get("features", []))
)

print(
    "Total observations available:",
    data.get("numberMatched")
)

# ============================================================
# DISPLAY OBSERVATION DATA
# ============================================================

features = data.get(
    "features",
    []
)

for i, feature in enumerate(
    features,
    start=1
):

    print()
    print("-" * 70)
    print(
        "OBSERVATION",
        i
    )
    print("-" * 70)

    print()
    print(
        "Station:",
        feature.get("properties", {}).get("name")
    )

    print(
        "Station ID:",
        feature.get("properties", {}).get(
            "wigos_station_identifier"
        )
    )

    print(
        "Observation time:",
        feature.get("properties", {}).get(
            "phenomenonTime"
        )
    )

    print()
    print("All properties:")

    properties = feature.get(
        "properties",
        {}
    )

    for key, value in properties.items():

        print(
            f"{key}: {value}"
        )

print()
print("=" * 70)
print("STEP 16B TEST COMPLETE")
print("=" * 70)