import requests
import urllib3

# ============================================================
# STEP 16E - TEST OGC FILTER FOR IMD STATION
# ============================================================

urllib3.disable_warnings(
    urllib3.exceptions.InsecureRequestWarning
)

URL = (
    "https://wis2box.imd.gov.in/"
    "oapi/collections/"
    "urn:wmo:md:in-imd:"
    "surface-based-observations.synop/items"
)

STATION_ID = "0-20000-0-43067"

print("=" * 70)
print("STEP 16E - TEST OGC FILTER FOR IMD STATION")
print("=" * 70)

print()
print("Testing station:")
print(STATION_ID)

# ============================================================
# OGC FILTER
# ============================================================

params = {
    "f": "json",
    "limit": 20,
    "filter":
        f"wigos_station_identifier='{STATION_ID}'",
    "filter-lang": "cql-text"
}

print()
print("Sending filtered request...")

response = requests.get(
    URL,
    params=params,
    timeout=(15, 45),
    verify=False
)

print()
print("HTTP Status:", response.status_code)

print()
print("Request URL:")
print(response.url)

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
    print("OBSERVATION", i)
    print("-" * 70)

    print(
        "Name:",
        properties.get("name")
    )

    print(
        "Station:",
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
print("STEP 16E COMPLETE")
print("=" * 70)