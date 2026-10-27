import requests
import urllib3

# ============================================================
# STEP 16F - TEST LATEST IMD OBSERVATIONS
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

print("=" * 70)
print("STEP 16F - TEST LATEST IMD OBSERVATIONS")
print("=" * 70)

print()
print("Requesting latest observations...")

params = {
    "f": "json",
    "limit": 20,
    "sortby": "-phenomenonTime"
}

response = requests.get(
    URL,
    params=params,
    timeout=(15, 60),
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
# DISPLAY
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
print("STEP 16F COMPLETE")
print("=" * 70)