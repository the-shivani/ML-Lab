import requests
import urllib3
from datetime import datetime, timedelta, timezone

# ============================================================
# STEP 16G - TEST IMD TIME FILTER
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
print("STEP 16G - TEST IMD TIME FILTER")
print("=" * 70)

# ============================================================
# CREATE RECENT TIME WINDOW
# ============================================================

now = datetime.now(timezone.utc)

start_time = now - timedelta(hours=6)

start = start_time.strftime(
    "%Y-%m-%dT%H:%M:%SZ"
)

end = now.strftime(
    "%Y-%m-%dT%H:%M:%SZ"
)

print()
print("Time window:")
print("Start:", start)
print("End  :", end)

# ============================================================
# REQUEST
# ============================================================

params = {
    "f": "json",
    "limit": 100,
    "datetime": f"{start}/{end}"
}

print()
print("Requesting recent IMD observations...")

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
# DISPLAY FIRST 20
# ============================================================

for i, feature in enumerate(
    features[:20],
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
print("STEP 16G COMPLETE")
print("=" * 70)