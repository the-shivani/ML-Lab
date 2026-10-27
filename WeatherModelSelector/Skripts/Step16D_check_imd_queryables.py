import requests
import urllib3

# ============================================================
# STEP 16D - CHECK IMD OBSERVATION QUERYABLES
# ============================================================

urllib3.disable_warnings(
    urllib3.exceptions.InsecureRequestWarning
)

URL = (
    "https://wis2box.imd.gov.in/"
    "oapi/collections/"
    "urn:wmo:md:in-imd:"
    "surface-based-observations.synop/"
    "queryables"
)

print("=" * 70)
print("STEP 16D - CHECK IMD OBSERVATION QUERYABLES")
print("=" * 70)

print()
print("Requesting IMD queryable fields...")

response = requests.get(
    URL,
    params={"f": "json"},
    timeout=(15, 45),
    verify=False
)

print()
print("HTTP Status:", response.status_code)

response.raise_for_status()

data = response.json()

print()
print("Available queryable fields:")
print("-" * 70)

properties = data.get(
    "properties",
    {}
)

for name, details in properties.items():

    print()
    print("FIELD:", name)

    if isinstance(details, dict):

        print(
            "  Type:",
            details.get("type")
        )

        if "title" in details:
            print(
                "  Title:",
                details.get("title")
            )

        if "description" in details:
            print(
                "  Description:",
                details.get("description")
            )

print()
print("=" * 70)
print("STEP 16D COMPLETE")
print("=" * 70)