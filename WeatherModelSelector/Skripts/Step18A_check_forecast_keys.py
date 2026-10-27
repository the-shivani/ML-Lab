import requests
import json
import urllib3

urllib3.disable_warnings(
    urllib3.exceptions.InsecureRequestWarning
)

# ============================================================
# STEP 18A - CHECK OPEN-METEO FORECAST KEYS
# ============================================================

latitude = 25.5941
longitude = 85.1376

url = "https://api.open-meteo.com/v1/forecast"

params = {
    "latitude": latitude,
    "longitude": longitude,

    "hourly": (
        "temperature_2m,"
        "relative_humidity_2m,"
        "wind_speed_10m,"
        "pressure_msl"
    ),

    "models": (
        "ecmwf_ifs,"
        "gfs_seamless,"
        "icon_seamless,"
        "gem_seamless"
    ),

    "forecast_days": 1,

    "timezone": "UTC"
}


print("=" * 70)
print("STEP 18A - CHECK OPEN-METEO FORECAST KEYS")
print("=" * 70)

response = requests.get(
    url,
    params=params,
    timeout=60,
    verify=False
)

print()
print("HTTP Status:", response.status_code)

data = response.json()

print()
print("HOURLY KEYS:")
print("=" * 70)

for key in data["hourly"].keys():

    print(key)


print()
print("=" * 70)
print("FIRST VALUE OF EACH KEY")
print("=" * 70)

for key, values in data["hourly"].items():

    if key != "time":

        print(
            f"{key:45} -> {values[0]}"
        )

print()
print("=" * 70)