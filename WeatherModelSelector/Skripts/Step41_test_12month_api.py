import requests

url = "https://previous-runs-api.open-meteo.com/v1/forecast"

params = {
    "latitude": 18.5204,
    "longitude": 73.8567,
    "start_date": "2025-10-01",
    "end_date": "2025-10-03",
    "hourly": "temperature_2m_previous_day1",
    "models": "ecmwf_ifs,gfs_seamless,icon_seamless,gem_seamless",
    "timezone": "UTC"
}

print("Testing Open-Meteo historical forecast API...")
print()

response = requests.get(url, params=params, timeout=60)

print("Status code:", response.status_code)

if response.status_code == 200:
    data = response.json()

    print("API test successful!")
    print("Hourly fields:", list(data.get("hourly", {}).keys()))
    print("Number of timestamps:",
          len(data.get("hourly", {}).get("time", [])))
else:
    print("API request failed.")
    print(response.text[:1000])
