import os
import json
import requests

BASE = r"C:\Projects\WeatherModelSelector"
OUTPUT = os.path.join(BASE, "data", "observations")
os.makedirs(OUTPUT, exist_ok=True)

URLS = {
    "maharashtra_aws": "https://api.imd.gov.in/api/v1/aws_data?sid=21",
    "aws_station_mapping": "https://api.imd.gov.in/api/v1/aws_data_mapping"
}

for name, url in URLS.items():
    print("\n" + "=" * 60)
    print("SOURCE:", name)
    print("=" * 60)

    try:
        response = requests.get(
            url,
            headers={
                "User-Agent": "WeatherModelSelector/1.0",
                "Accept": "application/json"
            },
            timeout=45
        )

        print("HTTP status:", response.status_code)
        print("Content type:", response.headers.get("Content-Type"))

        response.raise_for_status()
        data = response.json()

        print("Response type:", type(data).__name__)

        if isinstance(data, list):
            print("Total records:", len(data))
            sample = data[:2]
        elif isinstance(data, dict):
            print("Top-level keys:", list(data.keys())[:20])
            sample = {
                k: (f"{len(v)} records" if isinstance(v, list) else v)
                for k, v in list(data.items())[:10]
            }
        else:
            sample = str(data)[:1000]

        print("Sample:")
        print(json.dumps(sample, indent=2, ensure_ascii=False, default=str)[:3000])

        path = os.path.join(OUTPUT, name + ".json")

        with open(path, "w", encoding="utf-8") as file:
            json.dump(data, file, ensure_ascii=False, indent=2)

        print("Saved:", path)

    except Exception as error:
        print("Request failed:", error)

print("\nIMD SOURCE CHECK COMPLETED")
